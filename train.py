
'''
deepspeed --num_gpus 1 train.py
'''
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import deepspeed
import json
from torch.utils.data import Dataset, DataLoader
from deepspeed import comm as ds_dist

from src.model import EvoFill
from src.utils import load_config, export_config
from tqdm import tqdm

# 数据集类
class GeneticDataset(Dataset):
    def __init__(self,  data_path, n_var, site_overlap=0, is_train=True, cfg=None):

        # 使用内存映射方式加载数据
        self.data = torch.load(data_path, map_location='cpu')
        self.var_sites = self.data['var_site']          # (samples, var_sites)
        self.n_samples, self.total_sites = self.var_sites.shape
        self.n_var, self.site_overlap, self.is_train = n_var, site_overlap, is_train

        if bool(cfg.train.use_fst):
            fst_df       = pd.read_csv(cfg.data.fst_mat, index_col=0)
            pop2idx      = {p:i for i,p in enumerate(fst_df.index)}
            fst_np       = fst_df.values.astype(np.float32)   # (P,P)

            # 样本 → 群体
            sample_ids = self.data['sample_id']           # list of str
            pop_ids    = [s.split('_')[0] for s in sample_ids]
            pop_idx    = [pop2idx[p] for p in pop_ids]    # (N,)
            # 查表得样本间距离
            self.p_dis = torch.from_numpy(
                fst_np[np.ix_(pop_idx, pop_idx)]          # (N,N)
            )
        else:
            self.p_dis = None

        # 预计算所有位点组
        self.site_groups = self._precompute_site_groups()
        self.total_items = self.n_samples * len(self.site_groups)
    
    def _precompute_site_groups(self):
        """预计算位点分组"""
        site_groups = []
        start = 0
        end = 0
        while end < self.total_sites:
            end = min(start + self.n_var, self.total_sites)
            site_groups.append((start, end))
            if self.is_train:
                start = end - self.site_overlap
            else:
                start = end
        return site_groups
    
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        # 使用更高效的数据访问方式
        sample_idx = idx // len(self.site_groups)
        group_idx = idx % len(self.site_groups)
        start, end = self.site_groups[group_idx]
        
        # 直接索引，避免不必要的复制
        var_site = self.var_sites[sample_idx, start:end]  # 去掉 .clone()
        var_site = torch.cat([var_site, torch.zeros(self.n_var - len(var_site))]).long() 
        
        return {
            'var_site': var_site,
            'sample_idx': sample_idx,
            'site_range': (start, end)
        }

    def get_sample_indices(self):
        """返回所有样本的索引"""
        return torch.arange(self.n_samples)


class GeneticSubset(Dataset):
    def __init__(self, original_dataset, sample_indices):
        self.n_var   = original_dataset.n_var
        self.site_overlap = original_dataset.site_overlap
        self.is_train = original_dataset.is_train
        self.site_groups  = original_dataset.site_groups
        self.total_sites  = original_dataset.total_sites

        # 截取
        self.var_sites = original_dataset.var_sites[sample_indices].clone()
        if original_dataset.p_dis is not None:
            self.p_dis = original_dataset.p_dis[sample_indices][:, sample_indices].clone()
        else:
            self.p_dis = None

        self.n_samples = len(sample_indices)
        self.total_items = self.n_samples * len(self.site_groups)
        self.sample_mapping = sample_indices
    
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        sample_idx = idx // len(self.site_groups)
        group_idx = idx % len(self.site_groups)
        start, end = self.site_groups[group_idx]
        
        # 直接从当前数据集的 var_sites 获取数据
        var_site = self.var_sites[sample_idx, start:end].clone()
        
        # 填充到固定长度
        if var_site.shape[0] < self.n_var:
            padding = torch.zeros(self.n_var - var_site.shape[0], dtype=torch.long)
            var_site = torch.cat([var_site, padding])
        
        return {
            'var_site': var_site,
            'sample_idx': sample_idx,  # 这里使用在新数据集中的索引
            'site_range': (start, end)
        }

# --------------------------------------------------
# 数据加载器包装类
# --------------------------------------------------
class GeneticDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 创建内部的 DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def __iter__(self):
        # 返回 DataLoader 的迭代器
        return iter(self.dataloader)
    
    def __len__(self):
        """返回总批次数"""
        return len(self.dataloader)
    
    def collate_fn(self, batch):
        var_sites = torch.stack([item['var_site'] for item in batch]).long()
        sample_indices = torch.tensor([item['sample_idx'] for item in batch])
        site_ranges = [item['site_range'] for item in batch]

        if self.dataset.p_dis is not None:
            dist_mat = self.dataset.p_dis[sample_indices[:, None], sample_indices[None, :]].float()
        else:
            dist_mat = None          # 模型已支持

        return {
            'var_sites': var_sites,
            'dist_mat':  dist_mat,
            'sample_indices': sample_indices,
            'site_ranges': site_ranges
        }

# --------------------------------------------------
# 准确度计算函数
# --------------------------------------------------
def _make_valid_mask(site_ranges, L, device):
    """
    返回 (B, L) 的 bool mask：True 表示该位点在 site_ranges 内。
    """
    starts, ends = site_ranges[:, 0:1], site_ranges[:, 1:2]   # (B,1)
    pos = torch.arange(L, device=device).view(1, L)           # (1,L)
    return (pos >= starts) & (pos < ends)                     # (B,L)

@torch.no_grad()
def masked_accuracy(predictions, targets, site_ranges):
    """
    返回 batch 内所有有效位点的 top-1 平均准确率（scalar Tensor）。
    """
    B, L, D = predictions.shape
    device  = predictions.device
    mask = _make_valid_mask(site_ranges, L, device)           # (B,L)

    logits_valid = predictions[mask]                          # (N_valid, D)
    tgt_valid    = targets[mask]                              # (N_valid,)

    pred_class = logits_valid.argmax(dim=-1)
    correct    = (pred_class == tgt_valid).sum().float()
    total      = torch.tensor(tgt_valid.numel(), device=device)

    return correct / total.clamp(min=1)

def masked_cross_entropy(predictions, targets, site_ranges):
    """
    predictions : (B, L, D)  logits
    targets     : (B, L)     int64
    site_ranges : (B, 2)     int64  [[start, end], ...]
    return      : scalar Tensor
    """
    B, L, D = predictions.shape
    device  = predictions.device
    mask = _make_valid_mask(site_ranges, L, device)           # (B,L)

    # 仅保留有效位点的 logits/target
    logits_valid = predictions[mask]                          # (N_valid, D)
    tgt_valid    = targets[mask]                              # (N_valid,)

    # 计算 CE（reduction='sum' 后再除以总数 → 得到平均）
    loss = F.cross_entropy(logits_valid, tgt_valid, reduction='sum')
    return loss / mask.sum().clamp(min=1)

def masked_imputation_loss(predictions, targets, site_ranges, use_r2=True, group_size=4):
    """
    predictions: (B, L, D) logits
    targets:     (B, L) int64
    site_ranges: (B, 2) int64
    """
    B, L, D = predictions.shape
    device = predictions.device
    mask = _make_valid_mask(site_ranges, L, device)  # (B, L)

    logits_valid = predictions[mask]  # (N_valid, D)
    tgt_valid    = targets[mask]      # (N_valid,)

    # 1. CE loss
    ce_loss = F.cross_entropy(logits_valid, tgt_valid, reduction='sum')

    # 2. KL loss
    log_probs = F.log_softmax(logits_valid, dim=-1)
    # probs = F.softmax(logits_valid, dim=-1)
    tgt_onehot = F.one_hot(tgt_valid, num_classes=D).float()
    kl_loss = F.kl_div(log_probs, tgt_onehot, reduction='sum')

    total_loss = ce_loss + kl_loss

    # 3. Optional R² loss (Minimac-style)
    if use_r2:
        # 只支持 biallelic (D == 2)
        assert D == 2, "R² loss only supports biallelic variants"
        with torch.no_grad():
            # 分组计算 alt allele frequency
            n_full = len(tgt_valid) // group_size * group_size
            tgt_grouped = tgt_valid[:n_full].view(-1, group_size)  # (G, group_size)
            pred_probs = torch.softmax(logits_valid[:n_full], dim=-1)[:, 1]  # alt allele prob
            pred_grouped = pred_probs.view(-1, group_size)

            # 真实频率
            gt_af = tgt_grouped.float().mean(dim=1)  # (G,)
            # 预测频率
            pred_af = pred_grouped.mean(dim=1)

            # R² 损失
            denom = gt_af * (1 - gt_af)
            denom = torch.clamp(denom, min=0.01)
            r2_error = torch.square(pred_af - gt_af) / denom
            r2_loss = -r2_error.sum() * group_size  # 负号是因为我们要最小化负R²

        total_loss += r2_loss

    return total_loss / mask.sum().clamp(min=1)

# --------------------------------------------------
# 验证函数
# --------------------------------------------------
def validate_model(model, dataset, batch_size, device, mask_ratio=0.2, desc="Validation"):
    model.eval()
    total_accuracy = 0.0
    total_batches = 0
    
    # 使用包装的数据加载器
    val_loader = GeneticDataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    with torch.no_grad():
        for batch in (tqdm(val_loader, desc=desc, bar_format='{l_bar}{bar:20}{r_bar}') if ds_dist.get_rank() == 0 else val_loader):
            var_sites = batch['var_sites'].to(device)
            dist_mat = batch['dist_mat'].to(device) if batch['dist_mat'] is not None else None
            site_ranges = torch.tensor(
                [[s, e] for s, e in batch['site_ranges']], 
                dtype=torch.long, 
                device=device
            )
            
            batch_size, n_var = var_sites.shape
            mask = torch.rand(batch_size, n_var, device=device) < mask_ratio
            masked_input = var_sites.clone()
            masked_input[mask] = model.depth          # last one = missing 
            
            predictions = model(masked_input, dist_mat)
            accuracy = masked_accuracy(predictions, var_sites, site_ranges)
            total_accuracy += accuracy
            total_batches += 1

    ds_dist.barrier()  
    return total_accuracy / total_batches if total_batches > 0 else 0.0


# --------------------------------------------------
# 主训练函数
# --------------------------------------------------
def print_rank0(*args, **kwargs):
    """只在 rank0 打印信息"""
    if ds_dist.get_rank() == 0:
        print(*args, **kwargs)

def main():
    torch.manual_seed(42)          # CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # 加载配置
    cfg = load_config("config/config.json")
    
    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    # 加载并更新 DeepSpeed 配置
    with open("config/ds_config.json", 'r') as f:
        ds_config = json.load(f)

    ds_config['train_batch_size'] = ds_config['train_micro_batch_size_per_gpu'] * ds_config['gradient_accumulation_steps'] * num_gpus
    cfg.train.batch_size = ds_config['train_batch_size']
    
    # 加载数据
    var_index = torch.load(os.path.join(cfg.data.out_dir, "var_index.pt"))
    
    # 从 var_index 自动识别 depth
    depth = int(torch.max(var_index).item()) + 1  # add missing
    print(f"Automatically detected depth: {depth}")
    
    # 创建数据集
    train_dataset = GeneticDataset(
        os.path.join(cfg.data.out_dir, "train.pt"), 
        cfg.train.n_var, 
        cfg.train.site_overlap, 
        is_train=True,
        cfg=cfg
    )
    val_dataset = GeneticDataset(
        os.path.join(cfg.data.out_dir, "val.pt"), 
        cfg.train.n_var, 
        site_overlap=0,
        is_train=False,
        cfg=cfg
    )
    
    # 创建固定测试样本集
    g = torch.Generator().manual_seed(2024)
    train4test_sample_indices = torch.randperm(train_dataset.n_samples, generator=g)[:cfg.train.train4test]
    train4test_dataset = GeneticSubset(train_dataset, train4test_sample_indices)
    
    # 初始化模型
    model = EvoFill(
        depth=depth,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout
    )

    # 初始化 DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    var_index = var_index.to(model_engine.device) 

    if ds_dist.get_rank() == 0:
        os.makedirs(cfg.train.save_dir, exist_ok=True)
        os.makedirs(cfg.train.logs_dir, exist_ok=True)
        logs_path = Path(cfg.train.logs_dir, "train_loss.csv")
        # 第一次写时写表头
        if not os.path.exists(logs_path):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            logs_path = Path(cfg.train.logs_dir) / f"{timestamp}.csv"
            with open(logs_path, "w", newline="") as f:
                w = csv.writer(f, delimiter=",")
                w.writerow([f"# config.json={export_config(cfg, separators=(',', ':'))}"])
                w.writerow([f"# ds_config.json={json.dumps(ds_config, separators=(',', ':'))}"])
                w.writerow(["epoch","batch", "train_loss", "train_acc", "train4test_acc", "val_acc"])
    ds_dist.barrier()          # 等 rank0 建完目录
        
    # 使用包装的数据加载器
    train_loader = GeneticDataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader_workers,
        pin_memory=True  # 确保启用
    )

    # 初始化早停
    if cfg.train.early_stop.patience <= 0:
        cfg.train.early_stop.patience = float('inf')  # 永不早停
    best_val_acc = 0.0
    patience_counter = 0
    
    # 计算总批次数
    total_batches = len(train_loader)
    print_rank0(f"Total training batches per epoch: {total_batches}")
    
    try:
    # 训练循环
        for epoch in range(cfg.train.epochs):
            model_engine.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            processed_batches = 0
            
            # 只在 rank0 创建进度条
            if ds_dist.get_rank() == 0:
                progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{cfg.train.epochs}",bar_format='{l_bar}{bar:20}{r_bar}')
            else:
                progress_bar = None
            
            for batch_idx, batch in enumerate(train_loader):
                var_sites = batch['var_sites'].to(model_engine.device)
                dist_mat = batch['dist_mat'].to(model_engine.device) if batch['dist_mat'] is not None else None
                site_ranges = torch.tensor(
                    [[s, e] for s, e in batch['site_ranges']], 
                    dtype=torch.long, 
                    device=model_engine.device
                )
                
                # 确保数据类型正确
                if var_sites.dtype != torch.long:
                    var_sites = var_sites.long()
                
                mask = _make_valid_mask(site_ranges, var_sites.size(1), var_sites.device)
                if mask.sum() == 0:           # 本 batch 无有效位点
                    if ds_dist.get_rank() == 0:
                        progress_bar.update(1)
                    continue                  # 直接跳过本轮
                # 前向传播
                predictions = model_engine(var_sites, dist_mat)
                
                # 计算损失
                # loss = masked_cross_entropy(predictions, var_sites, site_ranges)
                loss = masked_imputation_loss(predictions, var_sites, site_ranges, use_r2=False)

                # 反向传播
                model_engine.backward(loss)
                model_engine.step()
                
                # 计算准确度
                accuracy = masked_accuracy(predictions, var_sites, site_ranges)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                processed_batches += 1
                
                # 只在 rank0 更新进度条
                if ds_dist.get_rank() == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy:.4f}'
                    })
                    with open(logs_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [epoch + 1, batch_idx+1, loss.item(), accuracy, None, None]
                        )
                    progress_bar.update(1)
            
            # 关闭进度条
            if ds_dist.get_rank() == 0:
                progress_bar.close()
            
            # 计算epoch统计
            avg_loss = epoch_loss / processed_batches
            avg_accuracy = epoch_accuracy / processed_batches
            
            print_rank0(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

            if ds_dist.get_rank() == 0:
                with open(logs_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [epoch + 1, "average", avg_loss, avg_accuracy, None, None]
                    )
            
            # ---------------- 验证 & 早停 ----------------
            if (epoch + 1) % cfg.train.val_interval == 0:
                train_accuracy = validate_model(
                    model_engine, train4test_dataset, cfg.train.batch_size,
                    model_engine.device, cfg.train.mask_ratio, "Train Validation"
                )

                val_accuracy = validate_model(
                    model_engine, val_dataset, cfg.train.batch_size,
                    model_engine.device, cfg.train.mask_ratio, "Val Validation"
                )

                print_rank0(f"Train Mask Accuracy: {train_accuracy:.4f}, Val Mask Accuracy: {val_accuracy:.4f}")
                if ds_dist.get_rank() == 0:
                    with open(logs_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [epoch + 1, "average", avg_loss, avg_accuracy, train_accuracy, val_accuracy]
                        )
                
                # ====== 早停逻辑（仅在 rank0 进行） ======
                save_best = torch.tensor(0, dtype=torch.long, device=model_engine.device)   # 1=需要保存
                stop_flag = torch.tensor(0, dtype=torch.long, device=model_engine.device)   # 1=需要停止
                if ds_dist.get_rank() == 0:
                    if val_accuracy > best_val_acc + cfg.train.early_stop.min_delta:
                        best_val_acc = val_accuracy
                        patience_counter = 0
                        save_best.fill_(1)                     # 通知所有 rank 保存 best
                    else:
                        patience_counter += 1

                    if patience_counter >= cfg.train.early_stop.patience:
                        print_rank0(f"Early stopping at epoch {epoch + 1}")
                        stop_flag.fill_(1)

                # 2. 广播决定
                ds_dist.broadcast(save_best, 0)
                ds_dist.broadcast(stop_flag, 0)

                # 3. 所有 rank 一起行动
                if save_best.item():
                    # 这里每个 rank 都会执行 save_checkpoint，内部会同步
                    model_engine.save_checkpoint(cfg.train.save_dir, tag="best")

                if stop_flag.item():
                    break

            # 保存检查点
            if (epoch + 1) % cfg.train.save_interval == 0:
                model_engine.save_checkpoint(
                    cfg.train.save_dir, 
                    tag=f"epoch_{epoch+1}"
                )
        
    finally:
        ds_dist.barrier()
        if ds_dist.get_rank() == 0:
            print("Training finished.")
        # 在退出前显式销毁进程组
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()