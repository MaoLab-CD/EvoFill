
'''
deepspeed --num_gpus 2 train.py
'''
import os
import torch
import torch.nn.functional as F
import deepspeed
import json
from torch.utils.data import Dataset, DataLoader
from deepspeed import comm as ds_dist

from src.model import EvoFill
from src.utils import load_config
from tqdm import tqdm

# --------------------------------------------------
# 数据集类
# --------------------------------------------------
class GeneticDataset(Dataset):
    def __init__(self, data_path, n_var, site_overlap=0, is_train=True):
        """
        优化内存使用的数据集类
        """
        # 使用内存映射方式加载数据
        self.data = torch.load(data_path, map_location='cpu')
        self.var_sites = self.data['var_site']  # (samples, var_sites)
        self.p_dis = self.data['p_dis']  # (samples, samples)
        
        self.n_samples = self.var_sites.shape[0]
        self.total_sites = self.var_sites.shape[1]
        self.n_var = n_var
        self.site_overlap = site_overlap
        self.is_train = is_train
        
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
    """直接从筛选的样本构建新数据集"""
    def __init__(self, original_dataset, sample_indices):
        """
        original_dataset: 原始 GeneticDataset
        sample_indices: 要选择的样本索引列表
        """
        # 复制原始数据集的属性
        self.n_var = original_dataset.n_var
        self.site_overlap = original_dataset.site_overlap
        self.is_train = original_dataset.is_train
        self.site_groups = original_dataset.site_groups
        self.total_sites = original_dataset.total_sites
        
        # 截取选中的样本数据
        self.var_sites = original_dataset.var_sites[sample_indices].clone()
        self.p_dis = original_dataset.p_dis[sample_indices][:, sample_indices].clone()
        
        # 更新样本数量和相关属性
        self.n_samples = len(sample_indices)
        self.total_items = self.n_samples * len(self.site_groups)
        
        # 保存样本映射关系（如果需要）
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
        var_sites = torch.stack([item['var_site'] for item in batch])
        # 确保 var_sites 是 long 类型
        var_sites = var_sites.long()
        
        sample_indices = torch.tensor([item['sample_idx'] for item in batch])
        site_ranges = [item['site_range'] for item in batch]
        
        # 直接使用数据集的距离矩阵
        dist_mat = self.dataset.p_dis[sample_indices[:, None], sample_indices[None, :]]
        dist_mat = dist_mat.float()
        
        return {
            'var_sites': var_sites,
            'dist_mat': dist_mat,
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
        for batch in (tqdm(val_loader, desc=desc) if ds_dist.get_rank() == 0 else val_loader):
            var_sites = batch['var_sites'].to(device)
            dist_mat = batch['dist_mat'].to(device)
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

    cfg.train.batch_size = ds_config['train_micro_batch_size_per_gpu']
    
    # 加载数据
    var_index = torch.load('data/var_index.pt')
    
    # 从 var_index 自动识别 depth
    depth = int(torch.max(var_index).item()) + 1  # add missing
    print(f"Automatically detected depth: {depth}")
    
    # 创建数据集
    train_dataset = GeneticDataset(
        'data/train_randomdis.pt', 
        cfg.train.n_var, 
        cfg.train.site_overlap, 
        is_train=True
    )
    val_dataset = GeneticDataset(
        'data/val_randomdis.pt', 
        cfg.train.n_var, 
        site_overlap=0,
        is_train=False
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
                progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
            else:
                progress_bar = None
            
            for batch_idx, batch in enumerate(train_loader):
                var_sites = batch['var_sites'].to(model_engine.device)
                dist_mat = batch['dist_mat'].to(model_engine.device)
                site_ranges = torch.tensor(
                    [[s, e] for s, e in batch['site_ranges']], 
                    dtype=torch.long, 
                    device=model_engine.device
                )
                
                # 确保数据类型正确
                if var_sites.dtype != torch.long:
                    var_sites = var_sites.long()
                
                weight_dtype = next(model_engine.parameters()).dtype
                if dist_mat.dtype != weight_dtype:
                    dist_mat = dist_mat.to(weight_dtype)
                
                mask = _make_valid_mask(site_ranges, var_sites.size(1), var_sites.device)
                if mask.sum() == 0:           # 本 batch 无有效位点
                    if ds_dist.get_rank() == 0:
                        progress_bar.update(1)
                    continue                  # 直接跳过本轮
                # 前向传播
                predictions = model_engine(var_sites, dist_mat)
                
                # 计算损失
                loss = masked_cross_entropy(predictions, var_sites, site_ranges)

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
                    progress_bar.update(1)
            
            # 关闭进度条
            if ds_dist.get_rank() == 0:
                progress_bar.close()
            
            # 计算epoch统计
            avg_loss = epoch_loss / processed_batches
            avg_accuracy = epoch_accuracy / processed_batches
            
            print_rank0(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
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