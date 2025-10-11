# OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py
# tensorboard --logdir ckpt/tensorboard --port 6066

import os
import shutil
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter   # 新增
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import random
from itertools import chain
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch_optimizer import Lamb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from src.utils import load_config
from src.model import EvoFill
from src.losses import ImputationLoss

MAF_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
        (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]

def unwrap(model):
    """返回真实模型，不管外面包了几层 DDP / FSDP."""
    return model.module if hasattr(model, 'module') else model

class GenotypeDataset(Dataset):
    def __init__(self, gts, coords, mask_int=None, mask_ratio=0.0):
        """
        gts: (N, L, A)  one-hot  float  —— 原始完整标签（含缺失向量）
        coords: (L, 4)
        mask_int: 缺失对应的类别下标，None 则默认为最后一维
        mask_ratio: 训练时额外随机遮掩的比例
        """
        self.gt_true = gts.float() 
        self.coords  = coords.float()
        self.mask_ratio = mask_ratio

        if mask_int is None:
            mask_int = self.gt_true.shape[-1] - 1
        self.mask_int = mask_int

    def __len__(self):
        return self.gt_true.shape[0]

    def __getitem__(self, idx):
        gt_true = self.gt_true[idx]                # (L, A)  只读
        gt_mask = gt_true.clone()                  # (L, A)  可修改

        # ---- 仅对 gt_mask 做随机遮掩 ----
        if self.mask_ratio > 0:
            L = gt_true.shape[0]
            rand_mask = torch.rand(L, device=gt_mask.device) < self.mask_ratio
            gt_mask[rand_mask] = 0.                    # 全 0
            gt_mask[rand_mask, self.mask_int] = 1.     # 缺失类别置 1

        return gt_mask, gt_true, self.coords


def collate_fn(batch):
    """
    batch: List[(gt_mask, gt_true, coords)] 每个 coords 形状相同
    返回：gt_mask, gt_true, coords（二维，直接取第 0 个即可
    """
    gt_mask  = torch.stack([b[0] for b in batch], 0)
    gt_true  = torch.stack([b[1] for b in batch], 0)
    coords   = batch[0][2]          # 全局共享
    return gt_mask, gt_true, coords


def build_loader(pt_path, batch_size, shuffle, mask_int, mask_ratio, sampler=None):
    data = torch.load(pt_path)
    dataset = GenotypeDataset(data['gts'], data['coords'], mask_int=mask_int, mask_ratio=mask_ratio)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),  # 有 sampler 时关闭 shuffle
        drop_last=shuffle,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

def precompute_maf(gts_oh, mask_int=None):
    """
    gts_oh: (N, L, A)  one-hot  float32/64
    mask_int: 缺失对应的类别下标，None 则默认为最后一维
    return:
        maf: (L,) float32
        bin_cnt: list[int] 长度 6
    """
    if mask_int is None:
        mask_int = gts_oh.shape[-1] - 1

    # 从 one-hot 反推类别索引
    gts_idx = gts_oh.argmax(-1).numpy()  # (N, L)  int

    N, L = gts_idx.shape
    maf = np.zeros(L, dtype=np.float32)
    bin_cnt = [0] * 6

    for l in range(L):
        alleles = gts_idx[:, l]
        alleles = alleles[alleles != mask_int]  # 去掉缺失
        if alleles.size == 0:
            maf[l] = 0.0
            continue

        uniq, cnt = np.unique(alleles, return_counts=True)
        total = cnt.sum()
        freq = cnt / total
        freq[::-1].sort()  # 降序
        maf_val = freq[1] if len(freq) > 1 else 0.0
        maf[l] = maf_val

        # 统计 bin
        for i, (lo, hi) in enumerate(MAF_BINS):
            if lo <= maf_val < hi:
                bin_cnt[i] += 1
                break

    return torch.from_numpy(maf), bin_cnt

def imputation_maf_accuracy_epoch(all_logits, all_gts_oh, all_mask, global_maf):
    """
    all_logits: (N, L, A)
    all_gts_oh: (N, L, A)  one-hot
    all_mask:   (N, L)     True 表示该位点被遮掩
    global_maf: (L,)       已预计算好
    """
    preds = torch.argmax(all_logits, dim=-1)        # (N, L)
    gts   = torch.argmax(all_gts_oh, dim=-1)        # (N, L)
    correct = (preds == gts) & all_mask             # (N, L)
    maf = global_maf.unsqueeze(0)                   # (1, L)

    accs = []
    for lo, hi in MAF_BINS:
        bin_mask = all_mask & (maf >= lo) & (maf < hi)
        n_cor = (correct & bin_mask).sum()
        n_tot = bin_mask.sum()
        accs.append((n_cor / n_tot).item() if n_tot > 0 else 0.0)
    return accs

def train_epoch(model, cfg, loader, criterion, optimizer, device, rank,
                global_maf, writer=None, global_step=0):
    model.train()
    total_loss = 0.0

    # 1. 先正常训练，不计算 MAF
    pbar = tqdm(loader, ncols=80, disable=(rank != 0), leave=False, desc='1/3')
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)

        logits = model(gt_mask, coords)
        loss, logs = criterion(logits, gt_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if writer:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            writer.add_scalar("Model/global_L2", grad_norm, global_step)
            # 3) 学习率（从优化器里取）
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Model/lr", lr, global_step)
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/CE", logs['ce'], global_step)
            writer.add_scalar("Train/KL", logs['kl'], global_step)
            writer.add_scalar("Train/R2", logs['r2'], global_step)
            global_step += 1

    # 2. 训练结束后，统一跑一次 forward（no_grad）收集全部结果
    model.eval()          # 训练结束后统一 forward
    all_logits, all_gts, all_mask = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, ncols=80, disable=(rank != 0), leave=False, desc='2/3')
        for gt_mask, gt_true, coords in pbar:
            gt_mask, gt_true = gt_mask.to(device), gt_true.to(device)
            logits = model(gt_mask, coords) 
            
            mask = gt_mask[..., model.mask_int] == 1 
            all_logits.append(logits.cpu())
            all_gts.append(gt_true.cpu())
            all_mask.append(mask.cpu())
    
    # 拼大矩阵
    all_logits = torch.cat(all_logits, 0)   # (N, L, A)
    all_gts    = torch.cat(all_gts, 0)      # (N, L)
    all_mask   = torch.cat(all_mask, 0)     # (N, L)
    # 直接用预计算好的 global_maf
    maf_accs = imputation_maf_accuracy_epoch(all_logits, all_gts, all_mask,
                                             global_maf.cpu())

    return {"loss": total_loss / len(loader),
            "maf_accs": maf_accs,
            "step": global_step}


@torch.no_grad()
def validate(model, cfg, loader, criterion, device, rank, global_maf, writer=None, global_step=0):
    model.eval()
    total_loss = 0.0

    all_logits, all_gts, all_mask = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, ncols=80, disable=(rank != 0), leave=False, desc='3/3')
        for gt_mask, gt_true, coords in pbar:
            gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)
            logits = model(gt_mask, coords)
            loss, logs = criterion(logits, gt_true)
            total_loss += loss.item()

            mask = gt_mask[..., model.mask_int] == 1 
            all_logits.append(logits.cpu())
            all_gts.append(gt_true.cpu())
            all_mask.append(mask.cpu())

            if writer:
                writer.add_scalar("Val/Loss", loss.item(), global_step)
                writer.add_scalar("Val/CE", logs['ce'], global_step)
                writer.add_scalar("Val/KL", logs['kl'], global_step)
                writer.add_scalar("Val/R2", logs['r2'], global_step)
                global_step += 1

    # 拼矩阵
    all_logits = torch.cat(all_logits, dim=0)
    all_gts    = torch.cat(all_gts, dim=0)
    all_mask   = torch.cat(all_mask, dim=0)
    coords     = coords.cpu()

    maf_accs = imputation_maf_accuracy_epoch(all_logits, all_gts, all_mask,
                                             global_maf.cpu())

    return {"loss": total_loss / len(loader),
            "maf_accs": maf_accs,
            "step": global_step}


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, mode='min', save_dir='./ckpt', rank=0):
        self.rank = rank
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.best_state = None
        self.save_dir = Path(save_dir)

    def __call__(self, metric, model):
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.rank == 0:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_state, self.save_dir / "evofill_best.pt")
                print(f"Model weights updated: {self.save_dir / 'evofill_best.pt'}")
            return False

        better = (metric < self.best - self.min_delta) if self.mode == 'min' else \
                 (metric > self.best + self.min_delta)

        if better:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.rank == 0:
                torch.save(self.best_state, self.save_dir / "evofill_best.pt")
                print(f"Model weights updated: {self.save_dir / 'evofill_best.pt'}")
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def main():
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1. 配置 & 随机种子
    cfg = load_config("config/config.json")
    seed = cfg.train.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. 数据 loader（训练集加 DistributedSampler）
    train_data = torch.load(Path(cfg.data.path) / "train.pt")
    global_maf, bin_cnt = precompute_maf(train_data['gts'].numpy(), mask_int=-1)
    if rank == 0:
        print('Train MAF-bin counts:', bin_cnt)

    train_sampler = DistributedSampler(
        GenotypeDataset(torch.load(Path(cfg.data.path) / "train.pt")['gts'],
                        torch.load(Path(cfg.data.path) / "train.pt")['coords'],
                        mask_int = -1,
                        mask_ratio=cfg.train.mask_ratio),
        shuffle=True,
    )
    train_loader = build_loader(
        Path(cfg.data.path) / "train.pt",
        batch_size=cfg.train.batch_size,
        shuffle=False,  # sampler 控制
        mask_int = -1,
        mask_ratio=cfg.train.mask_ratio,
        sampler=train_sampler,
    )
    val_loader = build_loader(
        Path(cfg.data.path) / "val.pt",
        batch_size=cfg.train.batch_size,
        shuffle=False,
        mask_int = -1,
        mask_ratio=cfg.train.mask_ratio,
        sampler=None,
    )

    # 3. 模型 -> DDP
    model = EvoFill(**vars(cfg.model)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ===== 新增：TensorBoard =====
    writer = None
    if rank == 0:
        tb_dir = Path(cfg.train.save) / "tensorboard" /  datetime.now().strftime("%m-%d %H:%M")
        # if tb_dir.exists():          # 目录已存在
        #     shutil.rmtree(tb_dir)    # 连目录带内容一起删掉
        tb_dir.mkdir(parents=True)   # 重新建一个空目录
        writer = SummaryWriter(tb_dir)

    # 4. 损失 & 优化器
    criterion = ImputationLoss(use_r2=cfg.train.use_r2_loss).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # optimizer = Lamb(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.train.scheduler_factor, patience=cfg.train.scheduler_patience, min_lr=cfg.train.min_lr)

    # 5. 早停
    save_dir = Path(cfg.train.save)
    early_stopper = EarlyStopper(
        patience=cfg.train.earlystop_patience,
        min_delta=cfg.train.min_delta,
        mode='min',
        save_dir=save_dir,
        rank=rank,
    )

    # 6. 训练循环
    train_step = 0
    val_step = 0
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metric = train_epoch(model, 
                                   cfg,
                                   train_loader,
                                   criterion,
                                   optimizer,
                                   device,
                                   rank,
                                   global_maf,
                                   writer=writer,
                                   global_step=train_step)
        train_step = train_metric['step']
        val_metric = validate(model,
                              cfg,
                              val_loader,
                              criterion,
                              device,
                              rank,
                              global_maf,
                              writer=writer,
                              global_step=val_step)
        val_step = val_metric['step']
        if rank == 0:
            print(f"Epoch {epoch:03d}   train loss: {train_metric['loss']:.5f}   val loss: {val_metric['loss']:.5f}")
            mask_acc = pd.DataFrame({
                'MAF_bin': ['(0.00, 0.05)', '(0.05, 0.10)', '(0.10, 0.20)', '(0.20, 0.30)', '(0.30, 0.40)', '(0.40, 0.50)'],
                'Counts': [f"{cts}" for cts in bin_cnt],
                'Train': [f"{acc:.3f}" for acc in train_metric['maf_accs']],
                'Val':   [f"{acc:.3f}" for acc in val_metric['maf_accs']]
            })
            print(mask_acc.to_string(index=False))

        scheduler.step(val_metric['loss'])

        # 同步各卡 val_loss 后再做早停判断（可选）
        val_loss_tensor = torch.tensor(val_metric['loss']).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        if early_stopper(val_loss_tensor.item(), model.module):
            if rank == 0:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    # 7. 结束
    if rank == 0:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()