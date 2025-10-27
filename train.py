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
from src.utils import load_config, precompute_maf, imputation_maf_accuracy_epoch
from src.model import EvoFill
from src.losses import ImputationLoss


def unwrap(model):
    """返回真实模型，不管外面包了几层 DDP / FSDP."""
    return model.module if hasattr(model, 'module') else model

class GenomicDataset(Dataset):
    """Dataset class for genomic data with masking for training"""

    def __init__(self, x_gts, x_extra=None, seq_depth=4,
                 mask=True, masking_rates=(0.5, 0.99)):
        self.gts = x_gts
        self.x_extra = x_extra
        self.seq_depth = seq_depth
        self.mask = mask
        self.masking_rates = masking_rates

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        x       = self.gts[idx].copy()
        y       = self.gts[idx]
        if self.x_extra is not None:
            x_extra = self.x_extra[idx]
        else:
            x_extra = None

        if self.mask:
            # Apply masking
            seq_len = len(x)
            masking_rate = np.random.uniform(*self.masking_rates)
            mask_size = int(seq_len * masking_rate)
            mask_indices = np.random.choice(seq_len, mask_size, replace=False)
            x[mask_indices] = self.seq_depth - 1  # Missing value token

        # Convert to one-hot
        x_onehot = np.eye(self.seq_depth)[x]
        y_onehot = np.eye(self.seq_depth - 1)[y]

        return torch.FloatTensor(x_onehot),torch.FloatTensor(x_extra), torch.FloatTensor(y_onehot)



def train_epoch(model, loader, criterion, optimizer, device, rank,
                global_maf, writer=None, global_step=0):
    model.train()
    total_loss = 0.0

    # 1. 先正常训练，不计算 MAF
    pbar = tqdm(loader, ncols=80, disable=(rank != 0), leave=False, desc='1/3')
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)

        logits = model(gt_mask, coords)
        
        # print('shape: gt_mask',gt_mask.shape,'gt_true',gt_true.shape,'logits', logits.shape)
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
    n_alleles = train_data['gts'].shape[-1]
    global_maf, bin_cnt = precompute_maf(train_data['gts'].numpy(), mask_int=-1)
    if rank == 0:
        print(f'Detected {n_alleles} alleles in training set (including missing label)')
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
        mask_ratio=cfg.train.mask_ratio,
        sampler=train_sampler,
    )
    val_loader = build_loader(
        Path(cfg.data.path) / "val.pt",
        batch_size=cfg.train.batch_size,
        shuffle=False,
        mask_ratio=cfg.train.mask_ratio,
        sampler=None,
    )

    # 3. 模型 -> DDP
    cfg.model.n_alleles = n_alleles
    model = EvoFill(**vars(cfg.model)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

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