# OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter   # 新增
from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.utils import load_config
from src.model import EvoFill
from src.losses import ImputationLoss

import torch.nn.functional as F
from mamba_ssm import Mamba2  # 官方实现


class GenotypeDataset(Dataset):
    def __init__(self, gts, coords, mask_ratio=0.0):
        self.gt_true = gts.long()          # 原始完整标签
        self.coords = coords.float()
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.gt_true.shape[0]

    def __getitem__(self, idx):
        gt_true = self.gt_true[idx]        # 完整标签
        coords = self.coords                # (L, 4)

        # 训练时额外随机遮掩
        gt_mask = gt_true.clone()
        if self.mask_ratio > 0:
            mask = torch.rand_like(gt_mask.float()) < self.mask_ratio
            gt_mask[mask] = -1             # 仅输入被遮掩

        # 返回：输入（含缺失）、原始标签、坐标
        return gt_mask, gt_true, coords 

def collate_fn(batch):
    """
    batch: List[(gt_mask, gt_true, coords)] 每个 coords 形状相同
    返回：gt_mask, gt_true, coords（二维，直接取第 0 个即可
    """
    gt_mask  = torch.stack([b[0] for b in batch], 0)
    gt_true  = torch.stack([b[1] for b in batch], 0)
    coords   = batch[0][2]          # 全局共享
    return gt_mask, gt_true, coords

def build_loader(pt_path, batch_size, shuffle, mask_ratio, sampler=None):
    data = torch.load(pt_path)
    dataset = GenotypeDataset(data['gts'], data['coords'], mask_ratio=mask_ratio)
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

def imputation_accuracy(logits, gts, mask):
    """仅在被 mask 位点计算 accuracy"""
    preds = torch.argmax(logits, dim=-1)  # (B, L)
    correct = (preds == gts) & mask
    return correct.sum().float() / mask.sum().float()

def train_epoch(model, loader, criterion, optimizer, device, rank, writer=None, global_step=0):
    model.train()
    total_loss, total_acc, total_mask = 0.0, 0.0, 0

    shared_params = list(model.module.length_proj.parameters()) + \
                    list(model.module.out_conv.parameters())

    pbar = tqdm(loader, disable=(rank != 0), leave=False)
    for batch_idx, (gt_mask, gt_true, coords) in enumerate(pbar):
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)

        logits = model(gt_mask, coords)
        loss = criterion(logits, gt_true)

        if criterion.use_gn:
            criterion.gn_loss.gradnorm_step(shared_params, retain_graph=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask = gt_mask == -1
        acc = imputation_accuracy(logits, gt_true, mask)
        total_loss += loss.item()
        total_acc += acc.item() * mask.sum().item()
        total_mask += mask.sum().item()

        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
            
        # ===== 新增：每 N 个 batch 记录一次（这里 N=10） =====
        if writer is not None and batch_idx % 10 == 0:
            # 1) 训练 loss（对数坐标更直观）
            writer.add_scalar("Loss/train", loss.item(), global_step)
            # 2) 梯度全局 L2-norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            writer.add_scalar("Grad/global_L2", grad_norm, global_step)
            # 3) 学习率（从优化器里取）
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR/lr", lr, global_step)
            # 4) 状态范数
            add_dt_logger(model, writer, global_step)
            global_step += 1
        # =========================================================

    return total_loss / len(loader), total_acc / total_mask, global_step   


@torch.no_grad()
def validate(model, loader, criterion, device, rank):
    model.eval()
    total_loss, total_acc, total_mask = 0.0, 0.0, 0
    pbar = tqdm(loader, disable=(rank != 0), leave=False, desc='validate')
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)
        logits = model(gt_mask, coords)
        loss = criterion(logits, gt_true)
        mask = gt_mask == -1
        acc = imputation_accuracy(logits, gt_true, mask)
        total_loss += loss.item()
        total_acc += acc.item() * mask.sum().item()
        total_mask += mask.sum().item()
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    return total_loss / len(loader), total_acc / (total_mask + 1e-8)


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
                print(f"Best model updated: {self.save_dir / 'evofill_best.pt'}")
            return False

        better = (metric < self.best - self.min_delta) if self.mode == 'min' else \
                 (metric > self.best + self.min_delta)

        if better:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.rank == 0:
                torch.save(self.best_state, self.save_dir / "evofill_best.pt")
                print(f"Best model updated: {self.save_dir / 'evofill_best.pt'}")
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def add_dt_logger(model, writer, global_step):
    """
    给每个 Mamba2 layer 注册 forward hook，记录 Δ=softplus(dt+dt_bias)
    仅在 rank-0 生效
    """
    if writer is None:
        return

    def _hook(m, _inp, _out):
        nonlocal global_step
        u = _inp[0]
        with torch.no_grad():
            zxbcdt = m.in_proj(u)           # (..., d_in_proj)
            nheads = m.nheads               # 官方属性
            # dt 总在最后 nheads 个通道
            dt = zxbcdt[..., -nheads:]      # (..., nheads)
            dt_bias = m.dt_bias             # (nheads,)
            delta = torch.nn.functional.softplus(dt + dt_bias)
            idx = m.layer_idx
            writer.add_scalar(f'Delta/L{idx}_min', delta.min().item(), global_step)
            writer.add_scalar(f'Delta/L{idx}_mean', delta.mean().item(), global_step)
            writer.add_scalar(f'Delta/L{idx}_max', delta.max().item(), global_step)

    # 递归找到所有 Mamba2 实例
    for name, m in model.named_modules():
        if m.__class__.__name__ == 'Mamba2':
            m.register_forward_hook(_hook)


def main():
    # 0. 环境变量由 torchrun 注入
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1. 配置 & 随机种子
    cfg = load_config("config/config.json")
    seed = cfg.train.seed + rank  # 不同卡用不同种子，防止数据增强重复
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. 数据 loader（训练集加 DistributedSampler）
    train_sampler = DistributedSampler(
        GenotypeDataset(torch.load(Path(cfg.data.path) / "train.pt")['gts'],
                        torch.load(Path(cfg.data.path) / "train.pt")['coords'],
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
    model = EvoFill(**vars(cfg.model)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ===== 新增：TensorBoard =====
    writer = None
    if rank == 0:
        tb_dir = Path(cfg.train.save) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_dir)

    # 4. 损失 & 优化器
    criterion = ImputationLoss(
        use_r2_loss=cfg.train.use_r2_loss,
        use_grad_norm=cfg.train.use_grad_norm,
        gn_alpha=0.8,
        gn_lr_w=cfg.train.lr / 10,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # 5. 早停
    save_dir = Path(cfg.train.save)
    early_stopper = EarlyStopper(
        patience=cfg.train.patience,
        min_delta=cfg.train.min_delta,
        mode='min',
        save_dir=save_dir,
        rank=rank,
    )

    # 6. 训练循环
    global_step = 0
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, rank,
            writer=writer, global_step=global_step)
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)

        # 每 epoch 再写一次验证集指标
        if rank == 0:
            writer.add_scalar("Loss/val", val_loss, global_step)
            writer.add_scalar("Acc/val", val_acc, global_step)
            print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        # 同步各卡 val_loss 后再做早停判断（可选）
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        if early_stopper(val_loss_tensor.item(), model.module):
            if rank == 0:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    # 7. 结束
    if rank == 0:
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()