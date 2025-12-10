#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-1 训练脚本
DeepSpeed ZeRO-3 多卡并行
运行：
    OMP_NUM_THREADS=8 accelerate launch --config_file ds_zero3.yaml train_stage1_deepspeed.py
"""
import math
import torch
import json
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from torch_optimizer import Lookahead
from timm.utils import ModelEmaV2

from src.model import EvoFill
from src.data import GenotypeEncoder, GenomicDataset
from src.loss import ImputationLoss

# ================= 1. 超参数 =================
MODEL_NAME         = "hg38_HLA"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251206_HLA/')
PRETRAIN_DIR       = WORK_DIR / "train"
MODEL_SAVE_DIR     = WORK_DIR / "models"

TEST_N_SAMPLES     = 128
BATCH_SIZE         = 16
MIN_MASK_RATE      = 0.05
MAX_MASK_RATE      = 0.95

CHUNK_SIZE         = 65536
OVERLAP            = 1024
D_MODEL            = 64
D_STATE            = 64
HEADDIM            = 64

MAX_EPOCHS         = 1000
EARLYSTOP_PATIENCE = 21
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-8
SEED               = 3047

# 优化器专属
LR          = 1e-3
BETAS       = (0.9, 0.999)
WD           = 1e-5
ACCUMULATE_STEPS = 4          # 梯度累积步数
EMA_DECAY        = 0.999      # EMA 衰减系数
LOOKAHEAD_K      = 5          # Lookahead 慢更新周期
LOOKAHEAD_ALPHA  = 0.5        # Lookahead 插值系数
# ==============================================

# -------------- 工具：打印只在主进程 --------------
def pprint(*args):
    if accelerator.is_main_process:
        print(*args)

def reset_lr(optimizer, init_lr):
    """把 optimizer 里所有 param_group 的 lr 一次性重置"""
    if optimizer is None:
        return
    for g in optimizer.param_groups:
        g['lr'] = init_lr

# -------------- 0. 初始化 Accelerator --------------
accelerator = Accelerator()
device = accelerator.device
world_size = accelerator.num_processes 
accelerate_set_seed(SEED)

SCHEDULER_PATIENCE = SCHEDULER_PATIENCE * world_size
# -------------- 1. 数据 --------------

gt_enc = GenotypeEncoder.loadfromdisk(PRETRAIN_DIR)
pprint(f"{gt_enc.n_samples:,} samples, {gt_enc.n_variants:,} variants, {gt_enc.seq_depth} seq-depth.")

train_idx, test_idx = train_test_split(
    range(gt_enc.n_samples),
    test_size=TEST_N_SAMPLES,
    random_state=SEED,
    shuffle=True,
)

train_ds = GenomicDataset(
    gt_enc, evo_mat=gt_enc.evo_mat, mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=train_idx
)
test_ds = GenomicDataset(
    gt_enc, evo_mat=gt_enc.evo_mat, mask=False, indices=test_idx
)

def collate_fn(batch, dataset):
    x_onehot = torch.stack([item[0] for item in batch])
    y_onehot = torch.stack([item[1] for item in batch])
    real_idx_list = [item[2] for item in batch]

    if dataset.evo_mat is not None:
        evo_mat_batch = dataset.evo_mat[np.ix_(real_idx_list, real_idx_list)]
        evo_mat_batch = torch.FloatTensor(evo_mat_batch)
    else:
        evo_mat_batch = torch.empty(0)
    return x_onehot, y_onehot, evo_mat_batch

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=train_ds)
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=test_ds)
)

# -------------- 2. 模型 --------------

model_raw = EvoFill(gt_enc.seq_depth, gt_enc.n_variants, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM)

if accelerator.is_main_process:
    meta = {"model_name":  MODEL_NAME,
            "alleles":     gt_enc.seq_depth,
            "total_sites": gt_enc.n_variants,
            "chunk_size":  CHUNK_SIZE,
            "overlap":     OVERLAP,
            "d_model":     D_MODEL,
            "d_state":     D_STATE,
            "headdim":     HEADDIM,}
    with open(MODEL_SAVE_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

# -------------- 3. 优化器 --------------
# ========== 分层 weight decay 分组 ==========
def param_groups_weight_decay(model, weight_decay=1e-4, skip_list=()):
    """
    返回两组参数：
      - 需要 decay 的：卷积核 + Mamba/SSD 矩阵
      - 不需要 decay的：bias、norm、pos_emb
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        elif "conv" in name and "weight" in name:        # 卷积核 -> 稍大 decay
            decay.append(param)
        elif "ssd" in name or "mamba" in name:          # SSM 矩阵 -> 小 decay
            decay.append(param)
        else:                                            # 其余不 decay
            no_decay.append(param)
    return [
        {'params': decay,   'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]

groups = param_groups_weight_decay(
    model_raw,
    weight_decay=WD,          # 你脚本里已经有的 1e-5 可改成 1e-4/1e-5 组合
    skip_list={"position_embedding.weight"}   # 想额外跳过可继续加
)

# 再喂给 AdamW
base_opt = AdamW(groups, lr=LR, betas=BETAS)
# 2. 包一层 Lookahead
optimizer = Lookahead(base_opt, k=LOOKAHEAD_K, alpha=LOOKAHEAD_ALPHA)
# 3. EMA：要在 accelerator.prepare 之前创建，但**不能**传给 prepare
ema_model = ModelEmaV2(model_raw, decay=EMA_DECAY, device='cpu')  # 先放 CPU，更新后手动同步
# 4. 学习率调度不变（仍可 Plateau）
scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                               patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
# 5. Accelerator 封装
model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
    model_raw, optimizer, scheduler, train_loader, test_loader
)
# -------------- 5. Loss --------------

criterion = ImputationLoss(use_r2=True, use_evo=True, r2_weight=1, evo_weight=4, evo_lambda=10)

# -------------- 6. 训练 --------------
best_loss = math.inf
patience_counter = 0
accum_cnt = 0                                      # 累积计数器

for epoch in range(MAX_EPOCHS):
    # ---- train ----
    model.train()
    tot_loss = 0.0
    train_pbar = tqdm(train_loader, disable=not accelerator.is_main_process,
                      desc=f'Epoch {epoch + 1}/{MAX_EPOCHS}')
    for _, (x, y, evo) in enumerate(train_pbar):
        x, y = x.to(device), y.to(device)
        evo = evo.to(device) if evo.numel() else None

        # 一个 batch 内按 chunk 顺序 forward
        batch_loss = 0.0
        for cid in range(model.module.n_chunks):
            optimizer.zero_grad()
            logits, prob, mask_idx = model(x, cid)
            loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                   y[:, mask_idx], evo)
            loss = loss / ACCUMULATE_STEPS          # 先缩放
            accelerator.backward(loss)
            optimizer.step()
            batch_loss += loss.item()

        avg_batch_loss = batch_loss / model.module.n_chunks
        tot_loss += avg_batch_loss
        accum_cnt += 1
        if accum_cnt % ACCUMULATE_STEPS == 0:       # 真正更新
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()                        # Lookahead 内部会处理快慢权重
            optimizer.zero_grad()
            # EMA 更新（只在主进程做，避免冗余）
            if accelerator.is_main_process:
                ema_model.update(model.module if hasattr(model, 'module') else model)

        train_pbar.set_postfix({'loss': avg_batch_loss,
                                'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})

    avg_train_loss = tot_loss / len(train_loader)

    avg_train_loss = tot_loss / len(train_loader)

    # ---- eval ----
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        for x, y, evo in test_loader:
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None
            batch_loss = 0.0
            for cid in range(model.module.n_chunks):
                logits, prob, mask_idx = model(x, cid)
                loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                batch_loss += loss.item()
            tot_loss += batch_loss / model.module.n_chunks
    avg_test_loss = tot_loss / len(test_loader)

    # scheduler / early-stop 保持不变
    scheduler.step(avg_test_loss)
    pprint(f"Epoch {epoch + 1}/{MAX_EPOCHS}: train={avg_train_loss:.2f} "
           f"test={avg_test_loss:.2f} lr={optimizer.param_groups[0]['lr']:.2e} pat={patience_counter}")

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        patience_counter = 0
        ts = datetime.datetime.now().strftime("%m%d-%H%M%S")
        ckpt_path = MODEL_SAVE_DIR / f"checkpoint-stage1-{ts}"
        accelerator.save_state(output_dir=ckpt_path)
        if accelerator.is_main_process:
            torch.save(ema_model.state_dict(), ckpt_path / "ema_model.pt")
        pprint(f"  --> {ts} checkpoint-stage1 + EMA updated.")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

accelerator.wait_for_everyone()
pprint("==> STAGE2 finished <==")