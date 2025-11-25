#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 训练脚本
DeepSpeed ZeRO-3 多卡并行
只训练 embedding + global_out，不分 chunk
运行：
    accelerate launch --config_file ds_zero3.yaml train_stage3_deepspeed.py
"""
import math
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim import SparseAdam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from src.data import GenotypeEncoder, GenomicDataset
from src.model import EvoFill
from src.loss import ImputationLoss


# ================= 1. 超参数 =================
# MODEL_NAME         = "chr22"
# WORK_DIR           = Path('/data/home/7240203/EvoFill_data/1kGP_chr22')
# MODEL_NAME         = "chr6"
# WORK_DIR           = Path('/data/home/7240325/EvoFill_data/1kGP_chr6')
MODEL_NAME         = "chr22_trim"
stage4_tag         = "CDX_BEB_ASW"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251121_chr22_v2/')
PRETRAIN_DIR       = WORK_DIR / "pretrain"
AUGMENT_DIR        = WORK_DIR / "augment"
FINETUNE_DIR        = WORK_DIR / "finetune"
MODEL_SAVE_DIR     = WORK_DIR / "models"

K_FOLD             = 5
BATCH_SIZE         = 4
MIN_MASK_RATE      = 0.3
MAX_MASK_RATE      = 0.7

CHUNK_SIZE         = 32768
OVERLAP            = 1024
D_MODEL            = 64
D_STATE            = 64
HEADDIM            = 64

MAX_EPOCHS         = 100
WARMUP_EPOCHS      = 3
ACCUM_GRAD   =2
EARLYSTOP_PATIENCE = 11
SEED               = 3047

# 优化器
LR           = 1e-4
BETAS        = (0.9, 0.999)
WD           = 1e-6
# ============================================================================

def pprint(*args):
    if accelerator.is_main_process:
        print(*args)

# -------------- 0. Accelerator --------------
accelerator = Accelerator()          # 全自动读取 yaml
device = accelerator.device
accelerate_set_seed(SEED)
world_size = accelerator.num_processes
EARLYSTOP_PATIENCE = EARLYSTOP_PATIENCE * world_size 

# -------------- 1. 数据 --------------
gt_enc_urp = GenotypeEncoder.loadfromdisk(FINETUNE_DIR)

def collate_fn(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    idx = [b[2] for b in batch]
    if gt_enc_urp.evo_mat is not None:
        evo = gt_enc_urp.evo_mat[np.ix_(idx, idx)]
        evo = torch.FloatTensor(evo)
    else:
        evo = torch.empty(0)
    return x, y, evo

# -------------- 2. 模型 --------------

model_raw = EvoFill(gt_enc_urp.seq_depth, gt_enc_urp.n_variants, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM)
ckpt = torch.load(f'{MODEL_SAVE_DIR}/{MODEL_NAME}_stage1.pth', map_location='cpu')
model_raw.load_state_dict(ckpt['model_state'])

# -------------- 3. 只解冻 embedding + global_out --------------
trainable_para= []
# 3.1 GlobalOut 全部
for name, p in model_raw.global_out.named_parameters():
        trainable_para.append(p)
# 3.2 Chunk-Embedding
for emb in model_raw.chunk_embeds:
    for p in emb.parameters():
        trainable_para.append(p)

# 先全部冻住
for p in model_raw.parameters():
    p.requires_grad = False
# 再放开需要的
for p in trainable_para:
    p.requires_grad = True

# -------------- 4. 优化器 --------------
optimizer  = AdamW(trainable_para, lr=LR, betas=BETAS, weight_decay=WD)

# 余弦退火 + 热身
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    return 0.5*(1+np.cos(np.pi*(epoch-WARMUP_EPOCHS)/(MAX_EPOCHS-WARMUP_EPOCHS)))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer,  lr_lambda)


# -------------- 6. Loss --------------

# criterion = ImputationLoss_Missing(use_r2=True, use_evo=False)
criterion = ImputationLoss(use_r2=True, use_evo=True, r2_weight=1, evo_weight=4, evo_lambda=10)

# -------------- 7. K-Fold 训练 --------------
urp_idx = np.arange(gt_enc_urp.n_samples)
kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

best_avg_val_loss = np.inf
patience_counter  = 0

for epoch in range(MAX_EPOCHS):
    fold_val_loss = []

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(kf.split(urp_idx), desc=f'Epoch {epoch+1}/{MAX_EPOCHS}', leave=False)
    ):
        # ---- 当前折数据 ----
        train_ds = GenomicDataset(
                gt_enc_urp, evo_mat=gt_enc_urp.evo_mat,mask=True,
                masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=train_idx)
        val_ds   = GenomicDataset(
                gt_enc_urp, evo_mat=gt_enc_urp.evo_mat, mask=True,
                masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=collate_fn)
        val_loader   = torch.utils.data.DataLoader(
            val_ds,   batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=collate_fn)

        # 每折重新 prepare（折内并行）
        model, optimizer, scheduler, train_loader, val_loader = \
            accelerator.prepare(model_raw, optimizer, scheduler, 
                                train_loader, val_loader)

        # ---- 训练 ----
        model.train()
        for step, (x, y, evo) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            evo  = evo.to(device) if evo.numel() else None

            with accelerator.accumulate(model, ACCUM_GRAD):
                logits, prob, mask_idx = model(x)
                loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        # ---- 验证 ----
        model.eval()
        val_loss = 0;  val_steps = 0
        with torch.no_grad():
            for x, y, evo in val_loader:
                x, y = x.to(device), y.to(device)
                evo  = evo.to(device) if evo.numel() else None
                logits, prob, mask_idx = model(x)
                loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                val_loss += loss.item()
                val_steps += 1
        val_loss = val_loss / val_steps
        fold_val_loss.append(val_loss)

    # ---- epoch 日志 & 调度 ----
    avg_val_loss = np.mean(fold_val_loss)
    accelerator.print(f'Epoch {epoch+1}: avg_val_loss={avg_val_loss:.4f}, '
                      f'lr={optimizer.param_groups[0]["lr"]:.1e}')
    scheduler.step()

    # ---- 早停 ----
    if avg_val_loss < best_avg_val_loss:
        best_avg_val_loss = avg_val_loss
        patience_counter = 0
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            torch.save({'model_state': unwrapped.state_dict(),
                        'epoch': int(epoch),
                        'avg_val_loss': float(avg_val_loss)},
                       WORK_DIR / 'models' / f'{MODEL_NAME}_{stage4_tag}.pth')
            print(f"  --> updated {MODEL_NAME}_{stage4_tag}.pth")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            accelerator.print('Early stopping triggered.')
            break

# -------------- 6. 最终保存 --------------
accelerator.wait_for_everyone()