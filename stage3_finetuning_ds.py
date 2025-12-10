#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 训练脚本
DeepSpeed ZeRO-3 多卡并行
只训练 embedding + global_out，不分 chunk
运行：
    accelerate launch --config_file ds_zero3.yaml train_stage3_deepspeed.py
"""
import torch
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from functools import partial
from torch.optim import  AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from src.data import GenotypeEncoder, GenomicDataset
from src.model import EvoFill
from src.loss import ImputationLoss


# ================= 1. 超参数 =================
MODEL_NAME         = "hg38_HLA"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251205_HLA/')
PRETRAIN_DIR       = WORK_DIR / "train"
AUGMENT_DIR        = WORK_DIR / "augment"
FINETUNE_DIR        = WORK_DIR / "finetune"
MODEL_SAVE_DIR     = WORK_DIR / "models"

K_FOLD             = 5
BATCH_SIZE         = 8
MIN_MASK_RATE      = 0.05
MAX_MASK_RATE      = 0.95

CHUNK_SIZE         = 65536
OVERLAP            = 1024
D_MODEL            = 64
D_STATE            = 64
HEADDIM            = 64

MAX_EPOCHS         = 100
WARMUP_EPOCHS      = 3
ACCUM_GRAD   = 2
EARLYSTOP_PATIENCE = 11
SEED               = 3047

# 优化器
LR           = 1e-5
BETAS        = (0.9, 0.999)
WD           = 1e-7
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

# -------------- 2. 模型 --------------

model = EvoFill(gt_enc_urp.seq_depth, gt_enc_urp.n_variants, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM)

state_dict = torch.load(f"{MODEL_SAVE_DIR}/pytorch_model_stage1.bin", map_location="cpu")
model.load_state_dict(state_dict)

# -------------- 3. 只解冻 embedding + global_out --------------
trainable_para= []
# 3.1 GlobalOut 全部
for name, p in model.global_out.named_parameters():
        trainable_para.append(p)
# 3.2 Chunk-Embedding
for emb in model.chunk_embeds:
    for p in emb.parameters():
        trainable_para.append(p)

# 先全部冻住
for p in model.parameters():
    p.requires_grad = False
# 再放开需要的
for p in trainable_para:
    p.requires_grad = True

print("Trainable params:", len(trainable_para))
# -------------- 4. 优化器 --------------
optimizer  = AdamW(trainable_para, lr=LR, betas=BETAS, weight_decay=WD)

# 余弦退火 + 热身
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS
    return 0.5*(1+np.cos(np.pi*(epoch-WARMUP_EPOCHS)/(MAX_EPOCHS-WARMUP_EPOCHS)))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer,  lr_lambda)

train_ds = GenomicDataset(
        gt_enc_urp, evo_mat=gt_enc_urp.evo_mat,mask=True,
        masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=None)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=train_ds))

model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)

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
        tqdm(kf.split(urp_idx),
            desc=f'Epoch {epoch+1}/{MAX_EPOCHS}',
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} fold')
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
            num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=train_ds))
        val_loader   = torch.utils.data.DataLoader(
            val_ds,   batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=val_ds))

        # 每折重新 prepare（折内并行）
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

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
        ts = datetime.datetime.now().strftime("%m%d-%H%M%S")
        ckpt_path = MODEL_SAVE_DIR / f"checkpoint-stage3-{ts}"
        accelerator.save_state(output_dir=ckpt_path)
        pprint(f"  --> {ts} checkpoint-stage3 updated.")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

accelerator.wait_for_everyone()
pprint("==> STAGE3 finished <==")