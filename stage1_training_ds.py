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

from src.model import EvoFill
from src.data import GenotypeEncoder, GenomicDataset
from src.loss_v1 import ImputationLoss

# ================= 1. 超参数 =================
MODEL_NAME         = "hg38_chr22_IGL"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251225_chr22_IGL/')
PRETRAIN_DIR       = WORK_DIR / "train"
MODEL_SAVE_DIR     = WORK_DIR / "models"

TEST_N_SAMPLES     = 64
BATCH_SIZE         = 8
MIN_MASK_RATE      = 0.05
MAX_MASK_RATE      = 0.95

CHUNK_SIZE         = 65536
OVERLAP            = 1024
D_MODEL            = 128
D_STATE            = 128
HEADDIM            = 128
N_BIMAMBA          = 4
N_STACK_MAMBA      = 4

MAX_EPOCHS         = 1000
EARLYSTOP_PATIENCE = 23
SCHEDULER_FACTOR   = 0.1
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR   = 1e-7
SEED               = 3047

# 优化器专属
LR          = 1e-3
BETAS       = (0.9, 0.999)
WD          = 1e-5
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

def find_latest_ckpt(save_dir: Path):
    if not save_dir.exists():
        return None
    dirs = [d for d in save_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-stage1-")]
    if not dirs:
        return None
    # 按时间戳排序：先拆出 “1211-193253” 这种子串
    dirs.sort(key=lambda x: datetime.datetime.strptime(
              x.name.split("-", 2)[2],   # 取第三段以后
              "%m%d-%H%M%S"))
    return dirs[-1]

# -------------- 1. 数据 --------------

gt_enc = GenotypeEncoder.loadfromdisk(PRETRAIN_DIR)
pprint(f"{gt_enc.n_samples:,} samples, {gt_enc.n_variants:,} variants, {gt_enc.seq_depth} seq-depth.")

train_idx, val_idx = train_test_split(
    range(gt_enc.n_samples),
    test_size=TEST_N_SAMPLES,
    random_state=SEED,
    shuffle=True,
)

train_ds = GenomicDataset(
    gt_enc, evo_mat=gt_enc.evo_mat, mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=train_idx
)
val_ds = GenomicDataset(
    gt_enc, evo_mat=gt_enc.evo_mat, mask=False, indices=val_idx
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
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=val_ds)
)

# -------------- 2. 模型 --------------

model_raw = EvoFill(gt_enc.seq_depth, gt_enc.n_variants, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM, N_BIMAMBA, N_STACK_MAMBA)

if accelerator.is_main_process:
    meta = {"model_name":  MODEL_NAME,
            "alleles":     gt_enc.seq_depth,
            "total_sites": gt_enc.n_variants,
            "chunk_size":  CHUNK_SIZE,
            "overlap":     OVERLAP,
            "d_model":     D_MODEL,
            "d_state":     D_STATE,
            "headdim":     HEADDIM,
            "bimamba_layers": N_BIMAMBA,
            "stack_mamba_layers": N_STACK_MAMBA}
    with open(MODEL_SAVE_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

# -------------- 3. 优化器 --------------

optimizer = AdamW(model_raw.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

scheduler  = ReduceLROnPlateau(optimizer,  mode='min', factor=SCHEDULER_FACTOR,
                                 patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# -------------- 4. Accelerator 封装 --------------
model, *opt_sch = accelerator.prepare(
    model_raw, optimizer, scheduler,
    train_loader, val_loader
)
optimizer, scheduler = opt_sch[:2]

latest_ckpt = find_latest_ckpt(MODEL_SAVE_DIR)

if latest_ckpt is not None:
    pprint(f"Find previous checkpoint: {latest_ckpt.name}, loading state...")
    accelerator.load_state(latest_ckpt)

    # 把 best_loss / patience_counter / start_epoch 读回来
    meta_json = latest_ckpt / "training_meta.json"
    if meta_json.exists():
        with open(meta_json, "r") as f:
            meta = json.load(f)
        best_loss        = meta.get("best_loss", math.inf)
        patience_counter = meta.get("patience_counter", 0)
        start_epoch      = meta.get("epoch", 0) + 1   # 下一 epoch
    pprint(f"Successfully resume from epoch {start_epoch}, best_loss={best_loss:.4f}, patience={patience_counter}")
else:
    start_epoch = 0
    best_loss = math.inf
    patience_counter = 0
    pprint("No checkpoint found, train from scratch.")
# -------------- 5. Loss --------------

# criterion = ImputationLoss(use_r2=True, use_evo=True, r2_weight=1, evo_weight=10, evo_lambda=3)
criterion = ImputationLoss(use_r2=True, use_evo=False)

# -------------- 6. 训练 --------------
for epoch in range(start_epoch, MAX_EPOCHS):
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
            accelerator.backward(loss)
            optimizer.step()
            batch_loss += loss.item()

        avg_batch_loss = batch_loss / model.module.n_chunks
        tot_loss += avg_batch_loss
        train_pbar.set_postfix({'loss': avg_batch_loss,
                                'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})

    avg_train_loss = tot_loss / len(train_loader)

    # ---- eval ----
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        for x, y, evo in val_loader:
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None
            batch_loss = 0.0
            for cid in range(model.module.n_chunks):
                logits, prob, mask_idx = model(x, cid)
                loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                batch_loss += loss.item()
            tot_loss += batch_loss / model.module.n_chunks
    avg_val_loss = tot_loss / len(val_loader)

    # scheduler / early-stop 保持不变
    scheduler.step(avg_val_loss)
    pprint(f"Epoch {epoch + 1}/{MAX_EPOCHS}: train={avg_train_loss:.2f} "
           f"test={avg_val_loss:.2f} lr={optimizer.param_groups[0]['lr']:.2e} pat={patience_counter}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        ts = datetime.datetime.now().strftime("%m%d-%H%M%S")
        ckpt_path = MODEL_SAVE_DIR / f"checkpoint-stage1-{ts}"
        accelerator.save_state(output_dir=ckpt_path)
        if accelerator.is_main_process:
            with open(ckpt_path / "training_meta.json", "w") as f:
                json.dump({"best_loss": best_loss,
                            "patience_counter": patience_counter,
                            "epoch": epoch}, f, indent=4)
        pprint(f"  --> {ts} checkpoint-stage1 updated.")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

# -------------- 7. 保存最终模型 --------------
accelerator.wait_for_everyone()
