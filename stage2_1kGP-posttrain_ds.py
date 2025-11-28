#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-1 训练脚本
DeepSpeed ZeRO-3 多卡并行 + 双优化器（SparseAdam / AdamW）
运行：
    OMP_NUM_THREADS=8 accelerate launch --config_file ds_zero3.yaml train_stage1_deepspeed.py
"""
import math
import torch
import json
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
from src.loss import ImputationLoss

# ================= 1. 超参数 =================
# MODEL_NAME         = "chr22"
# WORK_DIR           = Path('/data/home/7240203/EvoFill_data/1kGP_chr22')
# MODEL_NAME         = "chr6"
# WORK_DIR           = Path('/data/home/7240325/EvoFill_data/1kGP_chr6')
MODEL_NAME         = "aadr_chr22"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251127_chr22/')
TRAIN_DIR          = WORK_DIR / "posttrain"
MODEL_SAVE_DIR     = WORK_DIR / "models"

TEST_N_SAMPLES     = 128
BATCH_SIZE         = 32
MIN_MASK_RATE      = 0.05
MAX_MASK_RATE      = 0.95

CHUNK_SIZE         = 32768
OVERLAP            = 1024
D_MODEL            = 64
D_STATE            = 64
HEADDIM            = 64

MAX_EPOCHS         = 1000
EARLYSTOP_PATIENCE = 21
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-9
SEED               = 3047

# 优化器专属
LR          = 5e-4
BETAS       = (0.9, 0.999)
WD          = 1e-6
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

gt_enc = GenotypeEncoder.loadfromdisk(TRAIN_DIR)
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
ckpt = torch.load(f'{MODEL_SAVE_DIR}/{MODEL_NAME}_stage1.pth', map_location='cpu')
model_raw.load_state_dict(ckpt['model_state'])

# -------------- 3. 拆参数 → 双优化器 --------------

optimizer = AdamW(model_raw.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

scheduler  = ReduceLROnPlateau(optimizer,  mode='min', factor=SCHEDULER_FACTOR,
                                 patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# -------------- 4. Accelerator 封装 --------------
model, *opt_sch = accelerator.prepare(
    model_raw, optimizer, scheduler,
    train_loader, test_loader
)
optimizer, scheduler = opt_sch[:2]

# -------------- 5. Loss --------------

criterion = ImputationLoss(use_r2=True, use_evo=True, r2_weight=1, evo_weight=4, evo_lambda=10)

# -------------- 6. 训练 --------------
for cid in range(model.module.n_chunks):
    pprint(f"\n=== Chunk {cid + 1}/{model_raw.n_chunks} ===")

    # ---- 只重置学习率，不新建 Accelerator ----
    reset_lr(optimizer, LR)
    # 如果用了 ReduceLROnPlateau，把内部计数器也清掉
    scheduler.best = None

    best_loss = math.inf
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # ---- train ----
        model.train()
        tot_loss = 0.0
        train_pbar = tqdm(train_loader, disable=not accelerator.is_main_process,
                              desc=f'Chunk {cid + 1}/{model.n_chunks}, '
                                   f'Epoch {epoch + 1}/{MAX_EPOCHS}')
        for _, (x, y, evo) in enumerate(train_pbar):
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None

            optimizer.zero_grad()

            logits, prob, mask_idx = model(x, cid)
            loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                   y[:, mask_idx], evo)
            accelerator.backward(loss)
            optimizer.step()
            tot_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item(), 'ce':logs['ce'], 'r2':logs['r2'], 'evo':logs['evo']})

        avg_train_loss = tot_loss / len(train_loader)

        # ---- eval ----
        model.eval()
        tot_loss = 0.0
        with torch.no_grad():
            for x, y, evo in test_loader:
                x, y = x.to(device), y.to(device)
                evo = evo.to(device) if evo.numel() else None
                logits, prob, mask_idx = model(x, cid)
                loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                tot_loss += loss.item()
        avg_test_loss = tot_loss / len(test_loader)

        # scheduler
        scheduler.step(avg_test_loss)

        pprint(f"Chunk {cid + 1}/{model.n_chunks}, Epoch {epoch + 1}/{MAX_EPOCHS}: train={avg_train_loss:.2f} test={avg_test_loss:.2f}  "
               f"lr={optimizer.param_groups[0]['lr']:.2e}  pat={patience_counter}")

        # ---- early stop ----
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            ckpt = {
                "chunk_id": cid,
                "chunk_embed_state": unwrapped.chunk_embeds[cid].state_dict(),
                "chunk_module_state": unwrapped.chunk_modules[cid].state_dict(),
                "global_out_state": unwrapped.global_out.state_dict(),
                "best_test_loss": best_loss,
            }
            if accelerator.is_main_process:
                accelerator.save(ckpt, f"{MODEL_SAVE_DIR}/{MODEL_NAME}_chunk[{cid}].pth")
                pprint(f"  --> updated {MODEL_NAME}_chunk[{cid}].pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLYSTOP_PATIENCE:
                pprint("Early stop!")
                best_ckpt = torch.load(
                    f"{MODEL_SAVE_DIR}/{MODEL_NAME}_chunk[{cid}].pth",
                    map_location="cpu"
                )
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.chunk_embeds[cid].load_state_dict(best_ckpt["chunk_embed_state"])
                unwrapped.chunk_modules[cid].load_state_dict(best_ckpt["chunk_module_state"])
                unwrapped.global_out.load_state_dict(best_ckpt["global_out_state"])
                pprint(f"  --> chunk {cid} best weights reloaded (early-stop)")
                break

# -------------- 7. 保存最终模型 --------------
accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
final_ckpt = {
    "model_state": unwrapped.state_dict(),
    "n_chunks": unwrapped.n_chunks,
    "chunk_size": unwrapped.chunk_size,
    "chunk_overlap": unwrapped.chunk_overlap,
}
if accelerator.is_main_process:
    accelerator.save(final_ckpt, f"{MODEL_SAVE_DIR}/{MODEL_NAME}_stage2.pth")
    pprint(f"==> STAGE1 finished: {MODEL_SAVE_DIR}/{MODEL_NAME}_stage2.pth")