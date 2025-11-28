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

from src.data import GenotypeEncoder, GenomicDataset_Missing
from src.model import EvoFill
from src.loss import ImputationLoss_Missing

# ================= 1. 超参数 =================
# MODEL_NAME         = "chr22"
# WORK_DIR           = Path('/data/home/7240203/EvoFill_data/1kGP_chr22')
# MODEL_NAME         = "chr6"
# WORK_DIR           = Path('/data/home/7240325/EvoFill_data/1kGP_chr6')
MODEL_NAME         = "aadr_chr22"
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251127_chr22/')
TEST_DIR           = WORK_DIR / "posttrain"       # 测试集来源: 30% 1KGP
TRAIN_DIR          = WORK_DIR / "pretrain"      #
MODEL_SAVE_DIR     = WORK_DIR / "models"

TEST_RATIO         = 0.30 
BATCH_SIZE         = 32
MIN_MASK_RATE      = 0.1              # 假设古DNA样本本身有 50% 缺失
MAX_MASK_RATE      = 0.4

CHUNK_SIZE         = 32768
OVERLAP            = 1024
D_MODEL            = 64
D_STATE            = 64
HEADDIM            = 64

MAX_EPOCHS         = 1000
EARLYSTOP_PATIENCE = 11
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-8
SEED               = 3047

# 优化器
LR           = 1e-3
BETAS        = (0.9, 0.999)
WD           = 1e-5
# ============================================================================

def pprint(*args):
    if accelerator.is_main_process:
        print(*args)

# -------------- 0. Accelerator --------------
accelerator = Accelerator()          # 全自动读取 yaml
device = accelerator.device
accelerate_set_seed(SEED)
world_size = accelerator.num_processes

SCHEDULER_PATIENCE = SCHEDULER_PATIENCE * world_size
EARLYSTOP_PATIENCE = EARLYSTOP_PATIENCE * world_size 
# -------------- 1. 数据 --------------

# ### Stage-3 差异：训练集=augment，测试集=pretrain 随机 30%
gt_enc_train = GenotypeEncoder.loadfromdisk(TRAIN_DIR)
gt_enc_test  = GenotypeEncoder.loadfromdisk(TEST_DIR)

assert gt_enc_train.seq_depth  == gt_enc_test.seq_depth
assert gt_enc_train.n_variants == gt_enc_test.n_variants

TOTAL_SITES = gt_enc_train.n_variants
ALLELES     = gt_enc_train.seq_depth
pprint(f"Train: {gt_enc_train.n_samples:,} samples")
pprint(f"Test : {gt_enc_test.n_samples:,} samples")

# 测试集随机 30%
test_idx, _ = train_test_split(
    range(gt_enc_test.n_samples),
    test_size=1-TEST_RATIO,
    random_state=SEED,
    shuffle=True
)
pprint(f"Actually use {len(test_idx):,} samples for test")

train_ds = GenomicDataset_Missing(
    gt_enc_train, evo_mat=gt_enc_train.evo_mat, mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=None
)
test_ds = GenomicDataset_Missing(
    gt_enc_test, evo_mat=gt_enc_test.evo_mat, mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE), indices=test_idx
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

model_raw = EvoFill(ALLELES, TOTAL_SITES, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM)
# ckpt = torch.load(f'{MODEL_SAVE_DIR}/{MODEL_NAME}_stage1.pth', map_location='cpu')
# model_raw.load_state_dict(ckpt['model_state'])
if accelerator.is_main_process:
    meta = {"model_name":  MODEL_NAME,
            "alleles":     ALLELES,
            "total_sites": TOTAL_SITES,
            "chunk_size":  CHUNK_SIZE,
            "overlap":     OVERLAP,
            "d_model":     D_MODEL,
            "d_state":     D_STATE,
            "headdim":     HEADDIM,}
    with open(MODEL_SAVE_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

# -------------- 3. 只解冻 embedding + global_out --------------
# trainable_para= []
# # 3.1 GlobalOut 全部
# for name, p in model_raw.global_out.named_parameters():
#         trainable_para.append(p)
# # 3.2 Chunk-Embedding
# for emb in model_raw.chunk_embeds:
#     for p in emb.parameters():
#         trainable_para.append(p)

# # 先全部冻住
# for p in model_raw.parameters():
#     p.requires_grad = False
# # 再放开需要的
# for p in trainable_para:
#     p.requires_grad = True

# print("Trainable params:", len(trainable_para))
# -------------- 4. 优化器 --------------
# optimizer  = AdamW(trainable_para, lr=LR, betas=BETAS, weight_decay=WD)
optimizer = AdamW(model_raw.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                 patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# -------------- 5. Accelerator 封装 --------------
model, *opt_sch = accelerator.prepare(
    model_raw, optimizer, scheduler,
    train_loader, test_loader
)
optimizer, scheduler = opt_sch[:2]

# -------------- 6. Loss --------------

criterion = ImputationLoss_Missing(use_r2=True, use_evo=False)
# criterion = ImputationLoss_Missing(use_r2=True, use_evo=True, r2_weight=1, evo_weight=4, evo_lambda=10)

# -------------- 7. 训练 --------------
best_loss = math.inf
patience_counter = 0
predres_with_bestloss = None

for epoch in range(MAX_EPOCHS):
    # ---- train ----
    model.train()
    tot_loss = 0.0
    train_prob, train_gts, train_mask = [], [], []
    train_pbar = tqdm(train_loader, disable=not accelerator.is_main_process,
                          desc=f"Epoch {epoch+1} / {MAX_EPOCHS} [aDNA]")
    for _, (x, y, evo) in enumerate(train_pbar):
        x, y = x.to(device), y.to(device)
        evo = evo.to(device) if evo.numel() else None

        optimizer.zero_grad()

        # 不分 chunk → cid=None
        logits, prob, mask_idx = model(x, None)
        loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                               y[:, mask_idx], evo)
        accelerator.backward(loss)
        optimizer.step()
        tot_loss += loss.item()

        train_pbar.set_postfix({'loss': loss.item(), 'ce':logs['ce'], 'r2':logs['r2'], 'evo':logs['evo']})

        # 收集指标
        miss_mask = x[:, mask_idx][..., -1].bool()
        train_prob.append(prob[:, mask_idx].detach())
        train_gts.append(y[:, mask_idx].detach())
        train_mask.append(miss_mask)

    avg_train_loss = tot_loss / len(train_loader)
    train_prob = torch.cat(train_prob, dim=0)
    train_gts  = torch.cat(train_gts,  dim=0)
    train_mask = torch.cat(train_mask, dim=0)

    # ---- eval ----
    model.eval()
    tot_loss = 0.0
    test_prob, test_gts = [], []
    with torch.no_grad():
        for x, y, evo in tqdm(test_loader, disable=not accelerator.is_main_process,
                          desc=f"Epoch {epoch+1} / {MAX_EPOCHS} [TEST]"):
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None
            logits, prob, mask_idx = model(x, None)
            loss, _ = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                y[:, mask_idx], evo)
            tot_loss += loss.item()
            test_prob.append(prob[:, mask_idx].detach())
            test_gts.append(y[:, mask_idx].detach())

    avg_test_loss = tot_loss / len(test_loader)
    test_prob = torch.cat(test_prob, dim=0)
    test_gts  = torch.cat(test_gts,  dim=0)

    # scheduler
    scheduler.step(avg_test_loss)

    pprint(f"Epoch {epoch+1} / {MAX_EPOCHS}  aDNA={avg_train_loss:.3f}  test={avg_test_loss:.3f}  "
           f"lr={optimizer.param_groups[0]['lr']:.2e}  ")

    # ---- early stop ----
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        patience_counter = 0
        predres_with_bestloss = (train_prob, train_gts, test_prob, test_gts)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        ckpt = {
            "model_state": unwrapped.state_dict(),
            "best_test_loss": best_loss,
        }
        if accelerator.is_main_process:
            accelerator.save(ckpt, f"{MODEL_SAVE_DIR}/{MODEL_NAME}_stage1.pth")
            pprint(f"  --> updated {MODEL_NAME}_stage1.pth")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

# -------------- 8. 最终保存 --------------
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    pprint(f"==> STAGE3 finished: {MODEL_SAVE_DIR}/{MODEL_NAME}_stage1.pth")