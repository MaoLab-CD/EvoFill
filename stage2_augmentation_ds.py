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
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from src.data import GenotypeEncoder, GenomicDataset_AlignedMask, GenomicDataset_1240k
from src.model import EvoFill
from src.loss_v1 import ImputationLoss_Missing

# ================= 1. 超参数 =================
MODEL_NAME         = "hg38_chr22"
# WORK_DIR           = Path('/data/home/7240203/EvoFill_data/20251211_chr22/')
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251211_chr22/')
PRETRAIN_DIR       = WORK_DIR / "train"
AUGMENT_DIR        = WORK_DIR / "augment"
MODEL_SAVE_DIR     = WORK_DIR / "models"
ADNA_SITE_MAP      = AUGMENT_DIR / "aDNA-1kGP_sitesmap.npy"
PRETRAIN_BIN       = MODEL_SAVE_DIR / "pytorch_model_stage1.bin"

TEST_N_SAMPLES     = 2048
BATCH_SIZE         = 4
MIN_MASK_RATE      = 0.90
MAX_MASK_RATE      = 0.99

CHUNK_SIZE         = 65536
OVERLAP            = 1024
D_MODEL            = 128
D_STATE            = 128
HEADDIM            = 128
N_BIMAMBA          = 4
N_STACK_MAMBA      = 4

MAX_EPOCHS         = 200
EARLYSTOP_PATIENCE = 13
SCHEDULER_FACTOR   = 0.1
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR   = 1e-8
SEED               = 3047

# 优化器专属
LR          = 1e-4
BETAS       = (0.9, 0.999)
WD           = 1e-5

# 1240k 样本loss权重
AUG_LOSS_WEIGHT =  200
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

# -------------- 1. 数据 --------------
gt_enc_train = GenotypeEncoder.loadfromdisk(PRETRAIN_DIR)
gt_enc_val  = GenotypeEncoder.loadfromdisk(AUGMENT_DIR)

train_ds = GenomicDataset_AlignedMask(
    gt_enc_train, 
    site_map_path=str(ADNA_SITE_MAP),  # 传入AADR映射文件
    big_dim=gt_enc_train.n_variants,   # 19w
    mask_rate_range=(MIN_MASK_RATE, MAX_MASK_RATE),
    evo_mat=gt_enc_train.evo_mat,
    indices=None
)

pprint(f"1KGP Total variants    : {train_ds.n_total:,}, \n"
       f"1240K Shared variants: {train_ds.n_shared:,} ({train_ds.n_shared/train_ds.n_total:.1%})")


augment_idx, val_idx = train_test_split(
    range(gt_enc_val.n_samples),
    test_size=TEST_N_SAMPLES,
    random_state=SEED,
    shuffle=True,
)

augment_ds = GenomicDataset_1240k(
    gt_enc_val, evo_mat=gt_enc_val.evo_mat, 
    site_map = np.load(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask=False,
    indices=augment_idx
)

val_ds = GenomicDataset_1240k(
    gt_enc_val, evo_mat=gt_enc_val.evo_mat, 
    site_map = np.load(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask=False,
    indices=val_idx
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
augment_loader = torch.utils.data.DataLoader(
    augment_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=augment_ds)
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=partial(collate_fn, dataset=val_ds)
)

pprint(f"Train: {gt_enc_train.n_samples:,} samples\n"
        f"Aug  : {len(augment_idx):,} samples\n"
        f"Val  : {len(val_idx):,} samples")

# -------------- 2. 模型 --------------
meta = json.load(open(MODEL_SAVE_DIR / "model_meta.json"))
model_raw = EvoFill(
                    n_alleles=int(meta["alleles"]),
                    total_sites=int(meta["total_sites"]),
                    chunk_size=int(meta["chunk_size"]),
                    chunk_overlap=int(meta["overlap"]),
                    d_model=int(meta["d_model"]),
                    d_state=int(meta["d_state"]),
                    headdim=int(meta["headdim"]),
                    bimamba_layers=int(meta["bimamba_layers"]),
                    stack_mamba_layers=int(meta["stack_mamba_layers"])
                ).to(device)

state_dict = torch.load(PRETRAIN_BIN, map_location="cpu")
model_raw.load_state_dict(state_dict)

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

trainable = 0
frozen = 0
for p in model_raw.parameters():
    if p.requires_grad:
        trainable += p.numel()
    else:
        frozen += p.numel()

pprint(f"Trainable : {trainable:,}\n"
        f"Frozen    : {frozen:,}\n"
        f"Total     : {trainable + frozen:,}")

# -------------- 4. 优化器 --------------
optimizer  = AdamW(trainable_para, lr=LR, betas=BETAS, weight_decay=WD)
# optimizer = AdamW(model_raw.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                 patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# -------------- 5. Accelerator 封装 --------------
model, *opt_sch = accelerator.prepare(
    model_raw, optimizer, scheduler,
    train_loader, augment_loader, val_loader
)
optimizer, scheduler = opt_sch[:2]

def find_latest_ckpt(save_dir: Path):
    if not save_dir.exists():
        return None
    dirs = [d for d in save_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-stage2-")]
    if not dirs:
        return None
    # 按时间戳排序：先拆出 “1211-193253” 这种子串
    dirs.sort(key=lambda x: datetime.datetime.strptime(
              x.name.split("-", 2)[2],   # 取第三段以后
              "%m%d-%H%M%S"))
    return dirs[-1]

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
    pprint("No checkpoint found, train from previous stage.")

# -------------- 6. Loss --------------
criterion = ImputationLoss_Missing(use_r2=False, use_evo=True, evo_weight=1e5, evo_lambda=3)

# -------------- 7. 训练 --------------
best_loss = math.inf
patience_counter = 0
predres_with_bestloss = None

for epoch in range(start_epoch, MAX_EPOCHS):
    # ---- train ----
    model.train()
    tot_loss = 0.0
    train_pbar = tqdm(train_loader, disable=not accelerator.is_main_process,
                      desc=f'[TRAIN] Epoch {epoch + 1}/{MAX_EPOCHS}')
    for _, (x, y, evo) in enumerate(train_pbar):
        x, y = x.to(device), y.to(device)
        evo = evo.to(device) if evo.numel() else None

        # 一个 batch 内按 chunk 顺序 forward
        batch_loss = 0.0
        optimizer.zero_grad()
        for cid in range(model.module.n_chunks):
            
            logits, prob, mask_idx = model(x, cid)
            loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                   y[:, mask_idx], evo)
            accelerator.backward(loss)
            batch_loss += loss.item()
        optimizer.step()
        avg_batch_loss = batch_loss / model.module.n_chunks
        tot_loss += avg_batch_loss
        train_pbar.set_postfix({'loss': avg_batch_loss,
                                'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})

    avg_train_loss = tot_loss / len(train_loader)

    # ---- augmentation ----
    tot_loss = 0.0
    aug_pbar = tqdm(augment_loader, disable=not accelerator.is_main_process,
                      desc=f'[AUGMT] Epoch {epoch + 1}/{MAX_EPOCHS}')
    for _, (x, y, evo) in enumerate(aug_pbar):
        x, y = x.to(device), y.to(device)
        evo = evo.to(device) if evo.numel() else None

        # 一个 batch 内按 chunk 顺序 forward
        batch_loss = 0.0
        optimizer.zero_grad()
        for cid in range(model.module.n_chunks):

            logits, prob, mask_idx = model(x, cid)
            loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                   y[:, mask_idx], evo)
            # 损失按有效位点比例缩放
            loss = loss * AUG_LOSS_WEIGHT
            accelerator.backward(loss)
            batch_loss += loss.item()
        optimizer.step()
        avg_batch_loss = batch_loss / model.module.n_chunks
        tot_loss += avg_batch_loss
        aug_pbar.set_postfix({'loss': avg_batch_loss,
                                'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})

    avg_aug_loss = tot_loss / len(augment_loader)

    # ---- eval ----
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, disable=not accelerator.is_main_process,
                    desc=f'[VALID] Epoch {epoch + 1}/{MAX_EPOCHS}')
        for _, (x, y, evo) in enumerate(val_pbar):
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None
            batch_loss = 0.0
            for cid in range(model.module.n_chunks):
                logits, prob, mask_idx = model(x, cid)
                loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                loss = loss * AUG_LOSS_WEIGHT
                batch_loss += loss.item()
            avg_batch_loss = batch_loss / model.module.n_chunks
            tot_loss += avg_batch_loss
            val_pbar.set_postfix({'loss': avg_batch_loss,
                        'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})
    avg_val_loss = tot_loss / len(val_loader)

    # scheduler / early-stop 保持不变
    scheduler.step(avg_val_loss)
    pprint(f"Epoch {epoch + 1}/{MAX_EPOCHS}: train={avg_train_loss:.2e} "
           f"aug={avg_aug_loss:.2e} test={avg_val_loss:.2e} "
           f"lr={optimizer.param_groups[0]['lr']:.2e} pat={patience_counter}")

    # ---- early stop ----
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        ts = datetime.datetime.now().strftime("%m%d-%H%M%S")
        ckpt_path = MODEL_SAVE_DIR / f"checkpoint-stage2-{ts}"
        accelerator.save_state(output_dir=ckpt_path)
        if accelerator.is_main_process:
            with open(ckpt_path / "training_meta.json", "w") as f:
                json.dump({"best_loss": best_loss,
                            "patience_counter": patience_counter,
                            "epoch": epoch}, f, indent=4)
        pprint(f"  --> {ts} checkpoint-stage2 updated.")
    else:
        patience_counter += 1
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

# -------------- 8. 最终保存 --------------
accelerator.wait_for_everyone()
pprint("==> STAGE2 finished <==")
