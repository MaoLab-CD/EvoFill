#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本
DeepSpeed ZeRO-3 多卡并行
运行：
    OMP_NUM_THREADS=8 accelerate launch --config_file ds_zero3.yaml ds_training.py
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
from accelerate.utils import set_seed as accelerate_set_seed, find_latest_ckpt

from src.data import GenotypeEncoder, GenomicDataset, GenomicDataset_AlignedMask, GenomicDataset_1240k, MixedRatioSampler, MixedDataset
from src.model import EvoFill
from src.loss_v1 import ImputationLoss_Missing as ImputationLoss

from src.data import ImputationDataset
from src.utils import precompute_maf, metrics_by_maf, print_maf_stat_df
import os; os.environ['OMP_NUM_THREADS'] = '8'

# ================= 1. 超参数 =================
MODEL_NAME         = "hg38_chr22"
# WORK_DIR           = Path('/data/home/7240203/EvoFill_data/20251211_chr22/')
WORK_DIR           = Path('/mnt/qmtang/EvoFill_data/20251230_chr22/')
PRETRAIN_DIR       = WORK_DIR / "train"
AUGMENT_DIR        = WORK_DIR / "augment"
MODEL_SAVE_DIR     = WORK_DIR / "models"
ADNA_SITE_MAP      = AUGMENT_DIR / "sitesmap_1240k.npy"

TRAIN_N_SAMPLES    = 10000  # 每轮训练的样本数
VAL_N_SAMPLES      = 1000   # 验证集样本数
BATCH_SIZE         = 8
MIN_MASK_RATE      = 0.85
MAX_MASK_RATE      = 0.99

CHUNK_SIZE         = 65536
OVERLAP            = 1024
D_MODEL            = 128
D_STATE            = 128
HEADDIM            = 128
N_BIMAMBA          = 4
N_STACK_MAMBA      = 4

MAX_EPOCHS         = 1000
EARLYSTOP_PATIENCE = 13
SCHEDULER_FACTOR   = 0.1
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR   = 1e-8
SEED               = 3047
HYBRID_RATIO       = (0.34, 0.33, 0.33)
# 优化器专属
LR                 = 1e-3
BETAS              = (0.9, 0.999)
WD                 = 1e-5
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
gt_enc_aug  = GenotypeEncoder.loadfromdisk(AUGMENT_DIR)


gt_enc_imp  = GenotypeEncoder.loadfromdisk(WORK_DIR / "impute_in")
gt_enc_true = GenotypeEncoder.loadfromdisk(WORK_DIR / "impute_out")


# ========== 数据集构建 ==========
# 1. 1KGP随机mask数据集
train_1kgp_random = GenomicDataset(
    gt_enc_train, 
    evo_mat=gt_enc_train.evo_mat,
    mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE),
    indices=None
)

# 2. 1KGP按1240k panel mask数据集 (共享位点模式)
train_1kgp_panel = GenomicDataset_AlignedMask(
    gt_enc_train, 
    site_map_path=str(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask_rate_range=(0.0, 0.0),  # 不使用随机mask，使用固定panel mask
    evo_mat=gt_enc_train.evo_mat,
    indices=None
)

pprint(f"1KGP 总位点数    : {train_1kgp_panel.n_total:,}, \n"
        f"1240K 共享位点数: {train_1kgp_panel.n_shared:,} ({train_1kgp_panel.n_shared/train_1kgp_panel.n_total:.1%})")

# 3. 分割验证集（按比例分配各类型样本）
# 总验证样本数: VAL_N_SAMPLES = 1000
# 按40:30:30比例分配：
# - 1KGP随机mask: 400个样本
# - 1KGP panel: 300个样本  
# - 1240K: 300个样本

val_1kgp_samples = int(VAL_N_SAMPLES * HYBRID_RATIO[0])
val_panel_samples = int(VAL_N_SAMPLES * HYBRID_RATIO[1])
val_1240k_samples = VAL_N_SAMPLES - val_1kgp_samples - val_panel_samples

# 打印数据集大小信息
pprint("数据集大小信息:")
pprint(f"  1KGP训练样本总数: {len(train_1kgp_random)}")
pprint(f"  1KGP panel训练样本总数: {len(train_1kgp_panel)}")
pprint(f"  1240K训练样本总数: {gt_enc_aug.n_samples}")

# 分割1240K数据集为训练和验证
augment_idx, val_1240k_idx = train_test_split(
    range(gt_enc_aug.n_samples),
    test_size=val_1240k_samples,
    random_state=SEED,
    shuffle=True,
)

# 4. 1240K数据集 (训练和验证)
augment_ds = GenomicDataset_1240k(
    gt_enc_aug, evo_mat=gt_enc_aug.evo_mat, 
    site_map = np.load(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask=False,
    indices=augment_idx
)

val_1240k = GenomicDataset_1240k(
    gt_enc_aug, evo_mat=gt_enc_aug.evo_mat, 
    site_map = np.load(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask=False,
    indices=val_1240k_idx
)

# 5. 验证集的1KGP数据集 (按比例创建)
val_1kgp_random = GenomicDataset(
    gt_enc_train, 
    evo_mat=gt_enc_train.evo_mat,
    mask=True,
    masking_rates=(MIN_MASK_RATE, MAX_MASK_RATE),
    indices=np.random.RandomState(SEED).choice(gt_enc_train.n_samples, val_1kgp_samples, replace=False)
)

val_1kgp_panel = GenomicDataset_AlignedMask(
    gt_enc_train, 
    site_map_path=str(ADNA_SITE_MAP),
    big_dim=gt_enc_train.n_variants,
    mask_rate_range=(0.0, 0.0),
    evo_mat=gt_enc_train.evo_mat,
    indices=np.random.RandomState(SEED+1).choice(gt_enc_train.n_samples, val_panel_samples, replace=False)
)

def collate_fn(batch, datasets):
    """处理混合数据集的batch"""
    # 检查batch中每个item的长度，支持不同类型的数据集
    first_item_len = len(batch[0])
    
    x_onehot = torch.stack([item[0] for item in batch])
    y_onehot = torch.stack([item[1] for item in batch])
    real_idx_list = [item[2] for item in batch]
    
    if first_item_len == 4:
        # 混合数据集 (MixedDataset): 有dataset_type信息
        sample_types = [item[3] for item in batch]
        dataset_type = sample_types[0]
        dataset = datasets[dataset_type]
    else:
        # 单一数据集 (如GenomicDataset): 没有dataset_type信息
        dataset = datasets[0]  # 使用第一个数据集
    
    if dataset.evo_mat is not None:
        evo_mat_batch = dataset.evo_mat[np.ix_(real_idx_list, real_idx_list)]
        evo_mat_batch = torch.FloatTensor(evo_mat_batch)
    else:
        evo_mat_batch = torch.empty(0)
    
    return x_onehot, y_onehot, evo_mat_batch


# ========== 创建混合数据加载器 ==========
val_sampler = MixedRatioSampler(
    [val_1kgp_random, val_1kgp_panel, val_1240k],
    ratio=HYBRID_RATIO,
    num_samples=VAL_N_SAMPLES,
    shuffle=False,  # 验证集不需要shuffle
    batch_size=BATCH_SIZE
)

# 创建混合数据集
train_mixed_ds = MixedDataset([train_1kgp_random, train_1kgp_panel, augment_ds], None)
val_mixed_ds = MixedDataset([val_1kgp_random, val_1kgp_panel, val_1240k], val_sampler)

pprint(f"Train 混合数据集: {HYBRID_RATIO[0]*100:.1f}% 1KGP随机mask + {HYBRID_RATIO[1]*100:.1f}% 1KGP panel + {HYBRID_RATIO[2]*100:.1f}% 1240K\n"
        f"Val   混合数据集: {HYBRID_RATIO[0]*100:.1f}% 1KGP随机mask + {HYBRID_RATIO[1]*100:.1f}% 1KGP panel + {HYBRID_RATIO[2]*100:.1f}% 1240K\n"
        f"每轮训练样本数: {TRAIN_N_SAMPLES}\n"
        f"验证样本数: {VAL_N_SAMPLES}\n"
        f"批次大小: {BATCH_SIZE} (每个批次的样本来自同一数据集类型)\n"
        f"1KGP 训练样本总数: {gt_enc_train.n_samples:,}\n"
        f"1240K 训练样本总数: {len(augment_idx):,}\n"
        f"注: 为确保evo_mat有效性，每个批次的{BATCH_SIZE}个样本来自同一数据集类型")

#  -------------- URP 测试集
imp_dataset = ImputationDataset(
    x_gts_sparse=gt_enc_imp.X_gt,
    seq_depth=gt_enc_imp.seq_depth,
    indices=None                 # 可传入指定样本索引
)
imp_dataset.print_missing_stat()          # 查看原始缺失比例

def imp_collate_fn(batch):
    x_onehot = torch.stack([item[0] for item in batch])
    real_idx_list = [item[1] for item in batch]
    return x_onehot, real_idx_list   # 无 y
    
imp_loader = torch.utils.data.DataLoader(
    imp_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=imp_collate_fn
)
imp_y_true = gt_enc_true.X_gt.toarray()
imp_maf, imp_bin_cnt = precompute_maf(imp_y_true)
imp_y_true_oh = np.eye(gt_enc_true.seq_depth - 1)[imp_y_true]

# # -------------- 2. 模型 --------------
# model_raw = EvoFill(gt_enc_train.seq_depth, gt_enc_train.n_variants, CHUNK_SIZE, OVERLAP, D_MODEL, D_STATE, HEADDIM, N_BIMAMBA, N_STACK_MAMBA)

# if accelerator.is_main_process:
#     meta = {"model_name":  MODEL_NAME,
#             "alleles":     gt_enc_train.seq_depth,
#             "total_sites": gt_enc_train.n_variants,
#             "chunk_size":  CHUNK_SIZE,
#             "overlap":     OVERLAP,
#             "d_model":     D_MODEL,
#             "d_state":     D_STATE,
#             "headdim":     HEADDIM,
#             "bimamba_layers": N_BIMAMBA,
#             "stack_mamba_layers": N_STACK_MAMBA}
#     with open(MODEL_SAVE_DIR / "model_meta.json", "w") as f:
#         json.dump(meta, f, indent=4)

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

state_dict = torch.load("/mnt/qmtang/EvoFill_data/20251211_chr22/models/pytorch_model_stage1.bin", map_location="cpu")
model_raw.load_state_dict(state_dict)

# -------------- 4. 优化器 --------------
optimizer = AdamW(model_raw.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                 patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# -------------- 5. 创建训练数据加载器 --------------
initial_train_sampler = MixedRatioSampler(
    [train_1kgp_random, train_1kgp_panel, augment_ds],
    ratio=HYBRID_RATIO,
    num_samples=TRAIN_N_SAMPLES,
    shuffle=True,
    epoch_seed=SEED,  # 使用初始种子
    batch_size=BATCH_SIZE
)

train_mixed_ds.sampler = initial_train_sampler
initial_train_loader = torch.utils.data.DataLoader(
    train_mixed_ds, 
    batch_size=BATCH_SIZE, 
    num_workers=4, 
    pin_memory=True, 
    collate_fn=partial(collate_fn, datasets=[train_1kgp_random, train_1kgp_panel, augment_ds])
)

# 创建验证数据加载器（固定样本）
val_loader = torch.utils.data.DataLoader(
    val_mixed_ds,
    batch_size=BATCH_SIZE,
    sampler=val_sampler, 
    num_workers=4,
    pin_memory=True,
    collate_fn=partial(collate_fn, datasets=[val_1kgp_random, val_1kgp_panel, val_1240k])
)


# -------------- 6. Accelerator 封装 --------------
model, *opt_sch = accelerator.prepare(
    model_raw, optimizer, scheduler,
    initial_train_loader, val_loader, imp_loader
)
optimizer, scheduler = opt_sch[:2]

latest_ckpt = find_latest_ckpt(MODEL_SAVE_DIR, prefix="checkpoint-stage2-")

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

# -------------- 6. Loss --------------
criterion = ImputationLoss(use_r2=True, use_evo=True)


# -------------- 6. URP test --------------

y_prob = []
y_mask = []
with torch.no_grad():
    for x_onehot, real_idx in tqdm(imp_loader, disable=not accelerator.is_main_process,
                                        desc=f'[ CDX ]'):
        x_onehot = x_onehot.to(device)
        _, prob, _ = model(x_onehot)
        miss_mask = x_onehot[..., -1].bool()
        y_prob.append(prob)
        y_mask.append(miss_mask)

# 在主进程中执行增强的验证逻辑
if accelerator.is_main_process:
    try:
        # 转换预测结果
        y_prob = torch.cat(y_prob, dim=0).cpu().numpy()
        y_mask = torch.cat(y_mask, dim=0).cpu().numpy()

        bins_metrics = metrics_by_maf(y_prob, imp_y_true_oh, 
                                                hap_map=gt_enc_true.hap_map, 
                                                maf_vec=imp_maf, mask=y_mask)
        print_maf_stat_df(imp_bin_cnt, {'val': bins_metrics})
    except Exception as e:
        print(f'[TEST ] Enhanced validation failed: {str(e)}')


# -------------- 7. 训练 --------------
best_loss = math.inf
patience_counter = 0
predres_with_bestloss = None

for epoch in range(start_epoch, MAX_EPOCHS):
    # ========== 动态更新采样器（确保DeepSpeed兼容性） ==========
    # 更新采样器的epoch种子，确保每轮样本不同
    train_mixed_ds.sampler.set_epoch(epoch)
    
    # ---- train ----
    model.train()
    tot_loss = 0.0
    train_pbar = tqdm(initial_train_loader, disable=not accelerator.is_main_process,
                      desc=f'Epoch {epoch + 1}/{MAX_EPOCHS} [TRAIN]')
    for batch_idx, (x, y, evo) in enumerate(train_pbar):
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

    avg_train_loss = tot_loss / len(initial_train_loader)

    # ---- eval ----
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, disable=not accelerator.is_main_process,
                    desc=f'Epoch {epoch + 1}/{MAX_EPOCHS} [VALID]')
        for batch_idx, (x, y, evo) in enumerate(val_pbar):
            x, y = x.to(device), y.to(device)
            evo = evo.to(device) if evo.numel() else None
            batch_loss = 0.0
            for cid in range(model.module.n_chunks):
                logits, prob, mask_idx = model(x, cid)
                loss, logs = criterion(logits[:, mask_idx], prob[:, mask_idx],
                                    y[:, mask_idx], evo)
                batch_loss += loss.item()
            avg_batch_loss = batch_loss / model.module.n_chunks
            tot_loss += avg_batch_loss
            val_pbar.set_postfix({'loss': avg_batch_loss,
                        'ce': logs['ce'], 'r2': logs['r2'], 'evo': logs['evo']})
    avg_val_loss = tot_loss / len(val_loader)

    # ---- test ----
    model.eval()
    y_prob = []
    y_mask = []
    with torch.no_grad():
        for x_onehot, real_idx in tqdm(imp_loader, disable=not accelerator.is_main_process,
                                         desc=f'Epoch {epoch + 1}/{MAX_EPOCHS} [ CDX ]'):
            x_onehot = x_onehot.to(device)
            _, prob, _ = model(x_onehot)
            miss_mask = x_onehot[..., -1].bool()
            y_prob.append(prob)
            y_mask.append(miss_mask)
   
    # 在主进程中执行增强的验证逻辑
    if accelerator.is_main_process:
        try:
            # 转换预测结果
            y_prob = torch.cat(y_prob, dim=0).cpu().numpy()
            y_mask = torch.cat(y_mask, dim=0).cpu().numpy()

            bins_metrics = metrics_by_maf(y_prob, imp_y_true_oh, 
                                                    hap_map=gt_enc_true.hap_map, 
                                                    maf_vec=imp_maf, mask=y_mask)
            print_maf_stat_df(imp_bin_cnt, {'val': bins_metrics})
        except Exception as e:
            print(f'[ CDX ] Enhanced validation failed: {str(e)}')

    # scheduler / early-stop
    scheduler.step(avg_val_loss)
    pprint(f"Epoch {epoch + 1}/{MAX_EPOCHS}: train={avg_train_loss:.2f} "
           f"val={avg_val_loss:.2f} "
           f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # ---- early stop ----
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        ckpt_path = MODEL_SAVE_DIR / f"checkpoint-stage2-{epoch}"
        accelerator.save_state(output_dir=ckpt_path)
        if accelerator.is_main_process:
            with open(ckpt_path / "training_meta.json", "w") as f:
                json.dump({"best_loss": best_loss,
                            "patience_counter": patience_counter,
                            "epoch": epoch}, f, indent=4)
        pprint(f"  --> Checkpoint updated at epoch {epoch}.")
    else:
        patience_counter += 1
        pprint(f"patience counter={patience_counter}")
        if patience_counter >= EARLYSTOP_PATIENCE:
            pprint("Early stop!")
            break

# -------------- 8. 最终保存 --------------
accelerator.wait_for_everyone()
pprint("==> Stage2: Hybrid Training Finished <==")
