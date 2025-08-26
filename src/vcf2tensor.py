#!/usr/bin/env python3
"""
transform vcf to tensor in stici with p-distance
完全沿用 STICI 的 depth 计算策略：
1) 训练集一次性统计所有基因型 → depth
2) 验证集用同一 depth 编码
3) 训练/验证完全隔离（各读各的 vcf + dismat）
"""
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from cyvcf2 import VCF

from src.utils import load_config


# --------------------------- 读取 VCF --------------------------- #
def read_vcf(path: str, phased: bool):
    """
    支持字母 GT → 数字
    返回 (gts, samples, depth)
    """
    vcf = VCF(path)
    samples = list(vcf.samples)
    gts_list = []

    # 预取总行数，给 tqdm
    total = sum(1 for _ in VCF(path))

    for var in tqdm(vcf, total=total, desc="Parsing VCF"):
        alleles = [var.REF] + var.ALT
        allele2idx = {a: i for i, a in enumerate(alleles)}
        row = []
        for gt_str in var.gt_bases:
            if gt_str in ['.|.', './.']:
                row.append(None)
            else:
                a1, a2 = gt_str.split('|' if phased else '/')
                code = allele2idx[a1] + allele2idx[a2]
                row.append(code)
        gts_list.append(np.array(row, dtype=np.int32))

    gts = np.vstack(gts_list).T            # (n_samples, n_snps)
    flat = gts[gts >= 0]
    depth = int(flat.max()) + 2            # 统一 depth
    return gts.astype(np.int32), samples, depth

# --------------------------- 读取距离矩阵 --------------------------- #
def read_dismat(path: str, sample_order):
    dis = pd.read_csv(path, sep='\t', skiprows=[0], header=None, index_col=0)
    dis.index = dis.columns = [s.strip() for s in dis.index]
    return dis.loc[sample_order, sample_order].values.astype(np.float32)


# --------------------------- 统计 depth --------------------------- #
def compute_depth_and_map(gts: np.ndarray, phased: bool):
    """
    完全按 STICI 思路：
    1) 收集所有非缺失值
    2) 得到 depth = len(unique_alleles) + 1 (missing)
    3) 返回 depth 与映射字典
    """
    flat = gts[gts >= 0]          # 去掉缺失
    unique_vals = np.unique(flat)
    depth = int(unique_vals.max()) + 2   # 0..max  + missing
    return depth


# --------------------------- 编码 --------------------------- #
def encode_tensor(gts: np.ndarray, depth: int):
    """
    gts: int32 (n_samples, n_snps)
    depth: 由 compute_depth_and_map 给出
    返回 one-hot (n_samples, n_snps, depth)
    """
    gts = gts.copy()
    gts[gts < 0] = depth - 1
    tensor = torch.from_numpy(gts.astype(np.int64))
    onehot = torch.nn.functional.one_hot(tensor, num_classes=depth).float()
    return onehot

if __name__ == "__main__":
    cfg = load_config("config/config.json")
    os.makedirs(cfg.data.out_dir, exist_ok=True)

    phased = bool(cfg.data.tihp)

     # 1) 训练集一次性读取 + 计算 depth
    train_gts, train_samples, depth = read_vcf(cfg.data.train_vcf, phased)
    print(f"Inferred unified depth = {depth}")

    # 2) 处理训练集
    dis_train = read_dismat(cfg.data.train_dismat, train_samples)
    var_train = encode_tensor(train_gts, depth)
    torch.save({'var_site': var_train, 'p_dis': torch.from_numpy(dis_train)},
               os.path.join(cfg.data.out_dir, "train.pt"))
    print(f"Saved train.pt | var_site={tuple(var_train.shape)}")

    # 3) 处理验证集（用同一 depth）
    val_gts, val_samples, _ = read_vcf(cfg.data.val_vcf, phased)
    dis_val = read_dismat(cfg.data.val_dismat, val_samples)
    var_val = encode_tensor(val_gts, depth)
    torch.save({'var_site': var_val, 'p_dis': torch.from_numpy(dis_val)},
               os.path.join(cfg.data.out_dir, "val.pt"))
    print(f"Saved val.pt   | var_site={tuple(var_val.shape)}")