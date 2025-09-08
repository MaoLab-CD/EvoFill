#!/usr/bin/env python3
"""
transform vcf to tensor in stici with p-distance
Input: .vcf.gz, dismat.tsv
Ouput:
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from cyvcf2 import VCF

from src.utils import load_config


# --------------------------- 读取 VCF --------------------------- #
def read_vcf(path: str, phased: bool, out_dir: str):
    """
    返回 (gts, samples, depth)，
    同时在 out_dir 下保存 var_index.pt
    """
    vcf = VCF(path)
    samples = list(vcf.samples)

    gts_list = []
    var_index_list = []      # 新增：记录每个位点的离散值个数

    total = sum(1 for _ in VCF(path))

    for var in tqdm(vcf, total=total, desc="Parsing VCF"):
        alleles = [var.REF] + var.ALT
        allele2idx = {a: i for i, a in enumerate(alleles)}

        row = []
        for gt_str in var.gt_bases:
            if gt_str in ['.|.', './.']:
                row.append(-1)          # 统一用 -1 表示缺失
            else:
                a1, a2 = gt_str.split('|' if phased else '/')
                row.append(allele2idx[a1] + allele2idx[a2])

        row = np.array(row, dtype=np.int32)
        gts_list.append(row)

        # -------- 计算该位点的离散值个数 --------
        observed = row[row >= 0]
        var_index_list.append(int(np.unique(observed).size))
        # --------------------------------------

    gts = np.vstack(gts_list).T.astype(np.int32)

    flat = gts[gts >= 0]
    depth = int(flat.max()) + 2

    var_index = torch.tensor(var_index_list, dtype=torch.int8)

    return gts, samples, var_index, depth

# # --------------------------- 读取距离矩阵 --------------------------- #
# def read_dismat(path: str, sample_order):
#     dis = pd.read_csv(path, sep='\t', skiprows=[0], header=None, index_col=0)
#     dis.index = dis.columns = [s.strip() for s in dis.index]
#     return dis.loc[sample_order, sample_order].values

# def random_dismat(path: str, sample_order):
#     dis = pd.read_csv(path, sep='\t', skiprows=[0], header=None, index_col=0)
#     dis.index = dis.columns = [s.strip() for s in dis.index]

#     # 2. 生成随机、对角对称矩阵
#     n = len(dis)                         # 维度
#     rng = np.random.default_rng(42)      # 固定种子，结果可复现
#     rand_vals = rng.random((n, n))

#     sym = np.triu(rand_vals) + np.triu(rand_vals, k=1).T
#     np.fill_diagonal(sym, 0)   # 对角线0

#     random_dis = pd.DataFrame(sym,
#                         index=dis.index,
#                         columns=dis.columns)

#     return random_dis.values

# --------------------------- 编码 --------------------------- #
def encode_tensor(gts: np.ndarray, depth: int):
    """
    gts: int8 (n_samples, n_snps)
    """
    gts = gts.copy()
    gts[gts < 0] = depth - 1
    tensor = torch.from_numpy(gts.astype(np.int8))
    return tensor

if __name__ == "__main__":
    cfg = load_config("config/config.json")
    os.makedirs(cfg.data.out_dir, exist_ok=True)

    phased = bool(cfg.data.tihp)

     # 1) 训练集一次性读取 + 计算 depth
    train_gts, train_samples,var_index, depth = read_vcf(cfg.data.train_vcf, phased, cfg.data.out_dir)
    torch.save(var_index, os.path.join(cfg.data.out_dir, "var_index.pt"))
    print(f"Inferred unified depth = {depth}")

    # 2) 处理训练集
    var_train = encode_tensor(train_gts, depth)
    torch.save({"var_site":var_train, "sample_id":train_samples}, os.path.join(cfg.data.out_dir, "train.pt"))
    print(f"Saved train.pt | shape = {tuple(var_train.shape)}")

    # 3) 处理验证集（用同一 depth）
    val_gts, val_samples, _, _ = read_vcf(cfg.data.val_vcf, phased, cfg.data.out_dir)
    var_val = encode_tensor(val_gts, depth)

    torch.save({"var_site":var_val, "sample_id":val_samples}, os.path.join(cfg.data.out_dir, "val.pt"))

    print(f"Saved val.pt   | shape = {tuple(var_val.shape)}")
