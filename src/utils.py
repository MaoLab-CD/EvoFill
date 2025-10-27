import os
import shutil
import numpy as np
import torch
import json
from types import SimpleNamespace

MAF_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]

def load_config(path: str) -> SimpleNamespace:
    def boolify(obj):
        """把字符串 true/false 转成 Python 布尔值，其余原样返回。"""
        if isinstance(obj, str):
            lower = obj.lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
        return obj

    def hook(d):
        return SimpleNamespace(**{
            k: (hook(v) if isinstance(v, dict) else boolify(v))
            for k, v in d.items()
        })

    with open(path, encoding="utf-8") as f:
        return json.load(f, object_hook=hook)

def create_directories(save_dir, models_dir="models", outputs="out") -> None:
    """Create necessary directories"""
    for dd in [save_dir, f"{save_dir}/{models_dir}", f"{save_dir}/{outputs}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)

def clear_dir(path) -> None:
    """Clear directory if it exists"""
    if os.path.exists(path):
        shutil.rmtree(path)

def precompute_maf(gts_np, mask_int=-1):
    """
    gts_np: (N, L)  int64
    return:
        maf: (L,) float32
        bin_cnt: list[int] 长度 6，对应 6 个 bin 的位点数量
    """
    L = gts_np.shape[1]
    maf = np.zeros(L, dtype=np.float32)
    bin_cnt = [0] * 6

    for li in range(L):
        alleles = gts_np[:, li]
        alleles = alleles[alleles != mask_int]   # 去掉缺失
        if alleles.size == 0:
            maf[li] = 0.0
            continue

        uniq, cnt = np.unique(alleles, return_counts=True)
        total = cnt.sum()
        freq = cnt / total
        freq[::-1].sort()
        maf_val = freq[1] if len(freq) > 1 else 0.0
        maf[li] = maf_val

        # 统计 bin
        for i, (lo, hi) in enumerate(MAF_BINS):
            if lo <= maf_val < hi:
                bin_cnt[i] += 1
                break

    return maf, bin_cnt

def imputation_maf_accuracy_epoch(all_logits, all_gts, global_maf, mask=None):
    """
    all_logits: (N, L, C)
    all_gts:    (N, L, C) one-hot
    global_maf: (L,)
    mask:       (N, L) 或 None
    return:     list[float] 长度 6
    """
    # 1. 预测 vs 真实
    all_gts = all_gts.argmax(dim=-1)      # (N, L)
    preds   = all_logits.argmax(dim=-1)   # (N, L)

    # 2. 如果没有外部 mask，就默认全 1
    if mask is None:
        mask = torch.ones_like(all_gts, dtype=torch.bool)   # (N, L)
    correct = (preds == all_gts) & mask                   # (N, L)

    # 3. MAF 条件 -> (1, L) 再广播到 (N, L)
    maf = global_maf.unsqueeze(0)                         # (1, L)

    # 4. 分 bin 计算
    accs = []
    for lo, hi in MAF_BINS:
        maf_bin = mask & (maf >= lo) & (maf < hi)                # (1, L)
        n_cor = (correct & maf_bin).sum()
        n_tot = maf_bin.sum()
        accs.append(100*(n_cor / n_tot).item() if n_tot > 0 else 0.0)
    return accs