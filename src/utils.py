import os
import shutil
import numpy as np
import torch
import json
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd

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

def set_seed(seed=42):
    random.seed(seed)                # Python 内置 random 模块
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch 的 CPU 和 CUDA 的通用随机种子
    torch.cuda.manual_seed(seed)     # 当前 GPU
    torch.cuda.manual_seed_all(seed) # 所有 GPU（多卡训练时）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_workdir(save_dir, ) -> None:
    """Create necessary directories"""
    sub_dir = ["models", "train", "augment", "finetune", "impute_in", "impute_out"]
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for dd in sub_dir:
        if not os.path.exists(f"{save_dir}/{dd}"):
            os.makedirs(f"{save_dir}/{dd}")

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

    for l in range(L):
        alleles = gts_np[:, l]
        alleles = alleles[alleles != mask_int]   # 去掉缺失
        if alleles.size == 0:
            maf[l] = 0.0
            continue

        uniq, cnt = np.unique(alleles, return_counts=True)
        total = cnt.sum()
        freq = cnt / total
        freq[::-1].sort()
        maf_val = freq[1] if len(freq) > 1 else 0.0
        maf[l] = maf_val

        # 统计 bin
        for i, (lo, hi) in enumerate(MAF_BINS):
            if lo <= maf_val < hi:
                bin_cnt[i] += 1
                break

    return maf, bin_cnt

def build_geno3_map_from_hapmap(hap_map: dict) -> np.ndarray:
    # 定义一个映射：从基因型到分类
    gt_to_class = {}
    for gt in hap_map.keys():
        if gt in ('.|.', './.'):
            continue
        sep = '|' if '|' in gt else ('/' if '/' in gt else None)
        a, b = gt.split(sep) if sep else (gt, gt)
        try:
            ai, bi = int(a), int(b)
        except Exception:
            gt_to_class[gt] = 1
            continue
        if ai == bi == 0:
            gt_to_class[gt] = 0
        elif ai != bi:
            gt_to_class[gt] = 1
        else:
            gt_to_class[gt] = 2

    # 确保输出顺序为 [0, 1, 2]
    result = []
    for cls in [0, 1, 2]:
        found = False
        for gt, c in gt_to_class.items():
            if c == cls:
                result.append(c)
                found = True
                break
        if not found:
            raise ValueError(f"Missing class {cls} in hap_map")

    return np.array(result, dtype=np.int64)

# ---------- 2. 线程安全缓存 ----------
MAF_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]
_GENO3_CACHE: Dict[int, np.ndarray] = {}
_GENO3_LOCK = torch.multiprocessing.Lock()

def get_geno3_map(C_orig: int, hap_map) -> np.ndarray:
    key = int(C_orig)
    with _GENO3_LOCK:
        t = _GENO3_CACHE.get(key)
        if t is None:
            arr = build_geno3_map_from_hapmap(hap_map)  # 假设 gt_enc 已全局可见
            if arr.shape[0] != C_orig:
                raise RuntimeError(f"三分类映射长度{arr.shape[0]}与类别数{C_orig}不符")
            t = arr
            _GENO3_CACHE[key] = t
    return t

# ---------- 3. 三分类聚合 ----------
def aggregate_three_classes(prob: np.ndarray, y_true: np.ndarray, hap_map) -> Tuple[np.ndarray, np.ndarray]:
    _, _, C = prob.shape
    gmap = get_geno3_map(C,hap_map, )
    W = np.zeros((C, 3), dtype=prob.dtype)
    W[np.arange(C), gmap] = 1.0
    prob3 = prob @ W
    y3    = y_true @ W
    prob3 = prob3 / prob3.sum(-1, keepdims=True).clip(min=1e-8)
    return prob3, y3

# ---------- 4. 向量化计算 3 个指标 ----------
def _compute_site_metrics(prob3: np.ndarray,
                          y3: np.ndarray,
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一次性返回 (INFO, MaCH-Rsq, IQS) 三个 (L,) 向量
    prob3/y3: (N,L,3)  mask: (N,L)
    """
    p_ref, p_het, p_hom = prob3[..., 0], prob3[..., 1], prob3[..., 2]
    dosage  = p_het + 2 * p_hom
    W_score = p_het + 4 * p_hom

    n_valid = mask.sum(axis=0)                  # (L,)
    AF = 0.5 * (dosage * mask).sum(axis=0) / n_valid.clip(min=1)
    denom_info = AF * (1 - AF)

    var_want = ((W_score - dosage ** 2) * mask).sum(axis=0) / n_valid.clip(min=1)
    info = 1 - 0.5 * var_want / denom_info.clip(min=1e-8)
    info = info.clip(0, 1)

    # MaCH
    true_dos = y3[..., 1] + 2 * y3[..., 2]
    pred_dos = dosage
    mean_true = (true_dos * mask).sum(axis=0) / n_valid.clip(min=1)
    mean_pred = (pred_dos * mask).sum(axis=0) / n_valid.clip(min=1)

    num = ((pred_dos - mean_pred) ** 2 * mask).sum(axis=0) / n_valid.clip(min=1)
    AF2 = mean_true / 2.0
    denom = AF2 * (1 - AF2)
    mach = num / denom.clip(min=1e-8)
    mach = mach.clip(0, 1)

    # IQS
    pred_cls = prob3.argmax(axis=-1)
    true_cls = y3.argmax(axis=-1)
    agree = (pred_cls == true_cls) & mask
    Po = agree.sum(axis=0).astype(float) / n_valid.clip(min=1)
    Pe = np.zeros_like(Po)
    for c in range(3):
        p_c = ((pred_cls == c) & mask).sum(axis=0).astype(float) / n_valid.clip(min=1)
        t_c = ((true_cls == c) & mask).sum(axis=0).astype(float) / n_valid.clip(min=1)
        Pe += p_c * t_c
    iqs = (Po - Pe) / (1 - Pe).clip(min=1e-8)
    iqs = iqs.clip(-1, 1)

    # 无效位点填 0
    invalid = n_valid == 0
    info[invalid] = mach[invalid] = iqs[invalid] = 0
    return info, mach, iqs

# ---------- 5. 唯一对外接口 ----------
def metrics_by_maf(prob: torch.Tensor,
                   y_true: torch.Tensor,
                   hap_map : Dict,
                   maf_vec: np.ndarray,
                   bins: List[Tuple[float, float]] = MAF_BINS,
                   mask: Optional[torch.Tensor] = None
                   ) -> Dict[str, List[float]]:
    """
    返回 dict: {'Acc':[...], 'INFO':[...], 'MaCH':[...], 'IQS':[...]}
    顺序与 bins 一致
    """
    with torch.no_grad():
        if isinstance(prob, torch.Tensor):
            prob = prob.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if mask is None:
            mask = np.ones(prob.shape[:2], dtype=bool)
        else:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = mask.astype(bool)

        # 三分类
        prob3, y3 = aggregate_three_classes(prob, y_true, hap_map)

        # --- 5.1 accuracy 向量化 ---
        preds = prob3.argmax(-1)
        gts   = y3.argmax(-1)
        correct = (preds == gts) & mask                      # (N,L)
        acc_bins = []
        for lo, hi in bins:
            mbin = mask & (maf_vec >= lo) & (maf_vec < hi)
            n_cor = np.count_nonzero(correct & mbin)
            n_tot = np.count_nonzero(mbin)
            acc_bins.append(n_cor / n_tot if n_tot > 0 else 0.)

        # --- 5.2 其余 3 个指标 ---
        info_all, mach_all, iqs_all = _compute_site_metrics(prob3, y3, mask)
        info_bins, mach_bins, iqs_bins = [], [], []
        for lo, hi in bins:
            idx = (maf_vec >= lo) & (maf_vec < hi)
            if idx.sum() == 0:
                info_bins.append(0.); mach_bins.append(0.); iqs_bins.append(0.)
            else:
                info_bins.append(float(info_all[idx].mean()))
                mach_bins.append(float(mach_all[idx].mean()))
                iqs_bins.append(float(iqs_all[idx].mean()))

    return {'Acc': acc_bins, 'INFO': info_bins,
            'MaCH': mach_bins, 'IQS': iqs_bins}

# ---------- 6. 打印 ----------
def print_maf_stat_df(
    chunk_bin_cnt: List[int],
    bins_metrics: Dict[str, Dict[str, List[float]]],  # 外层 key 就是列前缀
    maf_bins: List[str] = None,
) -> None:
    """
    打印 MAF 分箱统计表。

    Args:
        chunk_bin_cnt: 每个 bin 的 SNP 数量，长度须与 maf_bins 一致。
        bins_metrics: 任意组 metrics，例如
                      {"train": train_bins_metrics,
                       "val"  : val_bins_metrics,
                       "test" : test_bins_metrics}
        maf_bins:    手动指定 MAF 区间，默认使用原文的 6 个区间。
    """
    if maf_bins is None:
        maf_bins = ['(0.00, 0.05)', '(0.05, 0.10)', '(0.10, 0.20)',
                    '(0.20, 0.30)', '(0.30, 0.40)', '(0.40, 0.50)']

    if len(chunk_bin_cnt) != len(maf_bins):
        raise ValueError("len(chunk_bin_cnt) != len(maf_bins)")

    # 1. 构造基础 DataFrame
    df = pd.DataFrame({'MAF_bin': maf_bins,
                       'Counts':  [f"{c}" for c in chunk_bin_cnt]})

    # 2. 动态拼接所有指标列
    #    指标名从第一组 metrics 里提取，保持顺序一致
    first_group = next(iter(bins_metrics.values()))
    metrics_names = sorted(first_group.keys())   # 例如 ['Acc', 'INFO', 'IQS', 'MaCH']

    for prefix in sorted(bins_metrics.keys()):   # 按字母序，保证列顺序稳定
        grp = bins_metrics[prefix]
        for m in metrics_names:
            col_name = f"{prefix}_{m}"
            df[col_name] = [f"{v:.3f}" for v in grp[m]]

    # 3. 打印（不带行索引）
    print(df.to_string(index=False))