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
    sorted_items = sorted(hap_map.items(), key=lambda kv: kv[1])
    three_class = []
    for gt, idx in sorted_items:
        if gt in ('.|.', './.'):
            continue
        sep = '|' if '|' in gt else ('/' if '/' in gt else None)
        a, b = (gt.split(sep) if sep else (gt, gt))
        try:
            ai, bi = int(a), int(b)
        except Exception:
            three_class.append(1); continue
        if ai == bi == 0:
            three_class.append(0)
        elif ai != bi:
            three_class.append(1)
        else:
            three_class.append(2)
    return np.array(three_class, dtype=np.int64)

# ---------- 2. 线程安全缓存 ----------
MAF_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]
_GENO3_CACHE: Dict[int, torch.Tensor] = {}
_GENO3_LOCK = torch.multiprocessing.Lock()

def get_geno3_map_tensor(C_orig: int, hap_map, device: torch.device) -> torch.Tensor:
    key = int(C_orig)
    with _GENO3_LOCK:
        t = _GENO3_CACHE.get(key)
        if t is None:
            arr = build_geno3_map_from_hapmap(hap_map)  # 假设 gt_enc 已全局可见
            if arr.shape[0] != C_orig:
                raise RuntimeError(f"三分类映射长度{arr.shape[0]}与类别数{C_orig}不符")
            t = torch.from_numpy(arr)
            _GENO3_CACHE[key] = t
    return t.to(device)

# ---------- 3. 三分类聚合 ----------
def aggregate_three_classes(prob: torch.Tensor, y_true: torch.Tensor, hap_map) -> Tuple[torch.Tensor, torch.Tensor]:
    N, L, C = prob.shape
    device = prob.device
    gmap = get_geno3_map_tensor(C,hap_map, device)
    W = torch.zeros(C, 3, device=device)
    W[torch.arange(C, device=device), gmap.long()] = 1.0
    prob3 = torch.einsum('nlc,ck->nlk', prob, W)
    y3    = torch.einsum('nlc,ck->nlk', y_true, W)
    prob3 = prob3 / prob3.sum(-1, keepdim=True).clamp(min=1e-8)
    return prob3, y3

# ---------- 4. 向量化计算 3 个指标 ----------
def _compute_site_metrics(prob3: torch.Tensor,
                          y3: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    一次性返回 (INFO, MaCH-Rsq, IQS) 三个 (L,) 向量
    prob3/y3: (N,L,3)  mask: (N,L)
    """
    # dosage / W / p_alt
    p_ref, p_het, p_hom = prob3.unbind(-1)
    dosage = p_het + 2*p_hom
    W_score = p_het + 4*p_hom

    # 按位点求平均
    n_valid = mask.sum(0)                        # (L,)
    AF = 0.5 * (dosage * mask).sum(0) / n_valid.clamp(min=1)
    denom_info = AF * (1 - AF)

    # INFO
    var_want = ((W_score - dosage.square()) * mask).sum(0) / n_valid.clamp(min=1)
    info = 1 - 0.5 * var_want / denom_info.clamp(min=1e-8)
    info = info.clamp(0, 1)

    # MaCH-Rsq
    # 真实剂量
    true_dosage = (y3[..., 1] + 2*y3[..., 2]).float()        # (N,L)
    # 预测剂量
    pred_dosage = dosage                                       # (N,L) 前面已算好
    # 有效样本均值
    mean_true = (true_dosage * mask).sum(0) / n_valid.clamp(min=1)
    mean_pred = (pred_dosage * mask).sum(0) / n_valid.clamp(min=1)

    # 分子：协方差（= 预测对真实解释的方差）
    num = ((pred_dosage - mean_pred.unsqueeze(0)).square() * mask).sum(0) / n_valid.clamp(min=1)

    # 分母：由真实剂量得到的 AF*(1-AF)
    AF = mean_true / 2.0
    denom = AF * (1 - AF)
    mach = num / denom.clamp(min=1e-8)
    mach = mach.clamp(0, 1)

    # IQS (Cohen's kappa)
    pred_cls = prob3.argmax(-1)                  # (N,L)
    true_cls = y3.argmax(-1)
    agree = (pred_cls == true_cls) & mask        # (N,L)
    Po = (agree.sum(0)).float() / n_valid.clamp(min=1)
    Pe = torch.zeros_like(Po)
    for c in range(3):
        p_c = ((pred_cls == c) & mask).sum(0).float() / n_valid.clamp(min=1)
        t_c = ((true_cls == c) & mask).sum(0).float() / n_valid.clamp(min=1)
        Pe += p_c * t_c
    iqs = (Po - Pe) / (1 - Pe).clamp(min=1e-8)
    iqs = iqs.clamp(-1, 1)

    # 无效位点填 0
    invalid = n_valid == 0
    info[invalid] = 0
    mach[invalid] = 0
    iqs[invalid]  = 0
    return info, mach, iqs

# ---------- 5. 唯一对外接口 ----------
def metrics_by_maf(prob: torch.Tensor,
                   y_true: torch.Tensor,
                   hap_map : Dict,
                   maf_vec: torch.Tensor,
                   bins: List[Tuple[float, float]] = MAF_BINS,
                   mask: Optional[torch.Tensor] = None
                   ) -> Dict[str, List[float]]:
    """
    返回 dict: {'Acc':[...], 'INFO':[...], 'MaCH':[...], 'IQS':[...]}
    顺序与 bins 一致
    """
    N, L, _ = prob.shape
    device = prob.device
    if mask is None:
        mask = torch.ones((N, L), dtype=torch.bool, device=device)

    # 三分类
    prob3, y3 = aggregate_three_classes(prob, y_true, hap_map)

    # --- 5.1 accuracy 向量化 ---
    preds = prob3.argmax(-1)
    gts   = y3.argmax(-1)
    correct = (preds == gts) & mask                      # (N,L)
    maf_b = maf_vec.unsqueeze(0)                         # (1,L)
    acc_bins = []
    for lo, hi in bins:
        mbin = mask & (maf_b >= lo) & (maf_b < hi)
        n_cor = (correct & mbin).sum()
        n_tot = mbin.sum()
        acc_bins.append((n_cor / n_tot).item() if n_tot > 0 else 0.)

    # --- 5.2 其余 3 个指标 ---
    info_all, mach_all, iqs_all = _compute_site_metrics(prob3, y3, mask)
    info_bins, mach_bins, iqs_bins = [], [], []
    for lo, hi in bins:
        idx = (maf_vec >= lo) & (maf_vec < hi)
        if idx.sum() == 0:
            info_bins.append(0.); mach_bins.append(0.); iqs_bins.append(0.)
        else:
            info_bins.append(info_all[idx].mean().item())
            mach_bins.append(mach_all[idx].mean().item())
            iqs_bins.append(iqs_all[idx].mean().item())

    return {'Acc': acc_bins, 'INFO': info_bins,
            'MaCH': mach_bins, 'IQS': iqs_bins}

# ---------- 6. 打印 ----------
def print_maf_stat_df(chunk_bin_cnt: List[int],
                      train_bins_metrics: Dict[str, List[float]],
                      val_bins_metrics: Dict[str, List[float]]):
    maf_df = pd.DataFrame({
        'MAF_bin': ['(0.00, 0.05)', '(0.05, 0.10)', '(0.10, 0.20)',
                    '(0.20, 0.30)', '(0.30, 0.40)', '(0.40, 0.50)'],
        'Counts':  [f"{c}" for c in chunk_bin_cnt],
        'Train_Acc':   [f"{v:.3f}" for v in train_bins_metrics['Acc']],
        'Val_Acc':     [f"{v:.3f}" for v in val_bins_metrics['Acc']],
        'Train_INFO':  [f"{v:.3f}" for v in train_bins_metrics['INFO']],
        'Val_INFO':    [f"{v:.3f}" for v in val_bins_metrics['INFO']],
        'Train_MaCH':  [f"{v:.3f}" for v in train_bins_metrics['MaCH']],
        'Val_MaCH':    [f"{v:.3f}" for v in val_bins_metrics['MaCH']],
        'Train_IQS':   [f"{v:.3f}" for v in train_bins_metrics['IQS']],
        'Val_IQS':     [f"{v:.3f}" for v in val_bins_metrics['IQS']],
    })
    print(maf_df.to_string(index=False))