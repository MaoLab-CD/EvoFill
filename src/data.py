
import os
import json
from typing import Optional, Dict
import numpy as np
import pandas as pd
from cyvcf2 import VCF
import scipy.sparse as sp
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler

class GenotypeEncoder:
    # ========= 1. 初始化 =========
    def __init__(self,  
                 phased: bool = True,
                 gts012: bool = False,
                 save2disk: bool = False,
                 save_dir: Optional[str] = None):
        self.gts012 = gts012
        self.phased = phased
        self.save2disk = save2disk
        self.save_dir = save_dir
        # 其余占位
        self.hap_map: Dict[str, int] = {}
        self.default_gt = 0
        self.seq_depth: Optional[int] = None
        self.n_samples = 0
        self.n_variants = 0
        self.sample_ids = []
        self.variant_ids = []
        self.X_gt: Optional[sp.csc_matrix] = None
        self.evo_mat: Optional[np.ndarray] = None

    # ========= 2. 首次编码：建立标准 =========
    def encode_new(self, vcf_path: str, evo_mat: Optional[str] = None):
        """第一次编码，允许新建 hap_map 与 seq_depth"""
        self.vcf_path = vcf_path
        # 清空头一次可能遗留的参照
        self.hap_map.clear()
        self.seq_depth = None
        self.evo_mat_path = evo_mat
        self.phased = self.phased if self.evo_mat_path is None else False
        # 真正干活
        self.X_gt = self._load_gt()
        self.evo_mat = self._load_evo_mat() if self.evo_mat_path else None

        self.seq_depth = self.X_gt.data.max() + 2 # 0=ref; -1=miss; 1,2,..=alt

        if self.save2disk:
            if self.save_dir:
                self._save_meta()
            else:
                print("[DATA] save_dir is not given")
        
        print(f'[DATA] Genotype matrix = {self.X_gt.shape}, sparsity = {self.X_gt.nnz / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}')
        print(f'[DATA] Genotype dictionary = {self.hap_map}, dictionary depth = {self.seq_depth}')
        return self

    # ========= 3. 参照编码：必须 100 % 一致 =========
    def encode_ref(self, ref_meta_json: str, vcf_path: str, evo_mat: Optional[str] = None):
        """按已有 meta 编码新 VCF，任何不一致都抛异常"""
        if not os.path.isfile(ref_meta_json):
            raise FileNotFoundError(ref_meta_json)
        with open(ref_meta_json) as f:
            meta = json.load(f)
        # 加载参照标准
        ref_hap_map = {k: int(v) for k, v in meta["hap_map"].items()}
        ref_seq_depth = int(meta["seq_depth"])
        ref_phased = meta["phased"].lower() == "true"
        ref_gts012 = meta["gts012"].lower() == "true"

        # 强制同步参数
        self.phased, self.gts012 = ref_phased, ref_gts012
        self.hap_map = ref_hap_map.copy()
        self.seq_depth = ref_seq_depth

        # 编码
        self.vcf_path = vcf_path
        self.X_gt = self._load_gt()

        self.evo_mat_path = evo_mat
        self.evo_mat = self._load_evo_mat() if self.evo_mat_path else None

        if self.save2disk:
            if self.save_dir:
                self._save_meta()
            else:
                print("[DATA] save_dir is not given")
        
        miss_rate = np.sum((self.X_gt.data == - 1)) 
        print(f'[DATA] Genotype matrix = {self.X_gt.shape}, sparsity = {self.X_gt.nnz / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}, missing rate = {miss_rate / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}')
        print(f'[DATA] Genotype dictionary = {self.hap_map}, dictionary depth = {self.seq_depth}')

        return self

    # ========= 4. 内部工具：encode_gt 仅增加“参照模式”判断 =========
    def _encode_gt(self, rec, phase=False, gts012=False):
        """参照模式下遇到新等位基因直接抛异常"""
        n = self.n_samples
        ref_mode = bool(self.hap_map) and self.seq_depth is not None

        if phase: # 基因型模式
            out = np.empty(2 * n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                for j, a in enumerate((a1, a2)):
                    key = '.' if a == -1 else str(a)
                    idx = 2 * i + j
                    if a == -1:
                        code = -1
                    else:
                        code = (0 if a == 0 else (2 if a >= 2 else 1)) if gts012 else a
                    if ref_mode:
                        if key not in self.hap_map:
                            raise ValueError(
                                f"[encode_ref] New allele '{key}' at {rec.CHROM}:{rec.POS} "
                                "not in reference hap_map."
                            )
                        if self.hap_map[key] != code:
                            raise ValueError(
                                f"[encode_ref] Allele '{key}' code mismatch: "
                                f"ref={self.hap_map[key]} vs current={code}"
                            )
                    else:
                        self.hap_map[key] = code
                    out[idx] = code
            if not ref_mode:
                self.hap_map['.'] = -1
            return out
        else: # 剂量模式
            out = np.empty(n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                sep = '|' if _phased else '/'
                if a1 == -1 or a2 == -1:
                    key = sep.join(sorted(['.', '.']))
                    code = -1
                else:
                    code = (1 if a1 > 0 else 0) + (1 if a2 > 0 else 0) if gts012 else \
                           (0 if a1 == 0 else a1) + (0 if a2 == 0 else a2)
                    a1s, a2s = ('.' if x == -1 else str(x) for x in (a1, a2))
                    key = sep.join(sorted([a1s, a2s]))
                if ref_mode:
                    if key not in self.hap_map:
                        raise ValueError(
                            f"[encode_ref] New diploid genotype '{key}' at {rec.CHROM}:{rec.POS} "
                            "not in reference hap_map."
                        )
                    if self.hap_map[key] != code:
                        raise ValueError(
                            f"[encode_ref] Diploid genotype '{key}' code mismatch: "
                            f"ref={self.hap_map[key]} vs current={code}"
                        )
                else:
                    self.hap_map[key] = code
                out[i] = code
            if not ref_mode:
                self.hap_map[sep.join(sorted(['.', '.']))] = -1
            return out

    def _load_gt(self):
        interval = 10000
        cols, data, indptr = [], [], [0]
        vcf = VCF(self.vcf_path, gts012=self.gts012)
        self.sample_ids = vcf.samples
        self.n_samples = len(self.sample_ids)
        self.n_variants = 0
        self.variant_ids = []
        for rec in vcf:
            vec = self._encode_gt(rec, phase=self.phased, gts012=self.gts012)
            nz_idx = np.flatnonzero(vec)
            cols.extend(nz_idx)
            data.extend(vec[nz_idx])
            indptr.append(indptr[-1] + len(nz_idx))
            self.n_variants += 1
            self.variant_ids.append(f"{rec.CHROM}:{rec.POS}_{rec.REF}/{','.join(rec.ALT)}")
            if self.n_variants % interval == 0:
                print(f'\r[DATA] Encoded {self.n_variants:,} variants', end='', flush=True)
        print(f'\r[DATA] Total {self.n_variants:,} variants  ', flush=True)
        vcf.close()
        n_rows = 2 * self.n_samples if self.phased else self.n_samples
        M = sp.csc_matrix((data, cols, indptr), shape=(n_rows, self.n_variants), dtype=np.int8)
        return M

    def _load_evo_mat(self):
        try:
            df = pd.read_csv(self.evo_mat_path, sep='\t', index_col=0)
            df = df.loc[self.sample_ids, self.sample_ids]
            np.fill_diagonal(df.values, 0)
            print(f"[DATA] EvoMat shape: {df.shape}")
            return df.values.astype(np.float32)
        except Exception as e:
            print(f"[DATA] EvoMat skipped: {e}")
            return None

    def _save_meta(self):
        def _make_json_safe(obj):
            if isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_make_json_safe(i) for i in obj]
            if isinstance(obj, np.ndarray):
                return _make_json_safe(obj.tolist())
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        os.makedirs(self.save_dir, exist_ok=True)
        meta = {
            "vcf_path": str(self.vcf_path),
            "evo_mat": str(self.evo_mat_path),
            "phased": str(self.phased),
            "gts012": str(self.gts012),
            "default_gt": str(self.default_gt),
            "n_samples": str(self.n_samples),
            "n_variants": str(self.n_variants),
            "seq_depth": str(self.seq_depth),
            "hap_map": _make_json_safe(self.hap_map),
        }
        with open(os.path.join(self.save_dir, "gt_enc_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if self.evo_mat is not None:
            np.save(os.path.join(self.save_dir, "evo_mat.npy"), self.evo_mat)

        # 同时落盘矩阵、样本、位点
        sp.save_npz(os.path.join(self.save_dir, "gt_matrix.npz"), self.X_gt)
        with open(os.path.join(self.save_dir, "gt_samples.txt"), "w") as f:
            for s in self.sample_ids:
                f.write(f"{s}_A\n{s}_B\n" if self.phased else f"{s}\n")
        with open(os.path.join(self.save_dir, "gt_variants.txt"), "w") as f:
            for vid in self.variant_ids:
                f.write(vid + "\n")
        print(f"[DATA] Results have been written to {self.save_dir}")



    def gts_toarray(self, idx=None):
        """
        把稀疏矩阵还原成 dense 数组。
        参数
        ----
        idx : None | int | list/ndarray
            None  -> 返回整个矩阵  (n_rows, n_variants)
            int   -> 返回单行      (1, n_variants)
            序列  -> 返回多行      (len(idx), n_variants)

        返回值
        ------
        dense : ndarray, dtype=int8
            缺失值 = -1，参考型 = 0，alt = 原值
        """
        # 1. 统一成“行索引数组”
        if idx is None:
            row_idx = np.arange(self.X_gt.shape[0])
        elif np.isscalar(idx):
            row_idx = np.array([idx], dtype=int)
        else:
            row_idx = np.asarray(idx, dtype=int)

        # 2. 拿子矩阵
        sub = self.X_gt[row_idx]          # scipy 支持 fancy indexing

        dense = sub.toarray().astype(np.int8)           # 形状 (len(row_idx), n_variants)

        return dense

    @classmethod
    def loadfromdisk(cls, work_dir: Path):
        """
        反向构造 GenotypeEncoder，要求 work_dir 里必须有：
            gt_matrix.npz      -> X_gt  (scipy.sparse.csc_matrix)
            gt_samples.txt     -> sample_ids
            gt_variants.txt    -> variant_ids
            gt_enc_meta.json   -> 其余标量 / 布尔 / 路径信息
        """
        # 1. 读 meta（构造 __init__ 需要的几个“外部”参数）
        meta_path = os.path.join(work_dir, "gt_enc_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} 不存在，无法反序列化")
        with open(meta_path) as f:
            meta = json.load(f)

        # 2. 先“假”构造一个对象（不触发 VCF 扫描）
        #    把关键字段先填进去，避免 __init__ 里再去读 VCF
        obj = cls.__new__(cls)  # 不调用 __init__
        obj.vcf_path    = meta["vcf_path"]
        obj.evo_mat     = meta["evo_mat"]
        obj.phased      = bool(meta["phased"])
        obj.gts012      = bool(meta["gts012"])
        obj.n_samples   = int(meta["n_samples"])
        obj.n_variants  = int(meta["n_variants"])
        obj.seq_depth   = int(meta["seq_depth"])
        obj.hap_map     = meta["hap_map"]

        # 3. 读样本 & 位点 ID 列表
        obj.sample_ids = [
            line.rstrip("\n") for line in open(os.path.join(work_dir, "gt_samples.txt"))
        ]
        obj.variant_ids = [
            line.rstrip("\n") for line in open(os.path.join(work_dir, "gt_variants.txt"))
        ]

        # 4. 读稀疏矩阵
        obj.X_gt = sp.load_npz(os.path.join(work_dir, "gt_matrix.npz"))
        if obj.X_gt.format != 'csc':
            obj.X_gt = obj.X_gt.tocsc() 

        # 5. 读 extra（如果有）
        evo_mat_path = os.path.join(work_dir, "evo_mat.npy")
        if os.path.exists(evo_mat_path):
            obj.evo_mat = np.load(evo_mat_path)
        else:
            obj.evo_mat = None

        return obj

class GenomicDataset(Dataset):
    """Dataset class for genomic data with masking for training"""
    def __init__(self, gt_encoder,evo_mat=None,
                 mask=True, masking_rates=(0.5, 0.99), indices=None):
        """
        x_gts_sparse: scipy.sparse.csr_matrix or similar
        evo_mat: numpy array or None
        indices: 可选，指定要使用的样本索引（如 train/valid 索引）
        """
        self.gt_enc = gt_encoder
        self.evo_mat = evo_mat
        self.seq_depth = gt_encoder.seq_depth
        self.mask = mask
        self.masking_rates = masking_rates
        self.indices = indices if indices is not None else np.arange(self.gt_enc.X_gt.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.gt_enc.gts_toarray(real_idx).squeeze()
        y = x.copy()

        if self.mask:
            seq_len = len(x)
            masking_rate = np.random.uniform(*self.masking_rates)
            mask_size = int(seq_len * masking_rate)
            mask_indices = np.random.choice(seq_len, mask_size, replace=False)
            x[mask_indices] = self.seq_depth - 1  # missing token

        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
        y_onehot = torch.FloatTensor(np.eye(self.seq_depth - 1)[y])

        return x_onehot, y_onehot, real_idx


class GenomicDataset_AlignedMask(Dataset):
    """
    为1KGP样本生成与AADR对齐的mask模式
    - AADR不存在的位点：100%缺失
    - AADR存在的位点：随机保留一部分（模拟AADR的~95%缺失率）
    """
    def __init__(self, 
                 gt_encoder, 
                 site_map_path: str,  # ADNA_SITE_MAP 文件路径
                 big_dim: int,         # 参考面板总位点数（19w）
                 mask_rate_range: tuple = (0.9, 0.99),  # 随机mask的比例范围
                 evo_mat=None,
                 indices=None):
        self.gt_enc = gt_encoder
        self.evo_mat = evo_mat
        self.seq_depth = gt_encoder.seq_depth
        self.indices = indices if indices is not None else np.arange(gt_encoder.X_gt.shape[0])
        
        # ========== 核心：预计算对齐掩码 ==========
        # 加载site_map并转换为布尔掩码
        site_map = np.load(site_map_path)  # shape: (small_n_var,), 值域: [-1, big_dim)
        self.shared_mask = np.zeros(big_dim, dtype=bool)  # 在19w维度上的掩码
        
        # site_map中>=0的索引对应AADR存在的位点
        valid_indices = site_map[site_map >= 0]
        self.shared_mask[valid_indices] = True  # 这1.7w位点是可以被观测的
        
        self.n_shared = self.shared_mask.sum()  # 约1.7w
        self.n_total = big_dim  # 19w
        self.mask_rate_range = mask_rate_range
        
    def __len__(self):
        return len(self.indices)
    
    def _generate_aligned_mask(self, batch_size):
        """生成对齐的mask矩阵 (bs, n_total)"""
        # 基础掩码：所有样本共享的盲区（17.3w位点为False）
        base_mask = np.tile(self.shared_mask, (batch_size, 1))
        
        # 计算需要在共享位点中保留的数量
        # 目标：总稀疏度 = random.uniform(0.9, 0.99)
        target_sparsity = np.random.uniform(*self.mask_rate_range)
        # 总保留数 = 总位数 * (1 - 缺失率)
        n_keep_total = int(self.n_total * (1 - target_sparsity))
        # 由于17.3w盲区已强制缺失，只能从1.7w中保留
        # 所以实际保留数 = min(目标保留数, 共享位数)
        n_keep = min(n_keep_total, self.n_shared)
        
        if n_keep > 0:
            # 对每条样本，在1.7w共享位点中随机选择n_keep个保留
            shared_indices = np.where(self.shared_mask)[0]  # 1.7w位点的索引
            for i in range(batch_size):
                # 随机选择保留位点
                keep_indices = np.random.choice(shared_indices, size=n_keep, replace=False)
                # 先全部置False（缺失），再设置保留位点为True
                base_mask[i, :] = False
                base_mask[i, keep_indices] = True
        
        return base_mask
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 1. 取出完整位点数据 (19w,)
        x = self.gt_enc.gts_toarray(real_idx).squeeze()
        y = x.copy()
        
        # 2. 生成对齐的mask (本样本)
        # 对于单样本，batch_size=1
        mask = self._generate_aligned_mask(batch_size=1)[0]  # shape: (n_total,)
        
        # 3. 应用mask：mask=False的位置设为缺失值
        x[~mask] = self.seq_depth - 1
        
        # 4. 转换为one-hot
        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
        y_onehot = torch.FloatTensor(np.eye(self.seq_depth - 1)[y])
        
        return x_onehot, y_onehot, real_idx
        

class GenomicDataset_1240k(Dataset):
    """
    读取 1240k 古 DNA 矩阵，getitem 时通过 site_map 映射到参考面板。
    site_map: length = 1240k 位点数，值 ∈ [-1, big_variants)，-1 表示参考面板没有该位点。
    ref_dim: 参考面板的位点总数（即 evo_mat.shape[1] 如果 evo_mat 不是 None）
    """
    def __init__(self, gt_encoder, evo_mat=None, site_map=None,
                 mask=True, masking_rates=(0.5, 0.99), indices=None,
                 big_dim=None):
        self.gt_enc = gt_encoder
        self.evo_mat = evo_mat
        self.seq_depth = gt_encoder.seq_depth
        self.mask = mask
        self.masking_rates = masking_rates

        # ---------- 处理 site_map ----------
        if site_map is None:
            raise ValueError('site_map 不能为空，它定义了 1240k->参考面板的列映射')
        self.site_map = np.asarray(site_map, dtype=np.int32)   # length = 1240k
        self.small_n_var = self.site_map.size
        self.big_n_var = big_dim if big_dim is not None else (
            evo_mat.shape[1] if evo_mat is not None else int(self.site_map.max()) + 1)

        # ---------- 样本索引 ----------
        self.indices = indices if indices is not None else np.arange(self.gt_enc.X_gt.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # 1. 取出 1240k 小矩阵的一行  (shape = small_n_var,)
        x_small = self.gt_enc.gts_toarray(real_idx).squeeze()        # 值域 0..，缺失=-1
        x_small = x_small.astype(np.int32)
        x_small[x_small==-1] = self.seq_depth - 1

        # 2. 映射到大矩阵维度
        x_big = np.full(self.big_n_var, self.seq_depth - 1, dtype=np.int32) # 默认缺失，缺失=self.seq_depth - 1
        # 只把 site_map 中 >=0 的列搬过去
        mask_valid = self.site_map >= 0
        x_big[self.site_map[mask_valid]] = x_small[mask_valid]

        # 3. 后续完全沿用你原来的逻辑
        y = x_big.copy()
        if self.mask:
            available = np.flatnonzero(y != self.seq_depth - 1)
            if available.size > 0:
                rate = np.random.uniform(*self.masking_rates)
                n_mask = max(1, int(available.size * rate))
                mask_idx = np.random.choice(available, n_mask, replace=False)
                x_big[mask_idx] = self.seq_depth - 1

        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x_big])
        y_onehot = torch.FloatTensor(np.eye(self.seq_depth)[y])
        return x_onehot, y_onehot, real_idx

class ImputationDataset(Dataset):
    """
    推理专用 Dataset
    - 初始化签名与 GenomicDataset 保持一致
    - 仅返回模型输入所需字段
    """
    def __init__(self,
                 x_gts_sparse,
                 seq_depth=4,
                 indices=None):
        self.gts_sparse = x_gts_sparse
        self.seq_depth = seq_depth
        self.indices = indices if indices is not None else np.arange(x_gts_sparse.shape[0])
        self._missing_stat = self._compute_missing_stat()

    # -----------------------------
    def _compute_missing_stat(self):
        total = 0
        missing = 0
        for idx in self.indices:
            row = self.gts_sparse[idx].toarray().ravel()
            total += row.size
            missing += (row == - 1).sum()
        return missing / total if total else 0.

    def print_missing_stat(self):
        print(f'[INFER] {len(self.indices):,} samples, '
              f'missing rate = {self._missing_stat:.2%}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.gts_sparse[real_idx].toarray().squeeze().astype(np.int8)
        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
        return x_onehot, real_idx   # 无 y

# ================= 自定义混合采样器 =================
class MixedRatioSampler(Sampler):
    """
    按指定比例混合三类样本的采样器
    确保每个批次内的样本都来自同一数据集类型，以保证evo_mat的有效性
    ratio: [1KGP随机mask, 1KGP按1240k panel mask, 1240K样本] 的比例
    支持DeepSpeed动态epoch种子更新，无需重建DataLoader
    """
    def __init__(self, datasets, ratio=(0.4, 0.3, 0.3), num_samples=None, shuffle=True, epoch_seed=None, batch_size=16):
        assert len(datasets) == 3, "需要三个数据集"
        assert abs(sum(ratio) - 1.0) < 1e-6, "比例之和必须为1.0"
        
        self.datasets = datasets
        self.ratio = ratio
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        # 支持动态epoch种子更新
        self._base_epoch_seed = epoch_seed
        self._current_epoch_seed = epoch_seed
        self._epoch = 0
        
        # 计算每个类型的样本数量
        if num_samples is None:
            # 使用最小数据集大小的整数倍
            min_size = min(len(ds) for ds in self.datasets)
            self.num_samples = min_size * 10  # 默认取10倍最小数据集大小
        
        self.n_per_type = []
        remaining = self.num_samples
        for i, r in enumerate(ratio):
            if i == len(ratio) - 1:  # 最后一个类型
                count = remaining
            else:
                count = int(self.num_samples * r)
                remaining -= count
            self.n_per_type.append(count)
        
        # 确保每个类型的样本数都能被batch_size整除，且不超过数据集大小
        for i in range(len(self.n_per_type)):
            max_samples = (len(self.datasets[i]) // batch_size) * batch_size
            self.n_per_type[i] = min(self.n_per_type[i], max_samples)
        
        # 重新计算总样本数
        self.total_samples = sum(self.n_per_type)
    
    def set_epoch(self, epoch):
        """动态更新epoch种子，确保DeepSpeed兼容性"""
        self._epoch = epoch
        if self._base_epoch_seed is not None:
            self._current_epoch_seed = self._base_epoch_seed + epoch
        else:
            self._current_epoch_seed = None
    
    def __iter__(self):
        # 使用当前epoch种子，确保每轮训练样本不同
        if self._current_epoch_seed is not None:
            np.random.seed(self._current_epoch_seed)
        elif self.shuffle:
            np.random.seed(None)  # 使用当前时间作为种子
        
        batches = []
        
        # 为每个数据集类型生成完整的批次
        for dataset_type, (dataset, count) in enumerate(zip(self.datasets, self.n_per_type)):
            if len(dataset) == 0 or count == 0:
                continue
                
            dataset_size = len(dataset)
            n_batches = count // self.batch_size
            
            for batch_idx in range(n_batches):
                # 生成当前批次的样本索引
                # 确保索引不超过数据集大小
                batch_indices = np.random.choice(dataset_size, self.batch_size, replace=True)
                
                # 为批次中的每个样本添加类型信息标记
                batch_with_type = [(dataset_type, idx) for idx in batch_indices]
                batches.append(batch_with_type)
        
        # 打乱所有批次（保持每个批次内的样本不变）
        if self.shuffle:
            np.random.shuffle(batches)
        
        # 将批次展平为单个样本索引迭代器，但保持批次内顺序
        flattened_indices = []
        for batch in batches:
            flattened_indices.extend(batch)
        
        # 返回索引迭代器
        return iter(flattened_indices)
    
    def __len__(self):
        return self.total_samples

# ========== 混合数据集 ==========
class MixedDataset:
    """
    混合数据集，统一接口
    """
    def __init__(self, datasets, sampler):
        self.datasets = datasets
        self.sampler = sampler
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            dataset_type, dataset_idx = key
            dataset = self.datasets[dataset_type]
            # 确保索引不超出数据集大小
            if dataset_idx >= len(dataset):
                dataset_idx = dataset_idx % len(dataset)
            x_onehot, y_onehot, real_idx = dataset[dataset_idx]
            # 添加数据集类型信息
            return x_onehot, y_onehot, real_idx, dataset_type
        else:
            # 直接索引时返回第一个数据集的样本，类型标记为0
            dataset = self.datasets[0]
            if key >= len(dataset):
                key = key % len(dataset)
            x_onehot, y_onehot, real_idx = dataset[key]
            return x_onehot, y_onehot, real_idx, 0
    
    def __len__(self):
        return len(self.sampler)