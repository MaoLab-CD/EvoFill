
import os
import json
from typing import Optional, Dict
import numpy as np
import pandas as pd
from cyvcf2 import VCF
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

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
        self.default_gt = None
        self.seq_depth: Optional[int] = None
        self.n_samples = 0
        self.n_variants = 0
        self.sample_ids = []
        self.variant_ids = []
        self.X_gt: Optional[sp.csc_matrix] = None
        self.evo_mat: Optional[np.ndarray] = None

    # ========= 2. 首次编码：建立标准 =========
    def encode_new(self, vcf_path: str, default_gt: str, evo_mat: Optional[str] = None,):
        """第一次编码，允许新建 hap_map 与 seq_depth"""
        self.vcf_path = vcf_path
        # 清空头一次可能遗留的参照
        self.hap_map.clear()
        self.seq_depth = None
        self.evo_mat_path = evo_mat
        self.phased = self.phased if self.evo_mat_path is None else False
        # 真正干活
        self.X_gt = self._load_gt(default_gt)
        self.evo_mat = self._load_evo_mat() if self.evo_mat_path else None

        if self.gts012:
            self.seq_depth = self.X_gt.data.max() + 1
        else:
            self.seq_depth = self.X_gt.data.max() + 2
            self.X_gt.data[self.X_gt.data == -1] = self.seq_depth - 1
            self.hap_map = {k: int(self.seq_depth - 1) if '.' in str(k) else v for k, v in self.hap_map.items()}

        if self.save2disk:
            if self.save_dir:
                self._save_meta()
            else:
                print("[DATA] save_dir is not given")
        
        print(f'[DATA] 位点矩阵 = {self.X_gt.shape}，稀疏度 = {self.X_gt.nnz / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}')
        print(f'[DATA] 位点字典 = {self.hap_map}，字典深度 = {self.seq_depth}')
        return self

    # ========= 3. 参照编码：必须 100 % 一致 =========
    def encode_ref(self, ref_meta_json: str, vcf_path: str, default_gt: str, evo_mat: Optional[str] = None):
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
        self.X_gt = self._load_gt(default_gt)

        self.evo_mat_path = evo_mat
        self.evo_mat = self._load_evo_mat() if self.evo_mat_path else None

        if self.save2disk:
            if self.save_dir:
                self._save_meta()
            else:
                print("[DATA] save_dir is not given")
        
        miss_rate = np.sum((self.X_gt.data == self.seq_depth - 1)) 
        print(f'[DATA] 位点矩阵 = {self.X_gt.shape}，稀疏度 = {self.X_gt.nnz / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}，缺失率 = {miss_rate / self.X_gt.shape[0] / self.X_gt.shape[1]:.2%}')
        print(f'[DATA] 位点字典 = {self.hap_map}，字典深度 = {self.seq_depth}')
        return self

    # ========= 4. 内部工具：encode_gt 仅增加“参照模式”判断 =========
    def _encode_gt(self, rec, n_samples, phase=False, gts012=False):
        """参照模式下遇到新等位基因直接抛异常"""
        n = n_samples
        ref_mode = bool(self.hap_map) and self.seq_depth is not None

        if phase: # 基因型模式
            out = np.empty(2 * n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                for j, a in enumerate((a1, a2)):
                    key = '.' if a == -1 else str(a)
                    idx = 2 * i + j
                    if a == -1:
                        if ref_mode:
                            code = self.seq_depth - 1
                        else:
                            code = 3 if gts012 else -1
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
                self.hap_map['.'] = 3 if gts012 else -1
            return out
        else: # 剂量模式
            out = np.empty(n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                sep = '|' if _phased else '/'
                if a1 == -1 or a2 == -1:
                    key = sep.join(sorted(['.', '.']))
                    if ref_mode:
                        code = self.seq_depth - 1
                    else:
                        code = 3 if gts012 else -1
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
                self.hap_map[sep.join(sorted(['.', '.']))] = 3 if gts012 else -1
            return out

    # ========= 5. 内部方法 =========
    def _load_gt(self, default_gt):
        self.default_gt = default_gt
        if self.default_gt not in {'ref', 'miss'}:
            raise ValueError('default_gt must be "ref" or "miss"')

        interval = 10000
        cols, data, indptr = [], [], [0]
        vcf = VCF(self.vcf_path, gts012=self.gts012)
        self.sample_ids = vcf.samples
        self.n_samples = len(self.sample_ids)
        self.n_variants = 0
        self.variant_ids = []

        for rec in vcf:
            vec = self._encode_gt(rec, self.n_samples, phase=self.phased, gts012=self.gts012)
            if self.default_gt == 'ref':          # 只存 alt（非 ref）
                nz_mask = vec != 0
            else:                                 # 只存非 miss
                miss_code = self.seq_depth - 1
                nz_mask = vec != miss_code
            nz_idx = np.flatnonzero(nz_mask)
            cols.extend(nz_idx)
            data.extend(vec[nz_idx])
            indptr.append(indptr[-1] + len(nz_idx))
            self.n_variants += 1
            self.variant_ids.append(f"{rec.CHROM}:{rec.POS}_{rec.REF}/{','.join(rec.ALT)}")
            if self.n_variants % interval == 0:
                print(f'\r[DATA] 已编码 {self.n_variants:,} 个位点', end='', flush=True)
        print(f'\r[DATA] 总计 {self.n_variants:,} 个位点  ', flush=True)
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
        print(f"[DATA] 结果已写入 {self.save_dir}")

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
            缺失值 = seq_depth-1，参考型 = 0，alt = 原值
        """
        # 1. 统一成“行索引数组”
        if idx is None:
            row_idx = np.arange(self.X_gt.shape[0])
        elif np.isscalar(idx):
            row_idx = np.array([idx], dtype=int)
        else:
            row_idx = np.asarray(idx, dtype=int)

        # 2. 拿子矩阵（仍是稀疏）
        sub = self.X_gt[row_idx]

        # 3. 先变成 dense，此时“没存的位置”全是 0
        dense = sub.toarray()          # shape (len(row_idx), n_variants)

        # 4. 构造一个与 dense 同形状的“缺失掩码”
        #    True 表示“当时没存”，需要回填
        mask = np.ones_like(dense, dtype=bool)
        # 把显式存的元素置 False
        for i, row in enumerate(row_idx):
            start = sub.indptr[i]
            end   = sub.indptr[i + 1]
            cols  = sub.indices[start:end]
            mask[i, cols] = False

        # 5. 按策略回填
        if self.default_gt == 'ref':
            # 没存的位置 → 参考型 0
            dense[mask] = 0
        else:  # default_gt == 'miss'
            # 没存的位置 → 缺失
            dense[mask] = self.seq_depth - 1

        return dense.astype(np.int8)

    @classmethod
    def loadfromdisk(cls, work_dir: str):
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
        obj.default_gt  = meta["default_gt"]
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

class GenomicDataset_Missing(Dataset):
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
            # 1. 找出“非缺失”下标
            miss_code = self.seq_depth - 1
            available = np.flatnonzero(y != miss_code) 
            if available.size == 0:
                pass
            else:
                # 2. 计算要遮多少
                masking_rate = np.random.uniform(*self.masking_rates)
                mask_size = int(len(available) * masking_rate)
                mask_size = max(1, mask_size) if available.size > 0 else 0
                # 3. 随机挑选并遮掉
                mask_indices = np.random.choice(available, mask_size, replace=False)
                x[mask_indices] = miss_code

        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
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
            missing += (row == self.seq_depth - 1).sum()
        return missing / total if total else 0.

    def print_missing_stat(self):
        print(f'[ImputationDataset] {len(self.indices):,} samples, '
              f'missing rate = {self._missing_stat:.2%}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.gts_sparse[real_idx].toarray().squeeze().astype(np.int8)
        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
        return x_onehot, real_idx   # 无 y