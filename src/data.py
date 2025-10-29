
import os
import json
from typing import Optional
import numpy as np
import pandas as pd
from cyvcf2 import VCF
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class GenotypeEncoder:
    def __init__(self,
                 save_dir: str,
                 vcf_path: str,
                 ref_extra: Optional[str] = None,
                 phased: bool = True,
                 gts012: bool = False):
        self.save_dir = save_dir
        self.vcf_path    = vcf_path
        self.ref_extra   = ref_extra
        self.phased      = phased if ref_extra is None else False # 是否把样本拆成单倍型
        self.gts012      = gts012
        
        # 其余成员先占位
        self.hap_map = {}
        self.n_samples   = 0
        self.n_variants  = 0
        self.sample_ids  = []   # 后面读 VCF 时填充
        self.variant_ids = []

        self.X_gt        = None   # 最终返回的张量
        self.X_extra     = None   # extra 信息
        self.seq_depth   = None

        # 1) 读 VCF
        self.X_gt = self.load_gt()
        # 2) 读 extra
        self.X_extra = self.load_extra() if self.ref_extra else None
        # 3) 保存 meta
        self.save_meta()

    def add_hap_map(self, key, val):
        if key in self.hap_map:
            if self.hap_map[key] != int(val):
                raise(f"[DATA] hap_map[{key}] inconsistent")
        else:
            self.hap_map[key] = int(val)

    def encode_gt(self, rec, n_samples, phase=False, gts012=True):
        """
        return:
            phase=False  -> (n_samples,)          剂量或基因型
            phase=True   -> (2*n_samples,)        单倍型

        encoding rule:
            gts012=True  -> 0/1/2/3  （3=missing）
            gts012=False -> 0/1/2/3/…/-1  （0=REF, 1+=ALT, -1=missing）
        """
        n = n_samples

        # ---------- 1. 单倍型模式 ----------
        if phase:
            out = np.empty(2 * n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                # 缺失
                if a1 is None:
                    out[2*i]   = 3 if gts012 else -1
                    a1 = '.'
                else:
                    if gts012:                      # 压缩成 0/1/2
                        out[2*i] = 0 if a1 == 0 else (2 if a1 >= 2 else 1)
                    else:                           # 原值保留
                        out[2*i] = a1
                self.add_hap_map(str(a1), out[2*i])
                if a2 is None:
                    out[2*i+1] = 3 if gts012 else -1
                    a2 = '.'
                else:
                    if gts012:
                        out[2*i+1] = 0 if a2 == 0 else (2 if a2 >= 2 else 1)
                    else:
                        out[2*i+1] = a2
                self.add_hap_map(str(a2),out[2*i+1])
            return out

        # ---------- 2. 剂量模式 ----------
        else:
            out = np.empty(n, dtype=np.int8)
            for i, gt in enumerate(rec.genotypes):
                a1, a2, _phased = gt
                phase = '|' if _phased else '/'
                # 缺失
                if a1 is None or a2 is None:
                    out[i] = 3 if gts012 else -1
                else:
                    if gts012:
                        # 0/1/2 剂量
                        out[i] = (1 if a1 > 0 else 0) + (1 if a2 > 0 else 0)
                    else:
                        # 多等位剂量：把 ALT 编号直接相加
                        out[i] = (0 if a1 == 0 else a1) + (0 if a2 == 0 else a2)
                a1 ='.' if a1 is None else str(a1)
                a2 ='.' if a2 is None else str(a2)
                a1, a2 = sorted([a1,a2])
                self.add_hap_map(a1+phase+a2, out[i])
            return out

    def load_extra(self) -> Optional[np.ndarray]:
        try:
            df = pd.read_csv(self.ref_extra, sep='\t', index_col=0)
            df = df.loc[self.sample_ids]          # 保证与 VCF 样本顺序一致
            print(f"[DATA] Extra dims: {df.shape}")
            return df.values.astype(np.float32)
        except Exception as e:
            print(f"[DATA] Extra features skipped: {e}")
            return None

    def load_gt(self):
        interval = 10000

        cols, data, indptr = [], [], [0]

        vcf = VCF(self.vcf_path, gts012 = self.gts012)
        self.sample_ids = vcf.samples
        self.n_samples = len(self.sample_ids)
        self.n_variants = 0

        for rec in vcf:
            vec = self.encode_gt(rec, self.n_samples, phase=self.phased, gts012=self.gts012)
            nz_idx = np.flatnonzero(vec)
            cols.extend(nz_idx)
            data.extend(vec[nz_idx])
            indptr.append(indptr[-1] + len(nz_idx))

            self.n_variants += 1
            self.variant_ids.append(f"{rec.CHROM}:{rec.POS}_{rec.REF}/{','.join(rec.ALT)}")
            if self.n_variants % interval == 0:
                print(f'\r[DATA] 已编码 {self.n_variants:,} 个位点', end='', flush=True)

        print(f'\r[DATA] 总计 {self.n_variants:,} 个位点  ', flush=True)
        vcf.close()

        # 根据 phase_mode 决定行数
        n_rows = 2 * self.n_samples if self.phased else self.n_samples
        M = sp.csc_matrix((data, cols, indptr),
                        shape=(n_rows,self.n_variants),
                        dtype=np.int8)

        print(f'[DATA] 位点矩阵 = {M.shape}，稀疏度 = {M.nnz / (M.shape[0] * M.shape[1]):.2%}')
        if self.gts012:
            self.seq_depth = M.data.max()+1
        else:
            self.seq_depth = M.data.max() + 2 
            M.data[M.data == -1] = M.data.max() + 1
            self.hap_map = {k: self.seq_depth-1 if '.' in str(k) else v for k, v in self.hap_map.items()}
        
        print("[DATA] Hap Map: ",self.hap_map)
        print(f'[DATA] gt alleles = [0 - {M.data.max()}], seq_depth = {self.seq_depth} ({self.seq_depth-1} 代表缺失)')

        os.makedirs(self.save_dir, exist_ok=True)          # 1. 不存在就创建

        # 2. 保存稀疏矩阵
        sp.save_npz(os.path.join(self.save_dir, "gt_matrix.npz"), M)

        # 3. 保存样本列表（顺序与矩阵行对应）
        with open(os.path.join(self.save_dir, "gt_samples.txt"), "w") as f:
            if self.phased:                      # 单倍型模式：写成 sample_A / sample_B
                for s in self.sample_ids:
                    f.write(f"{s}_A\n{s}_B\n")
            else:                               # 剂量模式
                for s in self.sample_ids:
                    f.write(f"{s}\n")

        # 4. 保存变异位点 ID（chr:pos/ref/alt）
        with open(os.path.join(self.save_dir, "gt_variants.txt"), "w") as f:
            for vid in self.variant_ids:
                f.write(vid + "\n")

        print(f"[DATA] 结果已写入 {self.save_dir}")
        return M

    def save_meta(self):
        def _make_json_safe(obj):
            """递归地把 numpy 数组、tuple、set、bytes 转成 list/str"""
            if isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_make_json_safe(i) for i in obj]
            if isinstance(obj, np.ndarray):
                return _make_json_safe(obj.tolist())
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, bytes):
                return obj.decode(errors='ignore')
            return obj
        meta = {
            "vcf_path"   : str(self.vcf_path),
            "ref_extra"  : str(self.ref_extra),
            "phased"     : str(self.phased),
            "gts012"     : str(self.gts012),
            "n_samples"  : str(self.n_samples),
            "n_variants" : str(self.n_variants),
            "seq_depth"  : str(self.seq_depth),
            "hap_map"    : _make_json_safe(self.hap_map),
        }
        with open(os.path.join(self.save_dir, "gt_enc_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # 如果 X_extra 不是 None，也可以落盘
        if self.X_extra is not None:
            np.save(os.path.join(self.save_dir, "gt_extra.npy"), self.X_extra)

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
        obj.ref_extra   = meta["ref_extra"]
        obj.phased      = bool(meta["phased"])
        obj.gts012      = bool(meta["gts012"])
        obj.n_samples   = int(meta["n_samples"])
        obj.n_variants  = int(meta["n_variants"])
        obj.seq_depth   = int(meta["seq_depth"])
        obj.hap_map     = meta["hap_map"]

        # 3. 读样本 & 位点 ID 列表
        obj.sample_ids = [
            l.rstrip("\n") for l in open(os.path.join(work_dir, "gt_samples.txt"))
        ]
        obj.variant_ids = [
            l.rstrip("\n") for l in open(os.path.join(work_dir, "gt_variants.txt"))
        ]

        # 4. 读稀疏矩阵
        obj.X_gt = sp.load_npz(os.path.join(work_dir, "gt_matrix.npz"))

        # 5. 读 extra（如果有）
        extra_path = os.path.join(work_dir, "gt_extra.npy")
        if os.path.exists(extra_path):
            obj.X_extra = np.load(extra_path)
        else:
            obj.X_extra = None

        return obj

class GenomicDataset(Dataset):
    """Dataset class for genomic data with masking for training"""
    def __init__(self, x_gts_sparse, x_extra=None, seq_depth=4,
                 mask=True, masking_rates=(0.5, 0.99), indices=None):
        """
        x_gts_sparse: scipy.sparse.csr_matrix or similar
        x_extra: numpy array or None
        indices: 可选，指定要使用的样本索引（如 train/valid 索引）
        """
        self.gts_sparse = x_gts_sparse
        self.x_extra = x_extra
        self.seq_depth = seq_depth
        self.mask = mask
        self.masking_rates = masking_rates
        self.indices = indices if indices is not None else np.arange(x_gts_sparse.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.gts_sparse[real_idx].toarray().squeeze().astype(np.int8)
        y = x.copy()

        if self.mask:
            seq_len = len(x)
            masking_rate = np.random.uniform(*self.masking_rates)
            mask_size = int(seq_len * masking_rate)
            mask_indices = np.random.choice(seq_len, mask_size, replace=False)
            x[mask_indices] = self.seq_depth - 1  # missing token

        x_onehot = torch.FloatTensor(np.eye(self.seq_depth)[x])
        y_onehot = torch.FloatTensor(np.eye(self.seq_depth - 1)[y])

        if self.x_extra is not None:
            x_extra = torch.FloatTensor(self.x_extra[real_idx])
        else:
            x_extra = torch.empty(0)

        return x_onehot, x_extra, y_onehot


class ImputationDataset(Dataset):
    """Dataset for imputation (no masking needed)"""

    def __init__(self, data, seq_depth):
        self.data = data
        self.seq_depth = seq_depth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Convert to one-hot without masking
        x_onehot = np.eye(self.seq_depth)[x]
        return torch.FloatTensor(x_onehot)