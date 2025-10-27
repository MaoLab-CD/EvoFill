import os
import numpy as np
from cyvcf2 import VCF
import scipy.sparse as sp

def encode_gt(rec, n_samples, phase=False, gts012=True):
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
            else:
                if gts012:                      # 压缩成 0/1/2
                    out[2*i] = 0 if a1 == 0 else (2 if a1 >= 2 else 1)
                else:                           # 原值保留
                    out[2*i] = a1

            if a2 is None:
                out[2*i+1] = 3 if gts012 else -1
            else:
                if gts012:
                    out[2*i+1] = 0 if a2 == 0 else (2 if a2 >= 2 else 1)
                else:
                    out[2*i+1] = a2
        return out

    # ---------- 2. 剂量模式 ----------
    else:
        out = np.empty(n, dtype=np.int8)
        for i, gt in enumerate(rec.genotypes):
            a1, a2, _phased = gt
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
        return out

 
vcf_path = "/mnt/NAS/Omics/DNA/1kGP/vcf/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
save_dir = "/mnt/qmtang/EvoFill/data/251023_chr22"
interval = 10000
gts012_switch = False
phase_mode = True

cols, data, indptr = [], [], [0]
variant_ids = []

vcf = VCF(vcf_path, gts012 = gts012_switch)
n_samples = len(vcf.samples)
sample_list = vcf.samples
n_samples = len(sample_list)
n_variants = 0

for rec in vcf:
    vec = encode_gt(rec, n_samples, phase=phase_mode, gts012=gts012_switch)
    nz_idx = np.flatnonzero(vec)
    cols.extend(nz_idx)
    data.extend(vec[nz_idx])
    indptr.append(indptr[-1] + len(nz_idx))

    n_variants += 1
    variant_ids.append(f"{rec.CHROM}:{rec.POS}_{rec.REF}/{','.join(rec.ALT)}")
    if n_variants % interval == 0:
        print(f'\r[DATA] 已编码 {n_variants:,} 个位点', end='', flush=True)

print(f'\r[DATA] 总计 {n_variants:,} 个位点', flush=True)
vcf.close()

# 根据 phase_mode 决定行数
n_rows = 2 * n_samples if phase_mode else n_samples
M = sp.csc_matrix((data, cols, indptr),
                  shape=(n_rows,n_variants),
                  dtype=np.int8)

print(f'[DATA] gt matrix = {M.shape}，稀疏度 = {M.nnz / (M.shape[0] * M.shape[1]):.2%}')
seq_depth = M.data.max()+1 if gts012_switch else M.data.max()+2
print(f'[DATA] gt alleles = [0 - {M.data.max()}], seq_depth = {seq_depth} (including missing)')

os.makedirs(save_dir, exist_ok=True)          # 1. 不存在就创建

# 2. 保存稀疏矩阵
sp.save_npz(os.path.join(save_dir, "gt_matrix.npz"), M)

# 3. 保存样本列表（顺序与矩阵行对应）
with open(os.path.join(save_dir, "gt_samples.txt"), "w") as f:
    if phase_mode:                      # 单倍型模式：写成 sample_A / sample_B
        for s in sample_list:
            f.write(f"{s}_A\n{s}_B\n")
    else:                               # 剂量模式
        for s in sample_list:
            f.write(f"{s}\n")

# 4. 可选：保存变异位点 ID（chr:pos/ref/alt）
with open(os.path.join(save_dir, "gt_variants.txt"), "w") as f:
    for vid in variant_ids:
        f.write(vid + "\n")

print(f"[DATA] 结果已写入 {save_dir}")