#!/usr/bin/env python3
"""
vcf2tensor.py
一次性完成：索引生成 → 分块转换
"""
import os
import math
import numpy as np
import torch
from cyvcf2 import VCF
from pyfaidx import Fasta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from src.utils import load_config

# -------------------------------------------------
# 0. 参数
# -------------------------------------------------
args = load_config("config/config.json")

# -------------------------------------------------
# 1. 全局常量
# -------------------------------------------------
BASE2INT = {'A': 1, 'C': 2, 'G': 3, 'T': 4,
            'a': 1, 'c': 2, 'g': 3, 't': 4,
            'N': 0, 'n': 0}

def seq_to_int(seq: str):
    return [BASE2INT.get(b, 0) for b in seq]

def fetch_seq_and_mask(fa, chrom: str, pos: int, k: int, snp_offset: int = 0):
    half = k // 2
    start = pos - half - 1 + snp_offset
    end   = pos + half + snp_offset
    try:
        seq = fa[chrom][start:end].seq
    except KeyError:
        seq = 'N' * k
    if len(seq) != k:
        seq = 'N' * k
    mask = [0] * k
    mask[half + snp_offset] = 1
    return seq_to_int(seq), mask

# -------------------------------------------------
# 2. 生成位点索引
# -------------------------------------------------
def build_pos_index(vcf_path, tsv_path):
    print(f"[index] 生成位点索引 {tsv_path} ...")
    vcf = VCF(vcf_path)
    with open(tsv_path, 'w') as fo:
        for idx, rec in enumerate(vcf, 1):
            fo.write(f"{idx}\t{rec.CHROM}\t{rec.POS}\n")
    vcf.close()
    return idx

# -------------------------------------------------
# 3. 单块转换
# -------------------------------------------------
def process_one_block(parms):
    n0, n1, s0, s1, sub_samples, S_all, stage, stage_vcf, stage_fasta, stage_pos_tsv, stage_chunks_dir = parms
    # 1. 读取位点列表
    block_vars = []
    with open(stage_pos_tsv) as f:
        for line in f:
            idx, chrom, pos = line.rstrip().split('\t')
            idx = int(idx)
            if s0 <= idx <= s1:
                block_vars.append((chrom, int(pos)))
    S_blk = len(block_vars)
    N_blk = n1 - n0

    # 2. 开 VCF 与 fasta
    vcf = VCF(stage_vcf, gts012=True, samples=sub_samples)
    fa  = Fasta(stage_fasta, one_based_attributes=False)

    seq   = torch.empty(N_blk, S_blk, args.k_mer, dtype=torch.int8)
    mask  = torch.empty(N_blk, S_blk, args.k_mer, dtype=torch.bool)
    label = torch.empty(N_blk, S_blk, dtype=torch.int8)

    vcf_iter = iter(vcf)
    for s, (chrom, pos) in enumerate(block_vars):
        snp_offset = np.random.randint(args.k_mer // 10, args.k_mer // 2)
        seq_s, mask_s = fetch_seq_and_mask(fa, chrom, pos, args.k_mer, snp_offset)
        seq[:, s]  = torch.tensor(seq_s,  dtype=torch.int8)
        mask[:, s] = torch.tensor(mask_s, dtype=torch.bool)

        rec = next(vcf_iter)
        gts_sum = rec.genotype.array().sum(axis=1, dtype=np.int8)[n0:n1]
        label[:, s] = torch.from_numpy(gts_sum)

    chunk_id = f"{(n0 // args.n_chunk) * math.ceil(S_all / args.s_chunk) + (s0 - 1) // args.s_chunk + 1:05d}"
    out_path = os.path.join(stage_chunks_dir, f"{stage}_{chunk_id}.pt")
    torch.save({'seq': seq, 'mask': mask, 'label': label}, out_path)
    vcf.close()
    return out_path

# -------------------------------------------------
# 4. 主流程
# -------------------------------------------------
def run_stage(stage: str):
    """
    stage: 'train' or 'val'
    """
    print(f"\n========== {stage.upper()} SET ==========")

    stage_vcf          = getattr(args, f"{stage}_vcf")
    stage_chunks_dir   = getattr(args, f"{stage}_chunks_dir")
    stage_pos_tsv      = os.path.join(stage_chunks_dir, f"{stage}_pos.tsv")

    os.makedirs(stage_chunks_dir, exist_ok=True)

    # 第 1 步：索引
    S_all = build_pos_index(stage_vcf, stage_pos_tsv)

    # 第 2 步：分块
    vcf = VCF(stage_vcf)
    sample_names = vcf.samples
    vcf.close()
    N_all = len(sample_names)

    tasks = []
    for n0 in range(0, N_all, args.n_chunk):
        n1 = min(n0 + args.n_chunk, N_all)
        sub_samples = sample_names[n0:n1]
        for s0 in range(1, S_all + 1, args.s_chunk):
            s1 = min(s0 + args.s_chunk - 1, S_all)
            tasks.append((n0, n1, s0, s1, sub_samples, S_all,
                          stage, stage_vcf, args.ref_fasta,
                          stage_pos_tsv, stage_chunks_dir))

    print(f"[convert] 开始转换：{len(tasks)} 个块，进程={args.jobs}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(tqdm(pool.map(process_one_block, tasks), total=len(tasks)))

def main():
    for stage in ('train', 'val'):
        run_stage(stage)
    print("\n全部完成。")

if __name__ == '__main__':
    main()