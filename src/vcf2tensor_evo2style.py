# #!/usr/bin python
# -*- coding: utf-8 -*-
'''
@File    :   vcf2tensor.py
@Time    :   2025/08/21 17:53:00
@Author  :   qmtang
@Version :   2.0
@Desc    :   读取VCF文件转为Pytorch张量（B,N,K）。
             B=样本数, N=位点数, K=位点毗邻序列长度。
             {'seq':Int(B,N,K),'mask':Bool(B,N,K)}
'''

import os
import random
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from cyvcf2 import VCF
from pyfaidx import Fasta
from tqdm import tqdm
from pathlib import Path

from src.utils import load_config


def base2int(base):
    """碱基字符编码: A->0, C->1, G->2, T->3, N或其他->4, 缺失用5。"""
    base = base.upper()
    if base == 'A':
        return 0
    elif base == 'C':
        return 1
    elif base == 'G':
        return 2
    elif base == 'T':
        return 3
    elif base == '.':  # 缺失标记
        return 5
    else:
        return 4  # N 或其他


def count_index(vcf_path):
    """生成位点索引，返回 variant_idx 和 sample_idx"""
    vcf = VCF(vcf_path)
    sample_idx = vcf.samples
    variant_idx = sum(1 for _ in vcf)  # 计算总变异位点数
    vcf.close()
    return variant_idx, sample_idx


def process_one_chunk(args):
    """
    每个子进程只处理一个 chunk。
    进度写入 {output_dir}/chunk_{chunk_id}.log
    """
    (chunk_id,
     chunk_samples,
     sample_idx,
     vcf_file,
     ref_fa,
     k,
     min_adj_len,
     output_dir,
     seed) = args

    # 每个进程独立随机种子
    random.seed(seed + chunk_id)

    # 把进度条写到独立文件，防止冲突
    log_path = Path(output_dir) / f'chunk_{chunk_id}.log'
    tqdm_kwargs = dict(file=open(log_path, 'w'), mininterval=2.0)

    # 一次性读整条染色体进内存
    ref = Fasta(ref_fa, sequence_always_upper=True)
    chrom_cache = {c: ref[c][:].seq for c in ref.keys()}

    vcf = VCF(vcf_file)
    sample_indices = [sample_idx.index(s) for s in chunk_samples]

    # 变异总数
    total_vars = sum(1 for _ in VCF(vcf_file))
    vcf = VCF(vcf_file)  # reopen

    sample_data = {s: [] for s in chunk_samples}
    for variant in tqdm(vcf, total=total_vars, desc=f'Chunk{chunk_id}', **tqdm_kwargs):
        chrom = 'chr' + str(variant.CHROM)
        pos = variant.POS
        ref_allele = variant.REF
        alt_alleles = variant.ALT
        ref_len = len(ref_allele)

        seq_source = chrom_cache[chrom]
        seq_len = len(seq_source)

        for local_idx, s in enumerate(chunk_samples):
            gt = variant.genotypes[sample_indices[local_idx]]
            if gt is None or len(gt) != 3 or all(a == -1 for a in gt[:2]):
                haplotypes = [None, None]
            else:
                haplotypes = [gt[0] if gt[0] != -1 else None, gt[1] if gt[1] != -1 else None]

            for hap_idx, hap in enumerate(haplotypes):
                if hap is None:
                    allele = '.' * ref_len
                elif hap == 0:
                    allele = ref_allele
                else:
                    allele_idx = hap - 1
                    if 0 <= allele_idx < len(alt_alleles):
                        allele = alt_alleles[allele_idx]
                    else:
                        allele = ref_allele

                allele = allele.upper()
                allele_len = len(allele)

                min_start = min_adj_len
                max_start = k - allele_len - min_adj_len
                if max_start < min_start:
                    variant_start = min_start
                else:
                    variant_start = random.randint(min_start, max_start)

                upstream_len = variant_start
                downstream_len = k - variant_start - allele_len

                ref_up_start = pos - 1 - upstream_len
                pad_up = max(0, -ref_up_start)
                actual_up_start = max(0, ref_up_start)
                upstream = 'N' * pad_up + seq_source[actual_up_start:pos - 1]

                end = pos - 1 + ref_len
                ref_down_end = end + downstream_len
                actual_down_end = min(ref_down_end, seq_len)
                downstream = seq_source[end:actual_down_end] + 'N' * (ref_down_end - actual_down_end)

                sequence = upstream + allele + downstream

                current_len = len(sequence)
                if current_len != k:
                    if current_len < k:
                        sequence += 'N' * (k - current_len)
                    else:
                        sequence = sequence[:k]

                seq_encoded = np.array([base2int(b) for b in sequence], dtype=np.uint8)
                mask = np.zeros(k, dtype=np.uint8)
                mask_end = min(k, variant_start + allele_len)
                mask[variant_start:mask_end] = 1

                sample_data[s].append({'seq': seq_encoded, 'mask': mask})

    vcf.close()

    # 组装成 tensor 并保存
    seq_list = [np.stack([d['seq'] for d in sample_data[s]]) for s in chunk_samples]
    mask_list = [np.stack([d['mask'] for d in sample_data[s]]) for s in chunk_samples]
    torch.save(
        {'sample':chunk_samples,
        'seq': torch.from_numpy(np.stack(seq_list)),
        'mask': torch.from_numpy(np.stack(mask_list))},
        Path(output_dir) / f'chunk_{chunk_id}.pt'
    )
    return f'chunk_{chunk_id} done'


def vcf2tensor_parallel(vcf_file, ref_fa, k=64, min_adj_len=0, chunk_size=10, seed=42, output_dir='output_chunks', max_workers=None):
    """并行处理 VCF 文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 先建位点索引
    variant_idx, sample_idx = count_index(vcf_file)
    print(f"Processing {len(sample_idx)} Samples with {variant_idx} variant sites...")

    # 构造任务列表
    tasks = [(chunk_id,
              sample_idx[chunk_start:chunk_start+chunk_size],
              sample_idx,
              vcf_file,
              ref_fa,
              k,
              min_adj_len,
              output_dir,
              seed)
             for chunk_id, chunk_start in enumerate(range(0, len(sample_idx), chunk_size))]

    # 并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
            future_to_chunk = {exe.submit(process_one_chunk, t): t[0] for t in tasks}
            for future in tqdm(as_completed(future_to_chunk), total=len(tasks), desc="Processing chunks"):
                future_to_chunk[future]


if __name__ == "__main__":
    cfg = load_config("config/config.json")
    print("Processing training set...")
    vcf2tensor_parallel(
        cfg.data.train_vcf,
        cfg.data.ref_fasta,
        k = cfg.data.k_mer,
        min_adj_len = 10,
        chunk_size = 10,
        seed = 42,
        output_dir = cfg.train_chunks_dir,
        max_workers = 16 # 内存受限
    )
    print("Processing validation set...")
    vcf2tensor_parallel(
        cfg.data.val_vcf,
        cfg.data.ref_fasta,
        k = cfg.data.k_mer,
        min_adj_len = 10,
        chunk_size = 10,
        seed = 42,
        output_dir = cfg.data.val_chunks_dir,
        max_workers = 16 # 内存受限
    )