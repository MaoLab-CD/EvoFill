#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 GRCh38 主组装 FASTA 生成 genome_coords.json
用法：
    python build_genome_json.py \
        -f /mnt/qmtang/GRCh38/Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa.gz \
        -o genome_coords.json
"""
import argparse
import json
import os
from pyfaidx import Fasta

# 染色体排序表（只保留 1-22,X,Y,MT，可按需增删）
ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--fasta", required=True,
                   help="GRCh38 primary assembly fasta (.fa/.fa.gz)")
    p.add_argument("-o", "--out", default="genome_coords.json",
                   help="输出 JSON 路径")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 加载 FASTA（支持 bgzip）
    fa = Fasta(args.fasta, read_ahead=int(1e5), sequence_always_upper=True)

    # 2. 收集 (chrom, length)
    chrom_len = {}
    for chrom in ORDER:
        if chrom in fa.keys():
            chrom_len[chrom] = len(fa[chrom])
        else:
            print(f"[warn] {chrom} not found in FASTA, skipped")

    # 3. 计算 chrom_start
    chrom_start = {}
    offset = 0
    for chrom in ORDER:
        if chrom in chrom_len:
            chrom_start[chrom] = offset
            offset += chrom_len[chrom]

    genome_len = offset

    # 4. 写 JSON
    out_dict = {
        "chrom_len": chrom_len,
        "chrom_start": chrom_start,
        "genome_len": genome_len
    }

    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"[done] 基因组总长度 {genome_len:,} bp")
    print(f"[done] 已写入 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()