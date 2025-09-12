#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Randomly mask 30 % or 50 % genotypes per sample in a VCF.
Usage:
    python random_mask.py /mnt/qmtang/EvoFill/data/sim_0910/PopB.vcf.gz
"""

import numpy as np
from cyvcf2 import VCF, Writer

def mask_and_write(vcf_reader, writer, missing_rate, seed0=0):
    """
    遍历 vcf_reader，对每个样本以 missing_rate 概率把 GT 设为缺失 .|.
    """
    for idx, variant in enumerate(vcf_reader):
        np.random.seed(seed0 + idx)          # 可重复
        # genotypes 是 list[list[int]]，例如 [[0, 0], [1, 2], [-1, -1], ...]
        gts = variant.genotypes
        for i in range(len(gts)):
            if np.random.rand() < missing_rate:
                gts[i] = [-1, -1, True]     # 第三个元素表示“是否 phased”
        variant.genotypes = gts              # 写回
        writer.write_record(variant)
    writer.close()

def main():
    in_vcf = "/mnt/qmtang/EvoFill/data/sim_0910/PopB.vcf.gz"
    base = in_vcf.replace(".gz", "").replace(".vcf", "")
    out30 = base + ".missing30.vcf.gz"
    out50 = base + ".missing50.vcf.gz"

    for missing_rate, out_name in [(0.3, out30), (0.5, out50)]:
        print(f"Creating {out_name}  (missing rate = {missing_rate})")
        vcf = VCF(in_vcf, threads=4)
        w = Writer(out_name, vcf, mode='wz')
        mask_and_write(vcf, w, missing_rate)
        vcf.close()
    print("All done.")

if __name__ == "__main__":
    main()