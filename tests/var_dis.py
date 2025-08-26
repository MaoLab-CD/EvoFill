#!/usr/bin/env python
"""
统计相邻变异位点距离的自定义区间分布
区间：0-8, 9-16, 17-32, 33-64, 65-128, 129-256, 257-512, 513-1024, >1024
"""

from cyvcf2 import VCF
import numpy as np
import pandas as pd

vcf_path = "data/hg19_chr22/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
out_tsv  = "chr22_dist_distribution.tsv"

# 1. 读取 vcf.gz
vcf = VCF(vcf_path)

# 2. 遍历计算相邻距离
dists = []          # 把所有距离先收集起来，便于后续多种统计
prev_chrom, prev_pos = None, None
for var in vcf:
    chrom, pos = var.CHROM, var.POS
    if chrom == prev_chrom:      # 只统计同一条染色体
        d = pos - prev_pos
        if d > 0:               # 忽略负/零距离
            dists.append(d)
    prev_chrom, prev_pos = chrom, pos

# 3. 按自定义区间分桶
bins   = [0, 9, 17, 33, 65, 129, 257, 513, 1025, np.inf]   # 左闭右开
labels = ['0-8', '9-16', '17-32', '33-64', '65-128',
          '129-256', '257-512', '513-1024', '>1024']

series = pd.cut(dists, bins=bins, labels=labels, right=False)
counts = series.value_counts().reindex(labels, fill_value=0)

# 4. 保存 TSV
# counts.to_csv(out_tsv, sep='\t', header=['count'])

# 5. 可选：打印或画图
print(counts)

# 画图（需 matplotlib）
try:
    import matplotlib.pyplot as plt
    counts.plot(kind='bar')
    plt.ylabel('Count')
    plt.xlabel('Distance range (bp)')
    plt.title('Adjacent variant distance distribution')
    plt.tight_layout()
    # plt.savefig('dist_distribution.png', dpi=150)
    plt.show()
except ImportError:
    pass

# 0-8         296799
# 9-16        186922
# 17-32       248077
# 33-64       231228
# 65-128      115533
# 129-256      21554
# 257-512       1554
# 513-1024       417
# >1024          296