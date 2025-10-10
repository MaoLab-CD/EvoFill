#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把训练日志中的 chunk/epoch/loss 以及 MAF 六区间统计
整理成一张宽表（csv 或 DataFrame）。
用法：
    python parse_log.py train.log  ->  在当前目录生成 train.log.csv
    也可以在脚本里直接改 LOG_PATH。
"""

import re
import pandas as pd
import sys
from pathlib import Path

# --------------------- 需要改的地方 ---------------------
LOG_PATH = '/home/qmtang/mnt_qmtang/EvoFill/logs/251005.logs'          # 日志路径
OUT_PATH = None                 # 输出 csv 路径，None 则自动生成
# ------------------------------------------------------

if OUT_PATH is None:
    OUT_PATH = Path(LOG_PATH).with_suffix('.csv')

# 正则：抓取 Chunk x, Epoch y/100, Train Loss: ..., Val Loss: ...
epoch_re = re.compile(
    r'Chunk\s+(\d+),\s+Epoch\s+(\d+)/\d+,\s+'
    r'Train Loss:\s+([\d\.]+),\s+Val Loss:\s+([\d\.]+)'
)

# 正则：抓取 MAF 表格里每一行  (0.00, 0.05)   7421 0.974 0.997
maf_re = re.compile(
    r'\(\d+\.\d+,\s*\d+\.\d+\)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)'
)

records = []
current = {}

with open(LOG_PATH, encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()

        # 遇到新的 epoch 行
        m = epoch_re.search(line)
        if m:
            # 如果上一个 epoch 已经抓完 MAF，就保存
            if current and 'maf_0_counts' in current:
                records.append(current)

            current = {
                'chunk': int(m.group(1)),
                'epoch': int(m.group(2)),
                'train_loss': float(m.group(3)),
                'val_loss': float(m.group(4))
            }
            continue

        # 在 MAF 表格区域内
        if current:
            m = maf_re.search(line)
            if m:
                idx = len([k for k in current.keys() if k.startswith('maf_')]) // 3
                current[f'maf_{idx}_counts'] = int(m.group(1))
                current[f'maf_{idx}_train'] = float(m.group(2))
                current[f'maf_{idx}_val'] = float(m.group(3))

# 别忘了最后一条
if current and 'maf_0_counts' in current:
    records.append(current)

# 拼表
df = pd.DataFrame(records)

# 让列顺序好看一点
cols = ['chunk', 'epoch', 'train_loss', 'val_loss']
maf_cols = sorted([c for c in df.columns if c.startswith('maf_')])
df = df[cols + maf_cols]

# 保存 & 打印
df.to_csv(OUT_PATH, index=False, float_format='%.6f')
print(f'已解析 {len(df)} 条记录，保存为 {OUT_PATH}')
print(df.head())