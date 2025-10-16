#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 VCF 的 POS 插值成遗传距离（cM）
用法：
    python vcf_add_gmap.py \
        --gmap_dir /mnt/qmtang/1KG_phase3_v5_hg19 \
        --vcf      input.hg19.vcf.gz \
        --out      output.hg19.gmap.vcf.gz
"""
import argparse, os, gzip, re
import numpy as np
from scipy.interpolate import interp1d
import cyvcf2
from tqdm import tqdm

def load_chrom_map(gmap_dir, chrom):
    """返回插值函数 f(pos) -> cM"""
    fname = os.path.join(gmap_dir, f"genetic_map_GRCh37_{chrom}.txt")
    if not os.path.exists(fname):
        fname += ".gz"   # 允许 .gz
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname, "rt") as f:
        df = np.loadtxt(f, skiprows=1,
                        dtype={"names": ("chr", "pos", "rate", "cM"),
                               "formats": ("U5", "i4", "f16", "f16")})
    return interp1d(df["pos"], df["cM"],
                    bounds_error=False, fill_value="extrapolate")

def main():
    parser = argparse.ArgumentParser(description="Add genetic map (cM) to VCF")
    parser.add_argument("--gmap_dir", required=True,
                        help="directory containing genetic_map_GRCh37_chr*.txt[.gz]")
    parser.add_argument("--vcf", required=True, help="input VCF")
    parser.add_argument("--out", required=True, help="output VCF")
    args = parser.parse_args()

    total = sum(1 for _ in cyvcf2.VCF(args.vcf))

    # 1. 打开输入 VCF
    vcf_in = cyvcf2.VCF(args.vcf, strict_gt=False)
    # 2. 新增 INFO 字段
    vcf_in.add_info_to_header(
        {"ID": "GeneticPos_cM",
         "Number": "1",
         "Type": "Float",
         "Description": "Cumulative genetic position (cM) on hg19/GRCh37"})
    vcf_out = cyvcf2.Writer(args.out, vcf_in)

    # 3. 预加载 1-22+X 的插值函数
    gmap = {}
    for ch in list(range(1, 23)) + ["X"]:
        gmap[str(ch)] = load_chrom_map(args.gmap_dir, f"chr{ch}")

    # 4. 逐记录处理 + 进度条
    for rec in tqdm(vcf_in, total=total, unit="rec"):
        chrom = rec.CHROM.replace("chr", "")
        if chrom not in gmap:          # 未找到的染色体跳过
            continue
        cm = float(gmap[chrom](rec.POS))
        rec.INFO["GeneticPos_cM"] = round(cm, 8)
        vcf_out.write_record(rec)

    vcf_out.close(); vcf_in.close()
    print("✅ 完成，输出：", args.out)

if __name__ == "__main__":
    main()
