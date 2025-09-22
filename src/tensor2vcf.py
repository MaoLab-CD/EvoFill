#!/usr/bin/env python3
"""
把 imputed 整数基因型张量写回 VCF
"""
import os
import cyvcf2
import numpy as np
import torch


def imputed_tensor2vcf(
    imp_tensor: torch.Tensor,
    orig_vcf_path: str,
    out_vcf_path: str,
    samples: list,
    phased: bool = True,
):
    """
    imp_tensor: int8 (n_samples, n_sites)  0/1/2/... 已填充
    其余信息（REF/ALT、POS、ID、QUAL、FILTER）完全照抄原始 VCF
    """
    vcf_rd = cyvcf2.VCF(orig_vcf_path)
    vcf_wr = cyvcf2.Writer(out_vcf_path, vcf_rd)

    # 把整数编码转回字符串
    for i, var in enumerate(vcf_rd):
        alleles = [var.REF] + var.ALT
        gt_int = imp_tensor[:, i].tolist()          # (n_samples,)  0/1/2/...
        gt_pairs = []
        for g in gt_int:
            if g < 0:                               # 防御性缺失
                gt_pairs.append([-1, -1])
            else:
                a1, a2 = g // 2, g % 2
                gt_pairs.append([int(a1), int(a2), phased])
        var.genotypes = gt_pairs
        vcf_wr.write_record(var)

    vcf_wr.close()
    vcf_rd.close()
    print(f"[tensor2vcf] saved -> {out_vcf_path}")


if __name__ == "__main__":
    """单测：python tensor2vcf.py"""
    import sys
    if len(sys.argv) != 5:
        print("usage: tensor2vcf.py imp.pt orig.vcf.gz out.vcf.gz phased<0|1>")
        exit(1)
    imp_pt, orig_vcf, out_vcf, phased = sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4]))
    tensor = torch.load(imp_pt)          # 假设就是 (N,L) 张量
    vcf_rd = cyvcf2.VCF(orig_vcf)
    samples = list(vcf_rd.samples)
    vcf_rd.close()
    imputed_tensor2vcf(tensor, orig_vcf, out_vcf, samples, phased)