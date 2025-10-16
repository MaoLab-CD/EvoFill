#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCF → tensor  with REF>ALT event token
保守事件 REF>REF = 0
缺失统一 <MISS> 且为字典最后一位
"""
import os
import math
import json
import numpy as np
import torch
from tqdm import tqdm
from cyvcf2 import VCF

# ---------- 配置 ----------
TOKEN_JSON = "config/variant_token.json"   # 字典落盘路径
EVENT2CODE = None   # 首次训练后不再修改
MISS_CODE  = None   # = len(EVENT2CODE)-1


# ---------- 辅助 ----------
def _log4(x):
    return math.log(x) / math.log(4)


def build_quaternion(chrom, pos, chrom_len_dict, chrom_start_dict, genome_len):
    chrom = str(chrom).strip('chr')
    pos   = int(pos)
    c_len = chrom_len_dict[chrom]
    c_start = chrom_start_dict[chrom]
    abs_pos = c_start + pos
    return [_log4(pos), _log4(c_len), _log4(abs_pos), _log4(genome_len)]


# ---------- 核心 ----------
def read_vcf(path: str, phased: bool, genome_json: str, is_train=False):
    """
    返回
        gts_onehot : torch.Tensor (n_samples, n_snps, n_events)
        samples    : list[str]
        var_depth_index : torch.Tensor (n_snps,)  int8
        quat_tensor     : torch.Tensor (n_snps, 4) float32
    """
    global EVENT2CODE, MISS_CODE

    # 0. 基因组元信息
    with open(genome_json) as f:
        gmeta = json.load(f)
    chrom_len  = gmeta["chrom_len"]
    chrom_start= gmeta["chrom_start"]
    genome_len = gmeta["genome_len"]

    vcf = VCF(path)
    samples = list(vcf.samples)

    # 1. 训练阶段：扫描所有 REF>ALT 事件
    if is_train:
        conservative = set()
        mutations    = set()
        for var in VCF(path):
            ref = var.REF
            conservative.add(ref)          # 仅记录有哪些 REF 碱基
            for alt in var.ALT:
                mutations.add(f"{ref}>{alt}")

        # 字典结构：{事件: 编号}
        event_dict = {"REF>REF": 0}        # 所有保守统一 0
        for idx, mut in enumerate(sorted(mutations), start=1):
            event_dict[mut] = idx
        event_dict["<MISS>"] = len(event_dict)  # 缺失最后

        EVENT2CODE = event_dict
        MISS_CODE  = EVENT2CODE["<MISS>"]
        os.makedirs(os.path.dirname(TOKEN_JSON), exist_ok=True)
        with open(TOKEN_JSON, 'w') as f:
            json.dump(EVENT2CODE, f, indent=2)
        print(f'[train] REF>ALT event dict saved -> {TOKEN_JSON}')
        print(f'       events: {list(EVENT2CODE.keys())}')
    else:
        # 验证阶段：直接读字典
        if EVENT2CODE is None:
            with open(TOKEN_JSON) as f:
                EVENT2CODE = json.load(f)
            MISS_CODE = EVENT2CODE["<MISS>"]

    # 2. 再次遍历：编码基因型
    gts_list, var_depth_list, quat_list = [], [], []
    total = sum(1 for _ in VCF(path))
    for var in tqdm(VCF(path), total=total, desc="Parsing VCF"):
        ref  = var.REF
        alts = [ref] + var.ALT          # index0 是 REF
        # 局部 allele -> 事件字符串
        allele2event = {}
        # 保守
        allele2event[ref] = "REF>REF"
        # 突变
        for alt in var.ALT:
            allele2event[alt] = f"{ref}>{alt}"
        allele2event['.'] = "<MISS>"

        row = []
        for gt_str in var.gt_bases:
            if gt_str in ['.|.', './.']:
                row.extend([MISS_CODE, MISS_CODE])
            else:
                sep = '|' if phased else '/'
                alleles = gt_str.split(sep)
                if len(alleles) == 2:
                    for a in alleles:
                        ev = allele2event.get(a, "<MISS>")
                        if ev not in EVENT2CODE:
                            print(f'[warning] unseen event "{ev}" at {var.CHROM}:{var.POS} -> MISS')
                            ev = "<MISS>"
                        row.append(EVENT2CODE[ev])
                elif len(alleles) == 1:  # 单倍体
                    a = alleles[0]
                    ev = allele2event.get(a, "<MISS>")
                    if ev not in EVENT2CODE:
                        print(f'[warning] unseen event "{ev}" at {var.CHROM}:{var.POS} -> MISS')
                        ev = "<MISS>"
                    row.extend([EVENT2CODE[ev], MISS_CODE])
                else:
                    row.extend([MISS_CODE, MISS_CODE])
        gts_list.append(np.array(row, dtype=np.int32))

        var_depth_list.append(len(alts))  # 含 REF
        quat_list.append(build_quaternion(var.CHROM, var.POS,
                                          chrom_len, chrom_start, genome_len))

    # 3. 组装 tensor
    gts = np.vstack(gts_list).T  # (n_snps, 2*n_samples) -> (n_samples, n_snps)
    gts = torch.tensor(gts, dtype=torch.long)
    gts_onehot = torch.nn.functional.one_hot(gts, num_classes=len(EVENT2CODE))

    var_depth_index = torch.tensor(var_depth_list, dtype=torch.int8)
    quat_tensor     = torch.tensor(quat_list, dtype=torch.float32)

    return gts_onehot, samples, var_depth_index, quat_tensor


# ---------- 主入口 ----------
if __name__ == "__main__":
    # 假设已有 load_config 函数
    from src.utils import load_config
    cfg = load_config("config/config.json")
    os.makedirs(cfg.data.path, exist_ok=True)

    phased      = bool(cfg.data.tihp)
    genome_json = cfg.data.genome_json

    # 训练集
    train_gts, train_samples, var_depth_index, quat_train = read_vcf(
        cfg.data.train_vcf, phased, genome_json, is_train=True)
    torch.save({'gts': train_gts, 'coords': quat_train, 'var_depths': var_depth_index},
               os.path.join(cfg.data.path, "train.pt"))
    print(f'Saved train.pt | gts={tuple(train_gts.shape)} | coords={tuple(quat_train.shape)}')

    # 验证集
    val_gts, val_samples, _, quat_val = read_vcf(
        cfg.data.val_vcf, phased, genome_json, is_train=False)
    torch.save({'gts': val_gts, 'coords': quat_val, 'var_depths': var_depth_index},
               os.path.join(cfg.data.path, "val.pt"))
    print(f'Saved val.pt   | gts={tuple(val_gts.shape)} | coords={tuple(quat_val.shape)}')