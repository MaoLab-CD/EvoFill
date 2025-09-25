#!/usr/bin/env python3
# tensor2vcf.py
import torch
import numpy as np

def logits2vcf_line(var, logits, n_sample):
    """
    把一条 variant 的 logits (n_sample, n_cats) 转成 VCF 所需字段
    返回 dict: {'GT': ..., 'GP': ..., 'DS': ..., 'IMPUTED': True/False}
    """
    # 1. softmax -> 概率
    probs = torch.softmax(logits, dim=-1).cpu().numpy()   # (n_sample, n_cats)

    # 2. 构造输出
    GT_list, GP_list, DS_list = [], [], []
    imputed_flag = False

    for s in range(n_sample):
        gt_orig = var.gt_types[s]        # 原始 -1/0/1/2
        if gt_orig != -1:                # 非缺失，保持原样
            gt_str = var.gt_bases[s]     # '0|0' 等
            # GP 用 one-hot；DS 用剂量
            idx = int(gt_orig)
            gp = np.zeros(probs.shape[1])
            gp[idx] = 1.0
            ds = float(idx) if idx <= 2 else 2.0
        else:                            # 缺失 -> 用模型
            imputed_flag = True
            gp = probs[s]                # 已经 softmax
            # 二倍体剂量 DS = P(0/1) + 2*P(1/1)
            if gp.shape[0] == 3:
                ds = gp[1] + 2*gp[2]
            else:
                ds = np.dot(np.arange(gp.shape[0]), gp)
            # MAP genotype
            idx = int(np.argmax(gp))
            a1, a2 = idx//2, idx%2
            gt_str = f"{a1}|{a2}"

        GT_list.append(gt_str)
        GP_list.append(gp)
        DS_list.append(ds)

    # 3. 组装成 cyvcf2 能接受的格式
    # GT:  list[str]
    # GP:  2-D numpy array (n_sample, n_genotypes)
    # DS:  1-D numpy array (n_sample,)
    return {
        'GT': np.array(GT_list, dtype='S10'),   # 必须字节数组
        'GP': np.vstack(GP_list),
        'DS': np.array(DS_list, dtype=np.float32),
        'IMPUTED': imputed_flag
    }