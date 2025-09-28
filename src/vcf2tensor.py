import os
import math
import json
import numpy as np
import torch
from tqdm import tqdm
from cyvcf2 import VCF

from src.utils import load_config


def build_quaternion(chrom, pos, chrom_len_dict, chrom_start_dict, genome_len):
    """
    返回 list[float32] 长度 4
    """
    def _log4(x):
        return math.log(x) / math.log(4)

    chrom = str(chrom).strip('chr')
    pos = int(pos)
    c_len = chrom_len_dict[chrom]
    c_start = chrom_start_dict[chrom]
    abs_pos = c_start + pos
    return [
        _log4(pos),
        _log4(c_len),
        _log4(abs_pos),
        _log4(genome_len),
    ]


def read_vcf(path: str, phased: bool, genome_json: str):
    """
    返回
        gts: np.ndarray (n_samples, n_snps)  int32
        samples: list[str]
        var_index: torch.Tensor (n_snps,)  int8
        depth: int
        pos_tensor: torch.Tensor (n_snps, 2)  str  // 染色体+坐标
        quat_tensor: torch.Tensor (n_snps, 4)  float32
    同时保存 var_index.pt
    """
    # ---- 0. 读基因组元信息 ----
    with open(genome_json) as f:
        gmeta = json.load(f)
    chrom_len = gmeta["chrom_len"]        # dict[str, int]
    chrom_start = gmeta["chrom_start"]    # dict[str, int]
    genome_len = gmeta["genome_len"]      # int

    vcf = VCF(path)
    samples = list(vcf.samples)

    gts_list = []
    var_depth_list = []
    quat_list = []

    total = sum(1 for _ in VCF(path))
    for var in tqdm(vcf, total=total, desc="Parsing VCF"):
        alleles = [var.REF] + var.ALT
        allele2idx = {a: i for i, a in enumerate(alleles)}

        row = []
        for gt_str in var.gt_bases:
            if gt_str in ['.|.', './.']:
                row.append([-1,-1])
            else:
                sep = '|' if phased else '/'
                for a in gt_str.split(sep):
                    row.append(allele2idx[a])
        row = np.array(row, dtype=np.int32)
        gts_list.append(row)
        
        var_depth_list.append(int(len(alleles)))

        # 变异位点位置坐标
        quat = build_quaternion(var.CHROM, var.POS, chrom_len, chrom_start, genome_len)
        quat_list.append(quat)

    gts = np.vstack(gts_list).T.astype(np.int32)
    flat = gts[gts >= 0]
    global_depth = int(flat.max())

    gts = torch.tensor(gts, dtype=torch.int8)
    var_depth_index = torch.tensor(var_depth_list, dtype=torch.int8)
    quat_tensor = torch.tensor(quat_list, dtype=torch.float32)

    return gts, samples, var_depth_index, global_depth, quat_tensor


if __name__ == "__main__":
    cfg = load_config("config/config.json")
    os.makedirs(cfg.data.path, exist_ok=True)

    phased = bool(cfg.data.tihp)
    genome_json = cfg.data.genome_json

    # ---------- 训练集 ----------
    train_gts, train_samples, var_depth_index, global_depth, quat_train = read_vcf(
        cfg.data.train_vcf, phased, genome_json)
    print(f"Inferred unified depth = {global_depth}")

    torch.save({'gts': train_gts, 'coords':quat_train, 'var_depths':var_depth_index},
               os.path.join(cfg.data.path, "train.pt"))

    print(f"Saved train.pt | gts={tuple(train_gts.shape)} | coords={tuple(quat_train.shape)}")


    # ---------- 验证集 ----------
    val_gts, val_samples, _, _, quat_val = read_vcf(
        cfg.data.val_vcf, phased, genome_json)

    torch.save({'gts': val_gts, 'coords':quat_val, 'var_depths':var_depth_index},
               os.path.join(cfg.data.path, "val.pt"))

    print(f"Saved val.pt   | gts={tuple(val_gts.shape)} | coords={tuple(quat_val.shape)}")
