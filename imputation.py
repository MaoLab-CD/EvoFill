#!/usr/bin/env python3
# imputation.py

import torch
import pandas as pd
from tqdm import tqdm
from cyvcf2 import VCF, Writer
from pathlib import Path

from src.utils import load_config
from src.model import EvoFill
from train import precompute_maf, imputation_maf_accuracy_epoch
from src.vcf2tensor import read_vcf


def load_model(cfg, device):
    model = EvoFill(**vars(cfg.model)).to(device)
    ckpt_path = Path(cfg.train.save) / "evofill_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def batch_inference(model, gts, coords, cfg, device, infer_batch=64):
    """
    分段 + 分样本 batch 推理
    返回: filled_gts (N, L)  ,  mask (N, L)
    """
    N, L = gts.shape
    chunk_size = cfg.model.chunk_size
    overlap = cfg.model.chunk_overlap
    filled = gts.clone()
    mask = gts == -1

    # 1. 按位点滑窗
    for start in tqdm(range(0, L, chunk_size - overlap), desc="chunk"):
        end = min(start + chunk_size, L)
        sub_coords = coords[start:end].to(device)          # (sub_L, 4)
        sub_gts = gts[:, start:end]                        # (N, sub_L)

        # 2. 对该 chunk 再按样本 batch 前向
        preds_chunk = torch.empty_like(sub_gts)
        for i in range(0, N, infer_batch):
            batch_gts = sub_gts[i:i + infer_batch].to(device)
            logits = model(batch_gts, sub_coords)          # (B, sub_L, A)
            preds = logits.argmax(-1).cpu()
            preds_chunk[i:i + infer_batch] = preds

        # 3. 仅更新缺失位点
        sub_mask = mask[:, start:end]
        filled[:, start:end][sub_mask] = preds_chunk[sub_mask]
    return filled, mask


def write_imputed_vcf(in_vcf_path, out_vcf_path, filled_gts, samples):
    """
    把填充后的基因型写回 VCF，仅替换原来 ./. 或 .|. 的行
    """
    # 先把 VCF 打开一次，取出长度
    vcf = VCF(in_vcf_path)
    total_sites = sum(1 for _ in vcf) 
    vcf.close()

    # 重新打开正式循环，并加进度条
    vcf = VCF(in_vcf_path)
    w = Writer(out_vcf_path, tmpl=vcf)
    n_sample_dip = len(samples) 
    miss_count = 0
    for var_idx, var in enumerate(tqdm(vcf, total=total_sites, desc="writing VCF")):
        gt_arr  = var.genotypes.copy()   # (N, 3)
        for i in range(n_sample_dip):    # 二倍体样本索引
            row = gt_arr[i]
            if row[0] == -1 and row[1] == -1:   # 缺失
                a1 = filled_gts[2 * i,     var_idx]
                a2 = filled_gts[2 * i + 1, var_idx]
                try:
                    row[0] = a1
                    row[1] = a2
                except IndexError:
                    miss_count += 1
                    pass

        var.genotypes = gt_arr
    w.write_record(var)
    print(f"{miss_count} sites fill failed.")
    w.close()
    vcf.close()


def evaluate_with_truth(cfg, imputed_gts, imputed_mask, device):
    """
    若提供了 ground-truth VCF，计算 MAF-bin 准确度
    """
    gt_gts, _, _, _, _ = read_vcf(
        cfg.infer.ground_true,
        phased=bool(cfg.data.tihp),
        genome_json=cfg.data.genome_json
    )
    assert gt_gts.shape == imputed_gts.shape
    maf, _ = precompute_maf(gt_gts.numpy(), mask_int=-1)
    A = cfg.model.n_alleles
    logits = torch.nn.functional.one_hot(imputed_gts.long(), num_classes=A).float()
    accs = imputation_maf_accuracy_epoch(
        logits.unsqueeze(0),
        gt_gts.unsqueeze(0),
        imputed_mask.unsqueeze(0),
        maf
    )
    bins = ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"]
    bin_edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    site_counts = [
        ((maf >= bin_edges[i]) & (maf < bin_edges[i+1])).sum().item()
        for i in range(6)
    ]

    # ---- 构造 DataFrame ----
    df = pd.DataFrame({
        "MAF_bin":  bins,
        "n_sites":  site_counts,
        "accuracy": accs
    })

    csv_path = Path(cfg.infer.impute_vcf).with_suffix(".maf_acc.csv")
    df.to_csv(csv_path, index=False)
    print("MAF-bin accuracy:")
    print(df)
    return df


def main():
    cfg = load_config("config/config.json")
    device = torch.device("cuda")

    # 1. 加载模型
    model = load_model(cfg, device)

    # 2. 读取待填充 VCF
    print("Reading input VCF …")
    gts, samples, _, _, coords = read_vcf(
        cfg.infer.input_vcf,
        phased=bool(cfg.data.tihp),
        genome_json=cfg.data.genome_json
    )
    print(f"gts shape: {gts.shape}, coords shape: {coords.shape}")

    # 3. 分段 + 分样本 batch 推理
    infer_batch = cfg.infer.batch_size
    filled, mask = batch_inference(model, gts, coords, cfg, device, infer_batch)

    # 4. 写回 VCF
    print("Writing imputed VCF …")
    write_imputed_vcf(cfg.infer.input_vcf, cfg.infer.impute_vcf, filled, samples)
    print(f"Saved: {cfg.infer.impute_vcf}")

    # 5. 若有真值，则评估
    if Path(cfg.infer.ground_true).exists():
        print("Evaluating against ground truth …")
        evaluate_with_truth(cfg, filled, mask, device)
    else:
        print("No ground truth provided, skip evaluation.")


if __name__ == "__main__":
    main()