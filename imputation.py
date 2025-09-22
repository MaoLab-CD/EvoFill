#!/usr/bin/env python3
"""
对已缺失 VCF 进行填充
python imputation.py
"""
import os
import torch
from tqdm import tqdm

from src.utils import load_config
from src.model import MambaImputer
from src.vcf2tensor import read_vcf, encode_tensor
from src.tensor2vcf import imputed_tensor2vcf


def main():
    cfg = load_config("config/config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. 加载模型 ----------
    n_cats = torch.load(os.path.join(cfg.data.out_dir, "train.pt"))["var_site"].max().item() + 1
    model = MambaImputer(
        n_cats=n_cats,
        d_model=cfg.model.d_model,
        d_state=cfg.model.d_state,
        d_conv=cfg.model.d_conv,
        n_layers=cfg.model.n_layers,
        expand=getattr(cfg.model, "expand", 2),
        headdim=getattr(cfg.model, "headdim", 64),
    ).to(device)
    ckpt = torch.load(os.path.join(cfg.train.save_dir, "best.pt"), map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    print("[IMP] model loaded ->", cfg.train.save_dir + "/best.pt")

    # ---------- 2. 读缺失 VCF ----------
    phased = bool(cfg.data.tihp)
    gts, samples, _, depth = read_vcf(cfg.infer.input_vcf, phased, cfg.data.out_dir)
    # 构造 mask：-1 位置需要 impute
    mask_missing = (gts == -1)
    tensor_full = encode_tensor(gts, depth).to(device)  # (n_samples, n_sites)

    # ---------- 3. 分批推理 ----------
    n_samples, n_sites = tensor_full.shape
    with torch.no_grad():
        # tensor_full: (n_samples, n_sites)
        for i in tqdm(range(n_samples), desc="Imputing"):
            seq = tensor_full[i:i+1]          # (1, L)  保持 batch=1
            logits = model(seq.long())        # (1, L, n_cats)
            imp = logits.argmax(dim=-1).squeeze(0)  # (L,)
            # 只更新缺失位点
            miss = mask_missing[i]            # (L,)
            tensor_full[i][miss] = imp[miss].to(tensor_full.dtype)
    # ---------- 4. 写回 VCF ----------
    os.makedirs(os.path.dirname(cfg.infer.output_vcf), exist_ok=True)
    return tensor_full.cpu().to(torch.int8)
    torch.save(tensor_full.cpu().to(torch.int8),
                os.path.join(cfg.data.out_dir, "imupte_res.pt"))
    imputed_tensor2vcf(
        tensor_full.cpu().to(torch.int8),
        cfg.infer.input_vcf,
        cfg.infer.output_vcf,
        samples,
        phased,
    )
    print("[IMP] all done ->", cfg.infer.output_vcf)


if __name__ == "__main__":
    tensor = main()