import os
import subprocess
import numpy as np
from cyvcf2 import VCF

def run(cmd):
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout

def add_format_headers_if_missing(header_lines):
    has_ds = any(l.startswith("##FORMAT=<ID=DS,") for l in header_lines)
    has_gp = any(l.startswith("##FORMAT=<ID=GP,") for l in header_lines)

    chrom_idx = None
    for i, l in enumerate(header_lines):
        if l.startswith("#CHROM"):
            chrom_idx = i
            break
    if chrom_idx is None:
        raise ValueError("Invalid VCF header: cannot find #CHROM line")

    extra = []
    if not has_ds:
        extra.append('##FORMAT=<ID=DS,Number=1,Type=Float,Description="ALT dosage = P(0/1)+2*P(1/1) from GP">')
    if not has_gp:
        extra.append('##FORMAT=<ID=GP,Number=3,Type=Float,Description="Genotype probabilities for 0/0,0/1,1/1">')

    if extra:
        header_lines = header_lines[:chrom_idx] + extra + header_lines[chrom_idx:]
    return header_lines

def normalize_probs(p):
    # p: (n_samples, 3)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = p.sum(axis=1, keepdims=True)
    bad = (row_sum[:, 0] <= 0.0)
    if bad.any():
        p[bad, :] = np.array([1.0, 0.0, 0.0], dtype=p.dtype)  # 兜底 0/0
        row_sum = p.sum(axis=1, keepdims=True)
    p = p / row_sum
    return p

def bgzip_and_index(plain_vcf, out_vcfgz):
    # bgzip plain_vcf -> plain_vcf.gz
    run(["bgzip", "-f", plain_vcf])
    gz = plain_vcf + ".gz"
    if gz != out_vcfgz:
        os.makedirs(os.path.dirname(out_vcfgz), exist_ok=True)
        os.replace(gz, out_vcfgz)
    run(["bcftools", "index", "-t", out_vcfgz])

def make_imputed_vcfgz_from_prob(prob_npy, mask_vcf_gz, out_vcfgz, digits=4):
    print("[INFO] Loading probs (mmap):", prob_npy)
    probs = np.load(prob_npy, mmap_mode="r")
    if probs.ndim != 3 or probs.shape[2] != 3:
        raise ValueError(f"Expected probs shape (N_sites,N_samples,3) or (N_samples,N_sites,3), got {probs.shape}")

    vcf_in = VCF(mask_vcf_gz)
    samples = vcf_in.samples
    n_samples = len(samples)
    print("[INFO] #samples in mask VCF =", n_samples)

    A, B, _ = probs.shape
    if A == n_samples:
        orientation = "sample_first"  # (N_samples, N_sites, 3)
        n_sites_prob = B
        print("[INFO] probs orientation: (N_samples, N_sites, 3)")
    elif B == n_samples:
        orientation = "site_first"    # (N_sites, N_samples, 3)
        n_sites_prob = A
        print("[INFO] probs orientation: (N_sites, N_samples, 3)")
    else:
        raise ValueError(f"Prob dims do not match sample count {n_samples}. probs.shape={probs.shape}")

    # 输出先写 plain VCF，再 bgzip
    out_plain = out_vcfgz[:-3] if out_vcfgz.endswith(".gz") else out_vcfgz
    if not out_plain.endswith(".vcf"):
        out_plain = out_plain + ".vcf"

    # Header：用 mask 的 header，并确保 DS/GP 存在
    header_lines = vcf_in.raw_header.strip("\n").splitlines()
    header_lines = add_format_headers_if_missing(header_lines)

    fmt_field = "GT:DS:GP"  # 固定顺序：DS 必须在第二列（适配你不改评估脚本）
    gt_map = np.array(["0|0", "0|1", "1|1"], dtype=object)
    ffmt = lambda x: format(float(x), f".{digits}f")

    print("[INFO] Writing plain VCF:", out_plain)
    os.makedirs(os.path.dirname(out_plain), exist_ok=True)

    with open(out_plain, "w", buffering=1024*1024) as out:
        out.write("\n".join(header_lines) + "\n")

        for i, rec in enumerate(vcf_in):
            if i >= n_sites_prob:
                raise ValueError(f"VCF has more sites than probs: i={i}, n_sites_prob={n_sites_prob}")

            # p: (n_samples,3) copy出来，避免修改 memmap view
            if orientation == "site_first":
                p = np.asarray(probs[i, :, :], dtype=np.float32).copy()
            else:
                p = np.asarray(probs[:, i, :], dtype=np.float32).copy()

            p = normalize_probs(p)

            gt_idx = np.argmax(p, axis=1)  # 0/1/2
            ds = p[:, 1] + 2.0 * p[:, 2]

            # prefix 8 列：CHROM POS ID REF ALT QUAL FILTER INFO
            fields = str(rec).rstrip("\n").split("\t")
            prefix8 = fields[:8]

            # sample columns
            samp_cols = []
            for s in range(n_samples):
                gt = gt_map[int(gt_idx[s])]  # 永远不会是缺失
                samp_cols.append(
                    f"{gt}:{ffmt(ds[s])}:{ffmt(p[s,0])},{ffmt(p[s,1])},{ffmt(p[s,2])}"
                )

            out.write("\t".join(prefix8 + [fmt_field] + samp_cols) + "\n")

            if (i + 1) % 10000 == 0:
                print(f"[INFO] processed {i+1} sites")

    vcf_in.close()

    print("[INFO] bgzip + index ->", out_vcfgz)
    bgzip_and_index(out_plain, out_vcfgz)

    # 自检
    print("[CHECK] FORMAT field:")
    run(["bash", "-lc", f"bcftools view -H {out_vcfgz} | head -n 1 | cut -f 9"])
    print("[CHECK] any '.|.' ?")
    run(["bash", "-lc", f"bcftools view -H {out_vcfgz} | head -n 5000 | grep -F '.|.' -m 1 || echo 'OK: no .|.'"])

    return out_vcfgz


