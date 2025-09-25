# imputation.py
import os, tempfile, shutil, subprocess
import torch
import json
from pathlib import Path
from tqdm import tqdm
from cyvcf2 import VCF, Writer
from src.utils import load_config
from src.model import EvoFill
from src.vcf2tensor import build_quaternion
from src.tensor2vcf import logits2vcf_line

def add_header_lines(in_vcf: Path, work_dir: Path) -> Path:
    """
    返回一个 *完整 VCF*（含记录），其 header 已追加 GT/GP/DS/IMPUTED。
    依赖 bcftools（默认服务器已有），没有就退化为文本拼接。
    """
    header_txt = [
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1">',
        '##FORMAT=<ID=DS,Number=A,Type=Float,Description="Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]">',
        '##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description="Marker was imputed">'
    ]
    hdr_file = work_dir / "extra.hdr"
    hdr_file.write_text('\n'.join(header_txt) + '\n')

    out_vcf = work_dir / f"{in_vcf.stem}.withHdr.vcf.gz"
    try:
        # 1. bcftools reheader 直接生成新文件（最快，且保留压缩）
        subprocess.check_call(
            ["bcftools", "reheader", "-h", str(hdr_file), "-o", str(out_vcf), str(in_vcf)],
            stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 2. fallback：文本级拼接（解压→改头→再压缩）
        tmp_vcf = work_dir / "tmp.vcf"
        with open(tmp_vcf, 'w') as fo:
            # 先写新头
            fo.write(open(hdr_file).read())
            # 再把原始记录流式拷过来（跳过旧头）
            with open(in_vcf) as fi:
                for line in fi:
                    if line.startswith('#'):
                        continue
                    fo.write(line)
        # 重新压缩索引（可选）
        subprocess.check_call(["bgzip", "-f", str(tmp_vcf)])
        shutil.move(str(tmp_vcf) + ".gz", str(out_vcf))

    return out_vcf


cfg   = load_config("config/config.json")
device= torch.device( cfg.train.device)
ckpt  = Path(cfg.train.save)/"evofill_best.pt"
out_vcf = Path(cfg.infer.output_vcf)

# ---- 0. 临时工作目录 ----
work_dir = out_vcf.parent / "tmp_impute"
work_dir.mkdir(exist_ok=True)

# ---- 1. 模型 ----
model = EvoFill(**vars(cfg.model)).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# ---- 2. 先给输入 VCF 加 header ----
tmp_vcf = add_header_lines(Path(cfg.infer.input_vcf), work_dir)

# ---- 3. 打开 VCF ----
vcf_in  = VCF(str(tmp_vcf), gts012=True)
vcf_out = Writer(str(out_vcf), vcf_in)

# ---- 4. 坐标张量 ----
gmeta = json.load(open(cfg.data.genome_json))

var_coords, var_idxs = [], []
for variant in vcf_in:
    var_coords.append(
        torch.tensor(build_quaternion(variant.CHROM, variant.POS,
                                        gmeta['chrom_len'], gmeta['chrom_start'], gmeta['genome_len']))
    )
    var_idxs.append(len(var_coords)-1)
var_coords = torch.stack(var_coords)  # (n_var, 4)

# ---- 5. 分批推理并写回 ----
batch_size = cfg.infer.batch_size
n_var = len(var_idxs)
for start in tqdm(range(0, n_var, batch_size), desc="Imputing"):
    end  = min(start + batch_size, n_var)
    idxs = var_idxs[start:end]
    coords_batch = var_coords[idxs].to(device)

    gts_batch = []
    for i in idxs:
        gts_batch.append(torch.tensor(vcf_in[i].gt_types, dtype=torch.int8))
    gts_batch = torch.stack(gts_batch).to(device)

    with torch.no_grad():
        logits = model(gts_batch, coords_batch)  # (B, nsamp, n_cats)

    for b, i in enumerate(idxs):
        var  = vcf_in[i]
        line = logits2vcf_line(var, logits[b], len(vcf_in.samples))
        var.set_format('GT', line['GT'])
        var.set_format('GP', line['GP'])
        var.set_format('DS', line['DS'])
        if line['IMPUTED']:
            var.INFO['IMPUTED'] = True
        vcf_out.write_record(var)

vcf_out.close(); vcf_in.close()
# 可选：删除临时文件
shutil.rmtree(work_dir, ignore_errors=True)
print(f"[IMPUTED] {out_vcf}")
