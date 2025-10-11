import os, math, json, numpy as np, torch
from tqdm import tqdm
from cyvcf2 import VCF
from src.utils import load_config

# ========= 新增：全局变量放字典 =========
TOKEN_JSON = None          # 首次调用 read_vcf 时写入
ALLELE2CODE = None         # 首次建立后不再修改
MISSING_CODE = None        # 等于 len(ALLELE2CODE)-1


def build_quaternion(chrom, pos, chrom_len_dict, chrom_start_dict, genome_len):
    def _log4(x): return math.log(x)/math.log(4)
    chrom = str(chrom).strip('chr')
    pos = int(pos)
    c_len = chrom_len_dict[chrom]
    c_start = chrom_start_dict[chrom]
    abs_pos = c_start + pos
    return [_log4(pos), _log4(c_len), _log4(abs_pos), _log4(genome_len)]


def read_vcf(path: str, phased: bool, genome_json: str, global_depth=None, is_train=False):
    """
    返回
        gts:  (n_samples, n_snps)  int32
        samples: list[str]
        var_index: torch.Tensor (n_snps,)  int8
        depth: int
        pos_tensor: torch.Tensor (n_snps, 2)  str
        quat_tensor: torch.Tensor (n_snps, 4)  float32
    同时保存 var_index.pt
    """
    global ALLELE2CODE, MISSING_CODE, TOKEN_JSON

    # ---- 0. 读基因组元信息 ----
    with open(genome_json) as f:
        gmeta = json.load(f)
    chrom_len = gmeta["chrom_len"]
    chrom_start = gmeta["chrom_start"]
    genome_len = gmeta["genome_len"]

    vcf = VCF(path)
    samples = list(vcf.samples)

    gts_list = []
    var_depth_list = []
    quat_list = []

    # ---- 1. 训练阶段：先扫描所有 allele ----
    if is_train:
        allele_set = set()
        for var in VCF(path):
            allele_set.update([var.REF] + var.ALT)
        if '.' not in allele_set:
            allele_set.add('.')
        allele_list = sorted(allele_set)          # 保证顺序稳定
        ALLELE2CODE = {a: i for i, a in enumerate(allele_list)}
        MISSING_CODE = ALLELE2CODE['.']           # 缺失一定在最后
        os.makedirs(os.path.dirname(TOKEN_JSON), exist_ok=True)
        with open(TOKEN_JSON, 'w') as f:
            json.dump(ALLELE2CODE, f, indent=2)
        print(f'[train] allele -> code dict saved to {TOKEN_JSON}')
        print(f'       allele list: {allele_list}')
    else:
        # ---- 2. 验证阶段：直接读字典 ----
        if ALLELE2CODE is None:
            with open(TOKEN_JSON) as f:
                ALLELE2CODE = json.load(f)
            MISSING_CODE = ALLELE2CODE['.']

    # ---- 3. 再次遍历，真正编码 ----
    total = sum(1 for _ in VCF(path))
    for var in tqdm(vcf, total=total, desc="Parsing VCF"):
        alleles = [var.REF] + var.ALT
        # 建立临时映射：本次位点的 allele -> 全局字典里的 code
        local2global = {}
        for a in alleles:
            if a in ALLELE2CODE:
                local2global[a] = ALLELE2CODE[a]
            else:
                # val 里出现 train 没见过的 allele
                print(f'[warning] unseen allele "{a}" at {var.CHROM}:{var.POS}, treat as missing.')
                local2global[a] = MISSING_CODE
        local2global['.'] = MISSING_CODE   # 强制缺失

        row = []
        for gt_str in var.gt_bases:
            if gt_str in ['.|.', './.']:
                row.extend([MISSING_CODE, MISSING_CODE])
            else:
                sep = '|' if phased else '/'
                alts = gt_str.split(sep)
                if len(alts) == 2:
                    row.extend([local2global.get(a, MISSING_CODE) for a in alts])
                elif len(alts) == 1:          # 单倍体
                    row.extend([local2global.get(alts[0], MISSING_CODE), MISSING_CODE])
                else:
                    row.extend([MISSING_CODE, MISSING_CODE])
        row = np.array(row, dtype=np.int32)
        gts_list.append(row)

        var_depth_list.append(len(alleles))
        quat_list.append(build_quaternion(var.CHROM, var.POS,
                                          chrom_len, chrom_start, genome_len))

    gts = np.vstack(gts_list).T.astype(np.int32)

    # ---- 4. 全局 depth 沿用旧逻辑 ----
    flat = gts[gts != MISSING_CODE]
    if global_depth is None:
        global_depth = int(flat.max())

    print(f'All missing site set to {MISSING_CODE} (internal code)')
    gts = torch.tensor(gts, dtype=torch.long)
    gts_onehot = torch.nn.functional.one_hot(gts, num_classes=len(ALLELE2CODE))

    var_depth_index = torch.tensor(var_depth_list, dtype=torch.int8)
    quat_tensor = torch.tensor(quat_list, dtype=torch.float32)

    return gts_onehot, samples, var_depth_index, global_depth, quat_tensor


if __name__ == "__main__":
    cfg = load_config("config/config.json")
    os.makedirs(cfg.data.path, exist_ok=True)

    phased = bool(cfg.data.tihp)
    genome_json = cfg.data.genome_json
    TOKEN_JSON = "config/variant_token.json"   # 全局路径

    # ---------- 训练集 ----------
    train_gts, train_samples, var_depth_index, global_depth, quat_train = read_vcf(
        cfg.data.train_vcf, phased, genome_json, is_train=True)
    print(f"Inferred unified depth: {list(range(global_depth + 1))}")

    torch.save({'gts': train_gts, 'coords': quat_train, 'var_depths': var_depth_index},
               os.path.join(cfg.data.path, "train.pt"))
    print(f"Saved train.pt | gts={tuple(train_gts.shape)} | coords={tuple(quat_train.shape)}")

    # ---------- 验证集 ----------
    val_gts, val_samples, _, _, quat_val = read_vcf(
        cfg.data.val_vcf, phased, genome_json, global_depth=global_depth, is_train=False)

    torch.save({'gts': val_gts, 'coords': quat_val, 'var_depths': var_depth_index},
               os.path.join(cfg.data.path, "val.pt"))
    print(f"Saved val.pt   | gts={tuple(val_gts.shape)} | coords={tuple(quat_val.shape)}")