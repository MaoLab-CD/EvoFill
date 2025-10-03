# train_chunk.py
# 用法：
# OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train_chunk.py

import os, math, random, warnings, numpy as np, pandas as pd, torch, torch.distributed as dist
from pathlib import Path
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import Lamb
from tqdm import tqdm

from src.utils import load_config
from src.model import EvoFill
from src.losses import ImputationLoss

warnings.filterwarnings("ignore")
MAF_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]

def unwrap(model):
    return model.module if hasattr(model, 'module') else model

# ---------------- Dataset / Sampler ----------------
class ChunkedGenotypeDataset(Dataset):
    def __init__(self, gts, coords, chunk_size, chunk_idx,
                 mask_int=-1, mask_ratio=0.0):
        self.gts        = gts.long()      # (N, L)
        self.coords     = coords.float()  # (L, 4)
        self.chunk_size = chunk_size
        self.chunk_idx  = chunk_idx
        self.mask_int   = mask_int
        self.mask_ratio = mask_ratio

        L = gts.shape[1]
        self.start = chunk_idx * chunk_size
        self.end   = min(self.start + chunk_size, L)
        self.slice = slice(self.start, self.end)

    def __len__(self):
        return self.gts.shape[0]

    def __getitem__(self, idx):
        gt_true = self.gts[idx, self.slice]
        coords  = self.coords[self.slice]
        gt_mask = gt_true.clone()
        if self.mask_ratio > 0:
            mask = torch.rand_like(gt_mask.float()) < self.mask_ratio
            gt_mask[mask] = self.mask_int
        return gt_mask, gt_true, coords


def collate_fn(batch):
    gt_mask = torch.stack([b[0] for b in batch], 0)
    gt_true = torch.stack([b[1] for b in batch], 0)
    coords  = batch[0][2]  # 共享
    return gt_mask, gt_true, coords


def build_loader(pt_path, chunk_size, chunk_idx,
                 batch_size, mask_int, mask_ratio, sampler=None):
    data = torch.load(pt_path)
    ds = ChunkedGenotypeDataset(data['gts'], data['coords'],
                                chunk_size, chunk_idx,
                                mask_int=mask_int, mask_ratio=mask_ratio)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(sampler is None),
                      drop_last=True,
                      collate_fn=collate_fn,
                      sampler=sampler,
                      num_workers=4,
                      pin_memory=True)


# ---------------- MAF ----------------
def precompute_maf_chunk(gts_np, slice_, mask_int=-1):
    sub = gts_np[:, slice_]
    L_chunk = sub.shape[1]
    maf = np.zeros(L_chunk, dtype=np.float32)
    bin_cnt = [0] * 6
    for l in range(L_chunk):
        alleles = sub[:, l]
        alleles = alleles[alleles != mask_int]
        if alleles.size == 0:
            maf[l] = 0.0
            continue
        uniq, cnt = np.unique(alleles, return_counts=True)
        total = cnt.sum()
        freq = cnt / total
        freq[::-1].sort()
        maf_val = freq[1] if len(freq) > 1 else 0.0
        maf[l] = maf_val
        for i, (lo, hi) in enumerate(MAF_BINS):
            if lo <= maf_val < hi:
                bin_cnt[i] += 1
                break
    return torch.from_numpy(maf), bin_cnt


def imputation_maf_accuracy_epoch(all_logits, all_gts, all_mask, global_maf):
    preds = torch.argmax(all_logits, dim=-1)
    correct = (preds == all_gts) & all_mask
    maf = global_maf.unsqueeze(0)
    accs = []
    for lo, hi in MAF_BINS:
        bin_mask = all_mask & (maf >= lo) & (maf < hi)
        n_cor = (correct & bin_mask).sum()
        n_tot = bin_mask.sum()
        accs.append((n_cor / n_tot).item() if n_tot > 0 else 0.0)
    return accs


# ---------------- train / validate ----------------
def train_one_epoch(model, cfg, loader, criterion, optimizer, device, rank,
                    global_maf, writer, global_step):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, ncols=80, disable=(rank != 0), leave=False)
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)

        logits = model(gt_mask, coords)
        raw_model = unwrap(model)
        shared = list(raw_model.embed.norm.parameters()) + \
                 list(raw_model.evo_embed.norm.parameters())
        loss, logs = criterion(logits, gt_true, shared_params=shared, retain_graph=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if writer:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/CE", logs['ce'], global_step)
            writer.add_scalar("Train/Focal", logs['focal'], global_step)
            writer.add_scalar("Train/R2", logs['r2'], global_step)
            writer.add_scalar("Model/global_L2", gn, global_step)
            writer.add_scalar("Model/lr", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

    # 计算本 chunk MAF-acc
    model.eval()
    all_logits, all_gts, all_mask = [], [], []
    with torch.no_grad():
        for gt_mask, gt_true, coords in loader:
            gt_mask, gt_true = gt_mask.to(device), gt_true.to(device)
            logits = model(gt_mask, coords)
            mask = gt_mask == -1
            all_logits.append(logits.cpu())
            all_gts.append(gt_true.cpu())
            all_mask.append(mask.cpu())
    all_logits = torch.cat(all_logits, 0)
    all_gts    = torch.cat(all_gts, 0)
    all_mask   = torch.cat(all_mask, 0)
    maf_accs   = imputation_maf_accuracy_epoch(all_logits, all_gts, all_mask,
                                               global_maf.cpu())
    return {"loss": total_loss / len(loader), "maf_accs": maf_accs}, global_step


@torch.no_grad()
def validate(model, cfg, loader, criterion, device, rank, global_maf, writer, global_step):
    model.eval()
    total_loss = 0.0
    all_logits, all_gts, all_mask = [], [], []
    for gt_mask, gt_true, coords in loader:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), coords.to(device)
        logits = model(gt_mask, coords)
        loss, logs = criterion(logits, gt_true)
        total_loss += loss.item()

        mask = gt_mask == -1
        all_logits.append(logits.cpu())
        all_gts.append(gt_true.cpu())
        all_mask.append(mask.cpu())

        if writer:
            writer.add_scalar("Val/Loss", loss.item(), global_step)
            writer.add_scalar("Val/CE", logs['ce'], global_step)
            writer.add_scalar("Val/Focal", logs['focal'], global_step)
            writer.add_scalar("Val/R2", logs['r2'], global_step)
            global_step += 1

    all_logits = torch.cat(all_logits, 0)
    all_gts    = torch.cat(all_gts, 0)
    all_mask   = torch.cat(all_mask, 0)
    maf_accs   = imputation_maf_accuracy_epoch(all_logits, all_gts, all_mask,
                                               global_maf.cpu())
    return {"loss": total_loss / len(loader), "maf_accs": maf_accs}, global_step


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, mode='min', save_dir='./ckpt', rank=0):
        self.rank = rank
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.save_dir = Path(save_dir)

    def __call__(self, metric, model):
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in unwrap(model).state_dict().items()}
            if self.rank == 0:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_state, self.save_dir / "evofill_best.pt")
                print(f"Model weights updated: {self.save_dir / 'evofill_best.pt'}")
            return False
        better = (metric < self.best - self.min_delta) if self.mode == 'min' else (metric > self.best + self.min_delta)
        if better:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in unwrap(model).state_dict().items()}
            if self.rank == 0:
                torch.save(self.best_state, self.save_dir / "evofill_best.pt")
                print(f"Model weights updated: {self.save_dir / 'evofill_best.pt'}")
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------- main ----------------
def main():
    # 1. 环境
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2. 配置 & 随机种子
    cfg = load_config("config/config.json")
    seed = cfg.train.seed + rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # 3. 数据常数
    train_data = torch.load(Path(cfg.data.path) / "train.pt")
    L = train_data['gts'].shape[1]
    chunk_size   = cfg.model.chunk_size
    num_chunks   = math.ceil(L / chunk_size)
    epochs_per_chunk = cfg.train.num_epochs

    # 4. 模型
    model = EvoFill(**vars(cfg.model)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. TensorBoard
    writer = None
    if rank == 0:
        tb_dir = Path(cfg.train.save) / "tensorboard" / datetime.now().strftime("%m-%d_%H:%M")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_dir)

    # 6. 损失 & 优化器 & 早停
    criterion = ImputationLoss(
        use_r2=cfg.train.use_r2_loss,
        use_focal=cfg.train.use_focal,
        group_size=cfg.train.r2_loss_groupsize,
        use_gradnorm=cfg.train.use_gradnorm,
        gn_alpha=cfg.train.gn_alpha,
        gn_lr_w=cfg.train.gn_lr_w).to(device)

    optimizer = Lamb(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.train.scheduler_factor,
                                  patience=cfg.train.scheduler_patience, min_lr=cfg.train.min_lr)
    early_stopper = EarlyStopper(patience=cfg.train.earlystop_patience,
                                 min_delta=cfg.train.min_delta,
                                 mode='min', save_dir=Path(cfg.train.save), rank=rank)

    # 7. 顺序：chunk -> epoch
    global_step = 0
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end   = min(start + chunk_size, L)
        slice_ = slice(start, end)
        global_maf, bin_cnt = precompute_maf_chunk(train_data['gts'].numpy(), slice_, mask_int=-1)
        if rank == 0:
            print(f"\n========== Chunk {chunk_id}/{num_chunks-1} | {end-start} sites ==========")

        # 为该 chunk 构建固定 sampler/loader
        train_sampler = DistributedSampler(
            ChunkedGenotypeDataset(train_data['gts'], train_data['coords'],
                                   chunk_size, chunk_id, mask_int=-1, mask_ratio=cfg.train.mask_ratio),
            num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(
            ChunkedGenotypeDataset(torch.load(Path(cfg.data.path)/"val.pt")['gts'],
                                   torch.load(Path(cfg.data.path)/"val.pt")['coords'],
                                   chunk_size, chunk_id, mask_int=-1, mask_ratio=cfg.train.mask_ratio),
            num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = build_loader(Path(cfg.data.path)/"train.pt", chunk_size, chunk_id,
                                    cfg.train.batch_size, -1, cfg.train.mask_ratio, train_sampler)
        val_loader   = build_loader(Path(cfg.data.path)/"val.pt",   chunk_size, chunk_id,
                                    cfg.train.batch_size, -1, cfg.train.mask_ratio, val_sampler)

        for inner_epoch in range(1, epochs_per_chunk + 1):
            train_sampler.set_epoch(inner_epoch)   # 保证 shuffle 不同

            train_metric, global_step = train_one_epoch(
                model, cfg, train_loader, criterion, optimizer, device, rank,
                global_maf, writer, global_step)
            val_metric, global_step   = validate(
                model, cfg, val_loader, criterion, device, rank,
                global_maf, writer, global_step)

            if rank == 0:
                print(f"Chunk {chunk_id} inner-epoch {inner_epoch}/{epochs_per_chunk} | "
                      f"train loss: {train_metric['loss']:.5f} | val loss: {val_metric['loss']:.5f}")
                df = pd.DataFrame({
                    'MAF_bin': [f"{lo}-{hi}" for lo, hi in MAF_BINS],
                    'Counts' : [f"{c}" for c in bin_cnt],
                    'Train'  : [f"{acc:.3f}" for acc in train_metric['maf_accs']],
                    'Val'    : [f"{acc:.3f}" for acc in val_metric['maf_accs']]
                })
                print(df.to_string(index=False))

            scheduler.step(val_metric['loss'])

            # 同步 val loss 后早停
            val_loss_tensor = torch.tensor(val_metric['loss']).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            if early_stopper(val_loss_tensor.item(), model):
                if rank == 0:
                    print(f"Early stopping at chunk {chunk_id} inner-epoch {inner_epoch}")
                break
        else:
            continue
        break          # 早停外层 chunk

    if rank == 0:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()