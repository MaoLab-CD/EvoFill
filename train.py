"""
单机多卡（token 并行）训练
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py
"""
import os
import torch
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.utils import load_config
from src.model import MambaImputer
from src.losses import ImputationLoss



def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


# ---------------- 数据集 ----------------
class GeneticDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(path, map_location="cpu")
        self.seq = data["var_site"].long()  # (N, L_raw)
        self.L_raw = self.seq.shape[1]

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx]  # (L,)


def collate_fn(batch, world_size: int, pad_idx: int):
    seq_list = batch
    max_len = max(s.shape[0] for s in seq_list)
    chunk_len = (max_len + world_size - 1) // world_size
    pad_len = chunk_len * world_size

    seq_pad = []
    for s in seq_list:
        if s.shape[0] < pad_len:
            s = torch.cat([s, torch.full((pad_len - s.shape[0],), pad_idx, dtype=s.dtype)])
        seq_pad.append(s)
    seq = torch.stack(seq_pad)  # (B, pad_len)
    labels = seq.clone()
    return seq, labels


# ---------------- 指标 ----------------
@torch.no_grad()
def mask_accuracy(model, loader, epoch, mask_ratio, pad_idx):
    model.eval()
    rank = dist.get_rank()
    total_acc, total_mask = 0.0, 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    for seq, labels in iterator:
        seq, labels = seq.cuda(), labels.cuda()
        B, L = seq.shape
        mask = torch.rand_like(seq.float()) < mask_ratio
        masked_seq = seq.clone()
        masked_seq[mask] = pad_idx
        logits = model(masked_seq)  # (B, L, n_cats)
        if rank == 0:
            acc = ((logits.argmax(-1) == labels) & mask).sum()
            total_acc += acc.item()
            total_mask += mask.sum().item()
    return total_acc / max(total_mask, 1) if rank == 0 else None


# ---------------- 主函数 ----------------
def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    cfg = load_config("config/config.json")

    # ---------- 数据 ----------
    train_ds = GeneticDataset(Path(cfg.data.out_dir,"train.pt"))
    val_ds = GeneticDataset(Path(cfg.data.out_dir,"val.pt"))
    n_cats = int(train_ds.seq.max().item()) + 1
    pad_idx = n_cats
    print0(f"[DATA] {train_ds.L_raw} variant sites | {n_cats} categories")

    # ---------- 模型 ----------
    model = MambaImputer(
        n_cats=n_cats,
        d_model=cfg.model.d_model,
        d_state=cfg.model.d_state,
        d_conv=cfg.model.d_conv,
        n_layers=cfg.model.n_layers,
        expand=getattr(cfg.model, "expand", 2),
        headdim=getattr(cfg.model, "headdim", 64),
    ).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    loss_fn = ImputationLoss(n_cats=n_cats, ignore_index=pad_idx, group_size=4, use_r2=False).cuda()
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ---------- DataLoader ----------
    bs = cfg.train.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, world_size, pad_idx),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, world_size, pad_idx),
        num_workers=4,
        pin_memory=True,
    )

    # ---------- 训练 ----------
    epochs = int(cfg.train.epochs)
    val_interval = int(cfg.train.val_interval)
    mask_ratio = float(cfg.train.mask_ratio)
    save_dir = cfg.train.save_dir
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    patience = int(cfg.train.early_stop.patience)
    patience_delta = float(cfg.train.early_stop.min_delta)
    patience_counter = 0
    stop_tensor = torch.tensor(0, dtype=torch.long, device="cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}", disable=(rank != 0))
        for seq, labels in iterator:
            seq, labels = seq.cuda(), labels.cuda()
            logits = model(seq)
            loss = loss_fn(logits.view(-1, n_cats), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0:
                epoch_loss += loss.item()
                iterator.set_postfix(loss=loss.item())

        if rank == 0:
            epoch_loss /= len(train_loader.dataset)
            print0(f"TRAIN {epoch:03d} | epoch_loss = {epoch_loss:.4f}")

        # ---------- 验证 ----------
        if epoch % val_interval == 0 or epoch == epochs:
            val_acc = mask_accuracy(model, val_loader, epoch, mask_ratio, pad_idx)
            if rank == 0:
                print0(f"TEST {epoch:03d} | test_mask_acc = {val_acc:.4f}")
                if val_acc > best_acc + patience_delta:
                    best_acc = val_acc
                    patience_counter = 0
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pt")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print0("Early stop!")
                    stop_tensor.fill_(1)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break

        # 常规保存
        if epoch % cfg.train.save_interval == 0 and rank == 0:
            torch.save(model.module.state_dict(), f"{save_dir}/epoch_{epoch}.pt")

    if rank == 0:
        print0("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()