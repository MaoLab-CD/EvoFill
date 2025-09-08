"""
单机多卡（token 并行）训练
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train2.py
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.utils import load_config
from src.model_condssm import DistributedMamba


# ---------------- 工具 ----------------
def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

# ---------------- 数据集 ----------------
class GeneticDataset(Dataset):
    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.var_site = data["var_site"].long()   # (N, L_raw)
        self.p_dis    = data["p_dis"].float()     # (N, k)
        self.L_raw    = self.var_site.shape[1]

    def __len__(self):
        return self.var_site.shape[0]

    def __getitem__(self, idx):
        return {"seq": self.var_site[idx],        # (L,)
                "p_dis": self.p_dis[idx]}        # (k,)
    

def collate_fn(batch, world_size: int, pad_idx: int):
    seq_list  = [b["seq"] for b in batch]
    feat_list = [b["p_dis"] for b in batch]

    max_len_raw = max(s.shape[0] for s in seq_list)
    chunk_len = (max_len_raw + world_size - 1) // world_size
    pad_len = chunk_len * world_size

    seq_pad = []
    for s in seq_list:
        if s.shape[0] < pad_len:
            s = torch.cat([s, torch.full((pad_len - s.shape[0],), 0, dtype=s.dtype)])   # 补零
        seq_pad.append(s)
    seq = torch.stack(seq_pad)        # (B, pad_len)
    p_dis = torch.stack(feat_list)    # (B, k)
    labels = seq.clone()              # 与输入对齐
    return seq, p_dis, labels

# ---------------- 指标 ----------------
@torch.no_grad()
def mask_accuracy(model, loader, epoch, mask_ratio, pad_idx):
    model.eval()
    rank = dist.get_rank()
    total_acc, total_mask = 0.0, 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    for seq, feat, labels in iterator:
        seq, feat, labels = seq.cuda(), feat.cuda(), labels.cuda()
        B, L = seq.shape
        mask = torch.rand_like(seq.float()) < mask_ratio
        masked_seq = seq.clone()
        masked_seq[mask] = pad_idx
        _, logits = model(masked_seq, feat, labels=None)  # 仅 rank0 有 logits
        if rank == 0:
            acc = ((logits.argmax(-1) == labels) & mask).sum()
            total_acc += acc.item()
            total_mask += mask.sum().item()
    if rank == 0:
        return total_acc / max(total_mask, 1)
    return None


# ---------------- 主函数 ----------------
def main():
    # dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # ---------- 读取 config ----------
    cfg = load_config("config/config.json")

    # ---------- 数据 ----------
    train_ds = GeneticDataset("data/train.pt")
    val_ds   = GeneticDataset("data/val.pt")
    n_sites = train_ds.var_site.shape[1]
    n_cats   = int(train_ds.var_site.max().item()) + 1
    k        = train_ds.p_dis.shape[-1]
    pad_idx  = n_cats
    print0(f"[MODEL] {n_sites} variants in samples")
    print0(f"[MODEL] {n_cats} category")
    print0(f"[MODEL] {k}-D egien of samples")
    # ---------- 模型 ----------
    model = DistributedMamba(n_cats=n_cats, k=k,
                             d_model=cfg.model.d_model,
                             d_state=cfg.model.d_state,
                             d_conv=cfg.model.d_conv,
                             n_layers=cfg.model.n_layers)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # ---------- DataLoader ----------
    batch_size = cfg.train.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, world_size, pad_idx),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
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
    stop_tensor = torch.tensor(0, dtype=torch.long, device='cuda')  # 0 表示继续，1 表示停止

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}", disable=(rank != 0))
        for seq, feat, labels in iterator:
            seq, feat, labels = seq.cuda(), feat.cuda(), labels.cuda()
            loss, _ = model(seq, feat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0:
                epoch_loss += loss.item()
                iterator.set_postfix(loss=loss.item())

        # ---------- 日志 ----------
        if rank == 0:
            epoch_loss /= len(train_loader.dataset)
            print0(f"Epoch {epoch:03d} | epoch_loss = {epoch_loss}")
        # ---------- Mask 验证 ----------
        if epoch % val_interval == 0 or epoch == epochs:
            test_acc = mask_accuracy(model, train_loader,"mask_train", mask_ratio, pad_idx)
            val_acc = mask_accuracy(model, val_loader,"mask_val", mask_ratio, pad_idx)
            if rank == 0:
                print0(f"Epoch {epoch:03d} | train_mask_acc = {test_acc:.4f} | val_mask_acc = {val_acc:.4f}")
                if val_acc > best_acc + patience_delta:
                    best_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{save_dir}/best.pt")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print0("Early stop!")
                    stop_tensor.fill_(1)
            if stop_tensor.item():
                dist.barrier()
                break
        # ---------- 常规保存 ----------
        if epoch % cfg.train.save_interval == 0 and rank == 0:
            torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
    
    dist.broadcast(stop_tensor, src=0)
    if rank == 0:
        print0("Training finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()