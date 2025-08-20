#!/usr/bin/env python
# deepspeed train.py --deepspeed config/ds_config.json
import os, json, math, torch, deepspeed, torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from src.model import EvoFill
from src.utils import load_config

# ---------- 数据集 -----------
class ChunkDataset(Dataset):
    def __init__(self, root_dir: str):
        import glob
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No *.pt found in {root_dir}")
    def __len__(self):   return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        return d["seq"].long(), d["mask"].bool(), d["label"].long()

# ---------- 单 epoch -----------
def run_one_epoch(model, engine, loader, is_train, mask_center_prob: float):
    model.train(is_train)
    tot = corr = loss_sum = 0
    for seq, mask, label in loader:
        seq, mask, label = (x.to(engine.device, non_blocking=True) for x in (seq, mask, label))
        B, N, K = seq.shape

        # 随机遮盖
        if mask_center_prob > 0:
            flat = mask.view(-1)
            idx  = flat.nonzero(as_tuple=False).squeeze(-1)
            n_mask = int(idx.numel() * mask_center_prob)
            if n_mask:
                flat[idx[torch.randperm(idx.size(0), device=seq.device)[:n_mask]]] = 0
                mask = flat.view(B, N, K)

        tgt = seq.masked_fill(~mask, 0)
        logits = model(tgt)                       # (B, N, K, V)
        logits_m = logits[mask]                   # (M, V)
        label_m  = label.view(-1).repeat_interleave(K)[mask.view(-1)]

        loss = torch.nn.functional.cross_entropy(logits_m, label_m, reduction='sum')
        if is_train:
            engine.backward(loss)
            engine.step()

        with torch.no_grad():
            loss_sum += loss.item()
            corr  += (logits_m.argmax(-1) == label_m).sum().item()
            tot   += label_m.numel()
    return loss_sum/max(tot,1), corr/max(tot,1)

# ---------- 主入口 -----------
def main():
    cfg = load_config("config/config.json")
    with open(cfg.deepspeed_config) as f:
        ds_cfg = json.load(f)

    # 根据实际 GPU 数量自动调整梯度累计步数
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    desired_global_batch = ds_cfg.get("global_batch_size", 256)   # 全局 batch 目标
    micro_per_gpu        = ds_cfg.get("micro_batch_per_gpu", 4)   # 每张卡 micro-batch
    # 计算累计步数 = ceil( 目标global / (卡数*micro) )
    grad_acc_steps = max(1, math.ceil(desired_global_batch / (world_size * micro_per_gpu)))
    ds_cfg["gradient_accumulation_steps"] = grad_acc_steps
    ds_cfg["train_micro_batch_size_per_gpu"] = micro_per_gpu
    if dist.get_rank() == 0:
        print(f"[INFO] world_size={world_size}, "
              f"micro_batch_per_gpu={micro_per_gpu}, "
              f"gradient_accumulation_steps={grad_acc_steps}, "
              f"=> global_batch={world_size*micro_per_gpu*grad_acc_steps}")

    model = EvoFill(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        k=cfg.k_mer
    )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg
    )

    train_ds = ChunkDataset(cfg.train_chunks_dir)
    val_ds   = ChunkDataset(cfg.val_chunks_dir)
    # 注意：batch_size=None，因为 DeepSpeed 的 DataLoader 自己按 micro-batch 分发
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=None, pin_memory=True)

    os.makedirs(cfg.save_dir, exist_ok=True)
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = run_one_epoch(model, engine, train_loader, True, 0.0)
        tr_loss_m, tr_acc_m = run_one_epoch(model, engine, train_loader, True, cfg.mask_ratio)
        val_loss, val_acc   = run_one_epoch(model, engine, val_loader,   False, cfg.mask_ratio)
        if engine.local_rank == 0:
            print(f"Epoch {epoch:02d}: "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                  f"tr_loss_m={tr_loss_m:.4f} tr_acc_m={tr_acc_m:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if (epoch+1)%5==0 or epoch+1==cfg.epochs:
                engine.save_checkpoint(cfg.save_dir, tag=f"ep{epoch}")

if __name__ == "__main__":
    main()