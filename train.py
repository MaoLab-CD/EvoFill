import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.utils import load_config
from src.model import EvoFill
from src.losses import ImputationLoss


# def print0(*args, **kwargs):
#     if dist.get_rank() == 0:
#         print(*args, **kwargs)


class GenotypeDataset(Dataset):
    def __init__(self, gts, coords, mask_ratio=0.0):
        self.gt_true = gts.long()          # 原始完整标签
        self.coords = coords.float()
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.gt_true.shape[0]

    def __getitem__(self, idx):
        gt_true = self.gt_true[idx]        # 完整标签
        coords = self.coords                # (L, 4)

        # 训练时额外随机遮掩
        gt_mask = gt_true.clone()
        if self.mask_ratio > 0:
            mask = torch.rand_like(gt_mask.float()) < self.mask_ratio
            gt_mask[mask] = -1             # 仅输入被遮掩

        # 返回：输入（含缺失）、原始标签、坐标
        return gt_mask, gt_true, coords 

def collate_fn(batch):
    """
    batch: List[(gt_mask, gt_true, coords)] 每个 coords 形状相同
    返回：gt_mask, gt_true, coords（二维，直接取第 0 个即可
    """
    gt_mask  = torch.stack([b[0] for b in batch], 0)
    gt_true  = torch.stack([b[1] for b in batch], 0)
    coords   = batch[0][2]          # 全局共享
    return gt_mask, gt_true, coords

def build_loader(pt_path, batch_size, shuffle, mask_ratio):
    data = torch.load(pt_path)
    dataset = GenotypeDataset(data['gts'], data['coords'], mask_ratio=mask_ratio)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_fn,
    )

def imputation_accuracy(logits, gts, mask):
    """仅在被 mask 位点计算 accuracy"""
    preds = torch.argmax(logits, dim=-1)  # (B, L)
    correct = (preds == gts) & mask
    return correct.sum().float() / mask.sum().float()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_mask = 0.0, 0.0, 0

    shared_params = list(model.length_proj.parameters()) + list(model.out_conv.parameters())
    assert len(shared_params) > 0
    assert all(p.requires_grad for p in shared_params)

    pbar = tqdm(loader, leave=False)
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), gt_true.to(device), \
                                               coords.to(device)
        
        logits = model(gt_mask, coords)  # (B, L, n_cats)
        loss = criterion(logits, gt_true) 

        if criterion.use_gn:
            criterion.gn_loss.gradnorm_step(shared_params, retain_graph=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy：只算被 mask 的位点
        mask = gt_mask == -1
        acc = imputation_accuracy(logits, gt_true, mask)
        total_loss += loss.item()
        total_acc += acc.item() * mask.sum().item()
        total_mask += mask.sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")

    return total_loss / len(loader), total_acc / total_mask


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_mask = 0.0, 0.0, 0

    pbar = tqdm(loader, leave=False, desc='validate')
    for gt_mask, gt_true, coords in pbar:
        gt_mask, gt_true, coords = gt_mask.to(device), \
                                     gt_true.to(device), \
                                     coords.to(device)

        logits = model(gt_mask, coords)          # (B, L, n_cats)
        loss   = criterion(logits, gt_true)      # 计算与真值差异

        # 只统计被 mask 的位点
        mask = gt_mask == -1
        acc  = imputation_accuracy(logits, gt_true, mask)

        total_loss += loss.item()
        total_acc  += acc.item() * mask.sum().item()
        total_mask += mask.sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")

    return total_loss / len(loader), total_acc / (total_mask + 1e-8)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        assert mode in {'min', 'max'}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.best_state = None

    def __call__(self, metric, model):
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        better = (metric < self.best - self.min_delta) if self.mode == 'min' else \
                 (metric > self.best + self.min_delta)

        if better:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def main():
    cfg = load_config("config/config.json")
    device = torch.device(cfg.train.device)
    torch.manual_seed(42)

    # 数据
    train_loader = build_loader(
        Path(cfg.data.path) / "train.pt",
        batch_size=cfg.train.batch_size,
        shuffle=True,
        mask_ratio=cfg.train.mask_ratio,
    )
    val_loader = build_loader(
        Path(cfg.data.path) / "val.pt",
        batch_size=cfg.train.batch_size,
        shuffle=False,
        mask_ratio=cfg.train.mask_ratio,
    )

    # 模型 & 优化器
    model = EvoFill(**vars(cfg.model)).to(device)

    criterion = ImputationLoss(use_r2_loss=True,
                           use_grad_norm=True,
                           gn_alpha=1.5,
                           gn_lr_w=cfg.train.lr/10).to(device) #权重学习率，比模型 lr 小 1~2 量级

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    early_stopper = EarlyStopper(patience=cfg.train.patience,
                                 min_delta=cfg.train.min_delta,
                                 mode='min')

    # 训练循环
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        if early_stopper(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 保存最优模型
    save_dir = Path(cfg.train.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(early_stopper.best_state, save_dir / "evofill_best.pt")
    print(f"Best model saved to {save_dir / 'evofill_best.pt'} (epoch {epoch - early_stopper.counter})")

if __name__ == "__main__":
    main()