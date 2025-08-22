# #!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2025/08/22 15:56:54
@Author  :   qmtang
@Version :   1.0
@Desc    :   None

deepspeed train.py

'''

import os
import json
import random
import torch
import deepspeed
import pandas as pd
import torch.nn.functional as F
import torch.distributed as torch_dist
from torch.utils.data import IterableDataset,DataLoader, get_worker_info

from src.model import EvoFill
from src.utils import load_config


# --------------------------------------------------
# 工具
# --------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class ChunkDataset(IterableDataset):
    """
    按需读取：每个 epoch 会遍历所有 chunk → 所有样本对 → 所有位点窗口
    """
    def __init__(self, chunks_dir, dismat_path, cfg):
        super().__init__()
        self.chunks_dir = chunks_dir
        self.chunk_files = sorted([os.path.join(chunks_dir, f)
                                   for f in os.listdir(chunks_dir) if f.endswith('.pt')])
        self.cfg = cfg
        self.k = cfg.data.k_mer
        self.n_var = cfg.model.n_var

        # 只读距离矩阵
        dismat = pd.read_csv(dismat_path, sep='\t', skiprows=[0], header=None, index_col=0)
        samples = [s.strip(' ') for s in dismat.index]
        dismat.index = dismat.columns = samples
        self.dismat = dismat

    def _iter_single_chunk(self, chunk_path):
        """给定 chunk，产生所有 (seq1, seq2, mask1, mask2, dist)"""
        chunk = torch.load(chunk_path)
        names = [s.strip(' ') for s in chunk['sample']]
        seqs  = chunk['seq']   # (B, N, k)
        masks = chunk['mask']  # (B, N, k)
        B, N = seqs.shape[0], seqs.shape[1]

        for i in range(B):
            for j in range(i + 1, B):
                dist = float(self.dismat.loc[names[i], names[j]])
                for pos in range(0, N - self.n_var + 1):
                    yield (
                        seqs[i, pos:pos + self.n_var, :].long(),
                        seqs[j, pos:pos + self.n_var, :].long(),
                        masks[i, pos:pos + self.n_var, :].float(),
                        masks[j, pos:pos + self.n_var, :].float(),
                        dist
                    )

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程：遍历所有 chunk
            chunk_ids = range(len(self.chunk_files))
        else:
            # 多进程：按 worker 切分 chunk
            per_worker = len(self.chunk_files) // worker_info.num_workers
            wid = worker_info.id
            start = wid * per_worker
            end = (wid + 1) * per_worker if wid != worker_info.num_workers - 1 \
                else len(self.chunk_files)
            chunk_ids = range(start, end)

        for cid in chunk_ids:
            yield from self._iter_single_chunk(self.chunk_files[cid])

# --------------------------------------------------
# 掩码函数
# --------------------------------------------------
def mask_tokens(mask, ratio):
    """
    mask : (n_var, k)  原始 0/1 指示变异位点
    ratio: 掩盖比例
    返回:
        masked : bool (n_var, k)  True 表示需要预测的位置
    """
    cand = (mask == 1).nonzero(as_tuple=False)          # (M,2)
    n_keep = max(1, int(len(cand) * (1-ratio)))
    idx = torch.randperm(len(cand))[:n_keep]
    keep = torch.zeros_like(mask, dtype=bool)
    keep[cand[idx][:,0], cand[idx][:,1]] = True
    return (~keep) & (mask.bool())     # True=需要预测


# --------------------------------------------------
# 测试 / 验证通用
# --------------------------------------------------
@torch.no_grad()
def run_test(engine, loader, cfg, mask=False):
    correct = total = 0
    for s1, s2, m1, m2, dist in loader:
        s1, s2, m1, m2, dist = [x.to(engine.device) for x in (s1, s2, m1, m2, dist)]
        logits1, logits2 = engine(s1, s2, m1, m2, dist)

        if mask:           # 只计算被掩盖位置的准确率
            mask1 = mask_tokens(m1[0], cfg.model.mask_ratio).unsqueeze(0)   # (1,n_var,k)
            mask2 = mask_tokens(m2[0], cfg.model.mask_ratio).unsqueeze(0)
        else:              # 全部 token
            mask1 = mask2 = None

        for logits, seq, msk in [(logits1, s1, mask1), (logits2, s2, mask2)]:
            if mask:
                idx = msk.view(-1)
                logits = logits.view(-1, cfg.model.vocab_size)[idx]
                seq    = seq.view(-1)[idx]
            else:
                logits = logits.view(-1, cfg.model.vocab_size)
                seq    = seq.view(-1)

            if logits.numel() == 0:
                continue
            pred = logits.argmax(-1)
            correct += (pred == seq).sum().item()
            total   += seq.numel()
    return correct / max(total, 1)


# --------------------------------------------------
# 主程序
# --------------------------------------------------
def main():
    cfg = load_config('config/config.json')
    ds_cfg = json.load(open('config/ds_config.json'))

    os.environ['TORCH_CUDA_ARCH_LIST']='90'
    set_seed(cfg.train.seed)

    model = EvoFill(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        k=cfg.data.k_mer
    )

    train_ds = ChunkDataset(cfg.data.train_chunks_dir,
                            cfg.data.train_dismat,
                            cfg)
    val_ds   = ChunkDataset(cfg.data.val_chunks_dir,
                            cfg.data.val_dismat,
                            cfg)

    # DeepSpeed 会自动切分 batch
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_cfg
    )
    
    # 根据单卡 batch 自动计算全局 batch
    world_size = torch_dist.get_world_size()
    ds_cfg["train_batch_size"] = (
        ds_cfg["train_micro_batch_size_per_gpu"] *
        ds_cfg["gradient_accumulation_steps"] *
        world_size
    )

    train_loader = DataLoader(train_ds,
                          batch_size=engine.train_micro_batch_size_per_gpu(),
                          num_workers=4,
                          pin_memory=True)
    test_loader = DataLoader(train_ds,
                            batch_size=engine.train_micro_batch_size_per_gpu(),
                            num_workers=4,
                            shuffle=False)
    val_loader = DataLoader(val_ds,
                          batch_size=engine.train_micro_batch_size_per_gpu(),
                          num_workers=4,
                          pin_memory=True)
    global_step = 0

    # --------------------------------------------------
    # 训练
    # --------------------------------------------------
    for epoch in range(cfg.train.epochs):
        # -------- train --------
        model.train()
        for s1, s2, m1, m2, dist in train_loader:
            s1, s2, m1, m2, dist = [x.to(engine.device) for x in (s1, s2, m1, m2, dist)]
            logits1, logits2 = engine(s1, s2, m1, m2, dist)
            # 所有 token 都参与 loss
            loss = F.cross_entropy(logits1.view(-1, cfg.model.vocab_size), s1.view(-1)) + \
                   F.cross_entropy(logits2.view(-1, cfg.model.vocab_size), s2.view(-1))
            engine.backward(loss)
            engine.step()
            global_step += 1
            if global_step % cfg.train.log_every == 0:
                print(f'Epoch {epoch} step {global_step} loss {loss.item():.4f}')

        # -------- test (train set)  --------
        model.eval()
        acc = run_test(engine, test_loader, cfg, mask=True)
        print(f'Epoch {epoch} train-set mask acc {acc:.4f}')

        # -------- val --------
        acc = run_test(engine, val_loader, cfg, mask=True)
        print(f'Epoch {epoch} val-set mask acc {acc:.4f}')

        # 保存
        if (epoch + 1) % cfg.train.save_every == 0:
            engine.save_checkpoint('ckpts', tag=f'ep{epoch}')


# --------------------------------------------------
if __name__ == '__main__':
    main()