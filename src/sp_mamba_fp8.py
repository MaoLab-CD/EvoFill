#!/usr/bin/env python3
"""
Sequence-Parallel FP8 SSM on 2×H100
=> 1 M tokens, batch_size=1
2×H100 FP8 序列并行
输入: (1, 1_000_000)  int8
输出: (1, 1_000_000, total_cats)

export PYTHONPATH=/mnt/qmtang/flash-attention/hopper/:$PYTHONPATH
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 sp_mamba_fp8.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    FP8_OK = True
except ImportError:
    FP8_OK = False


TOTAL_CATS = 15               # 0..9, 缺失类=10 不会出现在 softmax 维度
SEQ_LEN    = 1000000
BATCH      = 1
D_MODEL    = 512
N_LAYER    = 8

# ---------- 分布式 ----------
def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.cuda.current_device())
    return dist.get_rank(), dist.get_world_size()

# ---------- 极简 SSM ----------
class S4DLayer(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(d_model))
        self.B = torch.nn.Parameter(torch.randn(d_model))
        self.C = torch.nn.Parameter(torch.randn(d_model))
        self.D = torch.nn.Parameter(torch.randn(d_model))
    def forward(self, x, carry=None):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype) if carry is None else carry
        outs = []
        for t in range(L):
            h = h * torch.exp(self.A) + self.B * x[:, t]
            y = self.C * h + self.D * x[:, t]
            outs.append(y.unsqueeze(1))
        return torch.cat(outs, dim=1), h   # (B,L,D)

# ---------- 网络 ----------
class Net(torch.nn.Module):
    def __init__(self, total_cats, d_model, n_layer):
        super().__init__()
        self.embed = torch.nn.Embedding(total_cats+1, d_model)   # 含缺失类
        self.layers = torch.nn.ModuleList([S4DLayer(d_model) for _ in range(n_layer)])
        self.head = torch.nn.Linear(d_model, total_cats, bias=False)  # 不含缺失类
    def forward(self, x, carry=None):
        # x: (B, L_local) int8
        h = self.embed(x.long())  # -> fp8
        for lyr in self.layers:
            h, carry = checkpoint(lyr, h, carry, use_reentrant=False)
        logits = self.head(h)     # (B, L_local, total_cats)
        return logits


# ---------- 主函数 ----------
def main():
    rank, world_size = init_dist()
    assert world_size == 2
    L_local = SEQ_LEN // world_size
    start = rank * L_local
    # print(f"[rank{rank}] NCCL init done", flush=True)

    # 1. 随机 int8 数据

    ids = torch.randint(0, TOTAL_CATS + 1, (BATCH, SEQ_LEN),
                        dtype=torch.int8, device='cuda')

    dist.broadcast(ids, 0)

    ids_local = ids[:, start:start+L_local]

    # 2. 建模型（FP8 权重初始化）

    if FP8_OK:
        from transformer_engine.pytorch.fp8 import fp8_model_init
        with fp8_model_init(enabled=True):
            model = Net(TOTAL_CATS, D_MODEL, N_LAYER).cuda()
        recipe = DelayedScaling(fp8_format=Format.E4M3)
    else:
        model = Net(TOTAL_CATS, D_MODEL, N_LAYER).cuda()
        recipe = None

    model = DDP(model, device_ids=[rank])

    # 3. 前向 + 显存打印
    with te.fp8_autocast(enabled=FP8_OK,
                         fp8_recipe=recipe):
        logits = model(ids_local)
    # with te.fp8_autocast(enabled=False):
    #     logits = model(ids_local)
    print(f"[rank{rank}] ids_local: {ids_local.shape}, dtype={ids_local.dtype}")
    print(f"[rank{rank}] logits:    {logits.shape}, dtype={logits.dtype}")
    
    # 4. 虚拟 loss 与反向
    loss = logits.sum()   # dummy loss
    loss.backward()

    print(f"[rank{rank}] Peak mem:  {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()