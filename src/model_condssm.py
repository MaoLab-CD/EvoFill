"""
超长序列缺失填充（Mamba2 + 权重偏移 +多卡串联）
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 src/model_condssm.py
"""

import os
import torch
import torch.distributed as dist
from torch import nn
from src.mamba2_cond import Mamba2
from torch.nn.parallel import DistributedDataParallel as DDP

# 并行参数
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", device_id=torch.cuda.current_device())
world_size = dist.get_world_size()
rank = dist.get_rank()

# ---------- 2. 模型 ----------
class PerDeviceMamba(nn.Module):
    def __init__(self, n_cats=10, k=4, d_model=512, d_state=64, d_conv=4, n_layers=4, **mamba_kwargs):
        super().__init__()
        self.embed = nn.Embedding(n_cats + 1, d_model)
        self.layers = nn.ModuleList([
            Mamba2(k=k, d_model=d_model, d_state=d_state, d_conv=d_conv, **mamba_kwargs)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_cats)

    def forward(self, x_chunk, feat, h_prev=None):
        """
        x_chunk : (B, L)  int64
        feat    : (B, k)
        h_prev  : (B, d_model)  仅用于推理
        return:
            logits : (B, L, n_cats)
            h_next : (B, d_model)
        """
        B, L = x_chunk.shape
        tok = self.embed(x_chunk)
        if h_prev is not None:
            tok[:, 0] = tok[:, 0] + h_prev
        x = tok
        for layer in self.layers:
            x = layer(x, feat=feat)
        logits = self.head(x)
        h_next = x[:, -1]
        return logits, h_next


class DistributedMamba(nn.Module):
    """
    包装器：负责把序列切到不同 GPU, 并循环传递隐藏状态。
    超参全部在这里暴露，方便一键调参。
    """
    def __init__(self,
                 n_cats=10,
                 k=4,
                 d_model=512,
                 d_state=64,
                 d_conv=4,
                 n_layers=4,
                 **mamba_kwargs):
        """
        mamba_kwargs 留给未来扩展（如 expand, headdim 等）
        """
        super().__init__()
        self.subnet = PerDeviceMamba(
            n_cats=n_cats,
            k=k,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            n_layers=n_layers,
            **mamba_kwargs
        )
        # 用 DDP 包装，梯度会自动 all-reduce
        self.subnet = DDP(self.subnet.cuda(local_rank),
                          device_ids=[local_rank],
                          output_device=local_rank,
                          find_unused_parameters=True)
        self.n_cats = n_cats
        self.ignore_index = n_cats   # PAD token 不参与 loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='sum')

    def forward(self, seq, feat, labels=None):
        """
        seq    : (B, N)  int64
        feat   : (B, k)  float
        labels : (B, N)  int64  (与 seq 对齐，缺失位置=PAD)
        return:
            loss   : scalar (已跨卡平均)
            logits : (B, N, n_cats) 仅在 rank0 返回
        """
        B, N = seq.shape
        chunk_len = N // world_size
        my_seq   = seq[:, rank*chunk_len:(rank+1)*chunk_len].cuda(local_rank)
        my_feat  = feat.cuda(local_rank)
        logits, _ = self.subnet(my_seq, my_feat)   # (B, L, n_cats)

        # 收集全局 logits 到 rank0（用于评估/生成）
        if rank == 0:
            full_logits = torch.empty(B, N, self.n_cats, dtype=logits.dtype, device=logits.device)
            full_logits[:, :chunk_len] = logits
            for src in range(1, world_size):
                recv_buf = torch.empty_like(logits)
                dist.recv(recv_buf, src=src)
                full_logits[:, src*chunk_len:(src+1)*chunk_len] = recv_buf
        else:
            dist.send(logits, dst=0)
            full_logits = None

        # 计算损失
        if labels is not None:
            my_labels = labels[:, rank*chunk_len:(rank+1)*chunk_len].cuda(local_rank)
            loss = self.loss_fn(logits.view(-1, self.n_cats), my_labels.view(-1))
            
            loss = loss / B
        else:
            loss = None

        return loss, full_logits


# ---------- 3. 测试 ----------
if __name__ == "__main__":
    # ---------- 配置 ----------
    B = 1                # batch_size
    N = 1_000_000        # 序列长度
    n_cats = 15          # 类别数
    k = 4                # 样本特征维
    PAD = n_cats         # 缺失 token

    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
    torch.manual_seed(42)
    # 构造数据
    seq   = torch.randint(0, n_cats, (B, N))          # 无 PAD
    mask  = torch.rand(B, N) < 0.30
    seq[mask] = PAD
    labels = seq.clone()                              # 与输入对齐
    feat  = torch.randn(B, k)

    feat = torch.randn(B, k)

    model = DistributedMamba(n_cats=n_cats, k=k, 
                             d_model=512,   d_state=128,
                             d_conv=4,      n_layers=6)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 前向 + 反向
    loss, full_logits = model(seq, feat, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dist.barrier()
    if rank == 0:
        print(f"Loss: {loss.item():.4f}")
        print("输入序列 shape :", seq.shape)         # (1, 1_000_000)
        print("样本特征 shape :", feat.shape)        # (1, 4)
        print("输出 logits    :", full_logits.shape)   # (1, 1_000_000, 15)

    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[Rank {rank}] 显存使用: {max_memory:,.1f} MB")

    dist.destroy_process_group()