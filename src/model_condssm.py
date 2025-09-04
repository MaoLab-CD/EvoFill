"""
超长序列缺失填充（Mamba2 + 权重偏移 +多卡串联）
OMP_NUM_THREADS=8 python -m torch.distributed.run --nproc_per_node=2 model_condssm.py
"""

import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from mamba2_cond import Mamba2

# ---------- 1. 配置 ----------
B = 1                # batch_size
N = 1_000_000        # 序列长度
n_cats = 15          # 类别数
k = 4                # 样本特征维
PAD = n_cats         # 缺失 token

# 并行参数
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", device_id=torch.cuda.current_device())
world_size = dist.get_world_size()
rank = dist.get_rank()

chunk_len = N // world_size         # 每张卡处理的 token 数
assert N % world_size == 0

# ---------- 2. 模型 ----------
class PerDeviceMamba(nn.Module):
    def __init__(self,
                 n_cats=10,
                 k=4,
                 d_model=512,
                 d_state=64,
                 d_conv=4,
                 n_layers=4,
                 **mamba_kwargs):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.n_layers = n_layers

        # 1. token embedding
        self.embed = nn.Embedding(num_embeddings=n_cats + 1,  # [0, ...,n_cats-1, n_cats(missing tag)]
                                  embedding_dim=d_model)

        # 3. Mamba2 层
        self.layers = nn.ModuleList([
            Mamba2(k=k,
                   d_model=d_model,
                   d_state=d_state,
                   d_conv=d_conv,
                   **mamba_kwargs)
            for _ in range(n_layers)
        ])

        # 4. 输出头
        self.head = nn.Linear(d_model, n_cats)                # [0, ..., n_cats-1]

    def forward(self, x_chunk, feat, h_prev):
        """
        x_chunk : (B, L)  int64
        feat    : (B, k)
        h_prev  : (B, d_model)  来自上一 chunk 的输出残差
        return:
            logits : (B, L, n_cats)
            h_next : (B, d_model)
        """
        # B, L = x_chunk.shape
        # device = x_chunk.device

        # ---- 1. token 嵌入
        tok = self.embed(x_chunk)                 # (B, L, d_model)

        # ---- 2. 把上一 chunk 的残差加到第 0 个 token
        tok[:, 0] = tok[:, 0] + h_prev

        # ---- 4. 逐层前向
        x = tok
        for layer in self.layers:
            # 关键：把 feat 透传进去
            x = layer(x, feat=feat)   # 其余关键字保持默认

        # ---- 5. 输出
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
        ).cuda(local_rank)
        self.n_cats = n_cats
        self.k = k

    def forward(self, seq, feat):
        B, N = seq.shape
        chunk_len = N // dist.get_world_size()

        my_seq = seq[:, rank*chunk_len : (rank+1)*chunk_len].cuda(local_rank)
        feat = feat.cuda(local_rank)

        h_prev = torch.zeros(B, self.subnet.d_model, device=local_rank)
        logits, h_next = self.subnet(my_seq, feat, h_prev)

        # 状态传递：先同步再返回
        if rank < dist.get_world_size() - 1:
            dist.send(h_next, dst=rank+1)
        if rank > 0:
            dist.recv(h_prev, src=rank-1)

        # 收集 logits
        if rank == 0:
            full_logits = torch.empty(B, N, self.n_cats, dtype=torch.float32, device=local_rank)
            full_logits[:, :chunk_len] = logits
            for src in range(1, dist.get_world_size()):
                recv_buf = torch.empty(B, chunk_len, self.n_cats, dtype=torch.float32, device=local_rank)
                dist.recv(recv_buf, src=src)
                full_logits[:, src*chunk_len:(src+1)*chunk_len] = recv_buf
            return full_logits
        else:
            dist.send(logits, dst=0)
            # 关键：保持 logits 存活到通信完成
            return None


# ---------- 3. 测试 ----------
if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
    torch.manual_seed(42)
    # 随机输入
    seq = torch.randint(0, n_cats+1, (B, N), dtype=torch.int64)
    # 随机缺失
    mask = torch.rand(B, N) < 0.30
    seq[mask] = PAD

    feat = torch.randn(B, k)

    model = DistributedMamba(n_cats=n_cats, k=k, 
                             d_model=512,   d_state=128,
                             d_conv=4,      n_layers=6)
    model.eval()

    with torch.no_grad():
        full_logits = model(seq, feat)

    dist.barrier()

    if rank == 0:
        print("输入序列 shape :", seq.shape)         # (1, 1_000_000)
        print("类别数   :", model.n_cats)            # (15)
        print("特征维度 :", model.k)                 # (4)
        print("样本特征 shape :", feat.shape)        # (1, 4)
        print("输出 logits    :", full_logits.shape)   # (1, 1_000_000, 15)

    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[Rank {rank}] 显存使用: {max_memory:,.1f} MB")

    dist.destroy_process_group()