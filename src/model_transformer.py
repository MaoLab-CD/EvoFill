# model_transformer.py
import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder


# ---------- 1. 嵌入层（完全复用） ----------
class CatEmbeddings(nn.Module):
    """
    等位基因 + 坐标 嵌入
    -1 -> padding_idx (n_cats) -> 零向量
    """
    def __init__(self, n_cats: int, d_model: int, coord_dim: int = 4):
        super().__init__()
        self.allele_embed = nn.Embedding(n_cats + 1, d_model, padding_idx=n_cats)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        x = x.masked_fill(x == -1, self.allele_embed.padding_idx)
        e1 = self.allele_embed(x)                       # (B,L,d)
        e2 = self.coord_proj(x_coord).unsqueeze(0)      # (1,L,d)
        return self.norm(e1 + e2)


# ---------- 2. 双向 Transformer Block ----------
class BiTransformerBlock(nn.Module):
    """
    用标准 TransformerEncoderLayer 实现双向 Transformer。
    注意：我们不需要额外位置编码，因为坐标已经在 CatEmbeddings 里注入。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,               # 头数
        dim_feedforward: int = 1024,  # FFN 隐层
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        **kwargs  # 吸收其它 mamba 参数
    ):
        super().__init__()
        # 只取一层 EncoderLayer，不加 causal mask 即为双向
        self.layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,   # (B, L, d) 格式
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    (B, L, d)
        mask: (B, L)  0 表示 pad，1 表示有效；Transformer 用 key_padding_mask
        """
        # TransformerEncoderLayer 的 key_padding_mask 需要 (B, L) 的 BoolTensor
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None
        out = self.layer(x, src_key_padding_mask=key_padding_mask)
        return out


# ---------- 3. ChunkModule（只换 block） ----------
class ChunkModule(nn.Module):
    def __init__(self, d_model: int, n_layers: int, **trans_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiTransformerBlock(d_model=d_model, **trans_kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x_chunk):
        for blk in self.blocks:
            x_chunk = blk(x_chunk)
        return x_chunk


# ---------- 4. EvoFill（完全复用） ----------
class EvoFill(nn.Module):
    def __init__(
        self,
        n_cats: int,
        chunk_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        chunk_overlap: int = 64,
        **trans_kwargs,  # 传给 Transformer
    ):
        super().__init__()
        self.n_cats = n_cats
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed = CatEmbeddings(n_cats, d_model)
        self.chunk_module = ChunkModule(d_model, n_layers, **trans_kwargs)

        self.length_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv1d(d_model, n_cats, kernel_size=1)

    # 以下 forward 与 mamba 版本完全一致
    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        B, L_orig = x.shape
        device = x.device

        h = self.embed(x, x_coord)                      # (B, L_orig, d)

        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        step = chunk_size - overlap
        n_chunks = math.ceil((L_orig - overlap) / step)

        pad_len = n_chunks * step + overlap - L_orig
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))            # (B, L_pad, d)
        L_pad = h.shape[1]

        out_buf = torch.zeros(B, L_pad, h.shape[-1], device=device)
        count_buf = torch.zeros(B, L_pad, dtype=torch.long, device=device)

        for i in range(n_chunks):
            start = i * step
            end = start + chunk_size
            chunk = h[:, start:end, :]                  # (B, chunk_size, d)
            # 构造一个简单 mask：pad 位置为 0
            mask = torch.ones(B, chunk_size, dtype=torch.long, device=device)
            chunk_out = self.chunk_module(chunk)        # (B, chunk_size, d)

            out_buf[:, start:end, :] += chunk_out
            count_buf[:, start:end] += 1

        out_buf = out_buf / count_buf.unsqueeze(-1).clamp_min(1)

        out = self.length_proj(out_buf.transpose(1, 2))  # (B, d, L_pad)
        out = F.interpolate(out, size=L_orig, mode='linear', align_corners=False)

        logits = self.out_conv(out).transpose(1, 2)      # (B, L_orig, n_cats)
        return logits


# ---------- 5. 快速单元测试 ----------
if __name__ == "__main__":
    model = EvoFill(
        n_cats=4,
        chunk_size=512,
        chunk_overlap=64,
        d_model=256,
        n_layers=4,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
    ).cuda()

    x = torch.randint(-1, 4, (2, 1800)).cuda()
    x_coord = torch.randn(1800, 4).cuda()
    logits = model(x, x_coord)
    print(logits.shape)  # torch.Size([2, 1800, 4])