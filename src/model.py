# model.py
import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from mamba_ssm import Mamba2  # 官方实现



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
        """
        x:       (B, L)        long，-1 会被当成 padding_idx n_cats
        x_coord: (L, 4)        float
        return:  (B, L, d_model)
        """
        x = x.masked_fill(x == -1, self.allele_embed.padding_idx)
        e1 = self.allele_embed(x)                       # (B,L,d)
        e2 = self.coord_proj(x_coord).unsqueeze(0)      # (1,L,d)
        return self.norm(e1 + e2)

class BiMamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: str = "add",      # "add" | "ew_multiply"
        bidirectional_weight_tie: bool = True,
        # ---- 以下透传给 Mamba2 ----
        d_state: int = 128,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        headdim: int = 64,
        ngroups: int = 1,
        **mamba2_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy not in {"add", "ew_multiply"}:
            raise NotImplementedError(bidirectional_strategy)

        self.bidirectional = bidirectional
        self.strategy = bidirectional_strategy

        # 前向 SSM
        self.mamba_fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            conv_bias=conv_bias,
            bias=bias,
            headdim=headdim,
            ngroups=ngroups,
            **mamba2_kwargs,
        )

        if bidirectional:
            self.mamba_rev = Mamba2(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                conv_bias=conv_bias,
                bias=bias,
                headdim=headdim,
                ngroups=ngroups,
                **mamba2_kwargs,
            )
            if bidirectional_weight_tie:
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias   = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias   = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        out = self.mamba_fwd(x)
        if self.bidirectional:
            x_rev = x.flip(dims=[1])
            out_rev = self.mamba_rev(x_rev).flip(dims=[1])
            if self.strategy == "add":
                out = out + out_rev
            elif self.strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise RuntimeError(self.strategy)
        return out


class ChunkModule(nn.Module):
    def __init__(self, d_model: int, n_layers: int, **mamba_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiMamba2Block(d_model=d_model, **mamba_kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x_chunk):
        for blk in self.blocks:
            x_chunk = blk(x_chunk)
        return x_chunk

class EvoFill(nn.Module):
    def __init__(
        self,
        n_cats: int,
        chunk_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        chunk_overlap: int = 64,          # 直接指定 overlap 长度
        **mamba_kwargs,
    ):
        super().__init__()
        self.n_cats = n_cats
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed = CatEmbeddings(n_cats, d_model)
        # 只实例化一个 ChunkModule，所有 chunk 共享权重
        self.chunk_module = ChunkModule(d_model, n_layers, **mamba_kwargs)

        self.length_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv1d(d_model, n_cats, kernel_size=1)

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        """
        x:       (B, L)        long，-1 表示 padding
        x_coord: (L, 4)        float
        return:  (B, L, n_cats)
        """
        B, L_orig = x.shape
        device = x.device

        # 1. 嵌入
        h = self.embed(x, x_coord)                      # (B, L_orig, d)

        # 2. 滑窗切分
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        step = chunk_size - overlap
        n_chunks = math.ceil((L_orig - overlap) / step)

        # 3. 需要 pad 到能整除 step
        pad_len = n_chunks * step + overlap - L_orig
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))            # (B, L_pad, d)
        L_pad = h.shape[1]

        # 4. 收集每个 chunk 的输出，同时记录每个 token 被哪些 chunk 覆盖
        out_buf = torch.zeros(B, L_pad, h.shape[-1], device=device)
        count_buf = torch.zeros(B, L_pad, dtype=torch.long, device=device)

        for i in range(n_chunks):
            start = i * step
            end = start + chunk_size
            chunk = h[:, start:end, :]                  # (B, chunk_size, d)
            chunk_out = self.chunk_module(chunk)        # (B, chunk_size, d)

            # 累加重叠区域
            out_buf[:, start:end, :] += chunk_out
            count_buf[:, start:end] += 1

        # 5. 重叠区域取平均
        out_buf = out_buf / count_buf.unsqueeze(-1).clamp_min(1)

        # 6. 1D 卷积 + 插值回原始长度
        out = self.length_proj(out_buf.transpose(1, 2))  # (B, d, L_pad)
        out = F.interpolate(out, size=L_orig, mode='linear', align_corners=False)

        # 7. 输出 logits
        logits = self.out_conv(out).transpose(1, 2)      # (B, L_orig, n_cats)
        return logits


# unit test
if __name__ == "__main__":
    model = EvoFill(
        n_cats=2,
        chunk_size=512,
        chunk_overlap=64,
        d_model=256,
        n_layers=4,
        d_state=128,
        expand=2,
    ).cuda()

    x = torch.randint(-1, 4, (2, 1800)).cuda()
    x_coord = torch.randn(1800, 4).cuda()
    logits = model(x, x_coord)
    print(logits.shape)  # torch.Size([2, 1800, 4])