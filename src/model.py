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
        n_chunks: int,
        d_model: int = 256,
        n_layers: int = 4,
        chunk_overlap_ratio: float = 0.1,
        **mamba_kwargs,
    ):
        super().__init__()
        self.n_cats = n_cats
        self.n_chunks = n_chunks
        self.chunk_overlap_ratio = chunk_overlap_ratio

        self.embed = CatEmbeddings(n_cats, d_model)
        self.chunk_modules = nn.ModuleList([
            ChunkModule(d_model, n_layers, **mamba_kwargs)
            for _ in range(n_chunks)
        ])
        self.length_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv1d(d_model, n_cats, kernel_size=1)

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        B, L_orig = x.shape

        # 1. 动态计算 chunk_size
        chunk_size = math.ceil(L_orig / self.n_chunks)
        overlap = int(chunk_size * self.chunk_overlap_ratio)
        step = chunk_size - overlap

        # 2. 补 -1 到能被 step 整除
        pad_len = (step - L_orig % step) % step
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=-1)
            x_coord = F.pad(x_coord, (0, 0, 0, pad_len), value=0.0)
        L_pad = x.shape[-1]

        # 3. 嵌入
        h = self.embed(x, x_coord)  # (B, L_pad, d_model)

        # 4. 滑窗切 chunk -> 独立模块
        outs = []
        start = 0
        for i in range(self.n_chunks):
            end = min(start + chunk_size, L_pad)
            chunk = h[:, start:end, :]
            outs.append(self.chunk_modules[i](chunk))  # (B, chunk_len, d)
            if end == L_pad:
                break
            start += step

        # 5. concat (axis=-2) -> 投影回原始长度
        h_seq = torch.cat(outs, dim=-2)  # (B, total_len, d)
        h_seq = self.length_proj(h_seq.transpose(1, 2))  # (B, d, total_len)
        h_seq = F.interpolate(h_seq, size=L_orig, mode='linear', align_corners=False)  # (B, d, L_orig)

        # 6. 输出 logits
        logits = self.out_conv(h_seq).transpose(1, 2)  # (B, L_orig, n_cats)
        return logits

# unit test
if __name__ == "__main__":
    model = EvoFill(n_cats=4, n_chunks=4, d_model=256, n_layers=4,
                    chunk_size=512, chunk_overlap_ratio=0.1,
                    d_state=128, expand=2).cuda()

    x       = torch.randint(-1, 4, (2, 1800)).cuda()
    x_coord = torch.randn(1800, 4).cuda()
    logits  = model(x, x_coord)  # (2, 4*512, 4)