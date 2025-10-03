# model.py
import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from mamba_ssm import Mamba2  # 官方实现


class GenoEmbedding(nn.Module):
    def __init__(self, n_alleles: int, d_model: int, coord_dim: int = 4):
        super().__init__()
        self.n_alleles = n_alleles
        # 1. 基础嵌入
        self.allele_embedding = nn.Embedding(n_alleles + 1, d_model)  # 含缺失行
        self.coord_proj = nn.Linear(coord_dim, d_model)

        # 2. 低秩双线性（factorized bilinear）
        self.bilinear_l = nn.Linear(d_model, d_model, bias=False)  # W1
        self.bilinear_r = nn.Linear(d_model, d_model, bias=False)  # W2
        self.bilinear_out = nn.Linear(d_model, d_model)            # 降维 + 残差
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        """
        x:       (B, L)          整型 {-1,0,...,n-1}
        x_coord: (L, 4)          float  共享坐标
        """
        # --- 1. 基础嵌入 ---
        x = x.clone()
        x[x == -1] = self.n_alleles
        e1 = self.allele_embedding(x)            # (B,L,d)
        e2 = self.coord_proj(x_coord)            # (L,d)
        e2 = e2.unsqueeze(0)                     # (1,L,d)

        # --- 2. 逐位双线性融合 ---
        l = self.bilinear_l(e1)                  # (B,L,d)
        r = self.bilinear_r(e2)                  # (1,L,d)
        z = l * r                                # 逐位乘 → (B,L,d)  二阶交互
        z = self.bilinear_out(z)                 # 线性降回 d

        # --- 3. 残差 + Norm ---
        out = self.norm(z + e1 + e2)             # 双残差，平滑训练
        return out


class EvoEmbedding(nn.Module):
    """
    以样本为 token 做交叉注意力，捕捉“样本×样本”的演化关系。
    输入：
        x:  (B, L, d)  已融合位点+坐标信息的表示
        dist: (B, B) 或 None  样本间距离矩阵，作为注意力偏置
    输出：
        (B, L, d)  与输入同形状，可残差连接
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        assert self.d_head * n_heads == d_model

        # 交叉注意力：Q 来自自身，K/V 来自整个 batch
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                dist: torch.Tensor | None = None):
        """
        x:    (B, L, d)
        dist: (B, B)  距离矩阵，无则填 None
        """
        B, L, d = x.shape
        h = x.mean(dim=1)          # 简单 pooling 成样本级向量 (B,d)
        h = self.norm(h)

        # 交叉注意力：以每个样本为 query，整个 batch 为 key/value
        q = self.q(h).view(B, self.n_heads, self.d_head)   # (B,H,dH)
        k = self.k(h).view(B, self.n_heads, self.d_head)   # (B,H,dH)
        v = self.v(h).view(B, self.n_heads, self.d_head)

        # (H,B,B)
        scores = torch.einsum("bhd,chd->hbc", q, k) / (self.d_head ** 0.5)

        # 距离偏置
        if dist is not None:
            # 将 dist 归一化到 0~1 并取反（越近权重越大）
            bias = 1. - (dist / (dist.max() + 1e-6))
            bias = bias.unsqueeze(0).expand(self.n_heads, -1, -1)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)          # (H,B,B)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.einsum("hbc,chd->bhd", attn, v)  # (B,H,dH)
        out = out.reshape(B, d)
        out = self.o(out)                           # (B,d)

        # 广播回 L 个位点，残差连接
        out = out.unsqueeze(1).expand(-1, L, -1)    # (B,L,d)
        return x + out


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
        n_alleles: int,          # 真实 allele 类别数（不含缺失）
        chunk_size: int,
        d_model: int = 256,
        n_evo_heads: int = 8,
        n_layers: int = 4,
        chunk_overlap: int = 64,
        **mamba_kwargs,
    ):
        super().__init__()
        self.n_alleles = n_alleles
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed = GenoEmbedding(n_alleles, d_model)
        self.evo_embed = EvoEmbedding(d_model, n_evo_heads)   # ← 新增
        self.chunk_module = ChunkModule(d_model, n_layers, **mamba_kwargs)
        self.out_proj = nn.Linear(d_model, n_alleles)   # ← 输出 n_alleles 类概率

    def forward(self, x: torch.Tensor, 
                      x_coord: torch.Tensor, 
                      dist: torch.Tensor | None = None):
        """
        x:       (B, L)    基因型整型 {-1,0,...,n-1}      
        x_coord: (L, 4)    坐标
        dist:    (B, B)    距离矩阵，可选
        return:  (B, L, n_alleles) 概率，已 softmax
        """
        B, L_orig = x.shape
        device = x.device

        # 1. 嵌入
        h = self.embed(x, x_coord)                      # (B,L,d)
        h = self.evo_embed(h, dist)         # ← 样本间交叉注意力

        # 2. 滑窗切分、重叠合并（与你原来代码完全一致，省略）
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        step = chunk_size - overlap
        n_chunks = math.ceil((L_orig - overlap) / step)
        pad_len = n_chunks * step + overlap - L_orig
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))            # (B,L_pad,d)
        L_pad = h.shape[1]

        out_buf = torch.zeros(B, L_pad, h.shape[-1], device=device)
        count_buf = torch.zeros(B, L_pad, dtype=torch.long, device=device)

        for i in range(n_chunks):
            start = i * step
            end = start + chunk_size
            chunk = h[:, start:end, :]
            chunk_out = self.chunk_module(chunk)
            out_buf[:, start:end, :] += chunk_out
            count_buf[:, start:end] += 1

        out_buf = out_buf / count_buf.unsqueeze(-1).clamp_min(1).float()
        out_buf = out_buf[:, :L_orig, :]              # 去掉 pad
        logits = self.out_proj(out_buf)               # (B,L,n_alleles)
        probs = logits.softmax(dim=-1)
        return probs

# # 只在 EvoFill 里动刀，其余类不动
# class EvoFill(nn.Module):
#     def __init__(
#         self,
#         n_alleles: int,
#         chunk_size: int,
#         d_model: int = 256,
#         n_evo_heads: int = 8,
#         n_layers: int = 4,
#         chunk_overlap: int = 64,
#         **mamba_kwargs,
#     ):
#         super().__init__()
#         self.n_alleles = n_alleles
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.n_layers = n_layers
#         self.mamba_kwargs = mamba_kwargs

#         self.embed = GenoEmbedding(n_alleles, d_model)
#         self.evo_embed = EvoEmbedding(d_model, n_evo_heads)
#         # ① 不再用单个 chunk_module，而是 ModuleList，初始为空
#         self.chunk_modules = nn.ModuleList()
#         self.out_proj = nn.Linear(d_model, n_alleles)

#     def _get_chunk_module(self, idx: int, device: torch.device) -> ChunkModule:
#         """动态获取第 idx 个 chunk 的独立模块，不存在就现建，并保证在同设备。"""
#         while idx >= len(self.chunk_modules):
#             # 新建
#             new_mod = ChunkModule(
#                 d_model=self.out_proj.in_features,
#                 n_layers=self.n_layers,
#                 **self.mamba_kwargs,
#             )
#             # 关键：立即搬到指定设备
#             new_mod.to(device)
#             self.chunk_modules.append(new_mod)
#         return self.chunk_modules[idx]

#     def forward(self, x: torch.Tensor,
#                 x_coord: torch.Tensor,
#                 dist: torch.Tensor | None = None):
#         B, L_orig = x.shape
#         device = x.device

#         # 1. 嵌入
#         h = self.embed(x, x_coord)
#         h = self.evo_embed(h, dist)

#         # 2. 滑窗切分参数
#         chunk_size = self.chunk_size
#         overlap = self.chunk_overlap
#         step = chunk_size - overlap
#         n_chunks = math.ceil((L_orig - overlap) / step)
#         pad_len = n_chunks * step + overlap - L_orig
#         if pad_len > 0:
#             h = F.pad(h, (0, 0, 0, pad_len))
#         L_pad = h.shape[1]

#         out_buf = torch.zeros(B, L_pad, h.shape[-1], device=device)
#         count_buf = torch.zeros(B, L_pad, dtype=torch.long, device=device)

#         # 3. 每个 chunk 用独立权重
#         for i in range(n_chunks):
#             start = i * step
#             end = start + chunk_size
#             chunk = h[:, start:end, :]
#             chunk_out = self._get_chunk_module(i, device)(chunk)
#             out_buf[:, start:end, :] += chunk_out
#             count_buf[:, start:end] += 1

#         out_buf = out_buf / count_buf.unsqueeze(-1).clamp_min(1).float()
#         out_buf = out_buf[:, :L_orig, :]
#         logits = self.out_proj(out_buf)
#         probs = logits.softmax(dim=-1)
#         return probs

# unit test
if __name__ == "__main__":
    n_alleles = 3          # 真实 0,1,2 三类，-1 为缺失
    model = EvoFill(
        n_alleles=n_alleles,
        chunk_size=5120,
        chunk_overlap=64,
        d_model=256,
        n_layers=4,
        d_state=128,
        expand=2,
    ).cuda()

    B, L = 2, 56135
    x = torch.randint(-1, n_alleles, (B, L)).long().cuda()   # {-1,0,1,2}
    x_coord = torch.randn(L, 4).cuda()                   # (L,4)

    probs = model(x, x_coord)
    print(probs.shape)  # ✅ torch.Size([2, 1800, 3])