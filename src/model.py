
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Union

from mamba_ssm import Mamba2
from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2Block # 原Mamba2Block

class GenoEmbedding(nn.Module):
    """Genomic embedding layer with positional encoding"""

    def __init__(self, n_alleles, n_snps, d_model):
        super().__init__()
        self.d_model = d_model
        self.n_alleles = n_alleles
        self.n_snps = n_snps

        # Allele embedding
        self.allele_embedding = nn.Parameter(torch.randn(n_alleles, d_model))

        # Positional embedding
        self.position_embedding = nn.Embedding(n_snps, d_model)

        # Initialize parameters
        nn.init.xavier_uniform_(self.allele_embedding)

    def forward(self, x):
        # x shape: (batch, seq_len, n_alleles) - one-hot encoded
        _, seq_len, _ = x.shape

        # Allele embedding
        embedded = torch.einsum('bsn,nd->bsd', x, self.allele_embedding)

        # Positional embedding
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0)

        return embedded + pos_emb

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block for genomic sequence processing"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        # Forward and backward Mamba blocks
        self.mamba_forward = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.mamba_backward = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.GELU()
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        residual = x

        # Bidirectional processing
        x_norm = self.norm1(x)

        # Forward direction
        forward_out = self.mamba_forward(x_norm)

        # Backward direction (flip sequence)
        x_backward = torch.flip(x_norm, dims=[1])
        backward_out = self.mamba_backward(x_backward)
        backward_out = torch.flip(backward_out, dims=[1])

        # Concatenate bidirectional outputs
        bi_out = torch.cat([forward_out, backward_out], dim=-1)

        # FFN
        ffn_out = self.ffn(bi_out)
        ffn_out = self.dropout(ffn_out)

        # Residual connection
        out = self.norm2(residual + ffn_out)

        return out

class ConvBlock(nn.Module):
    """Convolutional block for local pattern extraction"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)

        self.conv_large1 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.conv_large2 = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7)

        self.conv_final = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv_reduce = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.gelu = nn.GELU()

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)

        xa = self.gelu(self.conv1(x))

        xb = self.gelu(self.conv2(xa))
        xb = self.gelu(self.conv3(xb))

        xc = self.gelu(self.conv_large1(xa))
        xc = self.gelu(self.conv_large2(xc))

        xa = xb + xc
        xa = self.gelu(self.conv_final(xa))
        xa = self.bn1(xa)
        xa = self.gelu(self.conv_reduce(xa))
        xa = self.bn2(xa)
        xa = self.gelu(xa)

        return xa.transpose(1, 2)  # (batch, seq_len, d_model)

class ExtraEmbedding(nn.Module):
    """
    输入:  (B, L)        L == extra_dim
    输出: (B, L, d_model)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int  = 4,
        expand: int  = 2,
        headdim: int = 128,
        ngroups: int = 1,
        dropout: float = 0.1,
        **mamba_kwargs,
    ):
        super().__init__()
        self.d_model   = d_model

        # 1. 把 (B, L) 的 1-d 标量升到 d_model
        self.in_proj = nn.Linear(1, d_model, bias=False)

        # 2. 官方 Mamba2Simple：把 L 当序列长度，建模 L↔L
        self.mamba = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            **mamba_kwargs
        )

        # 3. Norm
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor):
        """
        x: (B, L)  连续值或离散索引
        """
        # (B, L) -> (B, L, 1) -> (B, L, d_model)
        h = self.in_proj(x.unsqueeze(-1).float())   # 1-d 投影

        h = self.norm(h)

        # Mamba2Simple 要求输入 (B, L, d_model) 即可
        out = self.mamba(h)                           # SSD 全局建模
        return out

class StackMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        chunk_size=256,
        dropout=0.0,
        d_embed_dropout=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model

        # 距离矩阵嵌入
        self.extra_embed = ExtraEmbedding(d_model=d_model, dropout=d_embed_dropout)

        # 原归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # SSD 核心
        self.ssd = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_mem_eff_path=True,
            device=device,
            dtype=dtype,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, local_repr, global_repr, x_extra=None,
                start_offset=0, end_offset=0):
        """
        local_repr: (B, L, D)
        global_repr: (B, G, D)
        x_extra: 可选，(B,E) 
        """
        local_norm  = self.norm1(local_repr)
        global_norm = self.norm2(global_repr)

        # 1. 构造输入序列
        tokens = []
        if x_extra is not None:
            extra_token = self.extra_embed(x_extra)        # (B,E,D)
            tokens.append(extra_token)
        tokens.append(global_norm)
        tokens.append(local_norm)
        x = torch.cat(tokens, dim=1)               # [B, (E)+G+L, D]

        # 2. SSD 扫描
        x = self.ssd(x)                            # [B, (E)+G+L, D]

        # 3. 只取 local 部分
        local_len = local_norm.shape[1]
        x = x[:, -local_len:, :]                   # [B, L, D]

        # 4. pad 回原始长度
        if start_offset or end_offset:
            x = F.pad(x, (0, 0, start_offset, end_offset))

        # 5. 残差 + FFN
        x = x + local_norm
        x = self.norm3(x)
        x = self.ffn(x) + x
        return x

class ChunkModule(nn.Module):
    """Single chunk processing module with BiMamba"""

    def __init__(self, d_model, dropout_rate=0.2):
        super().__init__()
        self.d_model = d_model

        # BiMamba block
        self.bimamba_block = BiMambaBlock(d_model)

        # Convolutional blocks
        self.conv_block1 = ConvBlock(d_model)
        self.conv_block2 = ConvBlock(d_model)
        self.conv_block3 = ConvBlock(d_model)
        self.conv_block4 = ConvBlock(d_model)

        # Cross attention
        # self.cross_attention = CrossAttentionLayer(d_model, n_heads)
        self.cross_attention = StackMambaBlock(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=128,
            ngroups=1,
            chunk_size=256,
        )

        # Additional layers
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x, x_extra=None):
        # BiMamba processing
        xa0 = self.bimamba_block(x)

        # First conv block
        xa = self.conv_block1(xa0)
        xa_skip = self.conv_block2(xa)

        # Dense layer
        xa = self.gelu(self.dense(xa))
        xa = self.conv_block3(xa)

        # Cross attention
        xa = self.cross_attention(xa, xa0, x_extra)
        xa = self.dropout(xa)

        # Final conv block
        xa = self.conv_block4(xa)

        # Concatenate with skip connection
        xa = torch.cat([xa_skip, xa], dim=-1)

        return xa

class UltraLongRangeMamba(nn.Module):
    """
    线性复杂度 O(L) 全局建模，只激活 mask=1 的位点
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(d_model * expand)
        # 可选：多层 Mamba2
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        """
        x: (B, L_all, d_model)  全局张量，其余位置为 nan
        idx: (M,)  当前 mask=1 的坐标
        返回: (B, M, d_model//2)
        """
        # 只取有效 token
        x_in = x[:, idx] 
        x_in = self.dropout(x_in)                                   # (B, M, D)
        for layer in self.layers:
            x_in = layer(x_in)                              # BiMamba2
        return self.norm(x_in)

class GlobalOut(nn.Module):
    def __init__(self, d_model, n_alleles, total_sites, chunk_size,
                 kernel=5, pad=2, stripe=4096,
                 d_state=64, d_conv=4, expand=2, n_mamba_layers=2):
        super().__init__()
        self.k, self.p = kernel, pad
        self.stripe = stripe
        self.total_sites = total_sites
        self.n_alleles = n_alleles

        # -------------- 1) 局部卷积权重 --------------
        # Conv1: 2*d_model -> d_model//2
        self.w1 = nn.Parameter(torch.empty(d_model // 2, 2 * d_model, kernel))
        self.b1 = nn.Parameter(torch.zeros(d_model // 2))
        # Conv2: d_model//2 -> n_alleles-1
        self.w2 = nn.Parameter(torch.empty(n_alleles - 1, d_model // 2, kernel))
        self.b2 = nn.Parameter(torch.zeros(n_alleles - 1))
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)

        # -------------- 2) ulr 中间件（Mamba2） --------------
        self.ulr_mamba = UltraLongRangeMamba(
            d_model=d_model//2,          # 与 Conv1 输出同维
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_mamba_layers,
        )
        self.gate = nn.Linear(d_model, 2)   # [local; global] -> 2
        self.norm = nn.LayerNorm(d_model // 2)

        # -------------- 3) 开关 --------------
        self.skip_ulr = True
        self.set_ulr_enabled(False)

    # ============ 两阶段切换 ============
    def set_ulr_enabled(self, enabled: bool):
        self.skip_ulr = not enabled
        for p in self.ulr_mamba.parameters():
            p.requires_grad = enabled
        for p in self.gate.parameters():
            p.requires_grad = enabled

    # ============ 前向：ulr 是可插拔中间件 ============
    def forward(self, x, mask):
        """
        x:   (B, L,  2*d_model)
        mask:(L,) 0/1
        return: (B, L, n_alleles-1)
        """
        x = x.transpose(1, 2)  # (B, 2*d_model, L)
        device = x.device
        idx = torch.where(mask)[0]                # 有效坐标 M
        n = idx.shape[0]
        out = torch.full((x.shape[0], self.w2.shape[0], x.shape[2]), -float('inf'),
                         device=device, dtype=x.dtype)

        # ---- 1) 统一走 Conv1：2*d_model -> d_model//2 ----
        h_local = []                              # (B, d_model//2, M)
        for i in range(0, n, self.stripe):
            sl = slice(i, i + self.stripe)
            idx_i = idx[sl]
            x_i = x[..., idx_i].contiguous()      # (B, 2*d_model, stripe)

            y1 = checkpoint(self._band_conv1, x_i, self.w1, self.b1, use_reentrant=False)
            h_local.append(y1)
        h_local = torch.cat(h_local, dim=2).transpose(1, 2)  # (B, M, d_model//2)
        # ---- 2) ulr 中间件（可选） ----
        if self.skip_ulr:
            # 第一阶段：不做任何全局事，h_local 保持原样
            fused = h_local
        else:
            # 第二阶段：Mamba2 全局建模并融合
            h_global = self.ulr_mamba(h_local, idx)           # (B, M, d_model//2)
            gate_in = torch.cat([h_local, h_global], dim=-1)  # (B, M, d_model)
            w = torch.softmax(self.gate(gate_in), dim=-1)     # (B, M, 2)
            fused = w[..., 0:1] * h_local + w[..., 1:2] * h_global
            fused = self.norm(fused)                          # (B, M, d_model//2)

        # ---- 3) 统一走 Conv2：d_model//2 -> n_alleles-1 ----
        y_final = F.conv1d(fused.transpose(1, 2), self.w2, self.b2, padding=self.p)
        out[..., idx] = y_final
        return F.softmax(out.transpose(1, 2), dim=-1)

    # ---------- 辅助 ----------
    def _band_conv1(self, x, w, b):
        return F.gelu(F.conv1d(x, w, b, padding=self.p))

class EvoFill(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_alleles: int,
        total_sites: int,
        chunk_size: int = 8192,
        chunk_overlap: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_alleles = n_alleles
        self.total_sites = total_sites
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 1. chunk 边界
        stride = chunk_size - chunk_overlap
        starts = [i * stride for i in range((total_sites - 1) // stride + 1)]
        ends = [min(s + chunk_size, total_sites) for s in starts]
        self.register_buffer("starts", torch.tensor(starts, dtype=torch.long))
        self.register_buffer("ends", torch.tensor(ends, dtype=torch.long))
        self.n_chunks = len(starts)

        # 2. 每 chunk 一份嵌入 & 处理模块（常驻 GPU，但训练时只激活一个）
        self.chunk_embeds = nn.ModuleList(
            GenoEmbedding(n_alleles, e - s, d_model) for s, e in zip(starts, ends)
        )
        self.chunk_modules = nn.ModuleList(
            ChunkModule(d_model, dropout_rate) for s, e in zip(starts, ends)
        )

        # 3. 全局输出层
        self.global_out = GlobalOut(d_model, n_alleles, total_sites, chunk_size)

        # 4. chunk 掩码表  (n_chunks, L)
        masks = torch.stack(
            [torch.arange(total_sites).ge(s) & torch.arange(total_sites).lt(e)
             for s, e in zip(starts, ends)]
        ).float()
        self.register_buffer("chunk_masks", masks)

    def forward(self,
            x: torch.Tensor,                 # (B, L, n_alleles) one-hot
            chunk_id: Union[int, List[int]],
            x_extra: Optional[torch.Tensor] = None
            ):

        batch_size = x.shape[0]
        device = x.device
        if x_extra is not None and x_extra.shape[0] != batch_size:
            x_extra = None

        # 统一成 list
        if isinstance(chunk_id, int):
            mask = self.chunk_masks[chunk_id].bool()          # 单 chunk
            chunk_id = [chunk_id]
        else:
            mask = self.chunk_masks[chunk_id].sum(dim=0).bool()  # 多 chunk 并集

        z_acc   = torch.zeros(batch_size, self.total_sites, 2 * self.d_model, device=device)
        cnt_acc = torch.zeros(self.total_sites, device=device)

        # 1. 依次处理每个cid
        for cid in chunk_id:
            s, e = self.starts[cid].item(), self.ends[cid].item()
            x_chunk = x[:, s:e]
            z = self.chunk_embeds[cid](x_chunk)                    # (B, len, d_model)
            z = self.chunk_modules[cid](z, x_extra)                # (B, len, 2*d_model)
            z_acc[:, s:e] += z
            cnt_acc[s:e]  += 1

        # 2. 重叠平均
        cnt_acc = cnt_acc.clamp(min=1)
        z_full  = z_acc / cnt_acc.unsqueeze(0).unsqueeze(-1)     # (B, L, 2*d_model)

        # 3. 全局输出
        out  = self.global_out(z_full, mask)                     # (B, L, n_alleles-1)

        # 4. 返回并集区域
        # return out[:, torch.where(mask)[0]]
        return out, torch.where(mask)[0]


if __name__ == '__main__':
    # ---------- 假数据 ----------
    B, L, A = 8, 12345, 3
    d_model = 64
    chunk_size, overlap = 4096, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train = torch.zeros(B, L, A, device=device)
    allele = torch.randint(0, A, (B, L), device=device)
    x_train.scatter_(2, allele.unsqueeze(-1), 1)
    x_extra = torch.randn(B, 10, device=device)
    y_train = torch.randn(B, L, A-1, device=device)

    print(f"x_train: {x_train.shape}")
    print(f"x_extra: {x_extra.shape}")
    print(f"y_train: {y_train.shape}")
    print("")
    # ---------- 模型 &损失 ----------
    model = EvoFill(d_model, A, L, chunk_size, overlap).to(device)
    print(f"model chunks: {model.n_chunks}")

    print("单 chunk 测试")
    cid = 0
    model.global_out.set_ulr_enabled(False)
    pred, mask_idx = model(x_train, cid, x_extra)
    print(pred.shape)

    print("多 chunk 测试")
    cids= [0,2]
    model.global_out.set_ulr_enabled(True)
    pred, mask_idx = model(x_train, cid, x_extra)
    print(pred.shape)
