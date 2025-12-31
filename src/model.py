import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Union

from timm.layers import DropPath
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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.,init_value=1e-4):
        super().__init__()
        self.d_model = d_model

<<<<<<< HEAD
        # 只初始化一个 Mamba2 模块
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
        self.mamba_backward.conv1d.weight = self.mamba_forward.conv1d.weight
        self.mamba_backward.conv1d.bias   = self.mamba_forward.conv1d.bias
=======
    def forward(self, x,  dist_bias: torch.Tensor | None):
        """
        x: (N, L, D)          N = n_samples, L = seqlen
        dist_bias: (N, N)    遗传距离矩阵，作为注意力偏置
        """
        N, L, D = x.shape
        
        # 确保 dist_bias 与 x 的数据类型一致
        dist_bias = dist_bias.to(x.dtype) if dist_bias is not None else None
        
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # (3, N, H, L, Dh)
        
        # 确保 q, k, v 的数据类型一致
        q = q.to(x.dtype)
        k = k.to(x.dtype)
        v = v.to(x.dtype)
        
        # 转置为 (N, H, L, Dh) -> (H, L, N, Dh) 以便在样本维度计算注意力
        q = q.permute(1, 2, 0, 3)  # (H, L, N, Dh)
        k = k.permute(1, 2, 0, 3)  # (H, L, N, Dh)
        v = v.permute(1, 2, 0, 3)  # (H, L, N, Dh)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (H, L, N, N)
        
        if dist_bias is not None:                       # 只有传了才加偏置
            dist_bias = dist_bias.to(x.dtype)
            dist_bias = dist_bias.unsqueeze(0).unsqueeze(0)
            dist_bias = dist_bias.expand(self.num_heads, L, -1, -1)
            attn_scores = attn_scores + dist_bias
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 确保注意力权重和 v 的数据类型一致
        attn_weights = attn_weights.to(v.dtype)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # (H, L, N, Dh)
        
        # 重新排列维度
        out = out.permute(2, 0, 1, 3)  # (N, H, L, Dh)
        out = out.reshape(N, L, -1)  # (N, L, H * Dh)
        
        return self.proj(out)
>>>>>>> main

        self.gamma = nn.Parameter(init_value * torch.ones(d_model))
        # 其余部分保持不变
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.GELU()
        )
        self.dropout = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)

        # 把计算量大的部分包起来
        def _block(x_):
            f = self.mamba_forward(x_)
            b = self.mamba_backward(torch.flip(x_, [1]))
            b = torch.flip(b, [1])
            return torch.cat([f, b], -1)

        bi_out = checkpoint(_block, x_norm, use_reentrant=False)
        ffn_out = self.ffn(bi_out)
        ffn_out = self.dropout(ffn_out)
        out = self.norm2(residual + self.drop_path(self.gamma * ffn_out))
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

        self.bn1 = nn.GroupNorm(8, d_model)
        self.bn2 = nn.GroupNorm(8, d_model)

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

class StackMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model

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

<<<<<<< HEAD
    def forward(self, local_repr, global_repr,
                start_offset=0, end_offset=0):
=======
    def forward(self, x):
        return self.net(x)


class FlashTransformerBlock(nn.Module):
    """
    序列自注意力 + 样本注意力 + FFN
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.seq_attn = FlashMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.sample_attn = SampleFlashAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, dropout=dropout)

    def forward(self, x, dist_mat: torch.Tensor | None = None):
>>>>>>> main
        """
        local_repr: (B, L, D)
        global_repr: (B, G, D)
        """
<<<<<<< HEAD
        local_norm  = self.norm1(local_repr)
        global_norm = self.norm2(global_repr)

        # 1. 构造输入序列
        x = torch.cat([global_norm, local_norm], dim=1) # [B, G+L, D]

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
=======
        # 确保 dist_mat 与 x 的数据类型一致
        dist_mat = dist_mat.to(x.dtype) if dist_mat is not None else None
        
        # 1) 序列维度自注意力
        x = x + self.seq_attn(self.norm1(x))
        # 2) 样本维度交叉注意力（遗传距离作为偏置）
        x = x + self.sample_attn(self.norm2(x), dist_mat)
        # 3) FFN
        x = x + self.ffn(self.norm3(x))
>>>>>>> main
        return x

class ChunkModule(nn.Module):
    """Single chunk processing module with BiMamba"""

    def __init__(self, d_model, d_state=64, headdim=128, dropout_rate=0.2, bimamba_layers=1, stack_mamba_layers=1):
        super().__init__()
        self.d_model = d_model
        self.bimamba_layers = bimamba_layers
        self.stack_mamba_layers = stack_mamba_layers

        # BiMamba blocks (stacked)
        self.bimamba_blocks = nn.ModuleList([
            BiMambaBlock(d_model)
            for _ in range(bimamba_layers)
        ])

        # Convolutional blocks
        self.conv_block1 = ConvBlock(d_model)
        self.conv_block2 = ConvBlock(d_model)
        self.conv_block3 = ConvBlock(d_model)
        self.conv_block4 = ConvBlock(d_model)

        # Cross attention (stacked StackMamba blocks)
        # self.cross_attention = CrossAttentionLayer(d_model, n_heads)
        self.stack_mambas = nn.ModuleList([
            StackMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                headdim=headdim,
                ngroups=1,
                dropout=0.0,
            )
            for _ in range(stack_mamba_layers)
        ])

        # Additional layers
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        # BiMamba processing (stacked layers)
        xa0 = x
        for i in range(self.bimamba_layers):
            xa0 = self.bimamba_blocks[i](xa0)

        # First conv block
        xa = self.conv_block1(xa0)
        xa_skip = self.conv_block2(xa)

        # Dense layer
        xa = self.gelu(self.dense(xa))
        xa = self.conv_block3(xa)

        # Cross attention (stacked layers)
        for i in range(self.stack_mamba_layers):
            xa = self.stack_mambas[i](xa, xa0)
        xa = self.dropout(xa)

        # Final conv block
        xa = self.conv_block4(xa)

        # Concatenate with skip connection
        xa = torch.cat([xa_skip, xa], dim=-1)

        return xa

class UltraLongRangeMamba(nn.Module):
    """
    线性复杂度 O(M) 全局建模，只激活跨 chunk 且距离 > chunk_size 的位点关系。
    输出 (B,M,d_model//2)，可直接与 h_local 做门控融合。
    """
    def __init__(self, d_model, chunk_size=8192, total_sites=100_000, threshold=0.1,d_state=64, d_conv=4, expand=2,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.threshold  = threshold
        # ---- Mamba-2 堆叠 ----
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ---- idx 嵌入 ----
        self.idx_embed = nn.Embedding(total_sites, d_model // 2) #, sparse=True

        # ---- 门控 ----
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.Sigmoid()
        )
<<<<<<< HEAD

    def forward(self, h_local, idx):
=======
        
    def forward(self, x: torch.Tensor, dist_mat: torch.Tensor | None = None):
>>>>>>> main
        """
        h_local: (B, M, d_model//2)  局部特征
        idx:     (M,)                在 0~L-1 的坐标
        return:  (B, M, d_model//2)  已融合长程信号
        """
        # 1. 构造输入：坐标嵌入 + 局部特征
        idx_emb = self.idx_embed(idx).unsqueeze(0).expand(h_local.shape[0], -1, -1)                  # (B,M,D//2)
        x = torch.cat([idx_emb,h_local], dim=-1)       # (B,M,D)

        # 2. Mamba 全局编码（只激活 M 个位点，O(M)）
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        h_global = x[..., x.size(-1)//2:]               # 取后半截 (B,M,D//2)

<<<<<<< HEAD
        # 3. 门控 + 阈值过滤
        delta = h_global - h_local
        gate = self.gate_proj(delta)
        # 4. 硬阈值 mask：delta 太小就把 gate 置 0
        mask = (delta.abs() > self.threshold)      # (B,M,D//2)  逐元素
        gate = gate * mask                         # 不满足→ gate=0
        fused = h_local + gate * delta             # gate=0 时 fused=h_local
        return fused

class GlobalOut(nn.Module):
    def __init__(self, d_model, n_alleles, total_sites, chunk_size,
                 kernel=5, pad=2, stripe=4096):
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

    # ============ 前向：ulr 是可插拔中间件 ============
    def forward(self, x, mask):
=======
    def predict(self, x: torch.Tensor, dist_mat: torch.Tensor | None = None):
>>>>>>> main
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

        # ---- 3) 统一走 Conv2：d_model//2 -> n_alleles-1 ----
        y_final = F.conv1d(h_local.transpose(1, 2), self.w2, self.b2, padding=self.p)
        out[..., idx] = y_final

        return out.transpose(1, 2)

    # ---------- 辅助 ----------
    def _band_conv1(self, x, w, b):
        w = w.to(dtype=x.dtype)
        b = b.to(dtype=x.dtype)
        return F.gelu(F.conv1d(x, w, b, padding=self.p))

class EvoFill(nn.Module):
    def __init__(
        self,
        n_alleles: int,
        total_sites: int,
        chunk_size: int = 8192,
        chunk_overlap: int = 64,
        d_model: int = 64,
        d_state: int = 64, 
        headdim: int = 128,
        bimamba_layers: int = 4,
        stack_mamba_layers: int = 4,
        dropout_rate: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_alleles = n_alleles
        self.total_sites = total_sites
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bimamba_layers = bimamba_layers
        self.stack_mamba_layers = stack_mamba_layers

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
            ChunkModule(d_model, d_state, headdim, dropout_rate, self.bimamba_layers, self.stack_mamba_layers)
            for s, e in zip(starts, ends)
        )

        depth = 0
        for cid in range(self.n_chunks):
            # 为每个 bimamba 层设置 drop_path 率
            for i in range(self.bimamba_layers):
                rate = drop_path_rate * (depth / (self.n_chunks * (self.bimamba_layers + self.stack_mamba_layers)))
                self.chunk_modules[cid].bimamba_blocks[i].drop_path = DropPath(rate)
                depth += 1
            
            # 为每个 stack_mamba 层设置 drop_path 率
            for i in range(self.stack_mamba_layers):
                rate = drop_path_rate * (depth / (self.n_chunks * (self.bimamba_layers + self.stack_mamba_layers)))
                self.chunk_modules[cid].stack_mambas[i].drop_path = DropPath(rate)
                depth += 1

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
            chunk_id: Optional[Union[int, List[int]]] = None
            ):
        if x.dtype != next(self.parameters()).dtype:
            self.to(dtype=x.dtype)
        batch_size = x.shape[0]
        device = x.device
        if chunk_id is None:
            chunk_id = list(range(self.n_chunks))
        # 统一成 list
        if isinstance(chunk_id, int):
            chunk_id = [chunk_id]
        mask = self.chunk_masks[chunk_id].sum(dim=0).bool()  # 并集掩码

        z_acc   = torch.zeros(batch_size, self.total_sites, 2 * self.d_model, device=device)
        cnt_acc = torch.zeros(self.total_sites, device=device)

        # 1. 依次处理每个cid
        for cid in chunk_id:
            s, e = self.starts[cid].item(), self.ends[cid].item()
            x_chunk = x[:, s:e]
            z = self.chunk_embeds[cid](x_chunk)                    # (B, len, d_model)
            z = self.chunk_modules[cid](z)                # (B, len, 2*d_model)
            z_acc[:, s:e] += z
            cnt_acc[s:e]  += 1

        # 2. 重叠平均
        cnt_acc = cnt_acc.clamp(min=1)
        z_full  = z_acc / cnt_acc.unsqueeze(0).unsqueeze(-1)     # (B, L, 2*d_model)

        # 3. 全局输出
        logits  = self.global_out(z_full, mask)                     # (B, L, n_alleles-1)

        # 4. 返回并集区域
        prob = torch.full_like(logits, 0.)                  # 先占位
        prob[:, mask] = F.softmax(logits[:, mask], dim=-1)

        return logits, prob, torch.where(mask)[0]


if __name__ == '__main__':
    # ---------- 假数据 ----------
    B, L, A = 8, 1000000, 3
    d_model = 64
    d_state = 64
    headdim = 64
    chunk_size, overlap = 65536, 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train = torch.zeros(B, L, A, device=device)
    allele = torch.randint(0, A, (B, L), device=device)
    x_train.scatter_(2, allele.unsqueeze(-1), 1)
    # x_extra = torch.randn(B, 10, device=device)
    y_train = torch.randn(B, L, A-1, device=device)

    print(f"x_train: {x_train.shape}")
    # print(f"x_extra: {x_extra.shape}")
    print(f"y_train: {y_train.shape}")
    print("")
    # ---------- 模型 &损失 ----------
    # 添加层数参数，默认值为1
    bimamba_layers = 2
    stack_mamba_layers = 2
    model = EvoFill(A, L, chunk_size, overlap, d_model, d_state, headdim, bimamba_layers=bimamba_layers, stack_mamba_layers=stack_mamba_layers).to(device)
    print(f"model chunks: {model.n_chunks}")

    print("单 chunk 测试")
    cid = 0
    model.global_out.set_ulr_enabled(False)
    logits, prob, mask_idx = model(x_train, cid)
    print(prob.shape)

    print("多 chunk 测试")
    cids= [0,2]
    model.global_out.set_ulr_enabled(True)
    logits, prob, mask_idx = model(x_train, cid)
    print(prob.shape)
