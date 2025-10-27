
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LongRangeModule(nn.Module):
    """
    调用接口：
        x = self.long_range(x, mask)
    参数：
        total_sites   : 序列最大长度（决定 Embedding 词表大小）
        d_model   : 输入特征维
        chunk_size: 距离阈值
        cos_cutoff: 余弦相似度绝对值阈值
        d_emb     : embedding 维，默认 d_model//2
    """
    def __init__(self, total_sites, d_model, chunk_size=128, cos_cutoff=0.8, d_emb=None):
        super().__init__()
        self.total_sites = total_sites
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.cos_cutoff = cos_cutoff
        self.d_emb = d_emb or (d_model // 2)

        # 两个稀疏梯度 的 Embedding
        self.emb_i = nn.Embedding(self.total_sites, self.d_emb, sparse=True)
        self.emb_j = nn.Embedding(self.total_sites, self.d_emb, sparse=True)


    def forward(self, x, mask):
        """
        x    : (B, L, d_model)
        mask : (L,)  0/1 或 True/False
        return: 同 shape 的 x_out
        """
        # 1. 有效位点
        idx = torch.where(mask == 1)[0]          # (N_valid,)
        N_valid = idx.size(0)
        if N_valid == 0:
            return x

        # 2. 距离矩阵 & 是否有 far j
        dist = torch.abs(idx[:, None] - idx[None, :])          # (N_valid, N_valid)
        far_mask = dist > self.chunk_size
        if far_mask.sum() == 0:           # 一个 far j 都没有直接返回
            return x 

        emb_i_w = self.emb_i(idx)
        emb_j_w = self.emb_j(idx)
        cos_sim = torch.abs(F.cosine_similarity(emb_i_w.unsqueeze(1),
                                                emb_j_w.unsqueeze(0), dim=-1))

        # 4. 过滤
        valid_j_mask = far_mask & (cos_sim > self.cos_cutoff)

        # 5. 加权更新
        x_out = x.clone()
        for row, i_global in enumerate(idx):
            j_local_mask = valid_j_mask[row]
            num_j = j_local_mask.sum()
            if num_j == 0:
                continue
            j_local = torch.where(j_local_mask)[0]
            j_global = idx[j_local]
            weights = cos_sim[row, j_local] / num_j
            xj_weighted = (x[:, j_global] * weights.view(1, -1, 1)).sum(dim=1)
            x_out[:, i_global] = (x[:, i_global] + xj_weighted) / 2
        return x_out

class GlobalOut(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_alleles: int,
                 total_sites: int,
                 chunk_size: int,
                 cos_cutoff: float = 0.8):
        super().__init__()
        self.proj_in = nn.Linear(2 * d_model, d_model//2)
        self.long_range = LongRangeModule(
                total_sites=total_sites,
                d_model=d_model//2,
                chunk_size=chunk_size,
                cos_cutoff=cos_cutoff
            )
        self.proj_out = nn.Linear(d_model//2, n_alleles - 1)


    def forward(self, x, mask):
        x = self.proj_in(x)                       # (B, L, d_model)
        x = self.long_range(x, mask)              # + 稀疏长程信号
        x = self.proj_out(x)                      # (B, L, n_alleles-1)
        x = torch.where(mask.unsqueeze(0).unsqueeze(-1).bool(),
                        x, torch.tensor(-float('inf'), device=x.device))
        return F.softmax(x, dim=-1)

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

        # 3. 全局输出层（始终 GPU）
        self.global_out = GlobalOut(d_model, n_alleles, total_sites, chunk_size)

        # 4. chunk 掩码表  (n_chunks, L)
        masks = torch.stack(
            [torch.arange(total_sites).ge(s) & torch.arange(total_sites).lt(e)
             for s, e in zip(starts, ends)]
        ).float()
        self.register_buffer("chunk_masks", masks)

    def forward(self, x: torch.Tensor, chunk_id: int,
                x_extra: Optional[torch.Tensor] = None):
        """
        x:       (B, len, n_alleles)  对应chunk部分的序列 one-hot
        chunk_id: 0..n_chunks-1
        x_extra:  (B, extra_dim) or None
        return:   (B, len, n_alleles-1)  对应chunk部分的softmax 概率
        """
        B, _ , _ = x.shape
        device = x.device
        s, e = self.starts[chunk_id].item(), self.ends[chunk_id].item()
        mask = self.chunk_masks[chunk_id]          # (L,)  当前 chunk 覆盖区

        # 1. 确保输入 x 的形状与对应 chunk 吻合
        assert x.shape[1] == e-s

        # 2. 当前 chunk 嵌入 & 处理
        z = self.chunk_embeds[chunk_id](x)   # (B, len, d_model)
        z = self.chunk_modules[chunk_id](z, x_extra)  # (B, len, 2*d_model)

        # 3. 拼回全长度，其余 nan
        z_full = torch.full((B, self.total_sites, 2 * self.d_model), float('nan'), device=device)
        z_full[:, s:e] = z                        # (B, L, 2*d_model)

        # 4. 全局卷积只激活带状区
        out = self.global_out(z_full, mask)       # (B, L, n_alleles-1)
        
        # 5. 只返回对应chunk的logits
        return out[:, torch.where(mask)[0]]    # (B, len, n_alleles-1)
    
    @torch.no_grad()
    def infer(self, x: torch.Tensor,
              x_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        推理接口：遍历全部 chunk，重叠区特征取平均，再统一过全局卷积
        x:       (B, L, n_alleles)
        x_extra: (B, extra_dim)
        return:  (B, L, n_alleles-1)  softmax 概率
        """
        B, L, _ = x.shape
        device = x.device
        d_out = 2 * self.d_model          # z 通道数

        # 累加器
        z_sum = torch.zeros(B, d_out, L, device=device)   # (B, 2*d_model, L)
        z_cnt = torch.zeros(1, 1, L, device=device)        # (1, 1, L)

        # 1. 遍历所有 chunk，拼回全长度并累加
        for cid in range(self.n_chunks):
            mask = self.chunk_masks[cid]          # (L,)  0/1
            idx  = torch.where(mask)[0]           # 当前 chunk 位点
            s, e = self.starts[cid].item(), self.ends[cid].item()

            # 与 forward 完全相同：chunk -> z
            x_slice = x[:, s:e]
            z = self.chunk_embeds[cid](x_slice)
            z = self.chunk_modules[cid](z, x_extra)        # (B, len, 2*d_model)
            z = z.transpose(1, 2)                          # (B, 2*d_model, len)

            # 写回全长度 & 累加
            z_sum[..., idx] += z[..., idx - s]             # 局部→全局对齐
            z_cnt[..., idx] += 1

        # 2. 重叠区平均
        z_full = z_sum / z_cnt.clamp_min(1.0)              # (B, 2*d_model, L)

        # 3. 统一过全局卷积一次
        #    global_out 需要 mask：全 1 即可（所有位点都有效）
        full_mask = torch.ones(L, dtype=torch.float, device=device)
        return self.global_out(z_full, full_mask)          # (B, L, n_alleles-1)

if __name__ == '__main__':
    # ---------- 假数据 ----------
    B, L, A = 4, 12345, 3
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
    criterion = nn.MSELoss(reduction='mean')

    # ---------- 训练循环 ----------
    epochs_per_chunk = 3
    for cid in range(model.n_chunks):
        mask = model.chunk_masks[cid]
        idx = torch.where(mask)[0]
        x_band = x_train[:,idx]
        y_band = y_train[:,idx]
        print(f"x_band: {x_band.shape}")
        print(f"x_extra: {x_extra.shape}")
        print(f"y_band: {y_band.shape}")
        opt = torch.optim.AdamW(
            list(model.chunk_embeds[cid].parameters()) +
            list(model.chunk_modules[cid].parameters()) +
            list(model.global_out.parameters()), lr=3e-4)
        for epoch in range(epochs_per_chunk):
            opt.zero_grad()
            pred = model(x_band, cid, x_extra)

            loss = criterion(pred, y_band)        # 默认 reduction='mean'
            loss.backward()
            opt.step()
            print("")

            print(f"pred: {pred.shape}")
            print(f'chunk {cid}/{model.n_chunks-1} | epoch {epoch+1}/{epochs_per_chunk} | loss {loss.item():.4f}')