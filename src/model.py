# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2Block # 原Mamba2Block


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

class DistanceEmbed(nn.Module):
    """
    把 (B,B) 距离矩阵 -> (B, L*, D) 的时序表征
    L* = 1 或 L，这里用 1 个 token 代表整张图，可扩展
    """
    def __init__(self, max_len=1, d_model=256, dropout=0.0):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Linear(1, d_model)   # 把标量距离映成向量
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dismat):
        """
        x_dismat: (B,B)  对角=0
        返回: (B, max_len, D)  这里 max_len=1
        """
        # 取均值池化后作为全局距离向量 → 也可换成 GCN/Transformer 做更复杂编码
        z = x_dismat.mean(dim=1, keepdim=True)            # (B,1)
        z = z.unsqueeze(1)                         # (B,1,1)
        z = self.embed(z)                          # (B,1,D)
        z = self.norm(z)
        z = self.dropout(z)
        return z

class Mamba2CrossBlock(nn.Module):
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
        self.dist_embed = DistanceEmbed(max_len=1, d_model=d_model, dropout=d_embed_dropout)

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

    def forward(self, local_repr, global_repr,
                start_offset=0, end_offset=0,
                x_dismat=None):
        """
        local_repr: (B, L, D)
        global_repr: (B, G, D)
        D: 可选，(B,B) 距离矩阵，对角=0
        """
        local_norm  = self.norm1(local_repr)
        global_norm = self.norm2(global_repr)

        # 1. 构造输入序列
        tokens = []
        if x_dismat is not None:
            dist_token = self.dist_embed(self.d_model)        # (B,1,D)
            tokens.append(dist_token)
        tokens.append(global_norm)
        tokens.append(local_norm)
        x = torch.cat(tokens, dim=1)               # [B, (1)+G+L, D]

        # 2. SSD 扫描
        x = self.ssd(x)                            # [B, (1)+G+L, D]

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

class GenoEmbedding(nn.Module):
    """
    兼容 one-hot 输入的基因组嵌入层
    位置信息由 4 元浮点坐标实时生成
    """
    def __init__(self, n_alleles: int, d_model: int, coord_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        # 1. allele 嵌入：one-hot → d_model
        self.allele_embedding = nn.Parameter(torch.empty(n_alleles, d_model))
        nn.init.xavier_uniform_(self.allele_embedding)

        # 2. 坐标→位置向量网络
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor):
        """
        x:       (B, L, n_alleles)  one-hot 浮点
        x_coord: (L, 4)            每个 SNP 的 4 元坐标
        return:  (B, L, d_model)
        """
        # 1. allele 嵌入
        allele_emb = torch.einsum('bln,nd->bld', x, self.allele_embedding)  # (B,L,d)

        # 2. 坐标嵌入
        pos_emb = self.coord_proj(x_coord)           # (L,d)
        pos_emb = pos_emb.unsqueeze(0)               # (1,L,d)

        return allele_emb + pos_emb

class ChunkModule(nn.Module):
    """Single chunk processing module with BiMamba"""

    def __init__(self, d_model, start_offset=0, end_offset=0, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.start_offset = start_offset
        self.end_offset = end_offset

        # BiMamba block
        self.bimamba_block = BiMambaBlock(d_model)

        # Convolutional blocks
        self.conv_block1 = ConvBlock(d_model)
        self.conv_block2 = ConvBlock(d_model)
        self.conv_block3 = ConvBlock(d_model)

        # Cross attention
        # self.cross_attention = CrossAttentionLayer(d_model, n_heads)
        self.cross_attention = Mamba2CrossBlock(
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

    def forward(self, x, x_dismat=None):
        # BiMamba processing
        xa0 = self.bimamba_block(x)

        # First conv block
        xa = self.conv_block1(xa0)
        xa_skip = self.conv_block2(xa)

        # Dense layer
        xa = self.gelu(self.dense(xa))
        xa = self.conv_block2(xa)

        # Cross attention
        xa = self.cross_attention(xa, xa0, self.start_offset, self.end_offset, x_dismat)
        xa = self.dropout(xa)

        # Final conv block
        xa = self.conv_block3(xa)

        # Concatenate with skip connection
        xa = torch.cat([xa_skip, xa], dim=-1)

        return xa

class EvoFill(nn.Module):
    def __init__(self,
                 d_model,
                 n_alleles,
                 coord_dim = 4,
                 chunk_size=2048,
                 attention_range=64,
                 offset_before=0,
                 offset_after=0,
                 dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.attention_range = attention_range
        self.offset_before = offset_before
        self.offset_after = offset_after
        self.dropout_rate = dropout_rate
        self.n_alleles = n_alleles
        self.mask_int = n_alleles
        self.coord_dim = coord_dim

        # Embedding layer
        self.embedding = GenoEmbedding(n_alleles, self.d_model, self.coord_dim)

        # Create chunk modules
        self.chunk_module = ChunkModule(
            d_model=self.d_model,
            start_offset=0,
            end_offset=0,
            dropout_rate=self.dropout_rate
        )

        # Final layers
        self.final_conv = nn.Conv1d(self.d_model * 2, self.d_model // 2,
                                    kernel_size=5, padding=2)
        self.output_conv = nn.Conv1d(self.d_model // 2, n_alleles - 1,
                                     kernel_size=5, padding=2)

        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_coord, x_dismat=None):
        # x shape: (batch, seq_len, n_alleles)
        # Embedding
        _, seq_len, n_alleles = x.shape
        assert n_alleles == self.n_alleles
        x_embedded = self.embedding(x, x_coord)

        chunk_starts = list(range(0, seq_len, self.chunk_size))
        chunk_ends = [min(cs + self.chunk_size, seq_len) for cs in chunk_starts]
        mask_starts = [max(0, cs - self.attention_range) for cs in chunk_starts]
        mask_ends = [min(ce + self.attention_range, seq_len) for ce in chunk_ends]

        # Process chunks
        chunk_outputs = []
        for i in range(len(chunk_starts)):
            pad_left  = chunk_starts[i] - mask_starts[i]
            pad_right = mask_ends[i] - chunk_ends[i]
            chunk_input = x_embedded[:, mask_starts[i]:mask_ends[i]]
            chunk_output = self.chunk_module(chunk_input, x_dismat)   # 共享权重
            if pad_left or pad_right:
                chunk_output = F.pad(chunk_output, (0, 0, pad_left, pad_right))
            chunk_outputs.append(chunk_output)
        # Concatenate chunks along sequence dimension
        x_concat = torch.cat(chunk_outputs, dim=1)

        # # Final processing
        x_concat = x_concat.transpose(1, 2)  # (batch, features, seq_len)
        x_final = self.gelu(self.final_conv(x_concat))
        x_output = self.output_conv(x_final)
        x_output = x_output.transpose(1, 2)  # (batch, seq_len, n_alleles-1)

        # Apply offsets
        if self.offset_before > 0 or self.offset_after > 0:
            x_output = x_output[:, self.offset_before:self.seq_len - self.offset_after]

        x_output = self.softmax(x_output)

        return x_output

if __name__ == '__main__':
    n_alleles = 4  # 包含missing
    model = EvoFill(
        d_model=256,
        chunk_size=5120,
        n_alleles=n_alleles,
        attention_range=64, 
        offset_before=0,
        offset_after=0,
        dropout_rate=0.1,
    ).cuda()

    B, L = 2, 5120 

    # 1. 生成输入
    x = torch.randint(0, n_alleles, (B, L)).long().cuda()   # {0,1,2,3} 3=missing
    x_coord = torch.randn(L, 4).cuda()   

    # 2. -1 -> 3，并构造 one-hot（4 维）
    x_map = x.clone()
    x_onehot = torch.zeros(B, L, n_alleles, device='cuda')
    x_onehot.scatter_(2, x_map.unsqueeze(-1), 1)

    # 3. 前向
    with torch.no_grad():
        probs = model(x_onehot, x_coord)          # shape: (B,L,3)

    # 4. 简单校验
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, L, device='cuda'), atol=1e-5), \
        "概率未归一"
    print("✅ 含缺失数据前向通过，输出形状:", probs.shape)