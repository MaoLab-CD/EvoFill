import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("mamba-ssm not found. Please install with: pip install mamba-ssm")
    import sys
    sys.exit(1)

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block for genomic sequence processing"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        # Forward and backward Mamba blocks
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.mamba_backward = Mamba(
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

class CrossAttentionLayer(nn.Module):
    """Cross attention for integrating local and global features"""

    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU()
        )

    def forward(self, local_repr, global_repr, start_offset=0, end_offset=0):
        local_norm = self.norm1(local_repr)
        global_norm = self.norm2(global_repr)

        # Apply offsets
        if start_offset > 0 or end_offset > 0:
            query = local_norm[:, start_offset:local_norm.shape[1] - end_offset]
        else:
            query = local_norm

        key = value = global_norm

        # Cross attention
        attn_output, _ = self.cross_attention(query, key, value)

        # Skip connection
        attn_output = attn_output + query

        # FFN
        attn_output = self.norm3(attn_output)
        ffn_output = self.ffn(attn_output)
        output = ffn_output + attn_output

        return output

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
        batch_size, seq_len, _ = x.shape

        # Allele embedding
        embedded = torch.einsum('bsn,nd->bsd', x, self.allele_embedding)

        # Positional embedding
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0)

        return embedded + pos_emb

class ChunkModule(nn.Module):
    """Single chunk processing module with BiMamba"""

    def __init__(self, d_model, n_heads, start_offset=0, end_offset=0, dropout_rate=0.1):
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
        self.cross_attention = CrossAttentionLayer(d_model, n_heads)

        # Additional layers
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        # BiMamba processing
        xa0 = self.bimamba_block(x)

        # First conv block
        xa = self.conv_block1(xa0)
        xa_skip = self.conv_block2(xa)

        # Dense layer
        xa = self.gelu(self.dense(xa))
        xa = self.conv_block2(xa)

        # Cross attention
        xa = self.cross_attention(xa, xa0, self.start_offset, self.end_offset)
        xa = self.dropout(xa)

        # Final conv block
        xa = self.conv_block3(xa)

        # Concatenate with skip connection
        xa = torch.cat([xa_skip, xa], dim=-1)

        return xa

class EvoFill(nn.Module):
    """Main BiMamba model for genomic imputation"""

    def __init__(self,
                 d_model,
                 n_heads,
                 chunk_size=2048,
                 attention_range=64,
                 offset_before=0,
                 offset_after=0,
                 dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.attention_range = attention_range
        self.offset_before = offset_before
        self.offset_after = offset_after
        self.dropout_rate = dropout_rate

        # Will be set during build
        self.seq_len = None
        self.n_alleles = None
        self.embedding = None
        self.chunk_modules = nn.ModuleList()
        self.final_conv = None
        self.output_conv = None

    def build(self, seq_len, n_alleles):
        """Build the model with specific sequence length and allele count"""
        self.seq_len = seq_len
        self.n_alleles = n_alleles

        # Embedding layer
        self.embedding = GenoEmbedding(n_alleles, seq_len, self.d_model)

        # Calculate chunks
        chunk_starts = list(range(0, seq_len, self.chunk_size))
        chunk_ends = [min(cs + self.chunk_size, seq_len) for cs in chunk_starts]
        mask_starts = [max(0, cs - self.attention_range) for cs in chunk_starts]
        mask_ends = [min(ce + self.attention_range, seq_len) for ce in chunk_ends]

        # Create chunk modules
        for i, cs in enumerate(chunk_starts):
            start_offset = cs - mask_starts[i]
            end_offset = mask_ends[i] - chunk_ends[i]

            chunk_module = ChunkModule(
                d_model=self.d_model,
                n_heads=self.n_heads,
                start_offset=start_offset,
                end_offset=end_offset,
                dropout_rate=self.dropout_rate
            )
            self.chunk_modules.append(chunk_module)

        # Store chunk information
        self.chunk_starts = chunk_starts
        self.chunk_ends = chunk_ends
        self.mask_starts = mask_starts
        self.mask_ends = mask_ends

        # Final layers
        self.final_conv = nn.Conv1d(self.d_model * 2, self.d_model // 2,
                                    kernel_size=5, padding=2)
        self.output_conv = nn.Conv1d(self.d_model // 2, n_alleles - 1,
                                     kernel_size=5, padding=2)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch, seq_len, n_alleles)
        if self.embedding is None:
            raise RuntimeError("Model not built. Call build() first.")

        # Embedding
        x_embedded = self.embedding(x)

        # Process chunks
        chunk_outputs = []
        for i, chunk_module in enumerate(self.chunk_modules):
            chunk_input = x_embedded[:, self.mask_starts[i]:self.mask_ends[i]]
            chunk_output = chunk_module(chunk_input)
            chunk_outputs.append(chunk_output)

        # Concatenate chunks along sequence dimension
        x_concat = torch.cat(chunk_outputs, dim=1)

        # Final processing
        x_concat = x_concat.transpose(1, 2)  # (batch, features, seq_len)
        x_final = self.gelu(self.final_conv(x_concat))
        x_output = self.output_conv(x_final)
        x_output = x_output.transpose(1, 2)  # (batch, seq_len, n_alleles-1)

        # Apply offsets
        if self.offset_before > 0 or self.offset_after > 0:
            x_output = x_output[:, self.offset_before:self.seq_len - self.offset_after]

        x_output = self.softmax(x_output)

        return x_output

if __name__ == "__main__":
    n_alleles = 3
    model = EvoFill(
        d_model=256,
        n_heads=8,              # 原接口必填
        chunk_size=5120,
        attention_range=64,     # 原接口对应 chunk_overlap
        offset_before=0,
        offset_after=0,
        dropout_rate=0.1,
    ).cuda()

    B, L = 2, 5120
    model.build(seq_len=L, n_alleles=n_alleles+1)   # 关键 build
    model = model.cuda()  

    # 1. 生成含缺失的输入
    x = torch.randint(-1, n_alleles, (B, L)).long().cuda()   # {-1,0,1,2}

    # 2. -1 -> 3，并构造 one-hot（4 维）
    x_map = x.clone()
    x_map[x == -1] = n_alleles
    x_onehot = torch.zeros(B, L, n_alleles+1, device='cuda')
    x_onehot.scatter_(2, x_map.unsqueeze(-1), 1)

    # 3. 前向
    with torch.no_grad():
        probs = model(x_onehot)          # shape: (B,L,3)

    # 4. 简单校验
    assert probs.shape == (B, L, n_alleles), \
        f"期望 (B,L,3)，实际 {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, L, device='cuda'), atol=1e-5), \
        "概率未归一"
    print("✅ 含缺失数据前向通过，输出形状:", probs.shape)