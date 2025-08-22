
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_config


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, n_heads, L, d)
        B, n_heads, L, d = x.shape
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
        t = torch.arange(L, device=x.device).float()[:, None] * inv_freq[None, :]
        cos = torch.cos(t)   # (L, d//2)
        sin = torch.sin(t)

        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class DistanceToBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads))

    def forward(self, d):
        # d: scalar or (B,)
        d = torch.as_tensor(d).view(-1)  # (B,)
        return d.unsqueeze(-1) * self.weight  # (B, n_heads)


class MultiHeadSelfCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    @staticmethod
    def block_causal_mask(n_blocks, block_size, device):
        """
        n_blocks : 短序列条数 (n_snp)
        block_size: 每条长度 (k)
        return    : (n_blocks*block_size, n_blocks*block_size) 的 bool 掩码
        """
        L = n_blocks * block_size
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        for b in range(n_blocks):
            start = b * block_size
            end   = (b + 1) * block_size
            mask[start:end, start:end] = torch.tril(
                torch.ones(block_size, block_size, dtype=torch.bool, device=device)
            )
        return mask

    def forward(self, x, context=None, dist_bias=None):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if context is not None:
            context = context.to(dtype=x.dtype, device=x.device)
            kv = context.view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        q, k = self.rope(q), self.rope(k)
        att = (q @ k.transpose(-2, -1)) * self.scale
        n_blocks = L // self.block_size          # 这里 block_size=k
        mask = self.block_causal_mask(n_blocks, self.block_size, att.device)
        att = att.masked_fill(~mask, torch.finfo(att.dtype).min)

        if dist_bias is not None:
            dist_bias = dist_bias.to(dtype=att.dtype)
            att = att + dist_bias.view(-1, self.n_heads, 1, 1)

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        v = v.to(dtype=att.dtype)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, -1).to(dtype=self.out.weight.dtype)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EvoTLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout,block_size):
        super().__init__()
        self.self1 = MultiHeadSelfCrossAttention(d_model, n_heads, dropout, block_size)
        self.self2 = MultiHeadSelfCrossAttention(d_model, n_heads, dropout, block_size)
        self.cross1 = MultiHeadSelfCrossAttention(d_model, n_heads, dropout, block_size)
        self.cross2 = MultiHeadSelfCrossAttention(d_model, n_heads, dropout, block_size)
        self.ff1 = FeedForward(d_model, 4 * d_model, dropout)
        self.ff2 = FeedForward(d_model, 4 * d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dist_bias = DistanceToBias(n_heads)

    def forward(self, x1, x2, d):
        bias = self.dist_bias(d)
        x1 = self.norm(x1 + self.self1(x1, dist_bias=bias))
        x2 = self.norm(x2 + self.self2(x2, dist_bias=bias))
        cx1 = self.norm(x1 + self.cross1(x1, context=None, dist_bias=bias))
        cx2 = self.norm(x2 + self.cross2(x2, context=None, dist_bias=bias))
        cx1 = self.norm(cx1 + self.ff1(cx1))
        cx2 = self.norm(cx2 + self.ff2(cx2))
        return cx1, cx2


class EvoFill(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int,
                 n_heads: int, dropout: float, k: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = k

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EvoTLayer(d_model, n_heads, dropout, k) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head1 = nn.Linear(d_model, vocab_size, bias=False)
        self.head2 = nn.Linear(d_model, vocab_size, bias=False)
        self.head1.weight = self.embed.weight  # tie weight

    def forward(self, seq1, seq2, mask1, mask2, pdis):
        """
        seq1/seq2 : (1, n_snp, k)  int64
        mask1/2   : (1, n_snp, k)  float32  0/1 掩码
        pdis      : scalar  浮点数
        """
        # 把 batch 维度保留为 1
        ids1 = seq1.view(1, -1)   # (1, n_snp*k)
        ids2 = seq2.view(1, -1)

        x1 = self.embed(ids1)     # (1, L, d_model)
        x2 = self.embed(ids2)

        # pdis -> tensor
        d = torch.tensor(pdis, device=ids1.device, dtype=x1.dtype)

        for layer in self.layers:
            x1, x2 = layer(x1, x2, d)

        x1 = self.norm(x1)
        x2 = self.norm(x2)

        _, L, _ = x1.shape
        n_snp, k = seq1.shape[1], seq1.shape[2]
        logits1 = self.head1(x1).view(1, n_snp, k, self.vocab_size)
        logits2 = self.head2(x2).view(1, n_snp, k, self.vocab_size)

        return logits1, logits2


if __name__ == "__main__":
    cfg = load_config("config/config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EvoFill(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        k=cfg.data.k_mer 
    ).to(device)

    # 随机单条样本对
    n_var = cfg.model.n_var
    k = cfg.data.k_mer
    seq1 = torch.randint(0, cfg.model.vocab_size, (1, n_var, k), device=device)
    seq2 = torch.randint(0, cfg.model.vocab_size, (1, n_var, k), device=device)
    mask1 = torch.randint(0, 2, (1, n_var, k), device=device).float()
    mask2 = torch.randint(0, 2, (1, n_var, k), device=device).float()
    pdis = 0.042

    model.eval()
    with torch.no_grad():
        logits1, logits2 = model(seq1, seq2, mask1, mask2, pdis)
        print("logits1 shape:", logits1.shape)  # (1, n_var, k, vocab_size)
        print("logits2 shape:", logits2.shape)

    if device.type == "cuda":
        print(torch.cuda.memory_summary())

# logits1 shape: torch.Size([1, 500, 64, 5])
# logits2 shape: torch.Size([1, 500, 64, 5])
# |===========================================================================|
# |                  PyTorch CUDA memory summary, device ID 0                 |
# |---------------------------------------------------------------------------|
# |            CUDA OOMs: 1            |        cudaMalloc retries: 1         |
# |===========================================================================|
# |        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
# |---------------------------------------------------------------------------|
# | Allocated memory      | 682537 KiB |  65430 MiB |   3115 GiB |   3114 GiB |
# |       from large pool | 481280 KiB |  65235 MiB |   3114 GiB |   3114 GiB |
# |       from small pool | 201257 KiB |    196 MiB |      0 GiB |      0 GiB |
# |---------------------------------------------------------------------------|
# | Active memory         | 682537 KiB |  65430 MiB |   3115 GiB |   3114 GiB |
# |       from large pool | 481280 KiB |  65235 MiB |   3114 GiB |   3114 GiB |
# |       from small pool | 201257 KiB |    196 MiB |      0 GiB |      0 GiB |
# |---------------------------------------------------------------------------|
# | Requested memory      | 682012 KiB |  65430 MiB |   3115 GiB |   3114 GiB |
# |       from large pool | 480768 KiB |  65235 MiB |   3114 GiB |   3114 GiB |
# |       from small pool | 201244 KiB |    196 MiB |      0 GiB |      0 GiB |
# |---------------------------------------------------------------------------|
# | GPU reserved memory   |  66984 MiB |  66984 MiB |  67132 MiB | 151552 KiB |
# |       from large pool |  66786 MiB |  66786 MiB |  66934 MiB | 151552 KiB |
# |       from small pool |    198 MiB |    198 MiB |    198 MiB |      0 KiB |
# |---------------------------------------------------------------------------|
# | Non-releasable memory |   7639 KiB |   2381 MiB | 112073 MiB | 112065 MiB |
# |       from large pool |   6144 KiB |   2381 MiB | 111628 MiB | 111622 MiB |
# |       from small pool |   1495 KiB |      2 MiB |    444 MiB |    442 MiB |
# |---------------------------------------------------------------------------|
# | Allocations           |     502    |     517    |   33598    |   33096    |
# |       from large pool |       6    |      20    |    1544    |    1538    |
# |       from small pool |     496    |     499    |   32054    |   31558    |
# |---------------------------------------------------------------------------|
# | Active allocs         |     502    |     517    |   33598    |   33096    |
# |       from large pool |       6    |      20    |    1544    |    1538    |
# |       from small pool |     496    |     499    |   32054    |   31558    |
# |---------------------------------------------------------------------------|
# | GPU reserved segments |     113    |     113    |     118    |       5    |
# |       from large pool |      14    |      14    |      19    |       5    |
# |       from small pool |      99    |      99    |      99    |       0    |
# |---------------------------------------------------------------------------|
# | Non-releasable allocs |       8    |      16    |   16648    |   16640    |
# |       from large pool |       4    |      11    |     848    |     844    |
# |       from small pool |       4    |       6    |   15800    |   15796    |
# |---------------------------------------------------------------------------|
# | Oversize allocations  |       0    |       0    |       0    |       0    |
# |---------------------------------------------------------------------------|
# | Oversize GPU segments |       0    |       0    |       0    |       0    |
# |===========================================================================|