# model.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.utils import load_config
# from src.layers import xxxx

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm(x + self.mha(x, x, x, key_padding_mask=mask)[0])
        x = self.norm(x + self.dropout(self.ff(x)))
        return x


class EvoFill(nn.Module):
    """
    Autoregressive genomic infilling model.
    Input : (B, N, K)  token indices
    Output: (B, N, K, vocab_size) logits
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, k):
        super().__init__()
        self.k = k
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, N, K = x.shape
        x_flat = x.view(B, N * K)                       # (B, N*K)
        if mask is not None:
            mask = mask.view(B, N * K)                  # (B, N*K)
        h = self.token_embed(x_flat)                    # (B, N*K, d)
        for blk in self.blocks:
            if self.use_checkpoint:
                h = checkpoint(blk, h, mask, use_reentrant=False)
            else:
                h = blk(h, mask)
        logits = self.head(h)                           # (B, N*K, V)
        return logits.view(B, N, K, -1)                 # (B, N, K, V)

# ------------------------------------------------------------------
# 本地测试（读取 config.json）+ 模拟训练显存
# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch.cuda as cuda
    from torch.amp import autocast

    # 1. 读取配置
    cfg = load_config("config/config.json")

    # 2. 使用配置中的参数
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model = EvoFill(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        k=cfg.k_mer
    ).to(device)

    # 3. 构造随机数据（与 train.py 保持一致）
    B = 8
    N = cfg.n_snp
    K = cfg.k_mer
    x = torch.randint(0, cfg.vocab_size, (B, N, K), device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)          # 无 mask，全部参与

    # 4. 构造 optimizer（Adam 会额外占 2×参数量）
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 5. 前向 + 反向一次
    model.train()
    optimizer.zero_grad()
    with autocast("cuda",enabled=False):          # 需要 fp32 就关掉 amp
        logits = model(x, mask=mask)       # (B, N, K, V)
        # 随便造一个 label，保证能算 loss
        label = torch.randint(0, cfg.vocab_size, (B, N), device=device)
        # 把 label 扩展成 (B, N, K) 再展平，与 logits 对齐
        label_flat = label.unsqueeze(-1).expand(-1, -1, K).reshape(-1)
        logits_flat = logits.reshape(-1, cfg.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits_flat, label_flat)

    loss.backward()
    optimizer.step()

    # 6. 打印信息
    print("Input  shape :", x.shape)        # (B, N, K)
    print("Output shape :", logits.shape)   # (B, N, K, vocab_size)
    print("Loss :", loss.item())

    if device.type == "cuda":
        # 注意：nvidia-smi 看的是“已分配”，这里用 peak 更接近真实最大占用
        mb = cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"Peak GPU memory allocated: {mb:.2f} MB")
        cuda.reset_peak_memory_stats(device)   # 清掉计数器，方便下次测试