"""
EvoFill – 加入遗传距离的 STICI 改进版
依赖: flash-attn>=2.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from src.utils import load_config

# --------------------------------------------------
# 基础组件
# --------------------------------------------------
class FlashMultiHeadAttention(nn.Module):
    """
    序列维度 Flash-Attention；对 (batch, seqlen, dim) 做自注意力
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        # x: (B, L, D)
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, Dh)
        out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=False)
        out = out.reshape(B, L, -1)
        return self.proj(out)


class SampleFlashAttention(nn.Module):
    """
    样本维度 Flash-Attention：在样本维度做注意力，遗传距离作为偏置。
    使用手动实现的注意力来处理距离偏置。
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, dist_bias):
        """
        x: (N, L, D)          N = n_samples, L = seqlen
        dist_bias: (N, N)    遗传距离矩阵，作为注意力偏置
        """
        N, L, D = x.shape
        
        # 确保 dist_bias 与 x 的数据类型一致
        dist_bias = dist_bias.to(x.dtype)
        
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
        
        # 添加距离偏置，扩展维度以匹配注意力分数
        dist_bias = dist_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        dist_bias = dist_bias.expand(self.num_heads, L, -1, -1)  # (H, L, N, N)
        
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


class FFN(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

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

    def forward(self, x, dist_mat):
        """
        x: (N, L, D)
        dist_mat: (N, N)
        """
        # 确保 dist_mat 与 x 的数据类型一致
        dist_mat = dist_mat.to(x.dtype)
        
        # 1) 序列维度自注意力
        x = x + self.seq_attn(self.norm1(x))
        # 2) 样本维度交叉注意力（遗传距离作为偏置）
        x = x + self.sample_attn(self.norm2(x), dist_mat)
        # 3) FFN
        x = x + self.ffn(self.norm3(x))
        return x


# --------------------------------------------------
# 主模型
# --------------------------------------------------
class EvoFill(nn.Module):
    def __init__(self,
                 depth: int = 3,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.depth = depth  # 有效类别数（不包含 missing）
        self.embed = nn.Linear(depth + 1, embed_dim, bias=False)  # 输入维度 depth+1（包含 missing）
        self.blocks = nn.ModuleList([
            FlashTransformerBlock(embed_dim, num_heads, dropout) for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, depth)  # 输出 depth 个类别（不包含 missing）
        )
        
    def forward(self, x: torch.Tensor, dist_mat: torch.Tensor):
        """
        x: (N, L) 整型矩阵，数值范围 0-depth (depth 代表 missing)
        dist_mat: (N, N) 遗传距离矩阵
        return: (N, L, depth) 概率矩阵（不包含 missing）
        """

        # 创建 one-hot 编码（使用原始输入，包含 missing）
        if x.dtype != torch.long:
            x = x.long()
        x_one_hot = F.one_hot(x, num_classes=self.depth + 1).float()
        
        # 确保数据类型一致
        weight_dtype = next(self.parameters()).dtype
        if x_one_hot.dtype != weight_dtype:
            x_one_hot = x_one_hot.to(weight_dtype)
        
        # 嵌入层
        x_emb = self.embed(x_one_hot)  # (N, L, embed_dim)
        
        # 通过所有 transformer block
        for blk in self.blocks:
            x_emb = blk(x_emb, dist_mat)
        
        # 输出投影到 depth 维度（不包含 missing）
        logits = self.out_proj(x_emb)  # (N, L, depth)
        
        # 对于 missing 位置，我们可以选择保持预测或者进行特殊处理
        # 这里直接返回 softmax 结果
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, dist_mat: torch.Tensor):
        """
        预测方法：返回最可能的类别
        """
        probs = self.forward(x, dist_mat)
        return torch.argmax(probs, dim=-1)


# --------------------------------------------------
# 本地测试
# --------------------------------------------------
if __name__ == "__main__":
    import torch.cuda as cuda
    from torch.amp import autocast
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    cfg = load_config('config/config.json')
    depth = 3  # 有效类别数

    model = EvoFill(depth=depth,
                    embed_dim=cfg.model.embed_dim,
                    num_heads=cfg.model.n_heads,
                    n_layers=cfg.model.n_layers,
                    dropout=cfg.model.dropout).to(device)

    B, L = cfg.train.batch_size, cfg.train.n_var
    # 输入范围：0-3 (3 代表 missing)
    x = torch.randint(0, depth + 1, (B, L), device=device)
    dist = torch.rand(B, B, device=device)

    with autocast("cuda", enabled=True):
        out = model(x, dist)
        print("Input :", x.shape)      # (B, L)
        print("Input range:", x.min().item(), "-", x.max().item())  # 0-3
        print("Dist  :", dist.shape)   # (B, B)
        print("Output:", out.shape)    # (B, L, depth) - 不包含 missing
        print("Output range:", out.min().item(), "-", out.max().item())  # 概率值

    print(f"Peak GPU memory: {cuda.max_memory_allocated(device)/1024**2:.2f} MB")
    print(f"Output is probability distribution: {torch.allclose(out.sum(dim=-1), torch.ones(B, L, device=device), atol=1e-6)}")