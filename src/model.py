# model.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # 官方实现
from src.losses import ImputationLoss

class MambaImputer(nn.Module):
    """
    基于官方 Mamba2 的极简缺失填充模型。
    输入: (B, L)  long    元素 ∈ -1 … n_cats-1，-1 表示缺失
    输出: (B, L, n_cats)  对数几率
    """
    def __init__(self,
                 n_cats: int,
                 d_model: int = 512,
                 d_state: int = 128,
                 d_conv: int  = 4,
                 n_layers: int = 6,
                 **mamba_kwargs):
        super().__init__()
        self.n_cats   = n_cats
        self.ignore_idx = -100          # nn.CrossEntropyLoss 默认忽略值
        self.emb      = nn.Embedding(n_cats + 1, d_model, padding_idx=n_cats)  # 额外一行作为缺失嵌入
        self.layers   = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, **mamba_kwargs)
            for _ in range(n_layers)
        ])
        self.head     = nn.Linear(d_model, n_cats)
        self.loss_fn  = ImputationLoss(n_cats=n_cats, ignore_index=self.ignore_idx, group_size=4, use_r2=True)

    def forward(self, x, labels=None):
        """
        x: (B, L)  long
        labels: 与 x 同形状，缺失处可填 -1 或 -100，loss 会忽略
        return:
            logits: (B, L, n_cats)
            loss:   标量（仅当 labels 不为 None）
        """
        # 将 -1 映射到 n_cats，作为缺失专用索引
        x = x.masked_fill(x == -1, self.n_cats)
        h = self.emb(x)                # (B, L, d_model)

        for layer in self.layers:
            h = layer(h)               # (B, L, d_model)

        logits = self.head(h)          # (B, L, n_cats)

        if labels is not None:
            # 把 -1 转成 -100 以适配 CrossEntropyLoss
            labels = labels.masked_fill(labels == -1, self.ignore_idx)
            loss = self.loss_fn(logits.view(-1, self.n_cats), labels.view(-1))
            return logits, loss
        return logits


# ---------- 快速测试 ----------
if __name__ == "__main__":
    import torch
    device = torch.device("cuda", index=torch.cuda.current_device())

    B, L, n_cats = 2, 1024, 15
    model = MambaImputer(n_cats=n_cats).to(device)   # 1. 模型先上 GPU

    seq   = torch.randint(-1, n_cats, (B, L)).to(device)   # 2. 数据也上 GPU
    labels = seq.clone()
    logits, loss = model(seq, labels)
    print("logits shape:", logits.shape)
    print("loss item  :", loss.item())