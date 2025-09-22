import torch
import torch.nn as nn
import torch.nn.functional as F


def minimac_r2_loss(pred_probs, true_gt, group_size=4, eps=1e-6):
    """
    pred_probs : (B, L, n_alleles)  softmax 概率
    true_gt    : (B, L)             0/1/2/... 真实基因型
    group_size : 分块大小
    return     : scalar  负 R²（越小越好）
    """
    B, L = pred_probs.shape[:2]
    n_full = B // group_size
    n_rem  = B % group_size
    loss = 0.0

    # 计算 alt allele 概率（0 为 ref，其余为 alt）
    alt_probs = 1 - pred_probs[..., 0]  # (B, L)

    def _r2(alt_p, gt):
        # gt 0/1/2  alt_af ∈ [0,1]
        alt_af = gt.float() / 2.0
        mask = (alt_af == 0.0) | (alt_af == 1.0)
        alt_af = torch.where(mask, 0.5, alt_af)
        denom = alt_af * (1 - alt_af)
        denom = denom.clamp(min=0.01)
        mse = (alt_p - alt_af).pow(2).mean(0)  # 按位点平均
        r2 = torch.where(mask, 0.0, mse / denom)
        return r2.sum() * alt_p.shape[0]  # 乘回样本数

    # 完整组
    for i in range(n_full):
        sl = slice(i * group_size, (i + 1) * group_size)
        loss -= _r2(alt_probs[sl], true_gt[sl])

    # 剩余样本
    if n_rem:
        sl = slice(n_full * group_size, B)
        loss -= _r2(alt_probs[sl], true_gt[sl])

    return loss


class ImputationLoss(nn.Module):
    def __init__(self, n_cats, ignore_index=-100, group_size=4, use_r2=True):
        super().__init__()
        self.n_cats = n_cats
        self.ignore_index = ignore_index
        self.group_size = group_size
        self.use_r2 = use_r2
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.kl = nn.KLDivLoss(reduction="sum")

    def forward(self, logits, targets):
        """
        logits   : (B*L, n_cats)
        targets  : (B*L,)  long  0/1/2/...  -100 缺失
        """
        # 1. CE
        ce_loss = self.ce(logits, targets)

        # 2. KL  (log_softmax -> softmax)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        kl_loss = self.kl(log_probs, probs)

        total_loss = ce_loss + kl_loss

        # 3. R2 惩罚（仅对非缺失位点）
        if self.use_r2:
            mask = targets != self.ignore_index
            logits_valid = logits[mask]
            targets_valid = targets[mask]
            # reshape 回 (B, L)
            B, L = logits_valid.shape[0], 1
            probs_valid = F.softmax(logits_valid.view(B, L, -1), dim=-1)
            r2_loss = minimac_r2_loss(probs_valid, targets_valid.view(B, L), self.group_size)
            total_loss = total_loss + r2_loss

        return total_loss