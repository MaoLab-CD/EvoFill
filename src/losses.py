import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# | 特性                | `tf.keras.losses.KLDivergence` | `nn.KLDivLoss`                        |
# | ----------------- | ------------------------------ | ------------------------------------- |
# | 输入格式              | 概率                             | 输入：log-probabilities，目标：probabilities |
# | 是否需手动取 log        | ❌                              | ✅                                     |
# | 是否自动裁剪输入          | ✅                              | ❌                                     |
# | 默认归约方式            | `sum_over_batch_size`          | `batchmean`                           |
# | 是否支持 `log_target` | ❌                              | ✅（可选）                                 |


class GradNormLoss(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    参考原始论文 Chen et al. 2018 实现，适配 2 任务（CE + R²）
    """
    def __init__(self, num_tasks=2, alpha=1.5, lr_w=1e-3, eps=1e-8):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha   = alpha          # 恢复速度偏好，论文默认 1.5
        self.lr_w    = lr_w           # 权重学习率，比模型 lr 小 1~2 量级
        self.eps     = eps
        self.w       = nn.Parameter(torch.ones(num_tasks))   # 可训练权重
        self.register_buffer('L0', torch.zeros(num_tasks))   # 初始损失
        self.initialized = False

    def forward(self, losses: torch.Tensor):
        # losses: [ce, r2]  已经 detach-free
        if not self.initialized:
            self.L0 = losses.detach().clone()
            self.initialized = True

        self.L_t = losses
        weighted = self.w * losses          # w_i * L_i
        return weighted.sum()               # 返回给主优化器

    def gradnorm_step(self, shared_params, retain_graph=False):
        """
        在 model.loss_backward() 之后、optimizer.step() 之前调用一次
        shared_params:  ***共享部分*** 的参数（例如 encoder 最后一层）
        """
        if not self.initialized:
            return

        # 1. 清零 w 的 grad
        if self.w.grad is not None:
            self.w.grad.zero_()

        # 2. 计算每个任务对 shared 的梯度范数  G_i(t)
        G_t = []
        for i in range(self.num_tasks):
            g = grad(self.L_t[i], shared_params, retain_graph=True,
                     create_graph=True)[0]          # 返回 tuple
            G_t.append(torch.norm(g * self.w[i]) + self.eps)
        G_t = torch.stack(G_t)                      # [T]

        # 3. 相对逆训练速率  r_i(t)
        tilde_L_t = (self.L_t / self.L0).detach()
        r_t       = tilde_L_t / tilde_L_t.mean()

        # 4. 期望梯度范数
        bar_G_t = G_t.mean()

        # 5. GradNorm 损失：L_grad = sum|G_i(t) - bar_G_t * r_i(t)^α|
        l_grad = F.l1_loss(G_t, bar_G_t * (r_t ** self.alpha))

        # 6. 只更新 w
        self.w.grad = torch.autograd.grad(l_grad, self.w)[0]
        with torch.no_grad():
            new_w = self.w - self.lr_w * self.w.grad
            new_w = new_w * (self.num_tasks / new_w.sum())
            self.w.data = new_w  # ✅ 替换 copy_


class ImputationLoss(nn.Module):
    def __init__(self, use_r2_loss=True, group_size=4, eps=1e-8,
                 use_grad_norm=False, gn_alpha=1.5, gn_lr_w=1e-3):
        super().__init__()
        self.use_r2_loss = use_r2_loss
        self.group_size  = group_size
        self.eps         = eps
        self.use_gn      = use_grad_norm
        if self.use_gn:
            self.gn_loss = GradNormLoss(num_tasks=2, alpha=gn_alpha, lr_w=gn_lr_w)

    # ---------- 工具函数 ---------- #
    def _calc_r2(self, pred_alt_prob: torch.Tensor, gt_alt_af: torch.Tensor):
        mask = ((gt_alt_af == 0.0) | (gt_alt_af == 1.0))
        gt_alt_af = torch.where(mask, 0.5, gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.clamp(denom, min=0.01)
        r2 = ((pred_alt_prob - gt_alt_af) ** 2) / denom
        r2 = torch.where(mask, 0.0, r2)
        return r2

    # ---------- R2 loss（改为 1/log(r2)） ---------- #
    def _r2_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask_valid: torch.Tensor):
        B, V, C = y_pred.shape
        G = self.group_size
        num_full = B // G
        rem = B % G

        prob = F.softmax(y_pred, dim=-1)
        alt_prob = prob[..., 1] + 2.0 * prob[..., 2]

        r2_penalty = 0.0

        def one_group(sl):
            gt_sl   = y_true[sl]                     # (g_size, V)
            mask_sl = mask_valid[sl]                 # (g_size, V)
            alt_sl  = alt_prob[sl]                   # (g_size, V)

            gt_alt_cnt = (gt_sl * mask_sl).sum(dim=0)
            gt_alt_af  = gt_alt_cnt / (mask_sl.sum(dim=0) + self.eps)

            pred_alt_af = (alt_sl * mask_sl).sum(dim=0) / (mask_sl.sum(dim=0) + self.eps)

            r2 = self._calc_r2(pred_alt_af, gt_alt_af)          # (V,)
            return r2.sum() * (sl.stop - sl.start)      # 保持与原来相同的加权方式

        # 完整组
        for g in range(num_full):
            r2_penalty += one_group(slice(g * G, (g + 1) * G))

        # 剩余样本
        if rem:
            r2_penalty += one_group(slice(num_full * G, B))

        return 1.0 / torch.log(r2_penalty + self.eps)

    # ---------- 前向 ---------- #
    def forward(self, y_pred, y_true):
        mask_valid = (y_true != -1)
        y_true_m   = y_true.clone()
        y_true_m[~mask_valid] = 0

        # 1. MCE 改为 mean
        log_p = F.log_softmax(y_pred, dim=-1)
        ce = -log_p.gather(dim=-1, index=y_true_m.long().unsqueeze(-1)).squeeze(-1)
        ce = (ce * mask_valid).sum() / (mask_valid.sum() + self.eps)
        # 2. R²
        r2 = 0.
        if self.use_r2_loss:
            r2 = self._r2_loss(y_pred, y_true, mask_valid)

        # 3. GradNorm 或固定系数
        if self.use_gn:
            losses = torch.stack([ce, 10*r2])
            gn_loss = self.gn_loss(losses)
            # print('ce:',ce,'r2:',r2, 'gn_loss:', gn_loss)
            return gn_loss
        else:
            return ce + 10 * r2
