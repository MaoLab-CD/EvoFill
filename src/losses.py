import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# class GradNormLoss(nn.Module):
#     """
#     GradNorm: Gradient Normalization for Adaptive Loss Balancing
#     参考原始论文 Chen et al. 2018 实现，适配 2 任务（CE + R²）
#     """
#     def __init__(self, num_tasks=2, alpha=1.5, lr_w=1e-3, eps=1e-8):
#         super().__init__()
#         self.num_tasks = num_tasks
#         self.alpha   = alpha          # 恢复速度偏好，论文默认 1.5
#         self.lr_w    = lr_w           # 权重学习率，比模型 lr 小 1~2 量级
#         self.eps     = eps
#         self.w       = nn.Parameter(torch.ones(num_tasks))   # 可训练权重
#         self.register_buffer('L0', torch.zeros(num_tasks))   # 初始损失
#         self.initialized = False

#     def forward(self, losses: torch.Tensor):
#         # losses: [ce, r2]  已经 detach-free
#         if not self.initialized:
#             self.L0 = losses.detach().clone()
#             self.initialized = True

#         self.L_t = losses
#         weighted = self.w * losses          # w_i * L_i
#         return weighted.sum()               # 返回给主优化器

#     def gradnorm_step(self, shared_params, retain_graph=False):
#         """
#         在 model.loss_backward() 之后、optimizer.step() 之前调用一次
#         shared_params:  ***共享部分*** 的参数（例如 encoder 最后一层）
#         """
#         if not self.initialized:
#             return

#         # 1. 清零 w 的 grad
#         if self.w.grad is not None:
#             self.w.grad.zero_()

#         # 2. 计算每个任务对 shared 的梯度范数  G_i(t)
#         G_t = []
#         for i in range(self.num_tasks):
#             g = grad(self.L_t[i], shared_params, retain_graph=retain_graph,
#                      create_graph=True)[0]          # 返回 tuple
#             G_t.append(torch.norm(g * self.w[i]) + self.eps)
#         G_t = torch.stack(G_t)                      # [T]

#         # 3. 相对逆训练速率  r_i(t)
#         tilde_L_t = (self.L_t / self.L0).detach()
#         r_t       = tilde_L_t / tilde_L_t.mean()

#         # 4. 期望梯度范数
#         bar_G_t = G_t.mean()

#         # 5. GradNorm 损失：L_grad = sum|G_i(t) - bar_G_t * r_i(t)^α|
#         l_grad = F.l1_loss(G_t, bar_G_t * (r_t ** self.alpha))

#         # 6. 只更新 w
#         self.w.grad = torch.autograd.grad(l_grad, self.w)[0]
#         with torch.no_grad():
#             new_w = self.w - self.lr_w * self.w.grad
#             new_w = new_w * (self.num_tasks / new_w.sum())
#             self.w.data = new_w  # ✅ 替换 copy_

# class DiceLoss(nn.Module):
#     def __init__(self, n_classes, softmax=True):
#         super().__init__()
#         self.n_classes = n_classes
#         self.softmax = softmax

#     def forward(self, logits, target):
#         """
#         logits: (B, L, C)
#         target: (B, L)  0..C-1
#         """
#         # 1. 概率
#         if self.softmax:
#             probs = F.softmax(logits, dim=-1)        # (B, L, C)
#         else:
#             probs = logits

#         # 2. one-hot  (B, L, C)
#         target_onehot = F.one_hot(target.long(), self.n_classes).float()

#         # 3. 按类别求和 → (C,)
#         intersection = (probs * target_onehot).sum(dim=(0, 1))
#         cardinality  = (probs + target_onehot).sum(dim=(0, 1))

#         # 4. dice 分数 (C,)  → 1 - 平均 dice
#         dice = (2. * intersection + 1e-8) / (cardinality + 1e-8)
#         return 1. - dice.mean()

# class FocalLoss(nn.Module):
#     """
#     多类 Focal Loss，支持 ignore_index = -1
#     """
#     def __init__(self,
#                  n_classes,
#                  gamma=2.0,
#                  alpha=0.25,          # 可以是 float、list 或 None
#                  ignore_index=-1,
#                  reduction='mean'):
#         super().__init__()
#         self.n_classes = n_classes
#         self.gamma = gamma
#         self.alpha = alpha
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#         # 把 alpha 转成 tensor
#         if alpha is None:
#             self.alpha_t = None
#         else:
#             if isinstance(alpha, (float, int)):
#                 self.alpha_t = torch.tensor([alpha, 1 - alpha]).view(-1)
#             else:
#                 self.alpha_t = torch.tensor(alpha).float().view(-1)
#             if self.alpha_t.shape[0] != n_classes:
#                 raise ValueError('alpha length must match n_classes')

#     def forward(self, logits, target):
#         """
#         logits: (B, L, C)
#         target: (B, L)  含 -1 或 0
#         """
#         # 1. 计算每个位置的 CE
#         ce = F.cross_entropy(
#             logits.view(-1, self.n_classes),
#             target.view(-1),
#             ignore_index=self.ignore_index,
#             reduction='none')  # (B*L,)

#         # 2. pt
#         pt = torch.exp(-ce)

#         # 3. focal weight
#         focal_w = (1 - pt) ** self.gamma

#         # 4. alpha weight
#         if self.alpha_t is not None:
#             alpha_t = self.alpha_t.to(target.device).gather(
#                 0, target.view(-1).clamp(min=0))
#             alpha_t = torch.where(target.view(-1) == self.ignore_index,
#                                   torch.zeros_like(alpha_t),
#                                   alpha_t)
#         else:
#             alpha_t = 1.0

#         # 5. 组合
#         loss = alpha_t * focal_w * ce

#         # ✅ 新增：只保留 y_true > 0 的位置
#         mask = target.view(-1) > 0  # (B*L,)
#         loss = torch.where(mask, loss, torch.zeros_like(loss))

#         # 6. 还原 shape 并平均
#         if self.reduction == 'mean':
#             return loss.sum() / mask.sum().clamp(min=1)  # 防止除0
#         elif self.reduction == 'sum':
#             return loss.sum()
#         return loss.view_as(target)

# class R2Loss(nn.Module):
#     def __init__(self, group_size=4, eps=1e-6, ema_decay=0.8):
#         super().__init__()
#         self.group_size = group_size
#         self.eps = eps
#         self.decay = ema_decay

#         # EMA 缓冲区：首次 forward 时根据位点长度 resize
#         self.register_buffer("ema_alt_af", torch.zeros(1))
#         self.register_buffer("ema_cnt", torch.zeros(1))

#     def _calc_r2(self, pred_alt_af, gt_alt_af):
#         denom = gt_alt_af * (1.0 - gt_alt_af)
#         denom = torch.clamp(denom, min=0.01)
#         return (pred_alt_af - gt_alt_af) ** 2 / denom

#     def forward(self, y_pred, y_true, mask_valid):
#         """
#         y_pred : (B, L, 3)
#         y_true : (B, L)  0/1/2 或 -1
#         mask_valid: (B, L)
#         return : scalar  0-1 量级平滑 R²
#         """
#         B, L, _ = y_pred.shape
#         device = y_pred.device

#         # 首次使用或换长度时 resize EMA
#         if self.ema_alt_af.numel() != L:
#             self.ema_alt_af = torch.zeros(L, device=device)
#             self.ema_cnt = torch.zeros(L, device=device)

#         prob = F.softmax(y_pred, dim=-1)
#         alt_prob = prob[..., 1] + prob[..., 2]  # (B, L)

#         G = self.group_size
#         num_full = B // G
#         rem = B % G

#         r2_sum = 0.0
#         w_sum = 0.0

#         def one_group(sl):
#             gt_sl = y_true[sl].float()          # (g, L)
#             msk_sl = mask_valid[sl].float()     # (g, L)
#             alt_sl = alt_prob[sl]               # (g, L)

#             tot = msk_sl.sum(0)                 # (L,)
#             valid = tot > 0                     # 掩码非空位点

#             # 瞬时频率（仅用于 EMA 更新）
#             gt_alt_cnt = (gt_sl * msk_sl).sum(0)
#             inst_alt_af = gt_alt_cnt / (tot + self.eps)

#             # EMA 更新（detach，不反传）
#             with torch.no_grad():
#                 self.ema_alt_af = self.decay * self.ema_alt_af + (1 - self.decay) * inst_alt_af
#                 self.ema_cnt += tot

#             # 真正用于 r2 的目标：EMA 软频率
#             gt_alt_af = self.ema_alt_af.detach()  # 切断梯度
#             pred_alt_af = (alt_sl * msk_sl).sum(0) / (tot + self.eps)

#             # 单态 & 极小方差保护
#             mask_mono = (gt_alt_af == 0.0) | (gt_alt_af == 1.0)
#             r2 = self._calc_r2(pred_alt_af, gt_alt_af)
#             r2 = torch.where(mask_mono, torch.zeros_like(r2), r2)

#             # 只保留有效位点
#             r2 = r2[valid]
#             tot = tot[valid]

#             r2_mean = (r2 * tot).sum() / (tot.sum() + self.eps)
#             weight = tot.sum()
#             return r2_mean * weight, weight

#         for g in range(num_full):
#             r, w = one_group(slice(g * G, (g + 1) * G))
#             r2_sum += r
#             w_sum += w

#         if rem:
#             r, w = one_group(slice(num_full * G, B))
#             r2_sum += r
#             w_sum += w

#         # 加权平均 → 0-1
#         return 1 - r2_sum / (w_sum + self.eps) / L

# class ImputationLoss(nn.Module):
#     def __init__(self,
#                  use_r2=True,
#                  use_focal=True,
#                  group_size=16,
#                  gamma=2.0,
#                  alpha=None,
#                  eps=1e-6,
#                  use_gradnorm=False,     # <-- 新增开关
#                  gn_alpha=1.5,           # GradNorm 超参
#                  gn_lr_w=1e-3,
#                  ):
#         super().__init__()
#         self.use_r2   = use_r2
#         self.use_focal= use_focal
#         self.eps      = eps
#         self.r2       = R2Loss(group_size, eps) if use_r2 else None
#         self.focal    = FocalLoss(n_classes=3, gamma=gamma, alpha=alpha, ignore_index=-1) if use_focal else None

#         # ===== GradNorm =====
#         self.use_gradnorm = use_gradnorm
#         num_tasks = 0
#         if use_focal: num_tasks += 1
#         if use_r2:    num_tasks += 1
#         # 至少要有两个任务才用 GradNorm
#         self.gradnorm = GradNormLoss(num_tasks=num_tasks,
#                                      alpha=gn_alpha,
#                                      lr_w=gn_lr_w) if use_gradnorm and num_tasks >= 2 else None

#     def forward(self, logits, targets, shared_params=None, retain_graph=False):
#         """
#         对外接口保持不变，只新增两个可选关键字：
#         shared_params : 用于 GradNorm 的共享参数（例如 encoder 最后一层）
#         retain_graph:  是否保留计算图（训练时 True）
#         返回值依旧是 (loss, logs)
#         """
#         mask = targets != -1
#         ce   = F.cross_entropy(logits.view(-1, 3), targets.view(-1),
#                                ignore_index=-1, reduction='mean')
#         focal= self.focal(logits, targets) if self.use_focal else 0.
#         r2   = self.r2(logits, targets, mask) if self.use_r2 else 0.

#         # 1. 原始独立损失列表
#         task_losses = []
#         if self.use_focal: task_losses.append(focal)
#         if self.use_r2:    task_losses.append(r2)
#         task_losses = torch.stack(task_losses)          # [T]

#         # 2. 无 GradNorm 时直接相加
#         if self.gradnorm is None:
#             loss = ce + focal + r2
#         else:
#             # 2.1 先算加权损失（只用于反向，不改原始值）
#             loss = ce + self.gradnorm(task_losses)      # GradNorm 返回加权后的 focal+r2
#             # 2.2 更新 GradNorm 自己的权重 w
#             if shared_params is not None:
#                 self.gradnorm.gradnorm_step(shared_params, retain_graph=retain_graph)

#         logs = {'ce': ce.detach(),
#                 'focal': focal.detach() if isinstance(focal, torch.Tensor) else torch.tensor(focal),
#                 'r2': r2.detach() if isinstance(r2, torch.Tensor) else torch.tensor(r2)}
#         return loss, logs

class ImputationLoss(nn.Module):
    """Custom loss function for genomic imputation"""

    def __init__(self, use_r2=True):
        super().__init__()
        self.use_r2_loss = use_r2
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.kl_loss = nn.KLDivLoss(reduction='sum')

    def calculate_minimac_r2(self, pred_alt_allele_probs, gt_alt_af):
        """Calculate Minimac-style RÂ² metric"""
        mask = torch.logical_or(torch.eq(gt_alt_af, 0.0), torch.eq(gt_alt_af, 1.0))
        gt_alt_af = torch.where(mask, 0.5, gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, 0.01, denom)
        r2 = torch.mean(torch.square(pred_alt_allele_probs - gt_alt_af), dim=0) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2

    def forward(self, y_pred, y_true):
        y_true = y_true.float()

        # Convert to proper format for losses
        y_true_ce = torch.argmax(y_true, dim=-1)  # For CrossEntropy
        y_pred_log = torch.log(y_pred + 1e-8)  # For KL divergence

        # Basic losses
        ce_loss = self.ce_loss(y_pred.view(-1, y_pred.size(-1)), y_true_ce.view(-1))
        kl_loss = self.kl_loss(y_pred_log.view(-1, y_pred.size(-1)),
                               y_true.view(-1, y_true.size(-1)))
        total_loss = ce_loss + kl_loss

        if self.use_r2_loss:
            batch_size = y_true.size(0)
            group_size = 4
            num_full_groups = batch_size // group_size

            if num_full_groups > 0:
                y_true_grouped = y_true[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *y_true.shape[1:])
                y_pred_grouped = y_pred[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *y_pred.shape[1:])

                r2_loss = 0.0
                for i in range(num_full_groups):
                    gt_alt_af = torch.count_nonzero(
                        torch.argmax(y_true_grouped[i], dim=-1), dim=0
                    ).float() / group_size

                    pred_alt_allele_probs = torch.sum(y_pred_grouped[i][:, :, 1:], dim=-1)
                    r2_loss += -torch.sum(self.calculate_minimac_r2(
                        pred_alt_allele_probs, gt_alt_af)) * group_size

                total_loss += r2_loss

        logs = {'ce': ce_loss.detach(),
                'kl': kl_loss.detach(),
                'r2': r2_loss.detach() if self.use_r2_loss else 0}
        return total_loss, logs

if __name__ == "__main__":
    B, L = 12, 1000
    logits = torch.randn(B, L, 3).cuda()
    targets = torch.randint(0, 3, (B, L)).cuda()
    y_true_oh = F.one_hot(targets.long(), num_classes=3).float().cuda()
    # targets[0, :10] = -1   # 模拟缺失

    loss_fn = ImputationLoss(use_r2=True)
    loss = loss_fn(logits, y_true_oh)
    print(loss.item())          # 应在 0.x ~ 2.x 之间