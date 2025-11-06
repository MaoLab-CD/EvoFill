import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImputationLoss(nn.Module):
    """Custom loss function for genomic imputation"""

    def __init__(self, use_r2=True, use_evo=False, r2_weight=1, evo_weight=1, evo_lambda=1):
        super().__init__()
        self.use_r2_loss = use_r2
        self.use_evo_loss = use_evo
        self.r2_weight = r2_weight
        self.evo_weight = evo_weight  # 系数
        self.evo_lambda = evo_lambda  # “演化敏感度” = 控制模型对“系统发育距离”变化的敏感程度
        self.eps = 1e-8

    def calculate_minimac_r2(self, pred_alt_allele_probs, gt_alt_af):
        """Calculate Minimac-style RÂ² metric"""
        mask = torch.logical_or(torch.eq(gt_alt_af, 0.0), torch.eq(gt_alt_af, 1.0))
        gt_alt_af = torch.where(mask, 0.5, gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, 0.01, denom)
        r2 = torch.mean(torch.square(pred_alt_allele_probs - gt_alt_af), dim=0) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2

    def js_div(self, p, q, eps=1e-7):
        """Jensen–Shannon 散度  p,q: (..., C)  返回: (..., L)"""
        m = 0.5 * (p + q)
        kl_pm = (p * (p + eps).log() - p * (m + eps).log()).sum(-1)
        kl_qm = (q * (q + eps).log() - q * (m + eps).log()).sum(-1)
        return 0.5 * (kl_pm + kl_qm)          # (..., L)

    def forward(self, logits, prob, y_true, y_evo_mat = None):
        batch_size = y_true.size(0)
        lbl = y_true.argmax(dim=-1)   # 取最大索引
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                lbl.view(-1),
                                reduction='sum')
        total_loss = ce_loss

        r2_loss = torch.zeros((), device=prob.device)
        if self.use_r2_loss:

            group_size = 4
            num_full_groups = batch_size // group_size

            if num_full_groups > 0:
                y_true_grouped = y_true[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *y_true.shape[1:])
                y_pred_grouped = prob[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *prob.shape[1:])

                r2_loss = 0.0
                for i in range(num_full_groups):
                    gt_alt_af = torch.count_nonzero(
                        torch.argmax(y_true_grouped[i], dim=-1), dim=0
                    ).float() / group_size

                    pred_alt_allele_probs = torch.sum(y_pred_grouped[i][:, :, 1:], dim=-1)
                    r2_loss += -torch.sum(self.calculate_minimac_r2(
                        pred_alt_allele_probs, gt_alt_af)) * group_size

                r2_loss = self.r2_weight * r2_loss / batch_size
                total_loss += r2_loss

        evo_loss = torch.zeros((), device=prob.device)
        if self.use_evo_loss and y_evo_mat is not None:
            evo_loss = 0.0
            batchsize = prob.shape[0]
            device = prob.device
            # 相同位点掩码
            mask = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).all(dim=-1).float()  # (B, B, L)
            # norm = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # 避免除 0

            js = torch.zeros(batchsize, batchsize, device=device)
            # 向量化计算 JS（只遍历 B）
            for i in range(batchsize):
                for j in range(batchsize):
                    m_ij = mask[i, j]                  # (L,)
                    if m_ij.sum() == 0: continue
                    js_ij = self.js_div(prob[i], prob[j])   # (L,)
                    js[i, j] = (js_ij * m_ij).sum() #/ norm[i, j]

            # 演化权重
            w = torch.exp(-self.evo_lambda * y_evo_mat)
            w = w / (w.sum(dim=1, keepdim=True) + self.eps)

            evo_loss =  self.evo_weight * (w * js).sum()

            total_loss += evo_loss

        logs = {'ce':ce_loss.item(),
                'r2':r2_loss.item(),
                'evo':evo_loss.item()}

        return total_loss, logs

class GradNormImputationLoss(nn.Module):
    """
    包装你已有的 ImputationLoss，让 r2_weight / evo_weight 自动学。
    注意：ce 权重固定为 1，只学 r2 和 evo 两个标量。
    """
    def __init__(self, base_loss_fn: ImputationLoss, alpha=1.0, lr_w=0.025):
        super().__init__()
        self.base_loss_fn = base_loss_fn          # 复用你原来的 loss
        self.alpha = alpha
        # 可学习向量，w[0]=r2_weight, w[1]=evo_weight
        self.w = nn.Parameter(torch.tensor([base_loss_fn.r2_weight,
                                            base_loss_fn.evo_weight],
                                           dtype=torch.float32))
        self.optimizer_w = torch.optim.Adam([self.w], lr=lr_w)
        self.initial_L = None                     # 首次记录用

    def forward(self, logits, prob, y_true, y_evo_mat,
                shared_params, step):             # 多传一个 step 即可
        # 1. 先算裸 loss（用原来的权重 1，1，1 跑一遍，拿到日志）
        with torch.no_grad():
            _, logs = self.base_loss_fn(logits, prob, y_true, y_evo_mat)
            L_raw = [torch.tensor(logs['ce'], device=logits.device),
                     torch.tensor(logs['r2'], device=logits.device),
                     torch.tensor(logs['evo'], device=logits.device)]

        # 2. 记录初始 loss
        if self.initial_L is None and step == 0:
            self.initial_L = [l.detach() for l in L_raw]

        # 3. 归一化版本（数量级拉到 1 左右）
        L_hat = [L_raw[0] / 2.0e4,
                 L_raw[1] / 1.0e4,
                 L_raw[2] / 8.0e3]

        # 4. GradNorm 更新 w
        ratio = [(l / i) ** self.alpha for l, i in zip(L_raw, self.initial_L)]
        # 梯度范数
        grad_norms = []
        for i in range(3):
            g = torch.autograd.grad(L_hat[i], shared_params,
                                    retain_graph=True, create_graph=True)
            grad_norms.append(torch.cat([gg.flatten() for gg in g]).norm())
        grad_norms = torch.stack(grad_norms)
        target = grad_norms.mean() * torch.tensor(ratio, device=grad_norms.device)
        gradloss = F.l1_loss(grad_norms, target.detach())

        self.optimizer_w.zero_grad()
        gradloss.backward()
        self.optimizer_w.step()
        # 简单 clamp 防止负值
        with torch.no_grad():
            self.w.clamp_min_(0.01)
            self.w /= self.w.sum() / 2.0          # 保持 Σw = 2（ce 权重固定为 1）

        # 5. 用新 w 重新加权总损失
        r2_w, evo_w = self.w[0].item(), self.w[1].item()
        self.base_loss_fn.r2_weight = r2_w        # 热插拔到原 loss
        self.base_loss_fn.evo_weight = evo_w
        total_loss, logs = self.base_loss_fn(logits, prob, y_true, y_evo_mat)
        return total_loss, logs, self.w.data.clone()


if __name__ == "__main__":
    B, L = 12, 1000
    logits = torch.randn(B, L, 3)
    prob = F.softmax(logits, dim=-1)

    targets = torch.randint(0, 3, (B, L))
    y_true_oh = F.one_hot(targets.long(), num_classes=3).float()

    y_evo_mat = torch.rand(B, B)
    y_evo_mat = (y_evo_mat + y_evo_mat.T) / 2
    y_evo_mat.fill_diagonal_(0.)

    loss_fn = ImputationLoss(use_r2=True,use_evo=True)
    loss, logs= loss_fn(logits, prob, y_true_oh, y_evo_mat)
    print(loss.item())
    print(logs)