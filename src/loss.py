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
        lbl = y_true.argmax(dim=-1)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                lbl.view(-1),
                                reduction='sum')

        r2_loss = torch.zeros((), device=prob.device)
        if self.use_r2_loss:
            group_size = 4
            num_full_groups = batch_size // group_size
            r2_sum  =0.
            if num_full_groups > 0:
                y_true_grouped = y_true[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *y_true.shape[1:])
                y_pred_grouped = prob[:num_full_groups * group_size].view(
                    num_full_groups, group_size, *prob.shape[1:])
                for i in range(num_full_groups):
                    gt_alt_af = torch.count_nonzero(
                        torch.argmax(y_true_grouped[i], dim=-1), dim=0
                    ).float() / group_size

                    pred_alt_allele_probs = torch.sum(y_pred_grouped[i][:, :, 1:], dim=-1)
                    r2_sum += -torch.sum(self.calculate_minimac_r2(
                        pred_alt_allele_probs, gt_alt_af)) * group_size

                r2_loss = self.r2_weight * r2_sum / batch_size

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
            if not torch.isfinite(evo_loss):
                evo_loss = torch.tensor(0., device=evo_loss.device)

        logs = {'ce':ce_loss.item(),
                'r2':r2_loss.item(),
                'evo':evo_loss.item()}
        total_loss = ce_loss + r2_loss + evo_loss
        return total_loss, logs

class ImputationLoss_Missing(nn.Module):
    """
    基因组缺失填充自定义损失
    维度约定：
        BATCH_SIZE  : B
        N_LOCUS     : L   （位点数）
        N_CLASS     : C   （变异类型数，不含缺失）
        N_CLASS_FULL: C+1 （含缺失）
    """

    def __init__(self,
                 use_r2: bool = True,
                 use_evo: bool = False,
                 r2_weight: float = 1.0,
                 evo_weight: float = 1.0,
                 evo_lambda: float = 1.0):
        super().__init__()
        self.use_r2_loss = use_r2
        self.use_evo_loss = use_evo
        self.r2_weight = r2_weight
        self.evo_weight = evo_weight
        self.evo_lambda = evo_lambda
        self.eps = 1e-8

    # ------------- 私有工具 -------------
    def _calculate_minimac_r2(self,
                              pred_alt_allele_probs: torch.Tensor,
                              true_alt_af: torch.Tensor,
                              miss_mask: torch.Tensor):
        """
        pred_alt_allele_probs: (N_LOCUS,)
        true_alt_af          : (N_LOCUS,)
        miss_mask            : (N_LOCUS,)  True=缺失
        """
        # 缺失位点不参与计算
        true_alt_af = torch.where(miss_mask, 0.5, true_alt_af)
        mask = torch.logical_or(true_alt_af.eq(0.0), true_alt_af.eq(1.0))
        denom = true_alt_af * (1. - true_alt_af)
        denom = denom.clamp_min(0.01)
        r2 = (pred_alt_allele_probs - true_alt_af).square() / denom
        r2 = torch.where(torch.logical_or(mask, miss_mask), 0., r2)
        return r2
    
    def _js_div(self, p: torch.Tensor, q: torch.Tensor):
        """
        Jensen-Shannon 散度
        p, q: (N_LOCUS, N_CLASS)
        return: (N_LOCUS,)
        """
        eps = 1e-7
        p = p.clamp_min(eps)          # 防止真 0
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)
        kl_pm = (p * p.log() - p * m.log()).sum(-1)
        kl_qm = (q * q.log() - q * m.log()).sum(-1)
        return 0.5 * (kl_pm + kl_qm)

    # ------------- 前向 -------------
    def forward(self,
                logits: torch.Tensor,      # (BATCH_SIZE, N_LOCUS, N_CLASS)
                prob: torch.Tensor,        # (BATCH_SIZE, N_LOCUS, N_CLASS)
                y_true: torch.Tensor,      # (BATCH_SIZE, N_LOCUS, N_CLASS_FULL)
                y_evo_mat: torch.Tensor = None):  # (BATCH_SIZE, BATCH_SIZE)
        BATCH_SIZE, N_LOCUS, N_CLASS = prob.shape
        N_CLASS_FULL = y_true.shape[-1]          # = N_CLASS + 1
        DEVICE = prob.device

        # 1. 交叉熵：忽略缺失类
        lbl = y_true.argmax(-1)                  # (BATCH_SIZE, N_LOCUS)
        ce_loss = F.cross_entropy(logits.view(-1, N_CLASS),
                                  lbl.view(-1),
                                  ignore_index=N_CLASS,
                                  reduction='sum')
        
        # 2. R² 损失
        r2_loss = torch.zeros((), device=DEVICE)
        if self.use_r2_loss:
            GROUP_SIZE = 4
            N_FULL_GROUPS = BATCH_SIZE // GROUP_SIZE
            if N_FULL_GROUPS > 0:
                y_true_grp = y_true[:N_FULL_GROUPS * GROUP_SIZE].view(
                    N_FULL_GROUPS, GROUP_SIZE, N_LOCUS, N_CLASS_FULL)
                prob_grp = prob[:N_FULL_GROUPS * GROUP_SIZE].view(
                    N_FULL_GROUPS, GROUP_SIZE, N_LOCUS, N_CLASS)

                r2_sum = 0.
                for g in range(N_FULL_GROUPS):
                    # 非缺失掩码 (N_LOCUS,)
                    nonmiss = (y_true_grp[g].argmax(-1) != N_CLASS).any(0)
                    # 计算 AF（仅非缺失）
                    alt_cnt = y_true_grp[g, :, :, 1:N_CLASS].sum((0, 2))
                    af = alt_cnt / (GROUP_SIZE * nonmiss.sum().float()).clamp_min(1)
                    # 预测 ALT 剂量
                    pred_alt = prob_grp[g, :, :, 1:N_CLASS].sum(-1).mean(0)
                    r2_sum += -self._calculate_minimac_r2(pred_alt, af, ~nonmiss).sum() * GROUP_SIZE
                r2_loss = self.r2_weight * r2_sum / BATCH_SIZE

        # 3. 演化损失
        evo_loss = torch.zeros((), device=DEVICE)
        if self.use_evo_loss and y_evo_mat is not None:
            nonmiss = (y_true.argmax(-1) != N_CLASS)          # (BATCH_SIZE, N_LOCUS)
            same_gt = (lbl.unsqueeze(1) == lbl.unsqueeze(0))  # (BATCH_SIZE, BATCH_SIZE, N_LOCUS)
            mask = nonmiss.unsqueeze(1) & nonmiss.unsqueeze(0) & same_gt

            js = torch.zeros(BATCH_SIZE, BATCH_SIZE, device=DEVICE)
            for i in range(BATCH_SIZE):
                for j in range(BATCH_SIZE):
                    msk = mask[i, j]
                    if msk.sum() == 0:
                        continue
                    js[i, j] = (self._js_div(prob[i], prob[j]) * msk).sum() / msk.sum().clamp_min(1)

            w = torch.exp(-self.evo_lambda * y_evo_mat)
            w = w / (w.sum(1, keepdim=True) + self.eps)
            evo_loss = self.evo_weight * (w * js).sum()
            if not torch.isfinite(evo_loss):
                evo_loss = torch.tensor(0., device=evo_loss.device)

        logs = {'ce': ce_loss.item(),
                'r2': r2_loss.item(),
                'evo': evo_loss.item()}
        total_loss = ce_loss + r2_loss + evo_loss
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
    BATCH_SIZE = 16
    N_LOCUS = 32
    N_CLASS = 4          # 0=REF，1~3=ALT，不含缺失
    DEVICE = 'cpu'       # 无 GPU 也能跑

    # ---------- 1. 构造不带缺失的数据 ----------
    print('=== 测试 1：y_true 不含缺失 ===')
    y_true_clean = torch.zeros(BATCH_SIZE, N_LOCUS, N_CLASS, device=DEVICE)
    # 随机 one-hot
    for b in range(BATCH_SIZE):
        for l in range(N_LOCUS):
            cls = torch.randint(0, N_CLASS, (1,)).item()
            y_true_clean[b, l, cls] = 1.

    logits = torch.randn(BATCH_SIZE, N_LOCUS, N_CLASS, device=DEVICE)
    prob   = logits.softmax(-1)
    y_evo  = torch.rand(BATCH_SIZE, BATCH_SIZE, device=DEVICE)

    # loss_fn = ImputationLoss(use_r2=True, use_evo=True)
    loss_fn = ImputationLoss_Missing(use_r2=True, use_evo=True)
    
    loss, logs = loss_fn(logits, prob, y_true_clean, y_evo)
    print('total_loss =', loss.item())
    print('logs =', logs)

    # ---------- 2. 构造带缺失的数据 ----------
    print('\n=== 测试 2：y_true 含缺失 ===')
    N_CLASS_FULL = N_CLASS + 1
    y_true_missing = torch.zeros(BATCH_SIZE, N_LOCUS, N_CLASS_FULL, device=DEVICE)
    for b in range(BATCH_SIZE):
        for l in range(N_LOCUS):
            cls = torch.randint(0, N_CLASS_FULL, (1,)).item()
            y_true_missing[b, l, cls] = 1.

    # 同样重新生成 logits/prob（形状仍是 N_CLASS）
    logits = torch.randn(BATCH_SIZE, N_LOCUS, N_CLASS, device=DEVICE)
    prob   = logits.softmax(-1)

    loss, logs = loss_fn(logits, prob, y_true_missing, y_evo)
    print('total_loss =', loss.item())
    print('logs =', logs)