import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImputationLoss(nn.Module):
    """
    基因组缺失填充自定义损失
    维度约定：
        BATCH_SIZE  : B
        N_LOCUS     : L   （位点数）
        N_CLASS     : C   （变异类型数，不含缺失）
        N_CLASS_FULL: C(+1) （可能含缺失）
    """

    def __init__(self,
                 use_r2: bool = True,
                 use_evo: bool = False,
                 r2_weight: float = 0.1,
                 evo_weight: float = 10,
                 evo_lambda: float = 2.0):
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
        
        # 计算有效位点数（非缺失的位点数量）
        nonmiss_mask = (lbl != N_CLASS)
        valid_loci_count = nonmiss_mask.sum()
        
        # 按有效位点数平均交叉熵损失
        if valid_loci_count > 0:
            ce_loss = ce_loss / valid_loci_count
        
        # 2. R² 损失
        r2_loss = torch.zeros((), device=DEVICE)
        if self.use_r2_loss:
            GROUP_SIZE = 4
            N_FULL_GROUPS = BATCH_SIZE // GROUP_SIZE
            if N_FULL_GROUPS > 0:
                # 重塑成 (G, G_SIZE, L, C+1) 和 (G, G_SIZE, L, C)
                y_true_grp = y_true[:N_FULL_GROUPS * GROUP_SIZE].view(
                    N_FULL_GROUPS, GROUP_SIZE, N_LOCUS, N_CLASS_FULL)
                prob_grp = prob[:N_FULL_GROUPS * GROUP_SIZE].view(
                    N_FULL_GROUPS, GROUP_SIZE, N_LOCUS, N_CLASS)
                r2_sum = 0.
                total_valid_r2_loci = 0
                for g in range(N_FULL_GROUPS):
                    # 1) 非缺失掩码 (GROUP_SIZE, N_LOCUS)
                    nonmiss_mask = (y_true_grp[g].argmax(-1) != N_CLASS)
                    # 2) 该位点非缺失样本数  (N_LOCUS,)
                    nonmiss_cnt = nonmiss_mask.sum(0).float()
                    # 3) ALT 计数  (N_LOCUS,)
                    alt_cnt = y_true_grp[g, :, :, 1:N_CLASS].sum((0, 2))
                    # 4) 真实 ALT 频率
                    af = alt_cnt / nonmiss_cnt.clamp_min(1)
                    # 5) 预测 ALT 剂量
                    pred_alt = prob_grp[g, :, :, 1:N_CLASS].sum(-1)  # (G_SIZE, L)
                    pred_alt = pred_alt.mean(0)                      # (L,)
                    # 6) 计算 R²
                    group_r2 = -self._calculate_minimac_r2(
                        pred_alt, af, ~nonmiss_mask)  # (L,)
                    # 统计该group中有效的R²计算位点（非缺失且非0的位点）
                    group_valid_mask = (nonmiss_mask.sum(0) > 0) & (group_r2 != 0)
                    group_valid_count = group_valid_mask.sum()
                    # 累加有效的R²损失和位点数
                    r2_sum += group_r2.sum()
                    total_valid_r2_loci += group_valid_count

                # 按有效位点数平均R²损失
                if total_valid_r2_loci > 0:
                    r2_loss = self.r2_weight * (r2_sum / total_valid_r2_loci)
                else:
                    r2_loss = self.r2_weight * r2_sum

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
                    js[i, j] = (self._js_div(prob[i], prob[j]) * msk).sum()

            w = torch.exp(-self.evo_lambda * y_evo_mat)
            w = w / (w.sum(1, keepdim=True) + self.eps)
            evo_loss_sum = (w * js).sum()

            evo_loss = self.evo_weight * (evo_loss_sum / BATCH_SIZE)
            
            if not torch.isfinite(evo_loss):
                evo_loss = torch.tensor(0., device=evo_loss.device)

        logs = {'ce': ce_loss.item(),
                'r2': r2_loss.item(),
                'evo': evo_loss.item()}
        total_loss = 1000*(ce_loss + r2_loss + evo_loss)
        return total_loss, logs

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

    loss_fn = ImputationLoss(use_r2=True, use_evo=True)
    
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