
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return total_loss, None

if __name__ == "__main__":
    B, L = 12, 1000
    logits = torch.randn(B, L, 3).cuda()
    targets = torch.randint(0, 3, (B, L)).cuda()
    y_true_oh = F.one_hot(targets.long(), num_classes=3).float().cuda()
    # targets[0, :10] = -1   # 模拟缺失

    loss_fn = ImputationLoss(use_r2=True)
    loss, logs= loss_fn(logits, y_true_oh)
    print(loss.item())
    print(logs)