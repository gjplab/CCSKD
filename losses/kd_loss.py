import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        loss = F.kl_div(s, t, size_average=False) * (self.t**2) / stu_pred.shape[0]
        return loss


class KDLossv2(nn.Module):
    """Guided Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred, label, epoch):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        factor=0.9
        mask = (torch.eq(label, t_argmax).float()+1.0)*(factor ** (epoch // 30))
        count = stu_pred.shape[0]
        mask = mask.unsqueeze(-1)
        correct_s = s.mul(mask)
        correct_t = t.mul(mask)
        loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss