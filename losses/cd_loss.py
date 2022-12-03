import torch
import torch.nn as nn
from metric.loss import RkdDistance, RKdAngle

class CDLoss(nn.Module):
    """Channel Distillation Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        rkdloss = 0.
        for s, t in zip(stu_features, tea_features):
            dist_criterion = RkdDistance()
            angle_criterion = RKdAngle()
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            dist_loss = 1 * dist_criterion(s, t)
            angle_loss = 2 * angle_criterion(s, t)
            D_RKD = dist_loss + angle_loss
            loss += torch.mean(torch.pow(s - t, 2))
            rkdloss += D_RKD
        return loss, rkdloss
