import torch
import torch.nn.functional as F
import numpy as np


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, y, gt):
        B, C, H, W = y.shape
        y_flat = y.reshape(B, C, -1)
        gt_flat = gt.reshape(B, C, -1)
        y_norm = torch.norm(y_flat, 2, dim=1)
        gt_norm = torch.norm(gt_flat, 2, dim=1)
        numerator = torch.sum(gt_flat*y_flat, dim=1)
        denominator = y_norm * gt_norm
        sam = torch.div(numerator, denominator + 1e-5)
        sam = torch.sum(torch.acos(sam)) / (B * H * W) * 180 / 3.14159
        return sam

