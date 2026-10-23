from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = (y * torch.log(xs_pos.clamp(min=self.eps, max=1 - self.eps))
                + (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1 - self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * y + xs_neg * (1 - y)
            loss = loss * torch.pow(1 - pt, self.gamma_pos * y + self.gamma_neg * (1 - y))
        return -loss.sum() / x.shape[0]
