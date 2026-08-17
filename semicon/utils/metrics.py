from __future__ import annotations

import torch
from torch.nn import functional as F

from semicon.losses.composite import ssim_loss


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse <= 0:
        return pred.new_tensor(100.0)
    return 10.0 * torch.log10(pred.new_tensor(data_range**2) / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim_loss(pred, target)
