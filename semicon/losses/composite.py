from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).pow(2) + eps**2).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """1 - SSIM, averaged over the batch."""
    channels = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, pred.device, pred.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return 1.0 - ssim_map.mean()


def _sobel_kernels(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return gx, gy


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gx, gy = _sobel_kernels(pred.device, pred.dtype)
    channels = pred.shape[1]
    gx = gx.repeat(channels, 1, 1, 1)
    gy = gy.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(pred, gx, padding=1, groups=channels)
    pred_y = F.conv2d(pred, gy, padding=1, groups=channels)
    tgt_x = F.conv2d(target, gx, padding=1, groups=channels)
    tgt_y = F.conv2d(target, gy, padding=1, groups=channels)
    return F.l1_loss(pred_x, tgt_x) + F.l1_loss(pred_y, tgt_y)


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.log1p(torch.abs(torch.fft.rfft2(pred, norm="ortho")))
    tgt_mag = torch.log1p(torch.abs(torch.fft.rfft2(target, norm="ortho")))
    return F.l1_loss(pred_mag, tgt_mag)


class CompositeRestorationLoss(nn.Module):
    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 1.0,
        edge_weight: float = 0.1,
        fft_weight: float = 0.01,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l_char = charbonnier_loss(pred, target, self.charbonnier_eps)
        l_ssim = ssim_loss(pred, target)
        l_edge = edge_loss(pred, target)
        l_fft = fft_loss(pred, target)
        total = (
            self.charbonnier_weight * l_char
            + self.ssim_weight * l_ssim
            + self.edge_weight * l_edge
            + self.fft_weight * l_fft
        )
        return {
            "loss": total,
            "charbonnier": l_char.detach(),
            "ssim": l_ssim.detach(),
            "edge": l_edge.detach(),
            "fft": l_fft.detach(),
        }


def build_loss(cfg: dict[str, Any]) -> CompositeRestorationLoss:
    loss_cfg = cfg["loss"]
    return CompositeRestorationLoss(
        charbonnier_weight=float(loss_cfg["charbonnier_weight"]),
        ssim_weight=float(loss_cfg["ssim_weight"]),
        edge_weight=float(loss_cfg["edge_weight"]),
        fft_weight=float(loss_cfg["fft_weight"]),
        charbonnier_eps=float(loss_cfg["charbonnier_eps"]),
    )
