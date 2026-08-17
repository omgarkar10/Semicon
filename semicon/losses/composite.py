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


def _ssim_single_scale(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """SSIM at a single scale. Returns the SSIM map mean."""
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
    return ssim_map.mean()


def ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    num_scales: int = 3,
) -> torch.Tensor:
    """Multi-Scale SSIM loss (1 - MS-SSIM).
    
    Evaluates structural similarity at 3 spatial scales via downsampling,
    which is critical for capturing both fine defect lines and global structure.
    """
    weights = [0.2, 0.3, 0.5]  # higher weight on finest scale
    ms_ssim_val = 0.0
    p, t = pred, target
    for i in range(num_scales):
        ssim_val = _ssim_single_scale(p, t, data_range=data_range)
        ms_ssim_val += weights[i] * ssim_val
        if i < num_scales - 1:
            # Downsample by 2 for next scale
            p = F.avg_pool2d(p, kernel_size=2, stride=2)
            t = F.avg_pool2d(t, kernel_size=2, stride=2)
    return 1.0 - ms_ssim_val


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Backwards-compatible wrapper — now uses Multi-Scale SSIM."""
    return ms_ssim_loss(pred, target, data_range=data_range)


def _sobel_kernels(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return gx, gy


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel gradient edge loss — penalizes blurred boundaries."""
    gx, gy = _sobel_kernels(pred.device, pred.dtype)
    channels = pred.shape[1]
    gx = gx.repeat(channels, 1, 1, 1)
    gy = gy.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(pred, gx, padding=1, groups=channels)
    pred_y = F.conv2d(pred, gy, padding=1, groups=channels)
    tgt_x = F.conv2d(target, gx, padding=1, groups=channels)
    tgt_y = F.conv2d(target, gy, padding=1, groups=channels)
    return F.l1_loss(pred_x, tgt_x) + F.l1_loss(pred_y, tgt_y)


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Second-order Laplacian sharpness loss — critical for thin semiconductor line features."""
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred.device, dtype=pred.dtype
    ).view(1, 1, 3, 3)
    channels = pred.shape[1]
    lap_kernel = lap_kernel.repeat(channels, 1, 1, 1)
    pred_lap = F.conv2d(pred, lap_kernel, padding=1, groups=channels)
    tgt_lap = F.conv2d(target, lap_kernel, padding=1, groups=channels)
    return F.l1_loss(pred_lap, tgt_lap)


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase-aware frequency domain loss.
    
    Compares both amplitude AND phase of the FFT spectrum.
    Semiconductor patterns have highly structured phase relationships that
    pure amplitude comparison would miss (e.g., periodic circuit line spacing).
    """
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    tgt_fft = torch.fft.rfft2(target, norm="ortho")

    # Amplitude loss (log-compressed to balance low/high-frequency terms)
    pred_amp = torch.log1p(torch.abs(pred_fft))
    tgt_amp = torch.log1p(torch.abs(tgt_fft))
    amp_loss = F.l1_loss(pred_amp, tgt_amp)

    # Phase loss (cosine distance is numerically stable for angle comparison)
    pred_phase = torch.angle(pred_fft)
    tgt_phase = torch.angle(tgt_fft)
    # Use L1 on sin/cos representation to avoid phase wrapping issues
    phase_loss = F.l1_loss(torch.sin(pred_phase), torch.sin(tgt_phase)) + \
                 F.l1_loss(torch.cos(pred_phase), torch.cos(tgt_phase))

    return amp_loss + 0.1 * phase_loss


class CompositeRestorationLoss(nn.Module):
    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 1.5,
        edge_weight: float = 0.15,
        fft_weight: float = 0.05,
        laplacian_weight: float = 0.05,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight
        self.laplacian_weight = laplacian_weight
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l_char = charbonnier_loss(pred, target, self.charbonnier_eps)
        l_ssim = ssim_loss(pred, target)
        l_edge = edge_loss(pred, target)
        l_fft = fft_loss(pred, target)
        l_lap = laplacian_loss(pred, target)
        total = (
            self.charbonnier_weight * l_char
            + self.ssim_weight * l_ssim
            + self.edge_weight * l_edge
            + self.fft_weight * l_fft
            + self.laplacian_weight * l_lap
        )
        return {
            "loss": total,
            "charbonnier": l_char.detach(),
            "ssim": l_ssim.detach(),
            "edge": l_edge.detach(),
            "fft": l_fft.detach(),
            "laplacian": l_lap.detach(),
        }


def build_loss(cfg: dict[str, Any]) -> CompositeRestorationLoss:
    loss_cfg = cfg["loss"]
    return CompositeRestorationLoss(
        charbonnier_weight=float(loss_cfg["charbonnier_weight"]),
        ssim_weight=float(loss_cfg["ssim_weight"]),
        edge_weight=float(loss_cfg["edge_weight"]),
        fft_weight=float(loss_cfg["fft_weight"]),
        laplacian_weight=float(loss_cfg.get("laplacian_weight", 0.05)),
        charbonnier_eps=float(loss_cfg["charbonnier_eps"]),
    )
