from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.residual_scale + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, residual_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, residual_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.residual_scale + x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, num_feat: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(num_feat // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, hidden, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, num_feat, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SpatialAttention(nn.Module):
    """Spatial attention using max-pool + avg-pool along the channel dimension.
    
    Allows the network to focus on defect regions (high-contrast edges) and
    suppress homogeneous noise regions — critical for semiconductor inspection.
    """
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute max and mean pooling along channel dim
        max_pool = torch.amax(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Applies Channel Attention then Spatial Attention sequentially.
    This dual-attention mechanism allows the model to simultaneously select
    WHAT information is important (channels) and WHERE it is (spatial regions).
    """
    def __init__(self, num_feat: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(num_feat, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class NoiseEstimator(nn.Module):
    """Lightweight speckle/Gaussian noise map at LR resolution."""

    def __init__(self, in_channels: int = 1, num_feat: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, num_feat: int, scale: int = 2) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("This restorer is specified for exact 2x upsampling.")
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def bicubic_2x(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
