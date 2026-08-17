from __future__ import annotations

from typing import Any

import torch
from torch import nn

from semicon.models.blocks import (
    CBAM,
    NoiseEstimator,
    PixelShuffleUpsampler,
    RRDB,
    bicubic_2x,
)


class RRDBRestorer(nn.Module):
    """Joint speckle denoising + exact 2x super-resolution.

    Pipeline: noise estimate → subtract at LR → RRDB trunk (CBAM attention
    every ``attn_every`` blocks) → PixelShuffle 2x → residual over bicubic skip.
    
    Upgrades:
    - CBAM (Channel + Spatial attention) replaces pure ChannelAttention
    - Deeper NoiseEstimator for better speckle decoupling
    - Larger capacity (num_feat=96, num_block=16 by default)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_feat: int = 96,
        num_block: int = 16,
        num_grow_ch: int = 48,
        scale: int = 2,
        residual_scale: float = 0.2,
        attn_every: int = 4,
        noise_feat: int = 32,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("RRDBRestorer is locked to scale=2 (128→256).")
        self.scale = scale
        self.noise_head = NoiseEstimator(in_channels=in_channels, num_feat=noise_feat)
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        trunk: list[nn.Module] = []
        for i in range(num_block):
            trunk.append(RRDB(num_feat, num_grow_ch, residual_scale))
            if attn_every > 0 and (i + 1) % attn_every == 0:
                # CBAM: Channel Attention + Spatial Attention for defect-region focus
                trunk.append(CBAM(num_feat, reduction=16, spatial_kernel=7))
        self.rrdb_blocks = nn.Sequential(*trunk)
        self.conv_trunk = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = PixelShuffleUpsampler(num_feat, scale=scale)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, return_noise: bool = False):
        noise_map = self.noise_head(x)
        cleaned = x - noise_map
        feat = self.conv_first(cleaned)
        trunk = self.rrdb_blocks(feat)
        feat = feat + self.conv_trunk(trunk)
        feat = self.upsample(feat)
        residual = self.conv_last(feat)
        base = bicubic_2x(cleaned)
        out = base + residual
        if return_noise:
            return out, noise_map
        return out


def build_model(cfg: dict[str, Any]) -> RRDBRestorer:
    m = cfg["model"]
    return RRDBRestorer(
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        num_feat=int(m["num_feat"]),
        num_block=int(m["num_block"]),
        num_grow_ch=int(m["num_grow_ch"]),
        scale=int(m["scale"]),
        residual_scale=float(m["residual_scale"]),
        attn_every=int(m["attn_every"]),
        noise_feat=int(m["noise_feat"]),
    )
