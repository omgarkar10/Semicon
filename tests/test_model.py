from __future__ import annotations

import torch

from semicon.losses.composite import CompositeRestorationLoss
from semicon.models.restorer import RRDBRestorer
from semicon.utils.metrics import psnr, ssim


def test_restorer_2x_single_channel():
    model = RRDBRestorer(num_block=2, num_feat=16, num_grow_ch=8, noise_feat=8, attn_every=1)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 256, 256)
    y2, noise = model(x, return_noise=True)
    assert y2.shape == y.shape
    assert noise.shape == (2, 1, 128, 128)


def test_restorer_gradients():
    model = RRDBRestorer(num_block=1, num_feat=8, num_grow_ch=4, noise_feat=4, attn_every=1)
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    y = model(x)
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_composite_loss_finite():
    pred = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    criterion = CompositeRestorationLoss()
    out = criterion(pred, target)
    assert torch.isfinite(out["loss"])
    assert psnr(pred, pred).item() > 40
    assert ssim(target, target).item() > 0.99
