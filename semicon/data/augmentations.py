from __future__ import annotations

import random

import torch
from torch import Tensor


def paired_geometric(lr: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the same flip/90° rotation to a paired LR/GT tensor pair.

    Tensors are ``[C, H, W]``.
    """
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[-1])
        gt = torch.flip(gt, dims=[-1])
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[-2])
        gt = torch.flip(gt, dims=[-2])
    k = random.randint(0, 3)
    if k:
        lr = torch.rot90(lr, k, dims=[-2, -1])
        gt = torch.rot90(gt, k, dims=[-2, -1])
    return lr, gt


def additive_gaussian(lr: Tensor, std: float) -> Tensor:
    return lr + torch.randn_like(lr) * std


def multiplicative_speckle(lr: Tensor, std: float) -> Tensor:
    return lr * (1.0 + torch.randn_like(lr) * std)


def input_dropout(lr: Tensor, drop_ratio: float) -> Tensor:
    """Randomly zero a small fraction of LR pixels (OOD robustness)."""
    mask = torch.rand_like(lr) > drop_ratio
    return lr * mask.to(dtype=lr.dtype)


def augment_pair(
    lr: Tensor,
    gt: Tensor,
    *,
    extra_noise_prob: float = 0.15,
    extra_noise_std: float = 0.05,
    speckle_prob: float = 0.15,
    speckle_std: float = 0.08,
    input_dropout_prob: float = 0.10,
    input_dropout_ratio: float = 0.05,
) -> tuple[Tensor, Tensor]:
    lr, gt = paired_geometric(lr, gt)
    if extra_noise_prob > 0 and random.random() < extra_noise_prob:
        lr = additive_gaussian(lr, extra_noise_std)
    if speckle_prob > 0 and random.random() < speckle_prob:
        lr = multiplicative_speckle(lr, speckle_std)
    if input_dropout_prob > 0 and random.random() < input_dropout_prob:
        lr = input_dropout(lr, input_dropout_ratio)
    return lr, gt
