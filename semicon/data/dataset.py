from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from semicon.config import resolve_data_path
from semicon.data.augmentations import augment_pair


def _list_npy(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected image directory at {folder}")
    names = sorted(p.name for p in folder.glob("*.npy"))
    if not names:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    return names


def _load_chw(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D (H, W), got shape {arr.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return tensor.unsqueeze(0)


def split_filenames(names: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    n_val = max(1, int(round(len(names) * val_ratio)))
    val_idx = set(perm[:n_val].tolist())
    val = [names[i] for i in range(len(names)) if i in val_idx]
    train = [names[i] for i in range(len(names)) if i not in val_idx]
    return train, val


class PairedNpyDataset(Dataset):
    """Paired degraded LR (128x128) and GT (256x256) ``.npy`` arrays.

    Intensities are left as-is: GT is already in ``[0, 1]``; NoisyLR speckle
    may sit outside that range and must not be per-image min-max normalized.
    A channel axis is added so tensors are ``[1, H, W]``.
    """

    def __init__(
        self,
        lr_dir: str | Path,
        gt_dir: str | Path,
        filenames: list[str] | None = None,
        augment: bool = False,
        extra_noise_prob: float = 0.15,
        extra_noise_std: float = 0.05,
        speckle_prob: float = 0.15,
        speckle_std: float = 0.08,
        input_dropout_prob: float = 0.10,
        input_dropout_ratio: float = 0.05,
        expected_scale: int = 2,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.gt_dir = Path(gt_dir)
        self.filenames = filenames if filenames is not None else _list_npy(self.lr_dir)
        missing = [n for n in self.filenames if not (self.gt_dir / n).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} GT files missing under {self.gt_dir} (e.g. {preview})"
            )
        self.augment = augment
        self.extra_noise_prob = extra_noise_prob
        self.extra_noise_std = extra_noise_std
        self.speckle_prob = speckle_prob
        self.speckle_std = speckle_std
        self.input_dropout_prob = input_dropout_prob
        self.input_dropout_ratio = input_dropout_ratio
        self.expected_scale = expected_scale

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.filenames[index]
        lr = _load_chw(self.lr_dir / name)
        gt = _load_chw(self.gt_dir / name)
        if gt.shape[-2] != lr.shape[-2] * self.expected_scale or gt.shape[-1] != lr.shape[-1] * self.expected_scale:
            raise ValueError(
                f"{name}: expected GT to be {self.expected_scale}x LR, "
                f"got LR {tuple(lr.shape)} GT {tuple(gt.shape)}"
            )
        if self.augment:
            lr, gt = augment_pair(
                lr,
                gt,
                extra_noise_prob=self.extra_noise_prob,
                extra_noise_std=self.extra_noise_std,
                speckle_prob=self.speckle_prob,
                speckle_std=self.speckle_std,
                input_dropout_prob=self.input_dropout_prob,
                input_dropout_ratio=self.input_dropout_ratio,
            )
        return {"lr": lr, "gt": gt, "name": name}


class UnpairedNpyDataset(Dataset):
    """Test-time degraded images with no ground truth (root ``NoisyLR/``)."""

    def __init__(self, lr_dir: str | Path, filenames: list[str] | None = None) -> None:
        self.lr_dir = Path(lr_dir)
        self.filenames = filenames if filenames is not None else _list_npy(self.lr_dir)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.filenames[index]
        lr = _load_chw(self.lr_dir / name)
        return {"lr": lr, "name": name}


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    lr_dir = resolve_data_path(cfg, "lr_dir")
    gt_dir = resolve_data_path(cfg, "gt_dir")
    names = _list_npy(lr_dir)
    train_names, val_names = split_filenames(names, float(data_cfg["val_ratio"]), int(cfg["seed"]))
    common = dict(
        extra_noise_prob=float(data_cfg["extra_noise_prob"]),
        extra_noise_std=float(data_cfg["extra_noise_std"]),
        speckle_prob=float(data_cfg["speckle_prob"]),
        speckle_std=float(data_cfg["speckle_std"]),
        input_dropout_prob=float(data_cfg["input_dropout_prob"]),
        input_dropout_ratio=float(data_cfg["input_dropout_ratio"]),
        expected_scale=int(data_cfg["scale"]),
    )
    train_ds = PairedNpyDataset(lr_dir, gt_dir, train_names, augment=bool(data_cfg["augment"]), **common)
    val_ds = PairedNpyDataset(lr_dir, gt_dir, val_names, augment=False, **common)
    loader_kw = dict(
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_kw,
    )
    return train_loader, val_loader
