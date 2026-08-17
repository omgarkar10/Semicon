from __future__ import annotations

import numpy as np
import pytest
import torch

from semicon.data.augmentations import paired_geometric
from semicon.data.dataset import PairedNpyDataset, UnpairedNpyDataset, split_filenames


def test_split_is_disjoint_and_seeded():
    names = [f"{i:06d}.npy" for i in range(3200)]
    a_train, a_val = split_filenames(names, 0.1, 42)
    b_train, b_val = split_filenames(names, 0.1, 42)
    assert a_train == b_train and a_val == b_val
    assert set(a_train).isdisjoint(a_val)
    assert len(a_train) + len(a_val) == 3200
    assert len(a_val) == 320


def test_paired_dataset_adds_channel_and_keeps_values(tmp_path):
    lr_dir = tmp_path / "NoisyLR"
    gt_dir = tmp_path / "GT"
    lr_dir.mkdir()
    gt_dir.mkdir()
    lr = np.array([[-0.21, 1.94], [0.0, 0.5]], dtype=np.float32)
    # 2x GT for scale check: 4x4
    gt = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    np.save(lr_dir / "000000.npy", lr)
    np.save(gt_dir / "000000.npy", gt)

    ds = PairedNpyDataset(lr_dir, gt_dir, augment=False, expected_scale=2)
    sample = ds[0]
    assert sample["lr"].shape == (1, 2, 2)
    assert sample["gt"].shape == (1, 4, 4)
    assert sample["name"] == "000000.npy"
    assert torch.isclose(sample["lr"].min(), torch.tensor(-0.21))
    assert torch.isclose(sample["lr"].max(), torch.tensor(1.94))


def test_unpaired_dataset(tmp_path):
    lr_dir = tmp_path / "NoisyLR"
    lr_dir.mkdir()
    np.save(lr_dir / "t.npy", np.zeros((128, 128), dtype=np.float32))
    ds = UnpairedNpyDataset(lr_dir)
    item = ds[0]
    assert item["lr"].shape == (1, 128, 128)
    assert "gt" not in item


def test_paired_geometric_keeps_alignment():
    lr = torch.arange(4, dtype=torch.float32).view(1, 2, 2)
    gt = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    torch.manual_seed(0)
    lr2, gt2 = paired_geometric(lr, gt)
    assert lr2.shape == lr.shape and gt2.shape == gt.shape
