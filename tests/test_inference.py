from __future__ import annotations

import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from backend.inference import (
    CheckpointNotFoundError,
    InferenceService,
    InvalidInputArrayError,
    TTA_COUNT,
    postprocess_output,
    resolve_weights_path,
    validate_input_array,
)


class FakeRestorer(torch.nn.Module):
    """Lightweight 2x upsampler used in place of the real model."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)


class WideOutputRestorer(torch.nn.Module):
    """Returns out-of-range values so postprocessing clipping can be tested."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return out * 2.0 - 0.5


@pytest.fixture
def fake_service() -> tuple[InferenceService, FakeRestorer]:
    model = FakeRestorer(scale=2)
    service = InferenceService(model=model, require_checkpoint=False)
    return service, model


def test_validate_hw_shape():
    arr = np.zeros((64, 48), dtype=np.float32)
    out = validate_input_array(arr)
    assert out.shape == (64, 48)


def test_validate_hw1_shape():
    arr = np.zeros((64, 48, 1), dtype=np.float32)
    out = validate_input_array(arr)
    assert out.shape == (64, 48)


def test_validate_rejects_invalid_shape():
    arr = np.zeros((3, 64, 48), dtype=np.float32)
    with pytest.raises(InvalidInputArrayError, match="Unsupported shape"):
        validate_input_array(arr)


def test_validate_rejects_object_dtype():
    arr = np.array(["not-a-number"], dtype=object)
    with pytest.raises(InvalidInputArrayError, match="numeric arrays only"):
        validate_input_array(arr)


def test_restore_image_hw_output(fake_service: tuple[InferenceService, FakeRestorer]):
    service, model = fake_service
    arr = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    restored, metadata = service.restore_image(arr)

    assert restored.shape == (64, 64)
    assert restored.dtype == np.float32
    assert metadata["input_shape"] == (32, 32)
    assert metadata["output_shape"] == (64, 64)
    assert metadata["tta_count"] == TTA_COUNT
    assert metadata["inference_time"] >= 0.0
    assert model.call_count == TTA_COUNT


def test_restore_image_hw1_output(fake_service: tuple[InferenceService, FakeRestorer]):
    service, _model = fake_service
    arr = np.ones((16, 16, 1), dtype=np.float32)

    restored, metadata = service.restore_image(arr)

    assert restored.shape == (32, 32)
    assert metadata["input_shape"] == (16, 16)


def test_restore_image_finite_and_clipped():
    model = WideOutputRestorer()
    service = InferenceService(model=model, require_checkpoint=False)
    arr = np.ones((8, 8), dtype=np.float32)

    restored, _metadata = service.restore_image(arr)

    assert np.isfinite(restored).all()
    assert restored.min() >= 0.0
    assert restored.max() <= 1.0


def test_postprocess_output_clips_and_sanitizes():
    tensor = torch.tensor([[[[-1.0, 0.5, 2.0, float("nan")]]]])
    restored = postprocess_output(tensor)

    assert restored.dtype == np.float32
    assert restored.shape == (4,)
    assert np.isfinite(restored).all()
    assert restored.min() >= 0.0
    assert restored.max() <= 1.0


def test_tta_runs_eight_forward_passes(fake_service: tuple[InferenceService, FakeRestorer]):
    service, model = fake_service
    arr = np.zeros((12, 12), dtype=np.float32)

    _restored, metadata = service.restore_image(arr)

    assert metadata["tta_count"] == 8
    assert model.call_count == 8


def test_missing_checkpoint_raises():
    with pytest.raises(CheckpointNotFoundError, match="Checkpoint not found"):
        resolve_weights_path("/nonexistent/path/to/model.pth")


def test_resolve_weights_path_missing():
    with pytest.raises(CheckpointNotFoundError):
        resolve_weights_path("/path/that/does/not/exist/model.pth")
