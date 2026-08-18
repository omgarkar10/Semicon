"""Reusable inference service extracted from the submission ``run.py`` pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from semicon.config import ROOT, load_config
from semicon.models.restorer import build_model

logger = logging.getLogger(__name__)

TTA_COUNT = 8


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the trained model checkpoint is unavailable."""


class InvalidInputArrayError(ValueError):
    """Raised when an input array does not meet inference requirements."""


def validate_input_array(arr: np.ndarray) -> np.ndarray:
    """Validate and normalize an input array to grayscale ``(H, W)``."""
    if not isinstance(arr, np.ndarray):
        raise InvalidInputArrayError("Input must be a NumPy array.")

    if arr.dtype == np.object_ or not np.issubdtype(arr.dtype, np.number):
        raise InvalidInputArrayError(
            f"Unsupported dtype {arr.dtype!r}; numeric arrays only."
        )

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]

    raise InvalidInputArrayError(
        f"Unsupported shape {arr.shape}; expected (H, W) or (H, W, 1)."
    )


def preprocess_input_array(arr: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Convert a validated grayscale array to a ``[1, 1, H, W]`` tensor."""
    hw_arr = validate_input_array(arr)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(hw_arr))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return hw_arr, tensor


def _apply_transform(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of 8 geometric augmentations to a ``[1, 1, H, W]`` tensor."""
    if k == 0:
        return x
    if k == 1:
        return torch.flip(x, dims=[-1])
    if k == 2:
        return torch.flip(x, dims=[-2])
    if k == 3:
        return torch.rot90(x, 1, dims=[-2, -1])
    if k == 4:
        return torch.rot90(x, 2, dims=[-2, -1])
    if k == 5:
        return torch.rot90(x, 3, dims=[-2, -1])
    if k == 6:
        return torch.rot90(torch.flip(x, dims=[-1]), 1, dims=[-2, -1])
    return torch.rot90(torch.flip(x, dims=[-2]), 1, dims=[-2, -1])


def _invert_transform(x: torch.Tensor, k: int) -> torch.Tensor:
    """Invert the geometric transform applied by ``_apply_transform(k)``."""
    if k == 0:
        return x
    if k == 1:
        return torch.flip(x, dims=[-1])
    if k == 2:
        return torch.flip(x, dims=[-2])
    if k == 3:
        return torch.rot90(x, 3, dims=[-2, -1])
    if k == 4:
        return torch.rot90(x, 2, dims=[-2, -1])
    if k == 5:
        return torch.rot90(x, 1, dims=[-2, -1])
    if k == 6:
        return torch.flip(torch.rot90(x, 3, dims=[-2, -1]), dims=[-1])
    return torch.flip(torch.rot90(x, 3, dims=[-2, -1]), dims=[-2])


def tta_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    amp_device: str,
    amp_enabled: bool,
) -> torch.Tensor:
    """Run 8-fold TTA and return the averaged prediction."""
    predictions = []
    for k in range(TTA_COUNT):
        augmented = _apply_transform(tensor, k)
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            out = model(augmented)
        restored = _invert_transform(out, k)
        predictions.append(restored)
    return torch.stack(predictions, dim=0).mean(dim=0)


def postprocess_output(restored_tensor: torch.Tensor) -> np.ndarray:
    """Convert model output to clipped finite ``float32`` grayscale ``(H, W)``."""
    restored_arr = restored_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(restored_arr, 0.0, 1.0)


def resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    """Resolve checkpoint path with ``models/`` primary and ``weights/`` fallback."""
    if weights_path is not None:
        path = Path(weights_path)
        if path.is_file():
            return path
        raise CheckpointNotFoundError(f"Checkpoint not found at {path}")

    primary = ROOT / "models" / "best_model.pth"
    if primary.is_file():
        return primary

    fallback = ROOT / "weights" / "best_model.pth"
    if fallback.is_file():
        return fallback

    raise CheckpointNotFoundError(
        "Trained checkpoint not found. Expected one of:\n"
        f"  - {primary}\n"
        f"  - {fallback}\n"
        "Train the model with: python scripts/train.py --config configs/default.yaml"
    )


class InferenceService:
    """Load the restoration model once and run the submission inference pipeline."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        *,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        require_checkpoint: bool = True,
    ) -> None:
        cfg_path = Path(config_path) if config_path is not None else ROOT / "configs" / "default.yaml"
        self.cfg = load_config(cfg_path)
        self.config_path = Path(self.cfg["_config_path"])
        self.weights_path: Path | None = None

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_enabled = self.device.type == "cuda"

        if model is None:
            self.model = build_model(self.cfg).to(self.device)
            resolved_weights = resolve_weights_path(weights_path)
            self.weights_path = resolved_weights
            checkpoint = torch.load(resolved_weights, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            epoch = checkpoint.get("epoch", "unknown")
            best_psnr = checkpoint.get("best_psnr", "unknown")
            logger.info(
                "Loaded weights from %s (epoch=%s, best_psnr=%s)",
                resolved_weights,
                epoch,
                best_psnr,
            )
        else:
            if require_checkpoint and weights_path is not None:
                resolved_weights = resolve_weights_path(weights_path)
                self.weights_path = resolved_weights
            self.model = model.to(self.device)

        self.model.eval()

    @property
    def model_info(self) -> dict[str, Any]:
        """Basic model metadata safe to expose via the API."""
        model_cfg = self.cfg["model"]
        return {
            "name": model_cfg.get("name", "rrdb_restorer"),
            "scale": int(model_cfg["scale"]),
            "num_block": int(model_cfg["num_block"]),
            "num_feat": int(model_cfg["num_feat"]),
            "attn_every": int(model_cfg["attn_every"]),
            "checkpoint_loaded": self.weights_path is not None,
            "checkpoint_path": str(self.weights_path) if self.weights_path else None,
            "device": str(self.device),
            "tta_count": TTA_COUNT,
        }

    def restore_image(self, input_array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore one degraded image array using 8-fold TTA."""
        hw_arr, tensor = preprocess_input_array(input_array)
        tensor = tensor.to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            restored_tensor = tta_inference(
                self.model,
                tensor,
                self.amp_device,
                self.amp_enabled,
            )
        inference_time = time.perf_counter() - start

        restored_arr = postprocess_output(restored_tensor)
        metadata = {
            "input_shape": tuple(int(v) for v in hw_arr.shape),
            "output_shape": tuple(int(v) for v in restored_arr.shape),
            "inference_time": inference_time,
            "device": str(self.device),
            "tta_count": TTA_COUNT,
        }
        return restored_arr, metadata


_service: InferenceService | None = None


def get_inference_service(
    *,
    config_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    force_new: bool = False,
) -> InferenceService:
    """Return a process-wide singleton inference service."""
    global _service
    if force_new or _service is None:
        _service = InferenceService(config_path=config_path, weights_path=weights_path)
    return _service
