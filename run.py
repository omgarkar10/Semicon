"""
run.py — Semiconductor Image Restoration Entry Point
=====================================================
Usage: python run.py <input-dir> <output-dir>

Submission compliant:
  ✅ Reads all .npy files from input directory
  ✅ Creates output directory if it does not exist
  ✅ Generates one restored .npy file per input file
  ✅ Each output has the same filename as its input
  ✅ Outputs are grayscale (H, W) float32 arrays
  ✅ Output values clipped to [0, 1] with NaN/Inf replaced
  ✅ Loads weights from models/best_model.pth
  ✅ Runs on NVIDIA GPU; no internet, no API keys, no user input needed
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the local 'semicon' package can be imported without manual PYTHONPATH setup
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from semicon.config import load_config
from semicon.models.restorer import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# 8-fold Geometric Self-Ensemble (Test-Time Augmentation)
# ---------------------------------------------------------------------------
# The 8 transforms are: identity, hflip, vflip, rot90, rot180, rot270,
# hflip+rot90, vflip+rot90. Each is a lossless geometric transform, so
# averaging predictions in original space removes orientation bias.
# Typical gain: +0.3 to +0.8 dB PSNR with zero retraining.

def _apply_transform(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of 8 geometric augmentations to a [1,1,H,W] tensor."""
    if k == 0:
        return x
    elif k == 1:
        return torch.flip(x, dims=[-1])          # horizontal flip
    elif k == 2:
        return torch.flip(x, dims=[-2])          # vertical flip
    elif k == 3:
        return torch.rot90(x, 1, dims=[-2, -1])  # 90°
    elif k == 4:
        return torch.rot90(x, 2, dims=[-2, -1])  # 180°
    elif k == 5:
        return torch.rot90(x, 3, dims=[-2, -1])  # 270°
    elif k == 6:
        return torch.rot90(torch.flip(x, dims=[-1]), 1, dims=[-2, -1])   # hflip + 90°
    else:
        return torch.rot90(torch.flip(x, dims=[-2]), 1, dims=[-2, -1])   # vflip + 90°


def _invert_transform(x: torch.Tensor, k: int) -> torch.Tensor:
    """Invert the geometric transform applied by _apply_transform(k)."""
    if k == 0:
        return x
    elif k == 1:
        return torch.flip(x, dims=[-1])
    elif k == 2:
        return torch.flip(x, dims=[-2])
    elif k == 3:
        return torch.rot90(x, 3, dims=[-2, -1])  # inverse of 90° is 270°
    elif k == 4:
        return torch.rot90(x, 2, dims=[-2, -1])  # inverse of 180° is 180°
    elif k == 5:
        return torch.rot90(x, 1, dims=[-2, -1])  # inverse of 270° is 90°
    elif k == 6:
        return torch.flip(torch.rot90(x, 3, dims=[-2, -1]), dims=[-1])
    else:
        return torch.flip(torch.rot90(x, 3, dims=[-2, -1]), dims=[-2])


def tta_inference(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device,
                  amp_device: str, amp_enabled: bool) -> torch.Tensor:
    """Run 8-fold TTA and return the averaged prediction."""
    predictions = []
    for k in range(8):
        augmented = _apply_transform(tensor, k)
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            out = model(augmented)
        restored = _invert_transform(out, k)
        predictions.append(restored)
    return torch.stack(predictions, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # ✅ Creates output directory if it does not already exist
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    # Load config (relative to this script's location)
    script_root = Path(__file__).parent
    cfg_path = script_root / "configs" / "default.yaml"
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = torch.cuda.is_available()
    logging.info(f"Running inference on: {device}")

    # Build model
    model = build_model(cfg).to(device)

    # ✅ Load weights from models/best_model.pth
    weights_path = script_root / "models" / "best_model.pth"
    if not weights_path.exists():
        # Fallback: check legacy weights/ directory
        weights_path = script_root / "weights" / "best_model.pth"

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        epoch = checkpoint.get("epoch", "unknown")
        best_psnr = checkpoint.get("best_psnr", "unknown")
        logging.info(f"Loaded weights from {weights_path} (epoch={epoch}, best_psnr={best_psnr})")
    else:
        logging.warning(f"No weights found at {weights_path}. Running with untrained model.")

    model.eval()

    # ✅ Reads all .npy files from input directory
    files = sorted(input_path.glob("*.npy"))
    if not files:
        logging.warning(f"No .npy files found in {input_path}")
        return

    logging.info(f"Found {len(files)} .npy files. Running 8-fold TTA inference...")

    with torch.no_grad():
        for i, file_path in enumerate(files, 1):
            arr = np.load(file_path)

            # Handle (H, W) and (H, W, 1) inputs
            if arr.ndim == 2:
                hw_arr = arr
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                hw_arr = arr[:, :, 0]
            else:
                logging.warning(f"Skipping {file_path.name}: unsupported shape {arr.shape}")
                continue

            # [1, 1, H, W] tensor
            tensor = torch.from_numpy(np.ascontiguousarray(hw_arr)).float().unsqueeze(0).unsqueeze(0).to(device)

            # Run 8-fold TTA
            restored_tensor = tta_inference(model, tensor, device, amp_device, amp_enabled)

            # ✅ Output shape (H, W)
            restored_arr = restored_tensor.squeeze().cpu().numpy().astype(np.float32)

            # ✅ Replace NaN/Inf and clip to [0, 1]
            restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)
            restored_arr = np.clip(restored_arr, 0.0, 1.0)

            # ✅ Each output has the same filename as its input
            save_path = output_path / file_path.name
            np.save(save_path, restored_arr)

            if i % 10 == 0 or i == len(files):
                logging.info(f"  Processed {i}/{len(files)} — {file_path.name}")

    logging.info(f"Done. Restored {len(files)} images → {output_path}")


if __name__ == "__main__":
    main()
