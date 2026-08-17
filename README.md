# SEMICON India Hackathon 2026

AI restoration for degraded semiconductor inspection images: joint speckle denoising and exact **2×** super-resolution (128×128 → 256×256).

## Dataset (do not renormalize)

| Split | Path | Size | Values |
| --- | --- | --- | --- |
| Train LR | `train/NoisyLR/` | 3,200 × 128×128 `.npy` float32 | roughly `[-0.21, 1.94]` (speckle) |
| Train GT | `train/GT/` | 3,200 × 256×256 `.npy` float32 | `[0.0, 1.0]` |
| Test LR | `NoisyLR/` | 400 × 128×128 `.npy` float32 | no GT |

Pairs share filenames (`000000.npy` … `003199.npy`). Arrays are 2D; the loader adds a channel dim to `[1, H, W]`.

## Layout

```
configs/default.yaml     # hyperparameters
semicon/data/            # paired/unpaired datasets + augmentations
semicon/models/          # RRDB restorer (noise head + PixelShuffle 2x)
semicon/losses/          # Charbonnier + SSIM + Sobel + FFT
semicon/utils/metrics.py # PSNR / SSIM
tests/
```

## Setup

```bash
pip install -r requirements.txt
```

Run unit tests (synthetic tensors only; they do not scan the full dataset):

```bash
pytest -q
```

## Model

`RRDBRestorer` estimates a LR noise map, subtracts it, runs 8 residual-in-residual dense blocks with channel attention every 3 blocks, upsamples with PixelShuffle (×2), and adds a bicubic skip of the denoised LR. Output is a residual reconstruction at 256×256 (clamp to `[0, 1]` at inference time in a later phase).

```python
from semicon import load_config
from semicon.models import build_model
from semicon.data import build_dataloaders

cfg = load_config()
model = build_model(cfg)
train_loader, val_loader = build_dataloaders(cfg)
```

Training and inference scripts are the next phases.
