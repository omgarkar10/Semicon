"""Bootstrap a real checkpoint with a short CPU-friendly training run.

Uses the same model architecture, loss, and config as ``scripts/train.py``.
Intended for environments without a pre-trained ``models/best_model.pth``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semicon.config import load_config
from semicon.data.dataset import build_dataloaders
from semicon.losses.composite import build_loss
from semicon.models.restorer import build_model
from semicon.utils.metrics import psnr, ssim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bootstrap(config_path: str | None, max_batches: int) -> Path:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Bootstrap training on %s for up to %d batches", device, max_batches)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(device=amp_device, enabled=amp_enabled)

    model.train()
    seen = 0
    for batch in tqdm(train_loader, desc="Bootstrap train"):
        lr_img, gt = batch["lr"].to(device), batch["gt"].to(device)
        optimizer.zero_grad()
        with autocast(device_type=amp_device, enabled=amp_enabled):
            pred = model(lr_img)
            loss = criterion(pred, gt)["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        seen += 1
        if seen >= max_batches:
            break

    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            lr_img, gt = batch["lr"].to(device), batch["gt"].to(device)
            with autocast(device_type=amp_device, enabled=amp_enabled):
                pred = model(lr_img)
            val_psnr += psnr(pred, gt).item()
            val_ssim += ssim(pred, gt).item()
            val_batches += 1
            if val_batches >= 10:
                break

    val_psnr /= max(val_batches, 1)
    val_ssim /= max(val_batches, 1)

    models_dir = Path(cfg["_root"]) / "models"
    weights_dir = Path(cfg["_root"]) / cfg["train"]["save_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_psnr": val_psnr,
        "val_ssim": val_ssim,
        "cfg": cfg,
        "bootstrap_batches": seen,
    }
    model_path = models_dir / "best_model.pth"
    torch.save(checkpoint, model_path)
    torch.save(checkpoint, weights_dir / "best_model.pth")
    logging.info(
        "Saved bootstrap checkpoint to %s (batches=%d, val_psnr=%.2f, val_ssim=%.4f)",
        model_path,
        seen,
        val_psnr,
        val_ssim,
    )
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap a short training checkpoint")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()
    bootstrap(args.config, args.max_batches)
