import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Ensure the local 'semicon' package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semicon.config import load_config
from semicon.data.dataset import build_dataloaders
from semicon.losses.composite import build_loss
from semicon.models.restorer import build_model
from semicon.utils.metrics import psnr, ssim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train(config_path=None):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params / 1e6:.2f}M")

    criterion = build_loss(cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.999),
    )

    # Linear warmup for 5 epochs, then CosineAnnealingLR for the rest
    epochs = int(cfg["train"]["epochs"])  # FIXED: no longer capped at 2
    warmup_epochs = min(5, epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Modern AMP API (avoids deprecation warning)
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(cfg["train"]["amp"]) and torch.cuda.is_available()
    scaler = GradScaler(device=amp_device, enabled=amp_enabled)

    # Save to both weights/ (legacy) and models/ (submission format)
    save_dir = Path(cfg["_root"]) / cfg["train"]["save_dir"]
    models_dir = Path(cfg["_root"]) / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0
    logging.info(f"Starting training for {epochs} epochs (warmup: {warmup_epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for batch in pbar:
            lr_img, gt = batch["lr"].to(device), batch["gt"].to(device)
            optimizer.zero_grad()

            with autocast(device_type=amp_device, enabled=amp_enabled):
                pred = model(lr_img)
                loss_dict = criterion(pred, gt)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients with large RRDB stacks
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                lr_img, gt = batch["lr"].to(device), batch["gt"].to(device)

                with autocast(device_type=amp_device, enabled=amp_enabled):
                    pred = model(lr_img)
                    loss_dict = criterion(pred, gt)
                    loss = loss_dict["loss"]

                val_loss += loss.item()
                val_psnr += psnr(pred, gt).item()
                val_ssim += ssim(pred, gt).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        current_lr = scheduler.get_last_lr()[0]
        logging.info(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "val_ssim": val_ssim,
                "cfg": cfg,
            }
            # Save to both locations for compatibility
            torch.save(checkpoint, save_dir / "best_model.pth")
            torch.save(checkpoint, models_dir / "best_model.pth")
            logging.info(f"  ✅ Saved new best model → PSNR: {best_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

        # Also save latest checkpoint for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }, save_dir / "latest.pth")

    logging.info(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semiconductor image restorer")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
