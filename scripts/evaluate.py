import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the local 'semicon' package can be imported without manual PYTHONPATH setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semicon.config import load_config
from semicon.models.restorer import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate(input_dir: str, output_dir: str, config_path: str = None, weights_path: str = None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    # Load configuration
    cfg = load_config(config_path)
    
    # Auto-detect device (H100 GPU optimized)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running inference on device: {device}")

    # Build model and load weights
    model = build_model(cfg).to(device)
    
    if weights_path is None:
        weights_path = Path(cfg["_root"]) / cfg["train"]["save_dir"] / "best_model.pth"
    else:
        weights_path = Path(weights_path)
        
    if not weights_path.exists():
        logging.warning(f"Weights file not found at {weights_path}. Running with untrained weights (for testing only).")
    else:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded weights from {weights_path}")

    model.eval()

    # Process all .npy files in the input directory
    files = sorted(input_path.glob("*.npy"))
    if not files:
        logging.warning(f"No .npy files found in {input_path}")
        return

    logging.info(f"Found {len(files)} files for inference.")

    # Run inference loop
    with torch.no_grad():
        for file_path in files:
            # Load numpy array [H, W]
            arr = np.load(file_path)
            if arr.ndim != 2:
                logging.warning(f"Skipping {file_path.name}: Expected 2D array, got {arr.ndim}D")
                continue
                
            # Convert to tensor [1, 1, H, W]
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Fast inference with mixed precision
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=cfg["train"].get("amp", True)):
                restored_tensor = model(tensor)
                
            # Convert back to numpy [H, W]
            restored_arr = restored_tensor.squeeze().cpu().numpy()
            
            # Save output
            save_path = output_path / file_path.name
            np.save(save_path, restored_arr)
            
    logging.info(f"Evaluation complete. Restored images saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semicon Hackathon Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (optional)")
    
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir, args.config, args.weights)
