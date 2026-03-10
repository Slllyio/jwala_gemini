"""
Future Change Prediction Inference
===================================
Runs the temporal prediction head to generate a forest change risk map
for the next season/year over Fatehgarh AOI.

Usage:
    python src/inference/predict.py \
        --config config.yaml \
        --checkpoint outputs/checkpoints/best_predict.pth \
        --input data/raw/hls_2019.tif data/raw/hls_2020.tif ... \
        --output outputs/risk_map_2025.tif
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()

import yaml
import glob
import logging
import argparse
import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm

from src.model.full_model import build_model
from src.data.preprocess import load_hls_tif
from src.inference.detect import save_geotiff

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_prediction(cfg: dict, checkpoint: str, tif_paths: list,
                   output_path: str):
    """
    Generate future forest change risk map from temporal sequence.

    The model sees N historical composites and predicts the probability
    of forest loss at the next time step (T+1).

    Args:
        cfg: Config dict
        checkpoint: Trained predict checkpoint path
        tif_paths: Chronological list of N HLS GeoTIFF paths
        output_path: Output risk map GeoTIFF (float32, 0-1)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"🖥️  Running on: {device}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    log.info(f"✅ Model loaded from: {checkpoint}")

    # Load imagery sequence
    n_steps = cfg["prediction"]["num_time_steps"]
    tif_paths = tif_paths[-n_steps:]  # Use last N frames
    log.info(f"📂 Loading {len(tif_paths)} historical frames...")
    frames = [load_hls_tif(p) for p in tif_paths]  # List of (C, H, W)

    # Spatial size from first frame
    C, H, W = frames[0].shape
    tile_size = cfg["model"]["img_size"]
    overlap = 64
    stride = tile_size - overlap

    # Sliding window over spatial dims
    risk_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    row_starts = list(range(0, H - tile_size + 1, stride))
    col_starts = list(range(0, W - tile_size + 1, stride))
    if (H - tile_size) % stride != 0:
        row_starts.append(H - tile_size)
    if (W - tile_size) % stride != 0:
        col_starts.append(W - tile_size)

    positions = [(r, c) for r in row_starts for c in col_starts]
    log.info(f"  Running {len(positions)} tile predictions...")

    for r, c in tqdm(positions, desc="Predicting"):
        # Extract patch from each temporal frame
        patches = []
        all_valid = True
        for frame in frames:
            patch = frame[:, r:r + tile_size, c:c + tile_size]
            if patch.mean() < 0.01:
                all_valid = False
                break
            patches.append(torch.from_numpy(patch[np.newaxis]).float().to(device))  # (1, C, H, W)

        if not all_valid or len(patches) != n_steps:
            continue

        with torch.no_grad():
            risk = model(patches, mode="predict")  # (1, 1, H, W)
            risk_np = risk.cpu().numpy()[0, 0]

        risk_map[r:r + tile_size, c:c + tile_size] += risk_np
        count_map[r:r + tile_size, c:c + tile_size] += 1.0

    # Average and clip
    count_map = np.maximum(count_map, 1)
    risk_map /= count_map
    risk_map = np.clip(risk_map, 0, 1)

    # Save as float32 GeoTIFF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    reference_tif = tif_paths[-1]
    with rasterio.open(reference_tif) as ref:
        meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(risk_map.astype(np.float32), 1)

    # Also save a thresholded binary risk mask
    binary_out = output_path.replace(".tif", "_binary.tif")
    high_risk = (risk_map > 0.5).astype(np.uint8)
    save_geotiff(high_risk, reference_tif, binary_out)

    # Summary
    log.info(f"\n📊 Risk Prediction Results:")
    log.info(f"   Mean risk across AOI: {risk_map.mean():.3f}")
    log.info(f"   High risk pixels (>0.5): {high_risk.sum():,} "
             f"({100*high_risk.mean():.1f}%)")
    high_risk_ha = high_risk.sum() * 900 / 10000
    log.info(f"   High risk area: {high_risk_ha:.1f} ha")
    log.info(f"   Risk map: {output_path}")
    log.info(f"   Binary mask: {binary_out}")

    return risk_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+", help="Chronological HLS GeoTIFFs")
    parser.add_argument("--output", default="outputs/risk_map.tif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.input:
        tif_paths = args.input
    else:
        raw_dir = cfg["paths"]["raw_dir"]
        tif_paths = sorted(glob.glob(os.path.join(raw_dir, "hls_*.tif")))
        log.info(f"Auto-detected {len(tif_paths)} input files")

    run_prediction(cfg, args.checkpoint, tif_paths, args.output)
