"""
diag_beat.py — Quick false-negative diagnostic for a single beat chip.

Prints:
  • Chip spatial extent (bbox in lat/lon)
  • Max / mean model probability over the chip
  • Number of pixels above several low thresholds
  • Saves prob_map.tif and change_mask_001.tif (thresh=0.01) to the chip folder

Usage:
    python scripts/diag_beat.py \
        --chip outputs/alerts/beats/North_Guna/Mar_Ki_Mahu_dec2025/live_chip_2025-12-23_img.tif \
        --checkpoint outputs/checkpoints/best_detect.pth \
        --config config.yaml
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

def _pin_proj():
    try:
        import rasterio as _rt
        _p = os.path.join(os.path.dirname(_rt.__file__), "proj_data")
        if os.path.isdir(_p):
            os.environ["PROJ_DATA"] = _p
            os.environ["PROJ_LIB"]  = _p
            return
    except (ImportError, OSError):
        pass
    try:
        import pyproj
        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
        os.environ["PROJ_LIB"]  = os.environ["PROJ_DATA"]
    except (ImportError, OSError, AttributeError):
        pass
_pin_proj()

import argparse
import numpy as np
import rasterio
from rasterio.transform import array_bounds
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip",        required=True)
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--config",      default="config.yaml")
    args = ap.parse_args()

    chip_path = args.chip
    out_dir   = os.path.dirname(chip_path)

    # ── Print chip spatial info ──────────────────────────────────────────────
    with rasterio.open(chip_path) as src:
        profile  = src.profile.copy()
        bounds   = src.bounds
        crs      = src.crs
        h, w     = src.height, src.width
        px_m     = abs(src.transform.a)
        n_bands  = src.count

    print(f"\n{'='*60}")
    print(f"  CHIP DIAGNOSTICS: {os.path.basename(chip_path)}")
    print(f"{'='*60}")
    print(f"  CRS       : {crs}")
    print(f"  Bounds    : left={bounds.left:.5f}  bottom={bounds.bottom:.5f}")
    print(f"              right={bounds.right:.5f}  top={bounds.top:.5f}")
    print(f"  Size      : {w}x{h} px  @  {px_m:.1f} m/px  ({w*px_m/1000:.2f} x {h*px_m/1000:.2f} km)")
    print(f"  Bands     : {n_bands}")

    # ── Load model & run inference ───────────────────────────────────────────
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from src.inference.generate_alerts import load_model, load_v2_chip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device    : {device}")

    model = load_model(cfg, args.checkpoint, device)

    chip_arr = load_v2_chip(chip_path)
    if chip_arr is None:
        print("  ERROR: chip_arr is None — chip mostly nodata!")
        sys.exit(1)

    tensor = torch.from_numpy(chip_arr[np.newaxis]).float().to(device)
    with torch.no_grad():
        logits = model(tensor, mode="detect")
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_change = probs[1]   # class 1 = forest loss

    # ── Probability statistics ───────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  MODEL PROBABILITY MAP  (class=forest_loss)")
    print(f"  Max prob   : {prob_change.max():.4f}")
    print(f"  Mean prob  : {prob_change.mean():.4f}")
    print(f"  p95 prob   : {np.percentile(prob_change, 95):.4f}")
    print(f"  p99 prob   : {np.percentile(prob_change, 99):.4f}")
    print()
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    print(f"  {'Threshold':<12}  {'Pixels':>8}  {'Area (ha)':>10}")
    print(f"  {'─'*10:<12}  {'─'*8:>8}  {'─'*10:>10}")
    for t in thresholds:
        mask    = (prob_change >= t)
        npx     = int(mask.sum())
        area_ha = npx * (px_m ** 2) / 10_000
        print(f"  {t:<12.2f}  {npx:>8}  {area_ha:>10.3f} ha")

    # ── Save prob map GeoTIFF ────────────────────────────────────────────────
    prob_tif  = os.path.join(out_dir, "prob_map.tif")
    mask_tif  = os.path.join(out_dir, "change_mask_010.tif")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32")
    with rasterio.open(prob_tif, "w", **out_profile) as dst:
        dst.write(prob_change.astype("float32"), 1)

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8")
    with rasterio.open(mask_tif, "w", **mask_profile) as dst:
        dst.write((prob_change >= 0.10).astype("uint8"), 1)

    print(f"\n  Saved: {prob_tif}")
    print(f"  Saved: {mask_tif}  (threshold=0.10)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
