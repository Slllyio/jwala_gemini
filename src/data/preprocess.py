"""
Data Preprocessor — V2 (Anchor-Dated Chip Pipeline)
====================================================
Converts V2 GEE chip exports (anchor-dated 18-band stacks + binary labels)
into Prithvi-compatible .npy tensors and a CSV manifest.

V2 chip structure (from gee_fetch.py V2):
    chip_{pos|neg}_{type}_{year}_{id:05d}_img.tif  — 18 bands (3×6, T0/T1/T2)
    chip_{pos|neg}_{type}_{year}_{id:05d}_lbl.tif  — 1 band, binary

Shape contract: (T=3, C=6, H=224, W=224) where the three frames use
STANDARDISED look-backs relative to the S1-pinpointed anchor date:

    T0 = anchor − 365d  (same-season baseline, previous year)
    T1 = anchor − 30d   (recent context)
    T2 = anchor          (event / bare-soil signature)

Negative chips with summer anchor dates (April–June) teach the model that
natural dry-deciduous leaf-drop in Guna MP is NOT a felling event.

BACKWARD COMPATIBILITY:
    If no V2 chips are found, falls back to the V1 annual-composite pipeline
    (allows reuse while V2 GEE exports are still being submitted).

Usage:
    python src/data/preprocess.py --config config.yaml            # auto-detects V1 or V2
    python src/data/preprocess.py --config config.yaml --mode v2  # force V2
    python src/data/preprocess.py --config config.yaml --mode v1  # force V1 (legacy)
"""

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import os
import glob
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_PathLike = Union[str, "os.PathLike[str]"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Prithvi band ordering (must match gee_fetch band naming)
_N_BANDS  = 6
_N_FRAMES = 3


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Normalisation ─────────────────────────────────────────────────────────────

def percentile_normalize(arr: np.ndarray,
                          p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Normalize a single-band array using 2–98th percentile clipping."""
    valid = arr[arr > 0]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [p_low, p_high])
    arr = np.clip(arr, lo, hi)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def normalize_stack(data: np.ndarray) -> np.ndarray:
    """
    Normalize a (T*C, H, W) or (C, H, W) array per-band.
    For V2 chips the bands are already at native reflectance; we apply
    the same percentile normalization as V1 for consistency.
    """
    if data.max() > 10.0:
        data = data / 10_000.0      # HLS scale factor
    for b in range(data.shape[0]):
        data[b] = percentile_normalize(data[b])
    return data.astype(np.float32)


# ── V2 chip loading ───────────────────────────────────────────────────────────

def load_v2_chip(img_path: _PathLike) -> Optional[np.ndarray]:
    """
    Load a V2 18-band GeoTIFF (bands ordered Blue_0…SWIR2_2) into
    shape (T=3, C=6, H=224, W=224).

    Returns None if the chip is invalid (too many zeros).
    """
    with rasterio.open(img_path) as src:
        data = src.read().astype(np.float32)   # (18, H, W)

    if data.mean() < 0.005:
        return None     # mostly nodata — skip

    data = normalize_stack(data)   # (18, H, W)

    # Reshape to (T, C, H, W)
    H, W = data.shape[1], data.shape[2]
    reshaped = data.reshape(_N_FRAMES, _N_BANDS, H, W)
    return reshaped


def load_v2_label(lbl_path: _PathLike) -> np.ndarray:
    """
    Load a V2 label GeoTIFF (1 band, binary) → (H, W) uint8.
    Raises ValueError if the file contains only nodata.
    """
    with rasterio.open(lbl_path) as src:
        data = src.read(1).astype(np.uint8)
    return data


def process_v2_chips(cfg: dict):
    """
    Main V2 preprocessing pipeline.
    Reads per-chip img/lbl TIFs from chip_dir, normalizes, saves .npy pairs.
    """
    chip_dir  = cfg["paths"].get("chip_dir", "data/chips")
    out_dir   = cfg["paths"]["processed_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(chip_dir, "*_img.tif")))
    if not img_files:
        log.error(f"No V2 chip files found in {chip_dir}. Run gee_fetch.py --mode chips first.")
        return False

    log.info(f"Processing {len(img_files)} V2 chips from {chip_dir}...")

    manifest: List[dict] = []
    patch_id: int = 0
    skipped: int  = 0

    for img_path in tqdm(img_files, desc="V2 chips"):
        lbl_path = img_path.replace("_img.tif", "_lbl.tif")
        if not os.path.exists(lbl_path):
            log.warning(f"  No label for {img_path}, skipping.")
            skipped += 1
            continue

        stack = load_v2_chip(img_path)
        if stack is None:
            skipped += 1
            continue

        label = load_v2_label(lbl_path)

        # Parse metadata from filename:
        #   Positive: chip_pos_{year}_{id:05d}_img.tif   → parts: ['chip','pos',year,id]
        #   Negative: chip_neg_{type}_{year}_{id:05d}_img.tif → parts: ['chip','neg',type,year,id]
        stem  = Path(img_path).stem.replace("_img", "")
        parts = stem.split("_")
        polarity  = parts[1]                                  # 'pos' or 'neg'
        if polarity == "neg":
            neg_type = parts[2]                               # 'summer' | 'postmonsoon'
            year     = int(parts[3])
        else:
            neg_type = None
            year     = int(parts[2])

        # Save tensors
        img_save = os.path.join(out_dir, f"img_{patch_id:06d}.npy")
        lbl_save = os.path.join(out_dir, f"lbl_{patch_id:06d}.npy")
        np.save(img_save, stack.astype(np.float32))
        np.save(lbl_save, label.astype(np.uint8))

        change_ratio = float(label.mean())
        manifest.append({
            "id":                patch_id,
            "img_path":          img_save,
            "lbl_path":          lbl_save,
            "source_chip":       img_path,
            "polarity":          polarity,
            "neg_type":          neg_type or "",
            "year":              year,
            "forest_loss_ratio": change_ratio,
            "has_change":        int(change_ratio > 0.01),
            "temporal_schema":   "365_30_0",   # V2 standard
        })
        patch_id += 1  # type: ignore[operator]

    log.info(f"  Processed: {patch_id} chips | Skipped: {skipped}")

    df = pd.DataFrame(manifest)
    _save_splits(df, out_dir, cfg)
    return True


# ── V1 legacy pipeline ────────────────────────────────────────────────────────
# Retained for backward compatibility while V2 GEE exports are submitted.

def _load_hls_tif(tif_path: _PathLike) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = 0.0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_stack(data)


def _load_label_tif(tif_path: _PathLike) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        return src.read(1).astype(np.uint8)


def _tile(image: np.ndarray, tile_size: int = 224,
          overlap: int = 64) -> List[Tuple]:
    is_2d = image.ndim == 2
    if is_2d:
        image = image[np.newaxis]
    C, H, W = image.shape
    stride  = tile_size - overlap
    tiles   = []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            patch = image[:, r:r+tile_size, c:c+tile_size]
            tiles.append((patch[0] if is_2d else patch, (r, c)))
    return tiles


def process_v1_legacy(cfg: dict):
    """V1 pipeline: annual-composite sliding-window tiling (unchanged from original)."""
    log.warning(
        "⚠️  Running V1 (legacy) pipeline. Temporal intervals are NOT standardised.\n"
        "   This is suitable only for initial experimentation. Retrain with V2 chips\n"
        "   once GEE export completes to fix temporal mismatch at inference time."
    )

    raw_dir    = cfg["paths"]["raw_dir"]
    labels_dir = cfg["paths"]["labels_dir"]
    out_dir    = cfg["paths"]["processed_dir"]
    tile_size  = cfg["model"]["img_size"]
    num_frames = cfg["model"]["num_frames"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    hls_files = sorted([f for f in glob.glob(os.path.join(raw_dir, "hls_*.tif"))
                        if "_test" not in f])
    lbl_files = sorted([f for f in glob.glob(os.path.join(labels_dir, "loss_*.tif"))
                        if "_test" not in f])

    hls_dict: Dict[int, np.ndarray] = {}
    for f in tqdm(hls_files, desc="Loading HLS"):
        yr = int(Path(f).stem.split("_")[1])
        hls_dict[yr] = _load_hls_tif(f)

    lbl_dict: Dict[int, np.ndarray] = {}
    for f in tqdm(lbl_files, desc="Loading labels"):
        yr = int(Path(f).stem.split("_")[1])
        lbl_dict[yr] = _load_label_tif(f)

    years: List[int] = sorted(hls_dict.keys())
    samples = []
    for i in range(num_frames - 1, len(years)):
        fy: List[int] = list(years[i - (num_frames - 1): i + 1])  # type: ignore[index]
        if any(y not in hls_dict for y in fy) or fy[-1] not in lbl_dict:
            continue
        shapes = [hls_dict[y].shape for y in fy]
        if len(set(shapes)) > 1:
            continue
        stack = np.stack([hls_dict[y] for y in fy], axis=0)   # (T, C, H, W)
        samples.append({"image": stack, "label": lbl_dict[fy[-1]], "years": fy})

    manifest: List[dict] = []
    patch_id: int = 0
    for sample in tqdm(samples, desc="Tiling"):
        T  = sample["image"].shape[0]
        img_tiles = _tile(sample["image"][0], tile_size, 64)
        for _, (r, c) in img_tiles:
            mf = np.stack([sample["image"][t, :, r:r+tile_size, c:c+tile_size]
                           for t in range(T)])
            if mf.mean() < 0.01:
                continue
            lp = sample["label"][r:r+tile_size, c:c+tile_size]
            img_save = os.path.join(out_dir, f"img_{patch_id:06d}.npy")
            lbl_save = os.path.join(out_dir, f"lbl_{patch_id:06d}.npy")
            np.save(img_save, mf.astype(np.float32))
            np.save(lbl_save, lp.astype(np.uint8))
            manifest.append({
                "id": patch_id, "img_path": img_save, "lbl_path": lbl_save,
                "years": str(sample["years"]), "forest_loss_ratio": float(lp.mean()),
                "has_change": int(lp.mean() > 0.01), "temporal_schema": "v1_annual",
            })
            patch_id += 1  # type: ignore[operator]

    _save_splits(pd.DataFrame(manifest), out_dir, cfg)


# ── Split + save ──────────────────────────────────────────────────────────────

def _save_splits(df: pd.DataFrame, out_dir: str, cfg: dict):
    if len(df) == 0:
        log.error("No patches generated — check input files.")
        return

    manifest_path = os.path.join(out_dir, "manifest.csv")
    df.to_csv(manifest_path, index=False)

    train_df, test_df = train_test_split(
        df, test_size=cfg["training"]["test_split"],
        random_state=cfg["training"]["seed"], stratify=df["has_change"])
    train_df, val_df = train_test_split(
        train_df, test_size=cfg["training"]["val_split"],
        random_state=cfg["training"]["seed"], stratify=train_df["has_change"])

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir,   "val.csv"),   index=False)
    test_df.to_csv(os.path.join(out_dir,  "test.csv"),  index=False)

    pos = df["has_change"].sum()
    log.info(f"\n✅ Preprocessing complete!")
    log.info(f"   Mode:        {df.get('temporal_schema', ['?']).iloc[0]}")
    log.info(f"   Total:       {len(df)} patches ({pos} positive = {100*pos/len(df):.1f}%)")
    log.info(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    log.info(f"   Manifest:    {manifest_path}")

    if df.get("temporal_schema", pd.Series(["v1"])).iloc[0] == "365_30_0":
        log.info(f"\n   ✅ V2 temporal schema (365/30/0 look-backs) — inference-aligned.")
    else:
        log.warning(f"\n   ⚠️  V1 annual schema — temporal mismatch at inference time.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GEE data for Prithvi V2")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "v1", "v2"],
                        help="auto=try V2 first, fall back to V1")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "v1":
        process_v1_legacy(cfg)
    elif args.mode == "v2":
        process_v2_chips(cfg)
    else:
        # Auto: try V2 chips first, fall back to V1 annual composites
        chip_dir   = cfg["paths"].get("chip_dir", "data/chips")
        v2_chips   = glob.glob(os.path.join(chip_dir, "*_img.tif"))
        if v2_chips:
            log.info(f"Found {len(v2_chips)} V2 chips — running V2 pipeline.")
            success = process_v2_chips(cfg)
            if not success:
                log.warning("V2 failed, falling back to V1.")
                process_v1_legacy(cfg)
        else:
            log.info("No V2 chips found — running V1 (legacy) pipeline.")
            process_v1_legacy(cfg)
