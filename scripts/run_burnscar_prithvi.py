"""
Prithvi-EO-2.0 BurnScar Inference Pipeline for Guna Division
==============================================================
Runs ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars over all downloaded
HLS S30 granules for Guna Division, MP (2013 - present).

Model: 300M-parameter segmentation model (UNetDecoder head on Prithvi-EO-2.0
       ViT backbone), IoU=87.5% on HLS burn scar benchmark.

Performance:
  CPU:  ~1.75s/patch, ~28 patches/granule => ~49s per granule
  GPU:  ~0.3s/batch(4), ~7 batches/granule => ~2s per granule (DirectML/CUDA)

Strategy: FIRMS-guided (default). Only processes HLS granules where FIRMS
fire detections exist within ±30 days in the same tile's geographic extent.
Use --all to process every granule.

Output:
  data_lake/burn_scars/hls_s30/<tile>/<YYYY>/<MM>/<DD>/
      burnscar_<granule_id>.tif   (uint8: 0=unburned, 1=burn scar)

Usage:
  python scripts/run_burnscar_prithvi.py                     # FIRMS-guided
  python scripts/run_burnscar_prithvi.py --all               # all granules
  python scripts/run_burnscar_prithvi.py --year 2024
  python scripts/run_burnscar_prithvi.py --tile T43RGH
  python scripts/run_burnscar_prithvi.py --gpu                # DirectML/CUDA
  python scripts/run_burnscar_prithvi.py --gpu --batch-size 8 # larger batches
  python scripts/run_burnscar_prithvi.py --dry-run

Requirements:
  pip install terratorch einops albumentations
  Model checkpoint: models/prithvi_burnscar/Prithvi_EO_V2_300M_BurnScars.pt
"""

import os, sys, time, logging, argparse, warnings
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

# ── Fix PostgreSQL/PostGIS PROJ DB conflict (must happen before rasterio import)
# Override PROJ_DATA so rasterio/GDAL uses pyproj's own PROJ DB (version >=6)
# instead of the one bundled with PostgreSQL/PostGIS (version 2).
try:
    import pyproj
    _pyproj_data = str(Path(pyproj.datadir.get_data_dir()))
    os.environ["PROJ_DATA"] = _pyproj_data
    os.environ["PROJ_LIB"]  = _pyproj_data
except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
# Suppress verbose GDAL/PROJ error messages in log (warnings will be in file)
os.environ.setdefault("CPL_LOG_ERRORS", "OFF")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_LAKE  = ROOT / "data_lake"
MODEL_DIR  = ROOT / "models" / "prithvi_burnscar"
HLS_BASE   = DATA_LAKE / "satellite_imagery" / "hls_s30"
OUT_BASE   = DATA_LAKE / "burn_scars" / "hls_s30"
FIRMS_DIR  = DATA_LAKE / "fire_detections"
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
CKPT_PATH  = MODEL_DIR / "Prithvi_EO_V2_300M_BurnScars.pt"
CFG_PATH   = MODEL_DIR / "burn_scars_config.yaml"
HF_REPO    = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"

# ── Prithvi band config ───────────────────────────────────────────────────────
# HLS S30: B02=Blue, B03=Green, B04=Red, B8A=NIR_Narrow, B11=SWIR1, B12=SWIR2
BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Normalization statistics (from burn_scars_config.yaml)
MEANS = np.array([0.033349706, 0.057011855, 0.058897481,
                  0.232324511, 0.197285485, 0.119449142], dtype=np.float32)
STDS  = np.array([0.022691356, 0.026807560, 0.040041098,
                  0.077917324, 0.087087388, 0.072419795], dtype=np.float32)

NO_DATA_DN    = -9999        # HLS nodata in raw digital numbers
IMG_SIZE      = 512          # Prithvi chip size

# ── Logging ───────────────────────────────────────────────────────────────────
fh = logging.FileHandler(LOG_DIR / "burnscar_prithvi.log", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[fh, ch],
)
log = logging.getLogger("burnscar")


# ── Model download ────────────────────────────────────────────────────────────
def ensure_model():
    """Download model checkpoint and config if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if CKPT_PATH.exists() and CFG_PATH.exists():
        return
    log.info(f"Downloading Prithvi-EO-2.0-300M-BurnScars from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        if not CKPT_PATH.exists():
            hf_hub_download(HF_REPO, "Prithvi_EO_V2_300M_BurnScars.pt",
                            local_dir=MODEL_DIR)
            log.info(f"  Checkpoint: {CKPT_PATH}")
        if not CFG_PATH.exists():
            hf_hub_download(HF_REPO, "burn_scars_config.yaml",
                            local_dir=MODEL_DIR)
            log.info(f"  Config: {CFG_PATH}")
    except Exception as e:
        log.error(f"Model download failed: {e}")
        log.error("Run manually: pip install huggingface_hub && "
                  f"huggingface-cli download {HF_REPO}")
        sys.exit(1)


# ── Device detection ─────────────────────────────────────────────────────────
def _get_device(use_gpu):
    """Return best available device (DirectML > CUDA > CPU)."""
    if not use_gpu:
        return "cpu"
    try:
        import torch_directml
        dev = torch_directml.device()
        log.info("DirectML GPU: %s", torch_directml.device_name(0))
        return dev
    except ImportError:
        pass
    import torch
    if torch.cuda.is_available():
        log.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.warning("No GPU found, falling back to CPU")
    return "cpu"


def _patch_conv3d_for_directml(model):
    """Replace all Conv3d(kT=1) with Conv2d wrappers for DirectML.

    DirectML does not support F.conv3d. Prithvi patch_embed uses
    Conv3d(kernel=(1,H,W)) which is mathematically identical to Conv2d
    applied per temporal frame.
    """
    import torch
    import torch.nn as nn

    class _Conv3dShim(nn.Module):
        """Conv3d(kT=1) emulated as Conv2d for DirectML."""
        def __init__(self, conv3d):
            super().__init__()
            kH, kW = conv3d.kernel_size[1], conv3d.kernel_size[2]
            sH, sW = conv3d.stride[1], conv3d.stride[2]
            pH, pW = conv3d.padding[1], conv3d.padding[2]
            dH, dW = conv3d.dilation[1], conv3d.dilation[2]
            self.conv2d = nn.Conv2d(
                conv3d.in_channels, conv3d.out_channels,
                kernel_size=(kH, kW), stride=(sH, sW),
                padding=(pH, pW), dilation=(dH, dW),
                bias=(conv3d.bias is not None),
            )
            with torch.no_grad():
                self.conv2d.weight.copy_(conv3d.weight.squeeze(2))
                if conv3d.bias is not None:
                    self.conv2d.bias.copy_(conv3d.bias)

        def forward(self, x):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, C, H, W)
            x = self.conv2d(x)
            D, h, w = x.shape[1], x.shape[2], x.shape[3]
            x = x.reshape(B, T, D, h, w).permute(0, 2, 1, 3, 4).contiguous()
            return x

    count = 0
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and module.kernel_size[0] == 1:
            replacements.append(name)
    for name in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], _Conv3dShim(orig))
        count += 1
        log.info("  DML: Conv3d -> Conv2d at '%s'", name)
    return count


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(device="cpu"):
    """Load Prithvi-EO-2.0-300M-BurnScars via terratorch."""
    ensure_model()
    is_dml = str(device) not in ("cpu", "cuda")
    dev_label = "DirectML" if is_dml else str(device).upper()
    log.info("Loading Prithvi-EO-2.0-300M-BurnScars on %s...", dev_label)
    t0 = time.time()
    try:
        from terratorch.tasks import SemanticSegmentationTask
        # Always load to CPU first, then patch and move
        task = SemanticSegmentationTask.load_from_checkpoint(
            str(CKPT_PATH), map_location="cpu")

        # Patch Conv3d for DirectML before moving to device
        if is_dml:
            n = _patch_conv3d_for_directml(task.model)
            log.info("  Patched %d Conv3d layer(s) for DirectML", n)

        task.model.to(device)
        task.model.eval()
        log.info("Model ready on %s (%.1fs) | type=%s",
                 dev_label, time.time() - t0, type(task.model).__name__)
        return task, device
    except ImportError:
        log.error("terratorch not installed. Run: pip install terratorch")
        sys.exit(1)
    except Exception as e:
        log.error("Model load failed: %s", e)
        sys.exit(1)


# ── FIRMS data loading ────────────────────────────────────────────────────────
def load_firms_index():
    """
    Build a lookup: {(year, month)} -> GeoDataFrame of fire detections.
    Uses our downloaded VIIRS SNPP SP annual parquets.
    Returns dict: {year: pd.DataFrame} with lat/lon/acq_date columns.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}

    index = {}
    viirs_dir = FIRMS_DIR / "firms_viirs_snpp_sp"
    for parquet in sorted(viirs_dir.glob("*_annual.parquet")):
        try:
            yr = int(parquet.stem.split("_")[3])  # viirs_snpp_sp_2013_annual
            df = pd.read_parquet(parquet)
            if df.empty:
                continue
            # Parse acquisition date
            if "acq_date" in df.columns:
                df["acq_date"] = pd.to_datetime(df["acq_date"])
            elif "date" in df.columns:
                df["acq_date"] = pd.to_datetime(df["date"])
            index[yr] = df
            log.info(f"  FIRMS {yr}: {len(df):,} detections loaded")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
    return index


def tile_bbox_from_files(tile: str) -> tuple:
    """
    Derive approximate WGS84 bounding box of an HLS tile from existing TIF files.
    Returns (west, south, east, north) in degrees.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    tile_dir = HLS_BASE / tile
    for tif in tile_dir.rglob("*.tif"):
        try:
            with rasterio.open(tif) as src:
                bounds = transform_bounds(
                    src.crs, "EPSG:4326",
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                )
            return bounds  # (west, south, east, north)
        except Exception:
            continue
    return None


def has_firms_fire(firms_index: dict, tile_bbox: tuple,
                    acq_date: date, window_days: int = 30) -> bool:
    """
    Check if any FIRMS fire detection falls within tile_bbox within
    ±window_days of acq_date.
    """
    if not firms_index or tile_bbox is None:
        return True  # if no FIRMS data, don't filter

    west, south, east, north = tile_bbox
    # Expand bounding box slightly for edge fires
    buf = 0.1
    west -= buf; south -= buf; east += buf; north += buf

    start_dt = datetime.combine(acq_date - timedelta(days=window_days), datetime.min.time())
    end_dt   = datetime.combine(acq_date + timedelta(days=window_days), datetime.max.time())

    years = {acq_date.year}
    if acq_date.month <= window_days // 30 + 1:
        years.add(acq_date.year - 1)
    if acq_date.month >= 12 - window_days // 30:
        years.add(acq_date.year + 1)

    for yr in years:
        df = firms_index.get(yr)
        if df is None or df.empty:
            continue
        try:
            mask = (
                (df["latitude"]  >= south) & (df["latitude"]  <= north) &
                (df["longitude"] >= west)  & (df["longitude"] <= east)  &
                (df["acq_date"] >= start_dt) & (df["acq_date"] <= end_dt)
            )
            if mask.any():
                return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
    return False


# ── Band loading & normalization ──────────────────────────────────────────────
def load_and_normalize(band_paths: dict) -> tuple:
    """
    Load 6 band TIFs and return normalized (6, H, W) float32 array + rasterio meta.

    Preprocessing (matches Prithvi training pipeline):
      1. Stack 6 bands  →  (6, H, W) int16
      2. Clamp nodata (-9999) to 0
      3. Divide by 10000 → reflectance [0, 1]
      4. Z-score normalize with per-band mean/std from config
    """
    import rasterio

    arrays, meta = [], None
    for band in BAND_ORDER:
        path = band_paths.get(band)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Band {band} not found in {band_paths}")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
        arrays.append(arr)

    img = np.stack(arrays, axis=0)  # (6, H, W)
    # Clamp nodata
    img = np.where(img == NO_DATA_DN, 0.0, img)
    # Reflectance
    img = img / 10000.0
    img = np.clip(img, 0.0, 1.0)
    # Z-score
    img = (img - MEANS[:, None, None]) / STDS[:, None, None]
    return img.astype(np.float32), meta


# ── Sliding-window inference ──────────────────────────────────────────────────
def run_inference(model, img6hw: np.ndarray, device="cpu",
                  batch_size: int = 1) -> np.ndarray:
    """
    Run Prithvi BurnScar inference on (6, H, W) normalized array.
    Returns (H, W) uint8 mask: 0 = unburned, 1 = burn scar.
    Uses non-overlapping 512x512 sliding window with reflect-padding.

    Args:
        model: terratorch SemanticSegmentationTask
        img6hw: normalized (6, H, W) float32 array
        device: torch device (cpu, cuda, or DirectML)
        batch_size: patches per GPU batch (1 for CPU, 4-8 for GPU)
    """
    import torch
    from einops import rearrange

    original_h, original_w = img6hw.shape[1], img6hw.shape[2]
    pad_h = (IMG_SIZE - (original_h % IMG_SIZE)) % IMG_SIZE
    pad_w = (IMG_SIZE - (original_w % IMG_SIZE)) % IMG_SIZE

    # Pad: shape (1, 6, H+pad, W+pad)
    padded = np.pad(
        img6hw[np.newaxis],
        ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )

    full_img = torch.tensor(padded, dtype=torch.float32)  # (1, 6, H, W)

    # Build sliding windows -> (n_patches, 6, IMG_SIZE, IMG_SIZE)
    windows = full_img.unfold(2, IMG_SIZE, IMG_SIZE).unfold(3, IMG_SIZE, IMG_SIZE)
    h1, w1 = windows.shape[2], windows.shape[3]
    windows = rearrange(
        windows, "b c h1 w1 h w -> (b h1 w1) c h w",
        h=IMG_SIZE, w=IMG_SIZE,
    )

    n_patches = windows.shape[0]
    pred_imgs = []

    # Process in batches for GPU throughput
    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        batch = windows[start:end].to(device)        # (bs, 6, 512, 512)
        with torch.no_grad():
            out = model.model(batch)
        y_hat = out.output.argmax(dim=1, keepdim=True).float()  # (bs, 1, 512, 512)
        pred_imgs.append(y_hat.cpu())

    pred_imgs = torch.cat(pred_imgs, dim=0)  # (n_patches, 1, 512, 512)
    pred_imgs = rearrange(
        pred_imgs,
        "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
        h=IMG_SIZE, w=IMG_SIZE, b=1, c=1, h1=h1, w1=w1,
    )
    # Crop back to original size
    pred_imgs = pred_imgs[0, 0, :original_h, :original_w]
    return pred_imgs.numpy().astype(np.uint8)


# ── Save burn scar GeoTIFF ────────────────────────────────────────────────────
def save_mask(mask: np.ndarray, meta: dict, out_path: Path):
    """Save (H, W) uint8 burn scar mask as GeoTIFF."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta = meta.copy()
    save_meta.update(
        count=1, dtype="uint8", compress="lzw",
        nodata=255,   # 255 = masked/no-data
    )
    with rasterio.open(out_path, "w", **save_meta) as dst:
        dst.write(mask, 1)


# ── Granule discovery ─────────────────────────────────────────────────────────
def find_granules(year_filter=None, tile_filter=None):
    """
    Yield (acq_date, tile, band_dict, out_path) for all complete HLS S30 granules.
    band_dict: {band_id: Path}
    """
    if not HLS_BASE.exists():
        log.warning(f"HLS directory not found: {HLS_BASE}")
        return

    for tile_dir in sorted(HLS_BASE.iterdir()):
        if not tile_dir.is_dir() or not tile_dir.name.startswith("T"):
            continue
        tile = tile_dir.name
        if tile_filter and tile != tile_filter:
            continue

        for yr_dir in sorted(tile_dir.iterdir()):
            if not yr_dir.is_dir() or not yr_dir.name.isdigit():
                continue
            if year_filter and yr_dir.name != str(year_filter):
                continue
            for mo_dir in sorted(yr_dir.iterdir()):
                if not mo_dir.is_dir():
                    continue
                for day_dir in sorted(mo_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue

                    tifs = list(day_dir.glob("*.tif"))
                    # Group by granule ID (strip band suffix from filename)
                    granules: dict = defaultdict(dict)
                    for tif in tifs:
                        for band in BAND_ORDER:
                            if f".{band}." in tif.name:
                                gid = tif.name.split(f".{band}.")[0]
                                granules[gid][band] = tif
                                break

                    for gid, bands in granules.items():
                        if len(bands) < 6:
                            continue  # skip incomplete granules
                        try:
                            acq_date = date(
                                int(yr_dir.name),
                                int(mo_dir.name),
                                int(day_dir.name),
                            )
                        except ValueError:
                            continue

                        out_dir  = OUT_BASE / tile / yr_dir.name / mo_dir.name / day_dir.name
                        out_path = out_dir / f"burnscar_{gid}.tif"
                        yield acq_date, tile, bands, out_path


# ── Main processing loop ──────────────────────────────────────────────────────
def run_pipeline(
    year_filter:    int  = None,
    tile_filter:    str  = None,
    firms_guided:   bool = True,
    firms_window:   int  = 30,
    dry_run:        bool = False,
    all_granules:   bool = False,
    use_gpu:        bool = False,
    gpu_batch_size: int  = 4,
):
    """Process all HLS S30 granules and generate burn scar masks."""
    # Load FIRMS index for filtering
    firms_index = {}
    tile_bboxes  = {}
    if firms_guided and not all_granules:
        log.info("Loading FIRMS fire detection index for guided inference...")
        firms_index = load_firms_index()
        if not firms_index:
            log.warning("No FIRMS data found — processing all granules.")

    # Discover granules
    granules = list(find_granules(year_filter, tile_filter))
    log.info(f"Found {len(granules)} complete HLS S30 granules")

    # Apply FIRMS filter
    if firms_guided and firms_index and not all_granules:
        log.info(f"Applying FIRMS filter (+-{firms_window} days)...")
        import rasterio
        from rasterio.warp import transform_bounds

        filtered = []
        for acq_date, tile, bands, out_path in granules:
            # Get tile bbox (cached)
            if tile not in tile_bboxes:
                tile_bboxes[tile] = tile_bbox_from_files(tile)
            bbox = tile_bboxes.get(tile)
            if has_firms_fire(firms_index, bbox, acq_date, firms_window):
                filtered.append((acq_date, tile, bands, out_path))

        pct = 100 * len(filtered) / max(len(granules), 1)
        log.info(f"FIRMS filter: {len(filtered)}/{len(granules)} granules "
                 f"({pct:.0f}%) have nearby fire activity")
        granules = filtered
    else:
        log.info("Processing ALL granules (no FIRMS filter)")

    # Separate already-done vs pending
    pending = [(d, t, b, p) for d, t, b, p in granules if not p.exists()]
    done    = len(granules) - len(pending)
    log.info(f"Pending: {len(pending)} | Already done: {done}")

    if dry_run:
        for acq_date, tile, bands, out_path in pending[:5]:
            log.info(f"  DRY-RUN: {acq_date} {tile} -> {out_path.name}")
        log.info(f"  ... {len(pending)} granules total (dry run only shows 5)")
        return {"total": len(granules), "pending": len(pending),
                "done": done, "errors": 0}

    if not pending:
        log.info("All granules already processed!")
        return {"total": len(granules), "pending": 0, "done": done, "errors": 0}

    # Load model on selected device
    device = _get_device(use_gpu)
    bs = gpu_batch_size if str(device) != "cpu" else 1
    model, device = load_model(device)

    # Process granules
    t_start = time.monotonic()
    n_done = 0
    n_errors = 0
    total_burn_px = 0

    for i, (acq_date, tile, bands, out_path) in enumerate(pending):
        t0 = time.monotonic()
        try:
            img, meta = load_and_normalize(bands)
            mask = run_inference(model, img, device=device, batch_size=bs)
            save_mask(mask, meta, out_path)

            burn_px  = int(mask.sum())
            burn_pct = 100.0 * burn_px / mask.size
            total_burn_px += burn_px
            sz = out_path.stat().st_size // 1024
            n_done += 1

            elapsed  = time.monotonic() - t0
            total_e  = time.monotonic() - t_start
            rate     = n_done / total_e
            remaining = len(pending) - (i + 1)
            eta_min  = remaining / rate / 60 if rate > 0 else 0

            log.info(
                f"  {acq_date} {tile}: burn={burn_px:,}px ({burn_pct:.1f}%) | "
                f"{sz}KB | {elapsed:.1f}s | "
                f"[{n_done}/{len(pending)}] ETA {eta_min:.0f}min"
            )

        except Exception as exc:
            n_errors += 1
            log.warning(f"  {acq_date} {tile}: ERROR — {exc}")

    total_s = time.monotonic() - t_start
    log.info(
        f"\n=== BurnScar done: {n_done} masks | {n_errors} errors | "
        f"{total_burn_px/1e6:.2f}M burn pixels | {total_s/60:.1f}min ==="
    )
    return {"total": len(granules), "pending": len(pending),
            "done": n_done, "errors": n_errors}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Prithvi-EO-2.0 BurnScar inference for Guna Division"
    )
    p.add_argument("--year",   type=int, default=None,
                   help="Filter to a single year (e.g. 2024)")
    p.add_argument("--tile",   type=str, default=None,
                   help="Filter to a single MGRS tile (e.g. T43RGH)")
    p.add_argument("--all",    action="store_true",
                   help="Process all granules (no FIRMS filter)")
    p.add_argument("--firms-window", type=int, default=30,
                   help="FIRMS fire search window in days (default 30)")
    p.add_argument("--dry-run", action="store_true",
                   help="List granules without running inference")
    p.add_argument("--gpu", action="store_true",
                   help="Run inference on GPU (DirectML/CUDA)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Patches per GPU batch (default 4, ignored on CPU)")
    p.add_argument("--download-only", action="store_true",
                   help="Only download model checkpoint, then exit")
    args = p.parse_args()

    if args.download_only:
        ensure_model()
        log.info("Model ready.")
        return

    summary = run_pipeline(
        year_filter    = args.year,
        tile_filter    = args.tile,
        firms_guided   = not args.all,
        firms_window   = args.firms_window,
        dry_run        = args.dry_run,
        all_granules   = args.all,
        use_gpu        = args.gpu,
        gpu_batch_size = args.batch_size,
    )

    log.info("=== Final Summary ===")
    log.info(f"  Total granules  : {summary['total']}")
    log.info(f"  Processed       : {summary['done']}")
    log.info(f"  Errors          : {summary['errors']}")


if __name__ == "__main__":
    main()
