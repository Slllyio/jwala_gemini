"""
Van Suraksha Alert Generator
============================
Runs end-to-end inference for the Guna District forest monitor:

  1. Fetches a fresh 3-frame chip from GEE (365/30/0 schema)
  2. Runs Prithvi change-detection model → detection confidence map
  3. Suppresses false positives using Sentinel-1 VH radar (≥2 dB drop required)
  4. Scores alerts with a separate zone-level risk layer
  5. Exports: GeoTIFF mask · KMZ for Google Earth · PDF report
  6. Sends Telegram notification with PDF + KMZ attachments

Architecture note
-----------------
Detection confidence  (model output — WHAT is happening, how certain)
Zone risk score       (spatial prior — WHERE is the pressure highest)
These are kept SEPARATE and presented side-by-side — never blended —
so field officers can trust the signal without black-box risk magic.

Usage
-----
    python src/inference/generate_alerts.py \\
        --config config.yaml \\
        --checkpoint outputs/checkpoints/best_detect.pth \\
        [--date 2024-06-01]          # anchor date (default: today)
        [--threshold 0.35]           # detection confidence threshold
        [--telegram]                 # send Telegram notification
        [--out-dir outputs/alerts]

Environment variables for Telegram
-----------------------------------
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    TELEGRAM_CHAT_ID    — destination chat / group ID
"""

import sys
import os

# ── PROJ fix ────────────────────────────────────────────────────────────────
# PostgreSQL ships its own (old) PROJ.db that wins over the venv's newer one
# when both are on the PATH. Force GDAL + pyproj to use the venv-bundled data
# directory BEFORE rasterio / pyproj have a chance to initialise.
# Priority: rasterio's proj_data (ships with the GDAL wheel, version-matched)
#           → then pyproj's proj_dir (fallback)
try:
    import rasterio as _rasterio
    _rasterio_proj = os.path.join(os.path.dirname(_rasterio.__file__), "proj_data")
    if os.path.isdir(_rasterio_proj):
        _proj_dir = _rasterio_proj
    else:
        raise FileNotFoundError("rasterio proj_data not found")
except Exception:
    try:
        import pyproj as _pyproj
        _proj_dir = _pyproj.datadir.get_data_dir()
    except Exception:
        _proj_dir = None

if _proj_dir:
    os.environ["PROJ_DATA"] = _proj_dir   # PROJ ≥ 9
    os.environ["PROJ_LIB"]  = _proj_dir   # PROJ < 9 compat alias
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None  # type: ignore[union-attr]

import argparse
import logging
import json
import math
import struct
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _torch_available = False
import yaml
from matplotlib import cm, colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import torch_directml  # type: ignore[import]
    _dml = torch_directml.device()
except (ImportError, Exception):
    _dml = None

try:
    import ee
    _ee_available = True
except ImportError:
    _ee_available = False

try:
    from src.model.full_model import build_model as _build_model  # type: ignore[import]
    _model_available = True
except ImportError:
    _build_model = None  # type: ignore[assignment]
    _model_available = False

try:
    from src.data.preprocess import load_v2_chip, normalize_stack  # type: ignore[import]
    _preprocess_available = True
except ImportError:
    load_v2_chip = None  # type: ignore[assignment]
    normalize_stack = None  # type: ignore[assignment]
    _preprocess_available = False

from src.data.gee_fetch import (
    init_gee, make_three_frame_stack, export_local, load_config,
    _PRITHVI_BANDS, fetch_s1_monthly_vh_stack,
)

# CuSum scorer — imported at module level to avoid runtime sys.path surgery.
# enhanced_alert_windows lives in scripts/; add the project root (two levels up
# from src/inference/) to sys.path once at import time.
import sys as _sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
try:
    from scripts.enhanced_alert_windows import compute_sar_cusum_score as _cusum_score_fn
    _cusum_available = True
except ImportError:
    _cusum_score_fn = None   # type: ignore[assignment]
    _cusum_available = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants (defaults — overridden by config.yaml:inference at runtime) ──────
#
# All of these can be tuned via config.yaml under the  inference:  key.
# Module-level values serve as safe defaults when the key is absent.

_CHIP_BUFFER_M      = 1120    # half of 2240m = 224 px × 10 m
_S1_VH_DROP_DB      = 2.0     # minimum VH drop to accept a change detection
_S1_WINDOW_DAYS     = 45      # days either side of anchor for S1 composite
_DEFAULT_THRESHOLD  = 0.35    # detection confidence cut-off
_MIN_ALERT_AREA_HA  = 0.2     # ignore patches smaller than this (noise)
_CUSUM_SCORE_THRESH = 0.5     # CuSum zone-mean threshold to flag gradual SAR loss
_CUSUM_SENSITIVITY  = 1.5     # CuSum sensitivity (higher = more sensitive)


def load_inference_cfg(cfg: dict) -> dict:
    """
    Resolve inference knobs from config, falling back to module-level defaults.

    All keys live under ``config.yaml:inference``.  Call once at the top of
    run_alert_pipeline to get a single resolved settings dict.

    Returns
    -------
    dict with keys:
        threshold, min_area_ha, s1_drop_db, cusum_thresh, cusum_sensitivity,
        s1_window_days
    """
    inf = cfg.get("inference", {})
    return {
        "threshold":         float(inf.get("default_threshold",   _DEFAULT_THRESHOLD)),
        "min_area_ha":       float(inf.get("min_alert_area_ha",   _MIN_ALERT_AREA_HA)),
        "s1_drop_db":        float(inf.get("s1_vh_drop_db",       _S1_VH_DROP_DB)),
        "cusum_thresh":      float(inf.get("cusum_score_thresh",  _CUSUM_SCORE_THRESH)),
        "cusum_sensitivity": float(inf.get("cusum_sensitivity",   _CUSUM_SENSITIVITY)),
        "s1_window_days":    int(  inf.get("s1_window_days",      _S1_WINDOW_DAYS)),
    }

# Risk zone categories (purely spatial prior — independent of model)
_RISK_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
_RISK_COLORS = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c", 4: "#8e44ad"}

# ── Device helper ──────────────────────────────────────────────────────────────

def get_device():
    if not _torch_available or torch is None:
        return "cpu"   # rules mode — torch not needed
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _dml is not None:
        return _dml
    return torch.device("cpu")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(cfg: dict, checkpoint_path: str, device) -> "torch.nn.Module":
    """Load trained PrithviForestChange model from checkpoint."""
    if not _torch_available or torch is None or _build_model is None:
        raise RuntimeError(
            "torch / src.model.full_model not available — "
            "cannot load neural model. Use detection_mode: rules instead."
        )
    model = _build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    log.info(f"Model loaded: {checkpoint_path}")
    return model


# ── GEE chip fetching (live inference mode) ──────────────────────────────────

def fetch_live_chip(
    aoi: "ee.Geometry",
    anchor_date: str,
    out_tif: str,
    cloud_thresh: int = 20,
) -> bool:
    """
    Fetch a fresh 3-frame HLS chip from GEE centered on anchor_date.

    Uses 365/30/0 temporal schema (identical to training) so model
    temporal positional embeddings align correctly.

    Returns True on success.
    """
    if not _ee_available:
        log.error("earthengine-api not available. Cannot fetch live chip.")
        return False

    log.info(f"Fetching live chip: anchor={anchor_date}")
    chip_geom = aoi.centroid(maxError=10).buffer(_CHIP_BUFFER_M).bounds()

    stack  = make_three_frame_stack(chip_geom, anchor_date, cloud_thresh)
    all_bands = [f"{b}_{i}" for i in range(3) for b in _PRITHVI_BANDS]

    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    export_local(stack, out_tif, chip_geom, scale=10, bands=all_bands)
    log.info(f"Chip saved: {out_tif}")
    return True


# ── Sentinel-1 VH false-positive filter ──────────────────────────────────────

def check_s1_vh_drop(
    aoi: "ee.Geometry",
    anchor_date: str,
    min_drop_db: float = _S1_VH_DROP_DB,
) -> Tuple[bool, float]:
    """
    Query S1 VH backscatter before and after the anchor date.

    A real felling event shows ≥2 dB drop in VH (canopy gone → less volume
    scattering).  Phenological browning (no structural change) does NOT cause
    this drop — eliminating the dry-season false-positive problem.

    Returns
    -------
    (confirmed, drop_db)
        confirmed: True if VH drop meets threshold (real structural change)
        drop_db:   measured drop in dB (positive = drop)
    """
    if not _ee_available:
        log.warning("S1 filter skipped: earthengine-api not available.")
        return True, 0.0

    try:
        anchor = datetime.strptime(anchor_date, "%Y-%m-%d")
        before_start = (anchor - timedelta(days=_S1_WINDOW_DAYS + 30)).strftime("%Y-%m-%d")
        before_end   = (anchor - timedelta(days=15)).strftime("%Y-%m-%d")
        after_start  = anchor_date
        after_end    = (anchor + timedelta(days=_S1_WINDOW_DAYS)).strftime("%Y-%m-%d")

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select("VH")
        )

        vh_before = (
            s1.filterDate(before_start, before_end)
            .mean()
            .reduceRegion(ee.Reducer.mean(), aoi, scale=10, maxPixels=1e8)
            .get("VH").getInfo()
        )
        vh_after = (
            s1.filterDate(after_start, after_end)
            .mean()
            .reduceRegion(ee.Reducer.mean(), aoi, scale=10, maxPixels=1e8)
            .get("VH").getInfo()
        )

        if vh_before is None or vh_after is None:
            log.warning("S1 data unavailable for this chip — accepting without filter.")
            return True, 0.0

        drop_db = float(vh_before) - float(vh_after)
        confirmed = drop_db >= min_drop_db
        log.info(f"  S1 VH filter: before={vh_before:.2f} dB, after={vh_after:.2f} dB, "
                 f"drop={drop_db:.2f} dB → {'PASS' if confirmed else 'FAIL (phenological)'}")
        return confirmed, drop_db

    except Exception as exc:
        log.warning(f"S1 filter failed ({exc}) — accepting without filter.")
        return True, 0.0


# ── Model inference ───────────────────────────────────────────────────────────

def run_chip_inference(
    model:     "torch.nn.Module",
    chip_path: str,
    device,
    threshold: float = _DEFAULT_THRESHOLD,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Run model on a single V2 chip GeoTIFF.

    Returns
    -------
    (change_mask, prob_map, src_profile)
        change_mask : (H, W) uint8, 1=forest loss
        prob_map    : (H, W) float32, probability of change [0..1]
        src_profile : rasterio profile (for GeoTIFF writing)
    Returns (None, None, None) if chip is invalid.
    """
    chip = load_v2_chip(chip_path)
    if chip is None:
        log.warning(f"Chip invalid (mostly nodata): {chip_path}")
        return None, None, None

    # chip shape: (T=3, C=6, H=224, W=224)
    tensor = torch.from_numpy(chip[np.newaxis]).float().to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        logits = model(tensor, mode="detect")             # (1, num_classes, H, W)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (num_classes, H, W)

    prob_change  = probs[1]                                # class-1 = forest loss
    change_mask  = (prob_change >= threshold).astype(np.uint8)

    # Load spatial reference for output
    with rasterio.open(chip_path) as src:
        profile = src.profile.copy()

    return change_mask, prob_change, profile


# ── Zone risk scoring (spatial prior) ─────────────────────────────────────────

def compute_zone_risk(
    aoi:          "ee.Geometry",
    anchor_date:  str,
    cfg:          dict,
    cusum_score:  float = 0.0,
) -> int:
    """
    Compute a spatial risk tier (1–4) for the AOI.

    Scoring components
    ------------------
    Hansen historical loss density  × 4.0   (optical archive signal)
    Hansen recent loss fraction     × 3.0   (recency pressure)
    SAR CuSum gradual-loss boost    + 0.15  (when cusum_score ≥ 0.5)

    The CuSum boost catches slow canopy removals spread over 4–8 months
    that register weakly in the 5-year Hansen recency window because they
    are below the annual detection threshold.  Adding +0.15 to the score
    reliably pushes borderline Medium→High and High→Critical zones where
    the radar independently confirms an accumulating structural change.

    This is a SPATIAL PRIOR fused with a SAR temporal signal — it is
    independent of the optical model detection confidence and must be
    displayed separately to field officers.

    Parameters
    ----------
    aoi          : ee.Geometry of the chip bounding box
    anchor_date  : YYYY-MM-DD anchor date (used for recency window)
    cfg          : loaded config dict (needs gee.hansen_collection key)
    cusum_score  : [0..1] zone-mean CuSum score from compute_sar_cusum_score;
                   values ≥ 0.5 trigger the +0.15 SAR pressure boost

    Returns risk tier: 1=Low, 2=Medium, 3=High, 4=Critical
    """
    if not _ee_available:
        return 2  # fallback: medium

    try:
        hansen = ee.Image(cfg["gee"]["hansen_collection"])

        # Historical loss density (2001–lossyear)
        loss_area = (
            hansen.select("loss")
            .reduceRegion(ee.Reducer.mean(), aoi, scale=30, maxPixels=1e8)
            .get("loss").getInfo()
        )

        # Loss year recency — recent loss = higher pressure
        year  = int(anchor_date[:4])  # type: ignore[index]
        recent_loss = (
            hansen.select("lossyear")
            .gte(year - 5)
            .reduceRegion(ee.Reducer.mean(), aoi, scale=30, maxPixels=1e8)
            .get("lossyear").getInfo()
        )

        loss_density = float(loss_area)   if loss_area   is not None else 0.0
        recent_frac  = float(recent_loss) if recent_loss is not None else 0.0

        # Base score from optical archive
        score = loss_density * 4.0 + recent_frac * 3.0

        # ── SAR CuSum pressure boost ──────────────────────────────────────────
        # A CuSum score ≥ 0.5 means the VH time-series shows a persistent,
        # accumulating drop that is consistent with gradual canopy stripping.
        # This is an independent corroborating signal from a different sensor
        # modality, so we add it to the spatial score rather than gate on it.
        sar_boost = 0.0
        if cusum_score >= _CUSUM_SCORE_THRESH:
            sar_boost = 0.15
            log.info(f"  Zone risk: SAR CuSum boost applied (+{sar_boost:.2f}), "
                     f"cusum_score={cusum_score:.3f}")
        score += sar_boost

        log.info(f"  Zone risk: loss_density={loss_density:.4f}, "
                 f"recent_frac={recent_frac:.4f}, sar_boost={sar_boost:.2f}, "
                 f"final_score={score:.4f}")

        if   score >= 0.40: return 4   # Critical
        elif score >= 0.20: return 3   # High
        elif score >= 0.08: return 2   # Medium
        else:               return 1   # Low

    except Exception as exc:
        log.warning(f"Risk scoring failed ({exc}), defaulting to Medium.")
        return 2


# ── Alert object ──────────────────────────────────────────────────────────────

class Alert:
    """Container for a single forest-loss alert."""

    def __init__(
        self,
        anchor_date:      str,
        detect_conf:      float,      # mean detection confidence over changed pixels
        s1_drop_db:       float,      # VH backscatter drop (0 if S1 unavailable)
        s1_confirmed:     bool,       # passed the S1 VH filter?
        risk_tier:        int,        # 1–4 spatial risk (already includes SAR boost)
        area_ha:          float,      # estimated changed area in hectares
        change_mask:      np.ndarray, # (H, W) uint8
        prob_map:         np.ndarray, # (H, W) float32
        profile:          dict,       # rasterio profile for geocoding
        chip_path:        str,
        cusum_zone_score: float = 0.0,  # zone-mean SAR CuSum score [0..1]
    ):
        self.anchor_date      = anchor_date
        self.detect_conf      = detect_conf
        self.s1_drop_db       = s1_drop_db
        self.s1_confirmed     = s1_confirmed
        self.risk_tier        = risk_tier
        self.area_ha          = area_ha
        self.change_mask      = change_mask
        self.prob_map         = prob_map
        self.profile          = profile
        self.chip_path        = chip_path
        self.cusum_zone_score = cusum_zone_score
        self.timestamp        = datetime.utcnow()

    @property
    def risk_label(self) -> str:
        return _RISK_LABELS.get(self.risk_tier, "Unknown")

    @property
    def risk_color(self) -> str:
        return _RISK_COLORS.get(self.risk_tier, "#7f8c8d")

    def to_dict(self) -> dict:
        return {
            "anchor_date":  self.anchor_date,
            "generated_at": self.timestamp.isoformat() + "Z",
            "detection": {
                "confidence_mean_pct": round(self.detect_conf * 100, 1),  # type: ignore[call-overload]
                "area_ha":             round(self.area_ha, 2),             # type: ignore[call-overload]
                "s1_vh_drop_db":       round(self.s1_drop_db, 2),         # type: ignore[call-overload]
                "s1_confirmed":        self.s1_confirmed,
                "sar_cusum": {
                    "zone_score":         round(self.cusum_zone_score, 4),  # type: ignore[call-overload]
                    "gradual_loss_flag":  self.cusum_zone_score >= _CUSUM_SCORE_THRESH,
                },
            },
            "zone_risk": {
                "tier":              self.risk_tier,
                "label":             self.risk_label,
                "sar_boost_applied": self.cusum_zone_score >= _CUSUM_SCORE_THRESH,
            },
        }


# ── GeoTIFF export ────────────────────────────────────────────────────────────

def save_change_geotiff(alert: Alert, out_path: str) -> str:
    """Save change mask and probability map as a 2-band GeoTIFF."""
    profile = {**alert.profile, "count": 2, "dtype": "float32",
               "compress": "lzw", "driver": "GTiff"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(alert.change_mask.astype(np.float32), 1)
        dst.write(alert.prob_map,                       2)
        dst.update_tags(
            anchor_date       = alert.anchor_date,
            detect_conf       = f"{alert.detect_conf:.4f}",
            s1_drop_db        = f"{alert.s1_drop_db:.2f}",
            s1_confirmed      = str(alert.s1_confirmed),
            risk_tier         = str(alert.risk_tier),
            area_ha           = f"{alert.area_ha:.2f}",
            cusum_zone_score  = f"{alert.cusum_zone_score:.4f}",
            sar_boost_applied = str(alert.cusum_zone_score >= _CUSUM_SCORE_THRESH),
        )
    log.info(f"GeoTIFF saved: {out_path}")
    return out_path


# ── KMZ export ───────────────────────────────────────────────────────────────

def _change_mask_to_png(alert: Alert, tmp_dir: str) -> Tuple[str, Tuple]:
    """Render the change probability map as a transparent overlay PNG."""
    from PIL import Image  # type: ignore[import]

    H, W = alert.prob_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Red channel = probability, alpha = where detected
    rgba[:, :, 0] = (alert.prob_map * 255).astype(np.uint8)
    rgba[:, :, 3] = (alert.change_mask * 180).astype(np.uint8)  # semi-transparent

    img_path = os.path.join(tmp_dir, "overlay.png")
    Image.fromarray(rgba, "RGBA").save(img_path)

    # Get bounding box in WGS84 (west, south, east, north)
    with rasterio.open(alert.chip_path) as src:
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    # bounds = (west, south, east, north)
    return img_path, bounds


def save_kmz(alert: Alert, out_path: str) -> str:
    """Export alert as KMZ for Google Earth viewing."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            png_path, (west, south, east, north) = _change_mask_to_png(alert, tmp_dir)
            has_overlay = True
        except Exception as exc:
            log.warning(f"KMZ overlay skipped ({exc}) — writing placemark-only KMZ.")
            has_overlay = False
            west = south = east = north = 0.0

        lat_c = (south + north) / 2
        lon_c = (west + east) / 2

        overlay_kml = ""
        if has_overlay:
            overlay_kml = f"""
  <GroundOverlay>
    <name>Change Detection Overlay</name>
    <Icon><href>overlay.png</href></Icon>
    <LatLonBox>
      <north>{north}</north><south>{south}</south>
      <east>{east}</east><west>{west}</west>
    </LatLonBox>
    <color>aaffffff</color>
  </GroundOverlay>"""

        kml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Van Suraksha Alert — {alert.anchor_date}</name>
  <description>
    Detection confidence: {alert.detect_conf*100:.1f}%
    Area affected: {alert.area_ha:.2f} ha
    S1 VH drop: {alert.s1_drop_db:.2f} dB ({'confirmed' if alert.s1_confirmed else 'phenological — review'})
    Zone risk: {alert.risk_label} (tier {alert.risk_tier}/4)
  </description>
  <Placemark>
    <name>Alert centroid</name>
    <styleUrl>#alert_style</styleUrl>
    <Point><coordinates>{lon_c},{lat_c},0</coordinates></Point>
  </Placemark>
  <Style id="alert_style">
    <IconStyle>
      <color>ff0000ff</color>
      <scale>1.4</scale>
    </IconStyle>
  </Style>{overlay_kml}
</Document>
</kml>"""

        kml_path = os.path.join(tmp_dir, "doc.kml")
        with open(kml_path, "w", encoding="utf-8") as f:
            f.write(kml_body)

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as kmz:
            kmz.write(kml_path, "doc.kml")
            if has_overlay:
                kmz.write(png_path, "overlay.png")

    log.info(f"KMZ saved: {out_path}")
    return out_path


# ── PDF report ────────────────────────────────────────────────────────────────

def save_pdf_report(alert: Alert, out_path: str) -> str:
    """
    Generate a single-page PDF alert report using matplotlib.

    Layout:
      ┌─────────────────────────────────┐
      │  HEADER: Van Suraksha Alert     │
      │  Date / location / timestamp    │
      ├──────────────┬──────────────────┤
      │  Prob map    │  Metric cards    │
      │  (heatmap)   │  Detection conf  │
      │              │  S1 VH filter    │
      │              │  Zone risk tier  │
      │              │  Area estimate   │
      │              │  S1 note         │
      └──────────────┴──────────────────┘
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5), facecolor="#1a1a2e")
    ax_img  = fig.add_axes([0.04, 0.10, 0.44, 0.72])
    ax_cards = fig.add_axes([0.54, 0.10, 0.42, 0.72])
    ax_cards.axis("off")

    # — Probability heatmap —
    ax_img.imshow(alert.prob_map, cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax_img.contour(alert.change_mask, levels=[0.5], colors="white", linewidths=1.2)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("Detection confidence", color="white", fontsize=10, pad=6)

    # — Metric cards —
    risk_col = alert.risk_color
    cards = [
        ("DETECTION CONFIDENCE",
         f"{alert.detect_conf * 100:.1f}%",
         "#2980b9"),
        ("AREA AFFECTED",
         f"{alert.area_ha:.2f} ha",
         "#27ae60"),
        ("S1 VH RADAR DROP",
         f"{alert.s1_drop_db:.2f} dB  ({'real change' if alert.s1_confirmed else 'no structural change'})",
         "#27ae60" if alert.s1_confirmed else "#e74c3c"),
        ("ZONE RISK",
         f"{alert.risk_label.upper()}  (tier {alert.risk_tier}/4)",
         risk_col),
    ]

    card_h = 0.18
    card_gap = 0.04
    y = 0.98
    for title, value, color in cards:
        rect = mpatches.FancyBboxPatch(
            (0, y - card_h), 1.0, card_h - 0.01,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor=color, facecolor=color + "22",  # 13% alpha
            transform=ax_cards.transAxes, clip_on=False,
        )
        ax_cards.add_patch(rect)
        ax_cards.text(0.05, y - 0.04, title,
                      transform=ax_cards.transAxes,
                      fontsize=7, color="#aaaaaa", weight="bold")
        ax_cards.text(0.05, y - 0.13, value,
                      transform=ax_cards.transAxes,
                      fontsize=13, color="white", weight="bold")
        y -= card_h + card_gap

    # S1 interpretation note
    note = (
        "S1 VH drop CONFIRMED — structural canopy loss detected by radar."
        if alert.s1_confirmed else
        "S1 VH drop below threshold — likely seasonal leaf-drop (dry deciduous).\n"
        "Field verification recommended before filing FIR."
    )
    ax_cards.text(0.05, 0.06, note,
                  transform=ax_cards.transAxes,
                  fontsize=8, color="#f39c12",
                  style="italic", wrap=True)

    # — Header —
    fig.text(0.5, 0.90,
             f"Van Suraksha  |  Forest Loss Alert  |  {alert.anchor_date}",
             ha="center", fontsize=14, color="white", weight="bold")
    fig.text(0.5, 0.86,
             f"Guna District, Madhya Pradesh  ·  Generated {alert.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
             ha="center", fontsize=9, color="#aaaaaa")
    fig.text(0.5, 0.04,
             "Detection and risk scores are independent. "
             "Do NOT combine them numerically. Review field photos before legal action.",
             ha="center", fontsize=7, color="#777777", style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"PDF saved: {out_path}")
    return out_path


# ── Telegram notification ─────────────────────────────────────────────────────

def send_telegram_alert(alert: Alert, pdf_path: str, kmz_path: str) -> bool:
    """
    Send detection alert to a Telegram chat.

    Requires environment variables:
        TELEGRAM_BOT_TOKEN
        TELEGRAM_CHAT_ID

    Includes:
      - Formatted summary message
      - PDF report attachment
      - KMZ attachment (if file exists)

    Returns True on success.
    """
    import urllib.request
    import urllib.parse

    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        log.warning("Telegram env vars not set (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Skipping.")
        return False

    base_url = f"https://api.telegram.org/bot{token}"

    # --- Text message ---
    s1_line = (
        f"*S1 Radar:* {alert.s1_drop_db:.2f} dB drop — structural change CONFIRMED"
        if alert.s1_confirmed else
        f"*S1 Radar:* {alert.s1_drop_db:.2f} dB drop — likely seasonal (review)"
    )
    msg = (
        f"*Van Suraksha Forest Alert*\n"
        f"Guna District, Madhya Pradesh\n\n"
        f"*Date:* {alert.anchor_date}\n"
        f"*Area:* {alert.area_ha:.2f} ha\n"
        f"*Detection confidence:* {alert.detect_conf*100:.1f}%\n"
        f"{s1_line}\n"
        f"*Zone risk:* {alert.risk_label} (tier {alert.risk_tier}/4)\n\n"
        f"_Confidence and risk are separate metrics — do not combine._"
    )

    try:
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text":    msg,
            "parse_mode": "Markdown",
        }).encode()
        urllib.request.urlopen(f"{base_url}/sendMessage", payload, timeout=15)
        log.info("Telegram text message sent.")
    except Exception as exc:
        log.warning(f"Telegram text failed: {exc}")
        return False

    # --- PDF attachment ---
    def _send_file(file_path: str, caption: str) -> bool:
        if not os.path.exists(file_path):
            return False
        # multipart/form-data upload
        boundary = "---AlertBoundary7729"
        body_parts = []
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                          f"name=\"chat_id\"\r\n\r\n{chat_id}")
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                          f"name=\"caption\"\r\n\r\n{caption}")
        fname     = Path(file_path).name
        file_data = Path(file_path).read_bytes()
        file_part = (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"document\"; filename=\"{fname}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        )
        body = "\r\n".join(body_parts).encode() + b"\r\n" + \
               file_part.encode() + file_data + \
               f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{base_url}/sendDocument",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            urllib.request.urlopen(req, timeout=60)
            log.info(f"Telegram file sent: {fname}")
            return True
        except Exception as exc:
            log.warning(f"Telegram file send failed: {exc}")
            return False

    _send_file(pdf_path, f"Van Suraksha alert report — {alert.anchor_date}")
    _send_file(kmz_path, f"Forest change KMZ — open in Google Earth")
    return True


# ── Alert JSON log ────────────────────────────────────────────────────────────

def append_alert_log(alert: Alert, log_path: str) -> None:
    """Append alert metadata to a JSON-lines log file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Alert logged: {log_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _load_beat_geometry(beat_geojson: dict) -> "ee.Geometry":
    """Convert a GeoJSON Feature geometry dict into a GEE Geometry object."""
    import json
    return ee.Geometry(beat_geojson)


def _get_beat_feature(geojson_path: str, range_name: str, beat_name: str) -> Optional[dict]:
    """
    Look up a beat feature by Range + Beat name from a GeoJSON FeatureCollection.
    Returns the geometry dict, or None if not found.
    """
    import json
    with open(geojson_path, encoding="utf-8") as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        if (props.get("Range", "").strip().lower() == range_name.strip().lower() and
                props.get("Beat", "").strip().lower() == beat_name.strip().lower()):
            return feat.get("geometry")
    return None


def _run_rules_pipeline(
    cfg:           dict,
    anchor_date:   str,
    out_dir:       str,
    beat_geojson:  Optional[dict] = None,
    send_telegram: bool = False,
    dry_run:       bool = False,
) -> Optional["Alert"]:
    """
    GEE-SRC rules engine path for run_alert_pipeline(mode='rules').

    Calls run_gee_rules_engine() entirely server-side, then converts the
    returned dict into an Alert object.  No local chip is downloaded.

    The Alert's change_mask / prob_map / profile fields are synthetic
    scalars so that all downstream export functions (PDF, KMZ, log) work
    without modification.  The GeoJSON is also written as a separate file.
    """
    from src.inference.rules_engine import run_gee_rules_engine
    from src.data.gee_fetch import init_gee

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    gee_project = cfg["gee"].get("gee_project")
    try:
        init_gee(gee_project)
    except Exception as e:
        log.error(f"[rules] GEE init failed: {e} — aborting rules pipeline.")
        return None

    # Resolve beat geometry → GEE Geometry
    if beat_geojson is not None:
        beat_geom = _load_beat_geometry(beat_geojson)
        log.info("[rules] Using per-beat local GeoJSON geometry.")
    elif _ee_available:
        import ee as _ee
        beat_geom = _ee.FeatureCollection(cfg["gee"]["aoi_asset"]).geometry()
        log.info(f"[rules] Using GEE asset AOI: {cfg['gee']['aoi_asset']}")
    else:
        log.error("[rules] No beat geometry and ee not available.")
        return None

    # Load rules-engine config knobs
    inf_cfg = load_inference_cfg(cfg)
    min_area_ha = inf_cfg["min_area_ha"]

    # ── Run the rules engine ────────────────────────────────────────────────
    # Extract range label from beat GeoJSON for phenology model lookup
    range_label: Optional[str] = None
    if beat_geojson is not None:
        range_label = beat_geojson.get("properties", {}).get("Range") or None
        if range_label:
            log.info(f"[rules] Range label for phenology: '{range_label}'")

    log.info(f"[rules] Running GEE-SRC rules engine (anchor={anchor_date})...")
    result = run_gee_rules_engine(beat_geom, anchor_date, cfg, range_label=range_label)

    if result.get("skipped") or result.get("error"):
        log.error(f"[rules] Engine error: {result.get('error', 'unknown')}")
        return None

    area_ha    = float(result["area_ha"])
    final_conf = float(result["confidence"])
    final_tier = int(result["tier"])
    signals    = result.get("signals", {})
    geojson    = result.get("geojson", {"type": "FeatureCollection", "features": []})

    if area_ha < min_area_ha:
        log.info(f"[rules] No alert: area {area_ha:.3f} ha < minimum {min_area_ha} ha.")
        return None

    log.info(
        f"[rules] Engine result: tier={final_tier} conf={final_conf:.2f} "
        f"area={area_ha:.2f} ha  signals={signals}"
    )

    # ── Zone risk scoring (reuses same GEE function) ────────────────────────
    cusum_mean = float(signals.get("cusum_mean") or 0.0)
    if not dry_run:
        try:
            risk_tier = compute_zone_risk(
                beat_geom, anchor_date, cfg,
                cusum_score=cusum_mean,
            )
        except Exception as re:
            log.warning(f"[rules] Zone risk scoring failed ({re}), defaulting to Medium.")
            risk_tier = 2
    else:
        risk_tier = 2

    # Map rules tier → s1_confirmed / s1_drop for Alert / export compatibility
    # Tier 1 = all signals (optical + SAR + CuSum) → confirmed
    # Tier 2 = optical + SAR instantaneous, no long-term trend → yes
    # Tier 3 = SAR-only (no optical) → not confirmed
    s1_drop_db    = float(signals.get("dVH_mean_db") or 0.0)
    s1_confirmed  = final_tier in (1, 2)   # tier 3 = SAR-only, flag for review

    # ── Synthetic raster placeholders ────────────────────────────────────────
    # The rules engine is server-side (no local chip). We create minimal 1×1
    # NumPy arrays + a WGS84 rasterio profile so that save_change_geotiff /
    # save_kmz / save_pdf_report work unchanged.
    _stub_mask = np.ones((1, 1), dtype=np.uint8)
    _stub_prob = np.full((1, 1), final_conf, dtype=np.float32)
    try:
        _bbox = beat_geom.bounds().getInfo()["coordinates"][0]
        # GEE ring: [SW, SE, NE, NW, SW]  → coords[0]=SW=[lon,lat], coords[2]=NE=[lon,lat]
        _sw, _ne = _bbox[0], _bbox[2]
        _west, _south, _east, _north = _sw[0], _sw[1], _ne[0], _ne[1]
    except Exception:
        # Guna Division centroid fallback (~24.65°N, 77.30°E)
        _west, _south, _east, _north = 77.295, 24.645, 77.305, 24.655
    _stub_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 1,
        "height": 1,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(_west, _south, _east, _north, 1, 1),
    }

    # ── Build Alert ─────────────────────────────────────────────────────────
    alert = Alert(
        anchor_date      = anchor_date,
        detect_conf      = final_conf,
        s1_drop_db       = s1_drop_db,
        s1_confirmed     = s1_confirmed,
        risk_tier        = risk_tier,
        area_ha          = area_ha,
        change_mask      = _stub_mask,
        prob_map         = _stub_prob,
        profile          = _stub_profile,
        chip_path        = "",     # no chip in rules mode
        cusum_zone_score = cusum_mean,
    )

    log.info(
        f"\n{'='*60}\n"
        f"  [RULES ENGINE] ALERT SUMMARY — {anchor_date}\n"
        f"  Detection tier       : {final_tier}  (1=highest, 3=SAR-only)\n"
        f"  Detection confidence : {final_conf*100:.1f}%\n"
        f"  Area                 : {area_ha:.2f} ha\n"
        f"  S1 VH drop           : {s1_drop_db:.2f} dB  "
        f"({'confirmed' if s1_confirmed else 'SAR-only — no optical'})\n"
        f"  SAR CuSum mean       : {cusum_mean:.3f}\n"
        f"  Zone risk            : {alert.risk_label} (tier {risk_tier}/4)\n"
        f"  Signals              : {signals}\n"
        f"{'='*60}"
    )

    if dry_run:
        log.info("[rules] Dry-run mode — skipping file export.")
        return alert

    # ── Export outputs ───────────────────────────────────────────────────────
    date_slug  = anchor_date.replace("-", "")
    pdf_path   = os.path.join(out_dir, f"alert_{date_slug}_report.pdf")
    kmz_path   = os.path.join(out_dir, f"alert_{date_slug}.kmz")
    geojson_path = os.path.join(out_dir, f"alert_{date_slug}_rules.geojson")
    alert_log  = os.path.join(out_dir, "alert_log.jsonl")

    # Write GeoJSON alert polygons
    with open(geojson_path, "w", encoding="utf-8") as _gj:
        json.dump(geojson, _gj, ensure_ascii=False, indent=2)
    log.info(f"[rules] GeoJSON saved: {geojson_path}")

    # PDF report (uses stub prob_map — shows signal tier card instead of heatmap)
    try:
        save_pdf_report(alert, pdf_path)
    except Exception as ex:
        log.warning(f"[rules] PDF export failed ({ex}) — continuing.")

    # KMZ (stub-only; no visual overlay since no chip)
    try:
        save_kmz(alert, kmz_path)
    except Exception as ex:
        log.warning(f"[rules] KMZ export failed ({ex}) — continuing.")

    # JSON log entry
    alert_dict = alert.to_dict()
    alert_dict["mode"] = "rules"
    alert_dict["detection_tier"] = final_tier
    alert_dict["signals"] = signals
    alert_dict["geojson_path"] = geojson_path
    Path(alert_log).parent.mkdir(parents=True, exist_ok=True)
    with open(alert_log, "a", encoding="utf-8") as _al:
        _al.write(json.dumps(alert_dict, ensure_ascii=False) + "\n")
    log.info(f"[rules] Alert logged: {alert_log}")

    if send_telegram:
        send_telegram_alert(alert, pdf_path, kmz_path)

    return alert


def run_alert_pipeline(
    cfg:           dict,
    checkpoint:    str,
    anchor_date:   str,
    out_dir:       str,
    threshold:     float  = _DEFAULT_THRESHOLD,
    chip_path:     Optional[str] = None,
    send_telegram: bool   = False,
    s1_min_drop:   float  = _S1_VH_DROP_DB,
    dry_run:       bool   = False,
    beat_geojson:  Optional[dict] = None,   # geometry dict from local GeoJSON
    mode:          str    = "neural",       # "neural" | "rules" | "hybrid"
) -> Optional[Alert]:
    """
    End-to-end alert generation for one anchor date.

    Parameters
    ----------
    mode : str
        Detection mode: "neural" (default Prithvi model), "rules" (GEE-SRC same-
        season rules cascade, no chip download needed), or "hybrid" (rules engine
        result + neural probability map merged).

    Steps (neural mode)
    -------------------
    0  Resolve config-driven inference knobs (config.yaml:inference)
    1  Fetch or load chip
    2  Run model inference
    3  S1 VH false-positive filter + SAR CuSum gradual-loss scoring
    4  Zone risk scoring (fused with CuSum)
    5  Build Alert object
    6  Export GeoTIFF / KMZ / PDF
    7  Append JSON log
    8  Telegram notification (optional)

    Steps (rules mode)
    ------------------
    0  Resolve config-driven inference knobs
    1  GEE init + build AOI geometry
    2  run_gee_rules_engine → tier/area/confidence/geojson
    3  Zone risk scoring
    4  Build Alert object (no chip, no prob_map)
    5  Export KMZ / PDF / JSON log
    6  Telegram notification (optional)

    Returns Alert on success, None if no change detected or chip is invalid.
    """
    # Resolve mode from config if caller left at default
    _cfg_mode = cfg.get("inference", {}).get("detection_mode", "neural")
    if mode == "neural" and _cfg_mode in ("rules", "hybrid"):
        mode = _cfg_mode
    log.info(f"Detection mode: {mode}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()
    log.info(f"Device: {device}")

    # ── 0. Resolve config-driven inference knobs ──────────────────────────────
    inf_cfg = load_inference_cfg(cfg)
    # CLI args win only when the caller didn't leave them at the module default
    if threshold   == _DEFAULT_THRESHOLD:  threshold   = inf_cfg["threshold"]
    if s1_min_drop == _S1_VH_DROP_DB:      s1_min_drop = inf_cfg["s1_drop_db"]
    cusum_thresh      = inf_cfg["cusum_thresh"]
    cusum_sensitivity = inf_cfg["cusum_sensitivity"]
    min_area_ha       = inf_cfg["min_area_ha"]
    log.info(
        f"Inference cfg: threshold={threshold:.3f}  s1_drop={s1_min_drop:.1f} dB  "
        f"cusum_thresh={cusum_thresh:.2f}  min_area={min_area_ha:.2f} ha"
    )

    # ── RULES ENGINE PATH ─────────────────────────────────────────────────────
    if mode in ("rules", "hybrid"):
        return _run_rules_pipeline(
            cfg=cfg, anchor_date=anchor_date, out_dir=out_dir,
            beat_geojson=beat_geojson, send_telegram=send_telegram, dry_run=dry_run,
        )

    # ── 1. Chip  +  GEE initialisation ────────────────────────────────────────
    if chip_path is None:
        chip_path = os.path.join(out_dir, f"live_chip_{anchor_date}_img.tif")

    # Initialise GEE once, regardless of whether the chip already exists,
    # because S1 filter and CuSum both need an active GEE session (steps 3 & 3b).
    _gee_ready = False
    if _ee_available and not dry_run:
        gee_project = cfg["gee"].get("gee_project")
        try:
            init_gee(gee_project)
            _gee_ready = True
        except Exception as _gee_err:
            log.warning(f"GEE init failed ({_gee_err}); S1 / CuSum steps will be skipped.")

    # Resolve AOI as a GEE Geometry — prefer per-beat local GeoJSON over asset
    _aoi_geom: Optional["ee.Geometry"] = None
    if _gee_ready:
        if beat_geojson is not None:
            _aoi_geom = _load_beat_geometry(beat_geojson)
            log.info("Using per-beat local GeoJSON geometry as AOI.")
        else:
            _aoi_geom = ee.FeatureCollection(cfg["gee"]["aoi_asset"]).geometry()
            log.info(f"Using GEE asset AOI: {cfg['gee']['aoi_asset']}")

    if not os.path.exists(chip_path):
        if _aoi_geom is not None:
            ok = fetch_live_chip(_aoi_geom, anchor_date, chip_path,
                                 cloud_thresh=cfg["gee"]["cloud_threshold"])
            if not ok:
                return None
        else:
            log.error("Chip not found and GEE unavailable. Provide --chip-path.")
            return None
    else:
        log.info(f"Using existing chip: {chip_path}")

    # ── 2. Inference ─────────────────────────────────────────────────────────
    model = load_model(cfg, checkpoint, device)
    change_mask, prob_map, profile = run_chip_inference(
        model, chip_path, device, threshold
    )
    if change_mask is None:
        log.warning("Chip invalid — no alert generated.")
        return None

    # Guard Optional[np.ndarray] tuple members — proper if-checks, not asserts
    # (assert statements are stripped by the Python -O optimiser flag).
    if prob_map is None or profile is None:
        log.error("run_chip_inference returned None prob_map/profile — aborting.")
        return None

    # Area estimate — derive GSD from profile.
    # IMPORTANT: GEE exports in EPSG:4326 (degrees), so transform.a is in degrees
    # (~0.0001°), NOT metres.  We must convert to metres using the geodetic formula
    # at the chip's centre latitude, otherwise area comes out as effectively 0.000 ha.
    n_changed  = int(change_mask.sum())
    _tf = profile["transform"]
    _deg_per_px = abs(_tf.a)   # always positive
    try:
        import pyproj as _pyproj
        _crs_obj = _pyproj.CRS(profile.get("crs", "EPSG:4326"))
        _is_geographic = _crs_obj.is_geographic
    except Exception:
        # Fallback: assume geographic when pixel size looks like degrees (<< 1)
        _is_geographic = (_deg_per_px < 0.01)
    if _is_geographic:
        # Derive chip centre latitude from the transform (top-left corner + half height)
        _centre_lat = _tf.f + _tf.e * (profile["height"] / 2.0)
        import math as _math
        _m_per_deg_lat = 111_132.0 - 559.8 * _math.cos(2 * _math.radians(_centre_lat))
        _m_per_deg_lon = 111_412.8 * _math.cos(_math.radians(_centre_lat))
        _px_lat_m = abs(_tf.e) * _m_per_deg_lat   # tf.e is negative (north-up)
        _px_lon_m = _deg_per_px * _m_per_deg_lon
        pixel_m   = (_px_lat_m + _px_lon_m) / 2.0  # approximate square pixel in metres
        log.info(f"Geographic CRS — effective pixel size: {pixel_m:.2f} m/px "
                 f"(lat={_centre_lat:.4f}°)")
    else:
        pixel_m = _deg_per_px   # already in metres for projected CRS
    area_ha = n_changed * (pixel_m ** 2) / 10_000

    if area_ha < min_area_ha:
        log.info(f"Change area {area_ha:.3f} ha < minimum {min_area_ha} ha — skipping.")
        return None

    detect_conf = float(prob_map[change_mask == 1].mean()) if n_changed > 0 else 0.0
    log.info(f"Detection: {n_changed} px changed, {area_ha:.2f} ha, "
             f"mean confidence {detect_conf*100:.1f}%")

    # ── 3. S1 VH filter ──────────────────────────────────────────────────────
    # Re-use the AOI resolved in step 1 (per-beat or GEE asset)
    if _gee_ready and _aoi_geom is not None:
        aoi_geom = _aoi_geom
        s1_confirmed, s1_drop = check_s1_vh_drop(aoi_geom, anchor_date, s1_min_drop)
    else:
        aoi_geom = None
        # Safe-fail: mark as UNCONFIRMED so alert is flagged for manual review.
        # Do NOT default to True — that would treat every optical hit as SAR-confirmed.
        s1_confirmed, s1_drop = False, 0.0
        if dry_run:
            log.info("Dry-run: S1 filter bypassed.")

    # Log but don't suppress — field officers see filter result separately
    if not s1_confirmed:
        log.warning(
            "S1 VH drop below threshold — likely seasonal leaf-drop (dry deciduous). "
            "Alert will be flagged for manual review."
        )

    # ── 3b. SAR CuSum — gradual structural loss over the anchor year ──────────
    # Fetch monthly VH backscatter for the full anchor year, then score with
    # the CuSum detector from enhanced_alert_windows.  This catches slow
    # canopy removals (4–8 months) that the instantaneous VH drop gate misses.
    cusum_zone_score: float = 0.0
    cusum_map: Optional[np.ndarray] = None

    if _gee_ready and aoi_geom is not None:
        anchor_year = int(anchor_date[:4])
        sar_start   = f"{anchor_year}-01-01"
        sar_end     = f"{anchor_year}-12-31"
        chip_aoi    = aoi_geom.centroid(maxError=10).buffer(_CHIP_BUFFER_M).bounds()

        log.info("Fetching S1 monthly VH stack for CuSum analysis...")
        try:
            vh_stack = fetch_s1_monthly_vh_stack(
                chip_aoi, sar_start, sar_end, scale=20, chip_px=224
            )

            if vh_stack is not None and _cusum_available:
                # Use the module-level import (no runtime sys.path surgery)
                compute_sar_cusum_score = _cusum_score_fn

                cusum_map        = compute_sar_cusum_score(vh_stack, sensitivity=cusum_sensitivity)
                # Zone-mean over the detected change pixels only
                if change_mask.any() and cusum_map is not None:
                    cusum_zone_score = float(cusum_map[change_mask == 1].mean())
                else:
                    cusum_zone_score = float(cusum_map.mean()) if cusum_map is not None else 0.0

                log.info(
                    f"  SAR CuSum zone score: {cusum_zone_score:.3f} "
                    f"({'gradual loss detected' if cusum_zone_score >= cusum_thresh else 'no gradual loss'})"
                )
            else:
                log.info("  SAR CuSum: skipped (insufficient S1 data for this zone)")

        except Exception as _exc:
            log.warning(f"  SAR CuSum fetch/score failed: {_exc} — continuing without it")

    # ── 4. Zone risk (fused with SAR CuSum pressure) ───────────────────────────
    # Use _gee_ready (session actually initialised), not _ee_available
    # (library imported). If init_gee() failed, _ee_available is still True
    # but calling ee.* will crash on an uninitialised session.
    if _gee_ready and not dry_run:
        risk_tier = compute_zone_risk(
            aoi_geom, anchor_date, cfg,
            cusum_score=cusum_zone_score,        # ← SAR fusion point
        )
    else:
        risk_tier = 2  # medium fallback

    # ── 5. Alert object ───────────────────────────────────────────────────────
    alert = Alert(
        anchor_date      = anchor_date,
        detect_conf      = detect_conf,
        s1_drop_db       = s1_drop,
        s1_confirmed     = s1_confirmed,
        risk_tier        = risk_tier,
        area_ha          = area_ha,
        change_mask      = change_mask,
        prob_map         = prob_map,
        profile          = profile,
        chip_path        = chip_path,
        cusum_zone_score = cusum_zone_score,     # ← stored for logging/export
    )

    log.info(
        f"\n{'='*60}\n"
        f"  ALERT SUMMARY — {anchor_date}\n"
        f"  Detection confidence : {detect_conf*100:.1f}%\n"
        f"  Area                 : {area_ha:.2f} ha\n"
        f"  S1 VH drop (instant) : {s1_drop:.2f} dB  ({'confirmed' if s1_confirmed else 'REVIEW — phenological?'})\n"
        f"  SAR CuSum (gradual)  : {cusum_zone_score:.3f}  ({'gradual loss' if cusum_zone_score >= _CUSUM_SCORE_THRESH else 'no gradual trend'})\n"
        f"  Zone risk            : {alert.risk_label} (tier {risk_tier}/4)\n"
        f"{'='*60}"
    )

    if dry_run:
        log.info("Dry-run mode — skipping file export and notifications.")
        return alert

    # ── 6. Export outputs ─────────────────────────────────────────────────────
    date_slug  = anchor_date.replace("-", "")
    tif_path   = os.path.join(out_dir, f"alert_{date_slug}_change.tif")
    kmz_path   = os.path.join(out_dir, f"alert_{date_slug}.kmz")
    pdf_path   = os.path.join(out_dir, f"alert_{date_slug}_report.pdf")
    alert_log  = os.path.join(out_dir, "alert_log.jsonl")

    save_change_geotiff(alert, tif_path)
    save_kmz(alert, kmz_path)
    save_pdf_report(alert, pdf_path)
    append_alert_log(alert, alert_log)

    # ── 7. Telegram ───────────────────────────────────────────────────────────
    if send_telegram:
        send_telegram_alert(alert, pdf_path, kmz_path)

    return alert


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Van Suraksha — Forest Loss Alert Generator"
    )
    parser.add_argument("--config",      default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to trained model .pth checkpoint")
    parser.add_argument("--date",        default=None,
                        help="Anchor date YYYY-MM-DD (default: today)")
    parser.add_argument("--chip-path",   default=None,
                        help="Pre-downloaded chip GeoTIFF (skips GEE fetch)")
    parser.add_argument("--threshold",   type=float, default=_DEFAULT_THRESHOLD,
                        help=f"Detection confidence threshold (default: {_DEFAULT_THRESHOLD})")
    parser.add_argument("--s1-drop",     type=float, default=_S1_VH_DROP_DB,
                        help=f"S1 VH drop required to confirm structural change (default: {_S1_VH_DROP_DB} dB)")
    parser.add_argument("--out-dir",     default="outputs/alerts",
                        help="Output directory for alert files")
    parser.add_argument("--telegram",    action="store_true",
                        help="Send Telegram notification (needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run inference but skip file export and notifications")
    parser.add_argument(
        "--beat", default=None, metavar="RANGE/BEAT",
        help=(
            "Run for a specific beat, e.g. 'Aron/Goumukh'. "
            "Geometry is read from data/aoi/guna_beats.geojson — "
            "no GEE FeatureCollection asset required for the AOI."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    anchor_date = args.date or datetime.utcnow().strftime("%Y-%m-%d")
    log.info(f"Anchor date: {anchor_date}")

    # ── Per-beat AOI resolution ───────────────────────────────────────────────
    beat_geojson: Optional[dict] = None
    if args.beat:
        parts = args.beat.split("/", 1)
        if len(parts) != 2:
            log.error("--beat must be in the format 'RANGE/BEAT', e.g. 'Aron/Goumukh'")
            sys.exit(1)
        range_name, beat_name = parts
        _beats_path = os.path.join("data", "aoi", "guna_beats.geojson")
        beat_geojson = _get_beat_feature(_beats_path, range_name, beat_name)
        if beat_geojson is None:
            log.error(f"Beat '{range_name}/{beat_name}' not found in {_beats_path}")
            sys.exit(1)
        log.info(f"Beat AOI loaded: {range_name} / {beat_name}")
        # Per-beat outputs go into a sub-folder for cleaner organisation
        if args.out_dir == "outputs/alerts":
            args.out_dir = os.path.join(
                "outputs", "alerts", "beats",
                range_name.replace(" ", "_"),
                beat_name.replace(" ", "_"),
            )

    alert = run_alert_pipeline(
        cfg           = cfg,
        checkpoint    = args.checkpoint,
        anchor_date   = anchor_date,
        out_dir       = args.out_dir,
        threshold     = args.threshold,
        chip_path     = args.chip_path,
        send_telegram = args.telegram,
        s1_min_drop   = args.s1_drop,
        dry_run       = args.dry_run,
        beat_geojson  = beat_geojson,
    )

    if alert is None:
        log.info("No alert generated.")
        sys.exit(0)

    assert alert is not None  # Pyre2: sys.exit() is not recognized as NoReturn
    log.info(f"\nAlert complete. Risk: {alert.risk_label} | Conf: {alert.detect_conf*100:.1f}%")
    sys.exit(0)
