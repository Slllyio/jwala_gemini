"""
Alert Dynamics Visualization  (v3 — range-split, no vectorisation required)
=============================================================================
Plots 2025 simulation alerts — coloured by DW `Source` class — overlaid on
the 8-band DW probability delta raster (Dec year_a → Dec year_b).

KEY DESIGN DECISIONS:
  • No vectorisation step needed. The delta TIF IS the ground truth in raster
    space. Alert GeoJSONs overlay directly. GT binary rasters are shown as is.
  • Split by Forest Range. One per-range figure is ~200–300 sq km = readable.
    Full-division figure is also available but dense.
  • ML training samples pixels direct from GT raster — no polygon conversion.

Alert colours (DW Source class from ALERT_GUNA.ipynb):
    🟢 Trees          — green      (tree prob dropped)
    🟡 Crops          — yellow     (crop prob rose → encroachment)
    🟠 Bare           — orange     (bare prob rose → deforestation)
    🟤 Shrub/Scrub    — brown      (open-forest scrub)
    🔴 Built          — red        (built-up rose)
    🟩 Grass          — lime dash  (seasonal noise)
    ⬜ False Positive — white dot  (alert with no GT pixel overlap)
    🟡 Missed         — yellow dash (GT change pixel, no alert)

Input layout:
    data/ground_truth/{range}/
        dw_prob_delta_{year_a}_{year_b}_{range}.tif  ← primary viz base
        dw_prob_dec{year_a}_{range}.tif              ← optional
        dw_prob_dec{year_b}_{range}.tif              ← optional
        dw_gt_{year}_deforestation_{range}.tif       ← GT binary raster (optional)
        dw_gt_{year}_encroachment_{range}.tif        ← GT binary raster (optional)

    data/alerts/
        alert_{year}-MM-DD.geojson                   ← `Source` field required

Usage:
    # All ranges, one figure each
    python scripts/plot_alert_dynamics.py --year 2025

    # Single range
    python scripts/plot_alert_dynamics.py --year 2025 --range North_Guna

    # Filter to specific alert sources
    python scripts/plot_alert_dynamics.py --year 2025 --range Raghogarh --source Trees,Bare

    # Per-alert deep-dive panels
    python scripts/plot_alert_dynamics.py --year 2025 --range North_Guna --per-alert

    # Full-division mosaic (if --full-mosaic was exported)
    python scripts/plot_alert_dynamics.py --year 2025 --range FullGuna
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GT_DIR  = Path("data/ground_truth")
ALT_DIR = Path("data/alerts")
VAL_DIR = Path("data/validation")
OUT_DIR = Path("data/viz")

DW_BANDS = [
    "water", "trees", "grass", "flooded_veg",
    "crops", "shrub_and_scrub", "built", "bare",
]
# Bands 9 and 10 in the 10-band TIF:
CLD_BAND_BEFORE = 8   # index 8 = band 9 (cloud_fraction_before)
CLD_BAND_AFTER  = 9   # index 9 = band 10 (cloud_fraction_after)
CLD_THRESHOLD   = 0.4  # fraction above which a pixel is flagged as unreliable

DW_BAND_LABELS = {
    "water":           "Δ Water",
    "trees":           "Δ Trees",
    "grass":           "Δ Grass",
    "flooded_veg":     "Δ Flooded Veg",
    "crops":           "Δ Crops",
    "shrub_and_scrub": "Δ Shrub/Scrub",
    "built":           "Δ Built",
    "bare":            "Δ Bare",
}

DW_SOURCES = ["Trees", "Crops", "Bare", "Shrub and Scrub", "Built", "Grass"]

SOURCE_STYLE = {
    "Trees":           {"color": "#43A047", "lw": 1.8, "ls": "-"},
    "Crops":           {"color": "#FDD835", "lw": 1.8, "ls": "-"},
    "Bare":            {"color": "#FF7043", "lw": 1.8, "ls": "-"},
    "Shrub and Scrub": {"color": "#8D6E63", "lw": 1.5, "ls": "-"},
    "Built":           {"color": "#E53935", "lw": 1.5, "ls": "-"},
    "Grass":           {"color": "#C6EF9A", "lw": 1.2, "ls": "--"},
    "Unknown":         {"color": "#90A4AE", "lw": 1.0, "ls": ":"},
    "_false_positive": {"color": "#FFFFFF", "lw": 1.0, "ls": ":"},
    "_missed_gt":      {"color": "#FFEB3B", "lw": 1.5, "ls": "--"},
}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_CLR  = "#c9d1d9"
DIVERGING = "RdBu_r"

# Row 1 (4 bands): the four change-diagnostic classes
# Row 2 (3 bands + 1 cloud panel): remaining classes + cloud quality
DELTA_PANEL_ORDER = [
    "trees", "shrub_and_scrub", "crops", "bare",
    "water", "built", "grass",      # flooded_veg dropped — replaced by cloud panel
]
CLOUD_PANEL_SLOT = (2, 3)   # (row, col) in the 3×4 GridSpec for the cloud panel


# ── Raster helpers ────────────────────────────────────────────────────────────

def find_delta_tif(year_a: int, year_b: int, range_name: str) -> Path | None:
    """
    Searches for the delta TIF in a range subdirectory OR the flat GT_DIR.
    File naming: dw_prob_delta_{year_a}_{year_b}_{range}.tif
    """
    candidates = [
        GT_DIR / range_name / f"dw_prob_delta_{year_a}_{year_b}_{range_name}.tif",
        GT_DIR / f"dw_prob_delta_{year_a}_{year_b}_{range_name}.tif",
        GT_DIR / range_name / f"dw_prob_delta_{year_a}_{year_b}.tif",
        GT_DIR / f"dw_prob_delta_{year_a}_{year_b}.tif",           # flat full-division
    ]
    for p in candidates:
        if p.exists():
            log.info(f"  Delta TIF: {p}")
            return p
    return None


def find_comp_tifs(year: int, range_name: str) -> tuple[Path | None, Path | None]:
    """Find the individual Dec composite TIFs for computing delta locally."""
    def search(yr):
        for p in [
            GT_DIR / range_name / f"dw_prob_dec{yr}_{range_name}.tif",
            GT_DIR / f"dw_prob_dec{yr}_{range_name}.tif",
        ]:
            if p.exists():
                return p
        return None
    return search(year - 1), search(year)


def load_raster(path: Path) -> tuple[np.ndarray, object]:
    """Returns (data float32 [bands,H,W], rasterio bounds)."""
    with rasterio.open(path) as src:
        data   = src.read().astype(np.float32)
        bounds = src.bounds
    return data, bounds


def compute_delta(year_a: int, year_b: int, range_name: str):
    """Load or compute the 8-band delta raster for a given range."""
    delta_p = find_delta_tif(year_a, year_b, range_name)
    if delta_p:
        return load_raster(delta_p)

    prev_p, curr_p = find_comp_tifs(year_b, range_name)
    if prev_p and curr_p:
        log.info("  Computing delta from Dec composites ...")
        prev, bounds = load_raster(prev_p)
        curr, _      = load_raster(curr_p)
        return curr - prev, bounds

    log.error(f"  No DW rasters found for range '{range_name}'.")
    log.error(f"  Run: python scripts/export_dw_prob_rasters.py --range {range_name}")
    log.error(f"  Then place downloaded TIFs in data/ground_truth/{range_name}/")
    return None, None


# ── GT raster helpers (OPTIONAL — no vectorisation needed) ────────────────────

def load_gt_raster(year: int, cls: str, range_name: str,
                   bounds) -> np.ndarray | None:
    """
    Load a GT binary raster (1=change, 0=no change) for one class.
    Returns a 2-D array matching the delta raster extent — or None if not found.
    Note: GT raster is overlaid as a transparent mask, not vectorised.
    """
    candidates = [
        GT_DIR / range_name / f"dw_gt_{year}_{cls}_{range_name}.tif",
        GT_DIR / f"dw_gt_{year}_{cls}_{range_name}.tif",
        GT_DIR / range_name / f"dw_gt_{year}_{cls}.tif",
        GT_DIR / f"dw_gt_{year}_{cls}.tif",
    ]
    for p in candidates:
        if p.exists():
            with rasterio.open(p) as src:
                data = src.read(1).astype(np.float32)
            return data
    return None   # GT raster not downloaded yet — skip silently


# ── Alert loading ─────────────────────────────────────────────────────────────

def load_alerts(year: int, bounds=None) -> gpd.GeoDataFrame:
    """
    Load all daily alert GeoJSONs for the year.
    If bounds is given, clip to the range bbox.
    Source field is preserved as-is from ALERT_GUNA notebook.
    """
    files = sorted(ALT_DIR.glob(f"alert_{year}-*.geojson"))
    val_p = VAL_DIR / f"alert_validation_{year}.geojson"

    if val_p.exists():
        alerts = gpd.read_file(val_p).to_crs("EPSG:4326")
        log.info(f"  Validated alerts: {len(alerts)}")
    elif files:
        frames = []
        for f in files:
            try:
                gdf = gpd.read_file(f).to_crs("EPSG:4326")
                gdf["alert_date"] = pd.to_datetime(f.stem.replace("alert_", "")[:10])
                if "Source" not in gdf.columns:
                    gdf["Source"] = "Unknown"
                frames.append(gdf)
            except Exception as e:
                log.debug(f"  Skip {f.name}: {e}")
        if frames:
            alerts = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
            log.info(f"  Raw alerts: {len(alerts)}")
        else:
            return gpd.GeoDataFrame()
    else:
        log.warning(f"  No alert files found in {ALT_DIR}/ for {year}")
        return gpd.GeoDataFrame()

    # Clip to range bbox if given
    if bounds is not None:
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        alerts = alerts[alerts.geometry.intersects(bbox)].copy()
        log.info(f"  Alerts in range: {len(alerts)}")

    return alerts


# ── Legend ────────────────────────────────────────────────────────────────────

def make_legend(ax):
    handles = [
        Line2D([0], [0], color=s["color"], lw=s["lw"], ls=s["ls"],
               label=src.replace("_", " "))
        for src, s in SOURCE_STYLE.items()
        if not src.startswith("_")
    ] + [
        Line2D([0], [0], color=SOURCE_STYLE["_false_positive"]["color"],
               lw=1.0, ls=":", label="False Positive"),
        Line2D([0], [0], color=SOURCE_STYLE["_missed_gt"]["color"],
               lw=1.5, ls="--", label="Missed GT"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6,
              facecolor="#1a1a1a", edgecolor="#555",
              labelcolor=TEXT_CLR, framealpha=0.88, ncol=2)


# ── Cloud quality helpers ─────────────────────────────────────────────────────

def cloud_mask(delta: np.ndarray, threshold: float = CLD_THRESHOLD) -> np.ndarray | None:
    """
    Returns a boolean 2-D mask: True where the pixel is unreliable (cloud
    fraction above threshold in EITHER the before or after image).
    Returns None if the cloud bands are missing (8-band legacy TIF).
    """
    if delta.shape[0] < 10:
        return None
    cld_b = delta[CLD_BAND_BEFORE]
    cld_a = delta[CLD_BAND_AFTER]
    return (cld_b > threshold) | (cld_a > threshold)


def cloud_panel(ax, delta: np.ndarray, extent: list, alerts: gpd.GeoDataFrame,
                sources_filter, range_name: str):
    """
    Shows cloud fraction for before (blue) and after (red) images side-by-side
    in a single panel using a 2-channel display, plus a hatched unreliable-pixel
    overlay.  Alerts within high-cloud regions get a ⚠ annotation.
    """
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Cloud Fraction\nbefore=blue · after=red · hatch=unreliable",
                 fontsize=7, color=TEXT_CLR, pad=3)

    if delta.shape[0] < 10:
        ax.text(0.5, 0.5, "cloud bands not in TIF\n(re-export with updated script)",
                ha="center", va="center", color="#888", fontsize=8,
                transform=ax.transAxes)
        ax.tick_params(colors=TEXT_CLR, labelsize=5)
        return

    cld_b = delta[CLD_BAND_BEFORE]
    cld_a = delta[CLD_BAND_AFTER]

    # Composite: show max cloud of either image as a warm colour
    worst = np.maximum(cld_b, cld_a)
    im    = ax.imshow(worst, extent=extent, origin="upper",
                       cmap="YlOrRd", vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
    cb    = ax.get_figure().colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.ax.tick_params(labelsize=5, colors=TEXT_CLR)
    cb.set_label("max(cloud_before, cloud_after)", fontsize=5, color=TEXT_CLR)

    # Hatch unreliable pixels
    unreliable = worst > CLD_THRESHOLD
    if unreliable.any():
        hatch_img = np.zeros((*unreliable.shape, 4), dtype=np.float32)
        hatch_img[unreliable] = [1, 1, 1, 0.18]
        ax.imshow(hatch_img, extent=extent, origin="upper",
                   aspect="auto", zorder=3)

    overlay_alerts(ax, alerts, sources_filter)
    ax.tick_params(labelsize=5, colors=TEXT_CLR)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def overlay_alerts(ax, alerts: gpd.GeoDataFrame, sources_filter=None):
    if alerts is None or alerts.empty:
        return
    for src, style in SOURCE_STYLE.items():
        if src.startswith("_") or (sources_filter and src not in sources_filter):
            continue
        sub = alerts[alerts["Source"] == src]
        if sub.empty:
            continue
        if "matched" in sub.columns:
            matched = sub[sub["matched"] == True]
            fp      = sub[sub["matched"] == False]
            if not matched.empty:
                matched.plot(ax=ax, facecolor="none", edgecolor=style["color"],
                             linewidth=style["lw"], linestyle=style["ls"], zorder=7)
            if not fp.empty:
                fp.plot(ax=ax, facecolor="none",
                        edgecolor=SOURCE_STYLE["_false_positive"]["color"],
                        linewidth=1.0, linestyle=":", zorder=4)
        else:
            sub.plot(ax=ax, facecolor="none", edgecolor=style["color"],
                     linewidth=style["lw"], linestyle=style["ls"], zorder=7)


def overlay_gt_raster(ax, gt_arr: np.ndarray | None, extent: list, color: str):
    """Overlay GT binary raster as a semi-transparent colour mask."""
    if gt_arr is None:
        return
    mask = np.zeros((*gt_arr.shape, 4), dtype=np.float32)
    rgba = matplotlib.colors.to_rgba(color)
    mask[gt_arr > 0] = (*rgba[:3], 0.35)   # semi-transparent where GT=1
    ax.imshow(mask, extent=extent, origin="upper", aspect="auto", zorder=3)


# ── RGB composite ─────────────────────────────────────────────────────────────

def make_rgb(delta: np.ndarray) -> np.ndarray:
    trees_loss = np.clip(-delta[1], 0, 1)   # trees decreased
    crops_gain  = np.clip( delta[4], 0, 1)   # crops increased
    bare_gain   = np.clip( delta[7], 0, 1)   # bare increased

    r = np.clip(trees_loss * 0.9 + bare_gain * 0.6, 0, 1)
    g = np.clip(crops_gain * 0.85, 0, 1)
    b = np.clip(bare_gain  * 0.20 + crops_gain * 0.08, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    pos_vals = rgb[rgb > 0]
    if len(pos_vals):
        p99 = np.percentile(pos_vals, 99)
        rgb = np.clip(rgb / max(p99, 1e-6), 0, 1)
    return rgb


# ── Source count bar ──────────────────────────────────────────────────────────

def source_bar(ax, alerts: gpd.GeoDataFrame, range_name: str, year: int):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(f"{range_name}  ·  Alert Counts  ({year})",
                 fontsize=8, color=TEXT_CLR, pad=4)
    if alerts.empty or "Source" not in alerts.columns:
        ax.text(0.5, 0.5, "No alerts", ha="center", va="center",
                color=TEXT_CLR, fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        return

    src_order = [s for s in DW_SOURCES if s in alerts["Source"].values]
    counts    = alerts["Source"].value_counts()
    vals      = [counts.get(s, 0) for s in src_order]
    colors    = [SOURCE_STYLE.get(s, {"color": "#AAA"})["color"] for s in src_order]

    bars = ax.barh(src_order, vals, color=colors, edgecolor="#333", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", label_type="edge",
                 color=TEXT_CLR, fontsize=7, padding=2)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.spines[:].set_edgecolor("#444")
    ax.set_xlabel("Count", color=TEXT_CLR, fontsize=7)
    ax.invert_yaxis()

    if "matched" in alerts.columns:
        match_vals = [alerts[(alerts["Source"] == s) & (alerts["matched"] == True)]
                      .shape[0] for s in src_order]
        ax.barh(src_order, match_vals, color=colors, alpha=0.35,
                label="Matched", edgecolor="none")
        ax.legend(fontsize=6, labelcolor=TEXT_CLR, facecolor=PANEL_BG, loc="lower right")


# ── Main figure builder ───────────────────────────────────────────────────────

def build_range_figure(range_name: str, year: int, year_a: int,
                       delta: np.ndarray, bounds,
                       alerts: gpd.GeoDataFrame,
                       gt_defor: np.ndarray | None,
                       gt_encr:  np.ndarray | None,
                       sources_filter: list = None) -> plt.Figure:

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    plt.rcParams.update({
        "axes.facecolor":   PANEL_BG,
        "axes.edgecolor":   "#30363d",
        "text.color":       TEXT_CLR,
        "xtick.color":      TEXT_CLR,
        "ytick.color":      TEXT_CLR,
        "axes.labelcolor":  TEXT_CLR,
    })

    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.32, wspace=0.18,
                            left=0.04, right=0.97,
                            top=0.92, bottom=0.04)

    # ── Row 0 cols 0-1: RGB composite ─────────────────────────────────────────
    ax_rgb = fig.add_subplot(gs[0, :2])
    rgb    = make_rgb(delta)
    ax_rgb.imshow(rgb, extent=extent, origin="upper", aspect="auto",
                  interpolation="nearest")
    ax_rgb.set_title(
        f"RGB Change  ·  Dec {year_a} → Dec {year}  ·  {range_name}\n"
        f"Red=trees lost  |  Green=crops gained  |  Orange=bare exposed",
        fontsize=9, color=TEXT_CLR, pad=5)

    # GT rasters as semi-transparent mask (no vectorisation needed)
    overlay_gt_raster(ax_rgb, gt_defor, extent, "#EF5350")   # red = deforestation
    overlay_gt_raster(ax_rgb, gt_encr,  extent, "#FF9800")   # orange = encroachment

    overlay_alerts(ax_rgb, alerts, sources_filter)
    make_legend(ax_rgb)
    ax_rgb.tick_params(labelsize=6)

    # ── Row 0 col 2: GT mask panel (dedicated) ────────────────────────────────
    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.set_facecolor(PANEL_BG)
    ax_gt.set_title("GT Raster  (Dec→Dec DW)\nRed=Defor  ·  Orange=Encr",
                    fontsize=8, color=TEXT_CLR, pad=4)
    # Show trees delta as grayscale base
    ax_gt.imshow(delta[1], extent=extent, origin="upper", cmap="Greys_r",
                 aspect="auto", interpolation="nearest", alpha=0.6)
    overlay_gt_raster(ax_gt, gt_defor, extent, "#EF5350")
    overlay_gt_raster(ax_gt, gt_encr,  extent, "#FF9800")
    overlay_alerts(ax_gt, alerts, sources_filter)
    ax_gt.tick_params(labelsize=5)

    # ── Row 0 col 3: Source count bar ─────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 3])
    source_bar(ax_bar, alerts, range_name, year)

    # ── Rows 1-2: 7 delta panels + 1 cloud panel ─────────────────────────────
    for i, band_name in enumerate(DELTA_PANEL_ORDER):
        row = 1 + i // 4
        col = i  % 4
        ax  = fig.add_subplot(gs[row, col])

        band_idx = DW_BANDS.index(band_name)
        data     = delta[band_idx]
        vmax     = float(np.clip(
                       max(abs(np.nanpercentile(data,  2)),
                           abs(np.nanpercentile(data, 98))),
                       0.05, 1.0))

        im = ax.imshow(data, extent=extent, origin="upper",
                       cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
        cb.ax.tick_params(labelsize=5, colors=TEXT_CLR)
        cb.outline.set_edgecolor("#444")

        ax.set_title(f"{DW_BAND_LABELS[band_name]}\nblue=lost · red=gained",
                     fontsize=7, color=TEXT_CLR, pad=3)
        ax.tick_params(labelsize=5, colors=TEXT_CLR)
        ax.set_facecolor(PANEL_BG)

        # Hatch out unreliable (cloudy) pixels on each delta panel
        cld  = cloud_mask(delta)
        if cld is not None and cld.any():
            bad = np.zeros((*cld.shape, 4), dtype=np.float32)
            bad[cld] = [0.8, 0.8, 0.8, 0.25]   # light grey semi-transparent
            ax.imshow(bad, extent=extent, origin="upper", aspect="auto", zorder=4)

        overlay_alerts(ax, alerts, sources_filter)

    # ── Cloud quality panel (bottom-right slot) ────────────────────────────────
    ax_cld = fig.add_subplot(gs[CLOUD_PANEL_SLOT[0], CLOUD_PANEL_SLOT[1]])
    cloud_panel(ax_cld, delta, extent, alerts, sources_filter, range_name)

    # ── Header ────────────────────────────────────────────────────────────────
    n_total = len(alerts)
    n_match = int(alerts["matched"].sum()) if "matched" in alerts.columns else "?"
    src_summary = "  ".join(
        f"{s}:{(alerts['Source'] == s).sum()}"
        for s in DW_SOURCES
        if not alerts.empty and (alerts["Source"] == s).sum() > 0
    ) if not alerts.empty else "no alerts"

    fig.suptitle(
        f"eNetra  ·  Guna Division  ·  {range_name}  ·  "
        f"{year} Alert Dynamics on DW Probability Delta\n"
        f"Total alerts: {n_total}   Matched: {n_match}   |   {src_summary}",
        fontsize=10, color=TEXT_CLR, y=0.975, fontweight="bold"
    )
    return fig


# ── Per-alert deep-dive ────────────────────────────────────────────────────────

def plot_per_alert(row, delta: np.ndarray, bounds, idx: int,
                   year: int, range_name: str, out_dir: Path):
    geom = row.geometry
    buf  = geom.buffer(0.005)
    minx, miny, maxx, maxy = buf.bounds
    rH, rW = delta.shape[1], delta.shape[2]

    xs    = rW / (bounds.right  - bounds.left)
    ys    = rH / (bounds.top    - bounds.bottom)
    c0    = max(int((minx - bounds.left)  * xs), 0)
    c1    = min(int((maxx - bounds.left)  * xs), rW)
    r0    = max(int((bounds.top  - maxy)  * ys), 0)
    r1    = min(int((bounds.top  - miny)  * ys), rH)
    if c1 <= c0 or r1 <= r0:
        return

    crop   = delta[:, r0:r1, c0:c1]
    extent = [minx, maxx, miny, maxy]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=DARK_BG)
    axes = axes.flatten()

    src        = row.get("Source", "Unknown")
    matched    = row.get("matched", None)
    gt_cls     = row.get("gt_class", "—") or "—"
    cls_ok     = "✅ class match" if row.get("class_match") else "❌ class mismatch"
    area_d     = row.get("area_delta_ha")
    days_e     = row.get("days_early")
    alert_date = str(row.get("alert_date", ""))[:10]

    for i, band_name in enumerate(DW_BANDS):
        ax   = axes[i]
        data = crop[i]
        vmax = float(max(abs(float(data.max())), abs(float(data.min())), 0.05))
        im   = ax.imshow(data, extent=extent, origin="upper",
                          cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                          aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5)
        ax.set_title(DW_BAND_LABELS.get(band_name, band_name),
                     fontsize=8, color=TEXT_CLR, pad=3)
        ax.tick_params(labelsize=5, colors=TEXT_CLR)
        ax.set_facecolor(PANEL_BG)
        try:
            style = SOURCE_STYLE.get(src, SOURCE_STYLE["Unknown"])
            gpd.GeoDataFrame([{"geometry": geom}], crs="EPSG:4326").plot(
                ax=ax, facecolor="none",
                edgecolor=style["color"], linewidth=2.0, zorder=9)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    status    = "✅ MATCHED" if matched else "❌ FALSE POS" if matched is False else "❔"
    area_str  = f"  AreaΔ: {area_d:+.2f} ha" if area_d is not None else ""
    lead_str  = f"  Days early: {days_e}" if days_e is not None else ""
    fig.suptitle(
        f"Alert #{idx}  ·  {alert_date}  ·  Source: {src}  ·  {status}  ·  {cls_ok}\n"
        f"GT class: {gt_cls}{area_str}{lead_str}",
        fontsize=10, color=TEXT_CLR, y=0.99, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"alert_{year}_{range_name}_{src.replace(' ','_')}_{idx:04d}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info(f"  → {fname}")


# ── Discover available ranges ─────────────────────────────────────────────────

def discover_ranges(year_a: int, year_b: int) -> list[str]:
    """Find all range names for which a delta TIF exists."""
    found = []
    # Check range subdirectories
    for d in GT_DIR.iterdir():
        if d.is_dir():
            p = d / f"dw_prob_delta_{year_a}_{year_b}_{d.name}.tif"
            if p.exists():
                found.append(d.name)
    # Check flat files
    import glob
    flat = list(GT_DIR.glob(f"dw_prob_delta_{year_a}_{year_b}_*.tif"))
    for p in flat:
        rng = p.stem.replace(f"dw_prob_delta_{year_a}_{year_b}_", "")
        if rng not in found:
            found.append(rng)
    return sorted(found)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",      type=int, default=2025)
    parser.add_argument("--year-a",    type=int, default=None)
    parser.add_argument("--range",     type=str, default=None,
                        help="Single range name, or 'all' (default)")
    parser.add_argument("--source",    type=str, default=None,
                        help="Comma-separated Source filter e.g. Trees,Bare")
    parser.add_argument("--per-alert", action="store_true")
    args = parser.parse_args()

    year_b = args.year
    year_a = args.year_a or (year_b - 1)
    sources_filter = [s.strip() for s in args.source.split(",")] if args.source else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which ranges to plot
    if args.range and args.range.lower() != "all":
        ranges_to_plot = [args.range]
    else:
        ranges_to_plot = discover_ranges(year_a, year_b)
        if not ranges_to_plot:
            log.error("No delta TIFs found. Run export_dw_prob_rasters.py first.")
            log.error("Or place TIFs in data/ground_truth/{range}/ manually.")
            return
        log.info(f"Found {len(ranges_to_plot)} ranges: {ranges_to_plot}")

    for range_name in ranges_to_plot:
        log.info(f"\n{'─'*55}")
        log.info(f"  Range: {range_name}")
        log.info(f"{'─'*55}")

        delta, bounds = compute_delta(year_a, year_b, range_name)
        if delta is None:
            continue

        alerts  = load_alerts(year_b, bounds)
        gt_d    = load_gt_raster(year_b, "deforestation", range_name, bounds)
        gt_e    = load_gt_raster(year_b, "encroachment",  range_name, bounds)

        if gt_d is None and gt_e is None:
            log.info("  GT rasters not found — showing delta + alerts only")

        log.info("\n  Building figure ...")
        fig = build_range_figure(
            range_name, year_b, year_a, delta, bounds,
            alerts, gt_d, gt_e, sources_filter
        )

        out_path = OUT_DIR / f"alert_dynamics_{year_b}_{range_name}.png"
        fig.savefig(out_path, dpi=175, bbox_inches="tight", facecolor=DARK_BG)
        log.info(f"  Saved: {out_path}")
        plt.close(fig)

        if args.per_alert and not alerts.empty:
            alert_dir = OUT_DIR / f"per_alert_{year_b}_{range_name}"
            alert_dir.mkdir(exist_ok=True)
            log.info(f"  Per-alert panels → {alert_dir}/")
            for idx, (_, row) in enumerate(alerts.iterrows()):
                if sources_filter and row.get("Source") not in sources_filter:
                    continue
                plot_per_alert(row, delta, bounds, idx, year_b, range_name, alert_dir)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
