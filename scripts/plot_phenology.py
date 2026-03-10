"""
Guna Division — NDVI Phenology Time Series (2022–2025)
=======================================================
Extracts monthly median NDVI from HLS (Harmonized Landsat + Sentinel-2)
for FOREST PIXELS ONLY across Guna Division.

Forest mask = Hansen treecover2000 ≥ 5%  AND  ESRI LULC class = Trees (2)
This double-gates cropland, settlements, and rocky outcrops that share the
same AOI boundary as forest beats.

Generates:
  1. outputs/phenology/ndvi_phenology.csv          — monthly division-wide values
  2. outputs/phenology/ndvi_phenology.png          — yearly curves + delta analysis
  3. outputs/phenology/ndvi_heatmap.png            — year × month heatmap
  4. outputs/phenology/ndvi_by_range.png           — per-range phenology (8 ranges)
  5. outputs/phenology/forest_coverage.png         — forest % per range (diagnostic)
  6. outputs/phenology/ndvi_records_*.json         — cached GEE result (re-plot only)

Usage:
    python scripts/plot_phenology.py --config config.yaml
    python scripts/plot_phenology.py --config config.yaml --range Fatehgarh
    python scripts/plot_phenology.py --config config.yaml --beat "Mar Ki Mahu"
    python scripts/plot_phenology.py --config config.yaml --json outputs/phenology/ndvi_records_guna_division.json
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# ── Setup ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("outputs/phenology")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── GEE helpers ─────────────────────────────────────────────────────────────

def init_gee(project: str):
    import ee
    try:
        ee.Initialize(project=project)
        log.info(f"GEE initialised — project: {project}")
    except Exception as e:
        log.error(f"GEE init failed: {e}")
        log.info("Run: earthengine authenticate")
        sys.exit(1)


def get_modis_ndvi(aoi, start_date: str, end_date: str):
    """
    Return MODIS MOD13Q1 median NDVI image for the period.
    MOD13Q1: 250m, 16-day composite, Terra.
    NDVI band is pre-computed, scaled x10000 → divide by 10000.
    QA: keep pixel_reliability == 0 (good data) or 1 (marginal — usable).
    100% global coverage → no empty-collection problem.
    """
    import ee

    def mask_and_scale(img):
        qa   = img.select("SummaryQA")          # 0=good, 1=marginal, 2=snow, 3=cloudy
        good = qa.lte(1)                         # accept good + marginal
        ndvi = img.select("NDVI").multiply(0.0001)  # scale to [-1, 1]
        return ndvi.updateMask(good).rename("NDVI")

    return (ee.ImageCollection("MODIS/061/MOD13Q1")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .map(mask_and_scale)
              .median())


def get_dw_trees(aoi, start_date: str, end_date: str):
    """
    Return Dynamic World median tree probability image for the period.
    Dynamic World: 10m, Sentinel-2 based, near real-time (Google, 2022-present).

    Key insight: DW 'trees' probability varies seasonally in dry deciduous
    forests even though DW was trained globally, because the Sentinel-2 spectral
    inputs it sees change with leaf phenology. Same-season comparison cancels
    this local phenological bias, making DW locally calibrated for Guna.

    Band 'trees': probability [0..1] that pixel is tree-covered.
    Values near 0.7–0.9 = confirmed healthy forest.
    Values near 0.3–0.5 = degraded/sparse canopy.
    Drop of >0.20 same-season = structural forest loss signal.
    """
    import ee

    def cloud_filter(img):
        # DW is already cloud-masked at source (uses S2 cloud probability mask)
        # Select just the trees band
        return img.select("trees")

    return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .map(cloud_filter)
              .median())


def get_forest_mask(aoi):
    """
    Strict forest-only mask at 250m (MODIS resolution).
    Dual gate: Hansen treecover2000 ≥5% AND ESRI LULC class == Trees(2).
    Resampled to 250m to match MODIS NDVI pixel grid.
    """
    import ee
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    treecover = hansen.select("treecover2000").gte(5)

    esri_lulc = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
                   .filterDate("2022-01-01", "2023-12-31")
                   .mosaic()
                   .select("b1"))
    trees = esri_lulc.eq(2)

    # Combine gates and resample to 250m (MODIS pixel grid)
    forest_mask = treecover.And(trees).reproject(crs="EPSG:4326", scale=250)

    # Diagnostic: log forest % at 250m
    try:
        total  = forest_mask.unmask(0).reduceRegion(
            reducer=ee.Reducer.count(), geometry=aoi, scale=250, maxPixels=1e8,
            bestEffort=True).getInfo().get("treecover2000", 1)
        forest = forest_mask.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=250, maxPixels=1e8,
            bestEffort=True).getInfo().get("treecover2000", 0)
        pct = 100.0 * forest / max(total, 1)
        log.info(f"  Forest coverage in AOI: {pct:.1f}%  ({forest:,.0f} / {total:,.0f} px @ 250m)")
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    return forest_mask


def sample_ndvi_for_month(aoi, year: int, month: int, forest_mask, project: str):
    """
    Returns (mean, std, count) NDVI of forested 250-m MODIS pixels for a given month.
    Uses a 45-day window centred on the 15th so at least one 16-day composite falls in.
    Returns (None, None, 0) if no valid pixels.
    """
    import ee
    centre = datetime(year, month, 15)
    start  = (centre - timedelta(days=22)).strftime("%Y-%m-%d")
    end    = (centre + timedelta(days=22)).strftime("%Y-%m-%d")

    try:
        ndvi_img = get_modis_ndvi(aoi, start, end)
        masked   = ndvi_img.updateMask(forest_mask)

        stats = masked.reduceRegion(
            reducer   = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
                          .combine(ee.Reducer.count(), None, True),
            geometry  = aoi,
            scale     = 250,       # MODIS 250m native resolution
            maxPixels = 1e8,
            bestEffort= True,
        ).getInfo()
        mean  = stats.get("NDVI_mean")
        std   = stats.get("NDVI_stdDev")
        count = stats.get("NDVI_count")
        if mean is None or count is None or count < 5:   # 5 px @ 250m ≈ 31 ha
            return None, None, int(count or 0)
        return float(mean), float(std or 0), int(count)
    except Exception as e:
        log.warning(f"  [{year}-{month:02d}] GEE error: {e}")
        return None, None, 0


def sample_dw_for_month(aoi, year: int, month: int, forest_mask):
    """
    Returns (mean, std, count) Dynamic World 'trees' probability for
    confirmed forest pixels in the given month (45-day window, 10m pixels).

    DW is Sentinel-2 based (10m), so the forest mask is projected to 10m.
    Returns (None, None, 0) if no S2 observations exist (rare cloud-out).

    Why same-season matters: DW tree probability naturally dips from ~0.75
    (monsoon) to ~0.50 (dry season) in Guna's deciduous forests because the
    S2 spectral signature of leafless trees differs from the global training
    distribution. Same-season comparison removes this phenological drift and
    makes any probability drop attributable to structural change only.
    """
    import ee
    centre = datetime(year, month, 15)
    start  = (centre - timedelta(days=22)).strftime("%Y-%m-%d")
    end    = (centre + timedelta(days=22)).strftime("%Y-%m-%d")

    try:
        # DW available from 2016 globally; 2022 onward for India with good density
        if year < 2016:
            return None, None, 0

        dw_img = get_dw_trees(aoi, start, end)

        # Reproject forest mask to 10m to match DW native resolution
        forest_10m = forest_mask.reproject(crs="EPSG:4326", scale=10)
        masked     = dw_img.updateMask(forest_10m)

        stats = masked.reduceRegion(
            reducer   = ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
                          .combine(ee.Reducer.count(), None, True),
            geometry  = aoi,
            scale     = 10,        # DW 10m native resolution
            maxPixels = 1e9,
            bestEffort= True,
        ).getInfo()
        mean  = stats.get("trees_mean")
        std   = stats.get("trees_stdDev")
        count = stats.get("trees_count")
        if mean is None or count is None or count < 100:  # 100 px @ 10m ≈ 1 ha
            return None, None, int(count or 0)
        return float(mean), float(std or 0), int(count)
    except Exception as e:
        log.warning(f"  [{year}-{month:02d}] DW GEE error: {e}")
        return None, None, 0


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_phenology(aoi, aoi_label: str, cfg: dict, label: str = "division"):
    """
    Extract monthly NDVI + DW tree probability for 2022–2025 for confirmed
    forest pixels only.

    Two signals per month:
      ndvi_mean  — MODIS MOD13Q1 NDVI (250m) — broad spectral baseline
      dw_mean    — Dynamic World 'trees' prob (10m) — semantic tree identity

    Both use a 45-day median composite window centred on the 15th of each month.
    Same-season comparison of both signals makes the system locally calibrated
    for Guna's dry deciduous phenological cycle.

    Forest mask: Hansen treecover ≥5% AND ESRI LULC = Trees.
    """
    import ee
    forest_mask = get_forest_mask(aoi)

    records = []
    years   = [2022, 2023, 2024, 2025]

    # Feb 2026 is current date — only Jan–Feb available for 2025
    for year in years:
        max_month = 2 if year == 2025 else 12
        for month in range(1, max_month + 1):
            log.info(f"  [{label}] Sampling NDVI + DW: {year}-{month:02d} ...")
            ndvi_mean, ndvi_std, ndvi_count = sample_ndvi_for_month(
                aoi, year, month, forest_mask, cfg["gee"]["gee_project"]
            )
            dw_mean, dw_std, dw_count = sample_dw_for_month(
                aoi, year, month, forest_mask
            )
            records.append({
                "label":       label,
                "year":        year,
                "month":       month,
                "date_label":  f"{year}-{month:02d}",
                # NDVI signal (MODIS 250m)
                "ndvi_mean":   ndvi_mean,
                "ndvi_std":    ndvi_std,
                "pixel_count": ndvi_count,
                # Dynamic World tree probability (S2 10m)
                "dw_mean":     dw_mean,
                "dw_std":      dw_std,
                "dw_count":    dw_count,
            })
            ndvi_str = f"NDVI={ndvi_mean:.4f}±{ndvi_std:.4f}" if ndvi_mean is not None else "NDVI=--"
            dw_str   = f"DW_trees={dw_mean:.4f}±{dw_std:.4f}" if dw_mean is not None else "DW_trees=--"
            log.info(f"    {ndvi_str}  |  {dw_str}")

    return records


def extract_by_range(beats: list, cfg: dict):
    """
    Stratified extraction: run extract_phenology separately for each of the
    8 Guna ranges so we can see if phenology differs across the division.
    Only extracts peak dry-season month (March) and peak monsoon month (September)
    to keep GEE quota usage manageable.
    """
    import ee
    from collections import defaultdict

    # Dissolve each range into a single geometry
    by_range = defaultdict(list)
    for b in beats:
        by_range[b["range"]].append(b["geometry"])

    range_records = []
    peak_months   = [3, 9]   # March (dry peak) + September (monsoon peak)

    for range_name, geoms in sorted(by_range.items()):
        # Build dissolved geometry for this range
        coords = []
        for g in geoms:
            info = g.getInfo()
            if info["type"] == "Polygon":
                coords.append(info["coordinates"])
            elif info["type"] == "MultiPolygon":
                coords.extend(info["coordinates"])
        range_geom  = ee.Geometry.MultiPolygon(coords)
        forest_mask = get_forest_mask(range_geom)

        for year in [2023, 2024]:
            for month in peak_months:
                log.info(f"  Range {range_name}: {year}-{month:02d}")
                mean, std, count = sample_ndvi_for_month(
                    range_geom, year, month, forest_mask,
                    cfg["gee"]["gee_project"]
                )
                range_records.append({
                    "range":      range_name,
                    "year":       year,
                    "month":      month,
                    "ndvi_mean":  mean,
                    "ndvi_std":   std,
                    "pixel_count": count,
                })
    return range_records


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_phenology(records: list, aoi_label: str, save_dir: Path):
    """
    Plot 1: Full NDVI time series (each year + multi-year mean)
    Plot 2: Within-year phenology showing safe dNDVI thresholds
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.error("matplotlib not installed: pip install matplotlib")
        return

    # ── Organise data ────────────────────────────────────────────────────
    years     = sorted(set(r["year"] for r in records))
    by_year   = {y: {} for y in years}
    for r in records:
        if r["ndvi_mean"] is not None:
            by_year[r["year"]][r["month"]] = (r["ndvi_mean"], r["ndvi_std"])

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    colors = {2022: "#3B82F6", 2023: "#10B981", 2024: "#F59E0B", 2025: "#EF4444"}

    # ── FIGURE 1: Full yearly curves ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), facecolor="#0F172A")
    fig.suptitle(
        f"Guna Division — Forest NDVI + DW Phenology (same-season calibration)\n{aoi_label} | MODIS 250m + Dynamic World 10m | Forest pixels only (treecover ≥5%)",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    ax1 = axes[0]
    ax1.set_facecolor("#1E293B")
    ax1.set_title("Monthly Median NDVI by Year (2022–2025)", color="#94A3B8", fontsize=11)

    for year, monthly in by_year.items():
        months = sorted(monthly.keys())
        means  = [monthly[m][0] for m in months]
        stds   = [monthly[m][1] for m in months]

        ax1.plot(months, means, "o-",
                 color=colors[year], linewidth=2.5, markersize=6,
                 label=str(year), alpha=0.9)
        ax1.fill_between(months,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=colors[year], alpha=0.12)

    # Seasonal annotations
    ax1.axvspan(2, 5, alpha=0.07, color="#EF4444",  label="Dry season (Feb–May)")
    ax1.axvspan(7, 10, alpha=0.07, color="#10B981", label="Monsoon peak (Jul–Oct)")
    ax1.axhline(0.15, color="#94A3B8", linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(0.5, 0.16, "NDVI=0.15 (bare soil / near-death)", color="#94A3B8",
             fontsize=7.5, style="italic", transform=ax1.get_yaxis_transform())

    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels, color="#CBD5E1")
    ax1.set_ylabel("NDVI", color="#CBD5E1")
    ax1.set_ylim(0, 0.75)
    ax1.tick_params(colors="#CBD5E1")
    ax1.spines["bottom"].set_color("#334155")
    ax1.spines["left"].set_color("#334155")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", facecolor="#1E293B", labelcolor="white",
               edgecolor="#334155", fontsize=9)
    ax1.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)

    # ── FIGURE 2: Same-season dNDVI distribution ─────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1E293B")
    ax2.set_title(
        "Same-Season ΔNDVI: What's Normal vs Felling Signal\n"
        "(2024 vs 2023 same month — natural year-to-year variation)",
        color="#94A3B8", fontsize=11
    )

    # Compute same-season dNDVI between 2024 and 2023
    d_ndvi_normal = []
    shared_months = []
    for m in range(1, 13):
        if m in by_year.get(2024, {}) and m in by_year.get(2023, {}):
            d  = by_year[2024][m][0] - by_year[2023][m][0]
            d_ndvi_normal.append(d)
            shared_months.append(m)

    if d_ndvi_normal:
        bar_colors = ["#EF4444" if d < -0.15 else ("#F59E0B" if d < 0 else "#10B981")
                      for d in d_ndvi_normal]
        ax2.bar(shared_months, d_ndvi_normal, color=bar_colors, alpha=0.8, width=0.7)
        ax2.axhline(0, color="#94A3B8", linewidth=1)

        # Threshold bands
        ax2.axhline(-0.15, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.8)
        ax2.axhline(-0.10, color="#F59E0B", linestyle="--", linewidth=1.5, alpha=0.8)
        ax2.text(0.5, -0.14, "dNDVI = -0.15 (proposed detection threshold)",
                 color="#EF4444", fontsize=7.5, style="italic",
                 transform=ax2.get_yaxis_transform())
        ax2.text(0.5, -0.09, "dNDVI = -0.10",
                 color="#F59E0B", fontsize=7.5, style="italic",
                 transform=ax2.get_yaxis_transform())

        # Annotate unsafe zone
        ax2.fill_between([0.5, 12.5], [-0.30, -0.30], [-0.15, -0.15],
                         color="#EF4444", alpha=0.08)
        ax2.text(6.5, -0.23, "Felling signal zone (below -0.15)",
                 ha="center", color="#EF4444", fontsize=8, alpha=0.8)

        ax2.set_xticks(shared_months)
        ax2.set_xticklabels([month_labels[m-1] for m in shared_months], color="#CBD5E1")
        ax2.set_ylabel("ΔNDVI (2024 − 2023)", color="#CBD5E1")
        ax2.set_ylim(-0.35, 0.35)
        ax2.tick_params(colors="#CBD5E1")
        ax2.spines["bottom"].set_color("#334155")
        ax2.spines["left"].set_color("#334155")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)

        # Key insight text box
        natural_variation = max(abs(d) for d in d_ndvi_normal)
        ax2.text(0.98, 0.95,
                 f"Max natural same-season\nvariation: ±{natural_variation:.3f} NDVI\n"
                 f"Proposed threshold: 0.15",
                 transform=ax2.transAxes, ha="right", va="top",
                 color="white", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F172A",
                           edgecolor="#334155", alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "Insufficient data for 2023/2024 comparison",
                 ha="center", va="center", transform=ax2.transAxes,
                 color="#94A3B8", fontsize=12)

    # ── FIGURE 3: DW tree probability time-series (if data available) ─────
    # Build DW data from records (may be absent in old JSON — skip gracefully)
    dw_by_year = {y: {} for y in years}
    for r in records:
        dw_val = r.get("dw_mean")
        if dw_val is not None:
            dw_by_year[r["year"]][r["month"]] = (dw_val, r.get("dw_std", 0) or 0)

    has_dw = any(bool(m) for m in dw_by_year.values())

    if has_dw:
        ax3 = axes[2]
        ax3.set_facecolor("#1E293B")
        ax3.set_title(
            "Dynamic World 'trees' Probability — Monthly Median (same-season calibration)\n"
            "Drop in DW_trees vs same calendar month last year = structural canopy loss",
            color="#94A3B8", fontsize=10
        )

        for year, monthly in dw_by_year.items():
            if not monthly:
                continue
            months = sorted(monthly.keys())
            means  = [monthly[m][0] for m in months]
            stds   = [monthly[m][1] for m in months]
            ax3.plot(months, means, "o--",
                     color=colors[year], linewidth=2, markersize=5,
                     label=str(year), alpha=0.85)
            ax3.fill_between(months,
                             [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                             color=colors[year], alpha=0.10)

        # Seasonal reference lines
        ax3.axvspan(2, 5, alpha=0.07, color="#EF4444",  label="Dry (Feb–May)")
        ax3.axvspan(7, 10, alpha=0.07, color="#10B981", label="Monsoon (Jul–Oct)")

        # DW threshold for structural loss (0.20 drop from baseline)
        ax3.axhline(0.20, color="#F59E0B", linestyle=":", linewidth=1, alpha=0.7)
        ax3.text(0.5, 0.21, "ΔDW_trees = 0.20 (proposed deforestation threshold)",
                 color="#F59E0B", fontsize=7.5, style="italic",
                 transform=ax3.get_yaxis_transform())

        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(month_labels, color="#CBD5E1")
        ax3.set_ylabel("DW trees probability", color="#CBD5E1")
        ax3.set_ylim(0, 1.05)
        ax3.tick_params(colors="#CBD5E1")
        ax3.spines["bottom"].set_color("#334155")
        ax3.spines["left"].set_color("#334155")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.legend(loc="upper left", facecolor="#1E293B", labelcolor="white",
                   edgecolor="#334155", fontsize=9)
        ax3.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.6)
    else:
        ax3 = axes[2]
        ax3.set_facecolor("#1E293B")
        ax3.text(0.5, 0.5,
                 "DW tree probability: no data in this JSON\n"
                 "Re-run plot_phenology.py without --json to extract DW via GEE",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="#94A3B8", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = save_dir / "ndvi_phenology.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  Saved: {out_path}")
    return out_path



def plot_seasonal_heatmap(records: list, aoi_label: str, save_dir: Path):
    """Year × Month NDVI heatmap — shows the full phenological pattern at a glance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    years  = sorted(set(r["year"] for r in records))
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Build matrix
    matrix = np.full((len(years), 12), np.nan)
    for r in records:
        if r["ndvi_mean"] is not None and r["year"] in years:
            yi = years.index(r["year"])
            mi = r["month"] - 1
            matrix[yi, mi] = r["ndvi_mean"]

    fig, ax = plt.subplots(figsize=(13, 3.5), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.05, vmax=0.65, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("NDVI", color="#CBD5E1")
    cbar.ax.yaxis.set_tick_params(color="#CBD5E1")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#CBD5E1")

    # Annotate with values
    for yi, year in enumerate(years):
        for mi in range(12):
            val = matrix[yi, mi]
            if not np.isnan(val):
                ax.text(mi, yi, f"{val:.2f}", ha="center", va="center",
                        color="black" if 0.25 < val < 0.55 else "white",
                        fontsize=8, fontweight="bold")
            else:
                ax.text(mi, yi, "—", ha="center", va="center",
                        color="#64748B", fontsize=9)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, color="#CBD5E1")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    ax.set_title(
        f"Guna Division Forest NDVI — Year × Month Heatmap\n{aoi_label}",
        color="white", fontsize=11, fontweight="bold"
    )

    # Vertical lines marking season boundaries
    for x in [1.5, 4.5, 5.5, 9.5]:
        ax.axvline(x, color="#475569", linewidth=1.2, linestyle="--")

    ax.text(0, len(years)+0.1, "DRY SEASON →", color="#F87171",
            fontsize=7.5, ha="center")
    ax.text(7.5, len(years)+0.1, "← MONSOON →", color="#34D399",
            fontsize=7.5, ha="center")

    plt.tight_layout()
    out_path = save_dir / "ndvi_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  Saved: {out_path}")
    return out_path


def save_csv(records: list, save_dir: Path, aoi_label: str):
    import csv
    out_path = save_dir / "ndvi_phenology.csv"
    fieldnames = [
        "label", "year", "month", "date_label",
        # MODIS NDVI (250m)
        "ndvi_mean", "ndvi_std", "pixel_count",
        # Dynamic World tree probability (10m)
        "dw_mean", "dw_std", "dw_count",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info(f"  CSV saved: {out_path}")
    return out_path



# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Guna Division NDVI Phenology Extractor")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--range", default=None,
                    help="Extract only for a specific range (e.g. Fatehgarh)")
    ap.add_argument("--beat",  default=None,
                    help="Extract only for a specific beat (e.g. 'Mar Ki Mahu')")
    ap.add_argument("--json",  default=None,
                    help="Load pre-existing JSON instead of querying GEE (for re-plotting)")
    args = ap.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── If pre-computed JSON is given, skip GEE ──────────────────────────────
    if args.json:
        log.info(f"Loading pre-computed records from: {args.json}")
        with open(args.json) as f:
            records = json.load(f)
        aoi_label = Path(args.json).stem
        save_csv(records, OUT_DIR, aoi_label)
        plot_phenology(records, aoi_label, OUT_DIR)
        plot_seasonal_heatmap(records, aoi_label, OUT_DIR)
        return

    # ── GEE init ─────────────────────────────────────────────────────────────
    import ee
    project = cfg["gee"]["gee_project"]
    init_gee(project)

    # ── Build AOI ────────────────────────────────────────────────────────────
    sys.path.insert(0, ".")
    from src.data.gee_fetch import load_guna_beats, load_guna_division_aoi, load_config
    load_config(args.config)

    if args.beat:
        # Single beat
        beats = load_guna_beats()
        beat_name = args.beat.lower()
        matched = [b for b in beats if b["beat"].lower() == beat_name]
        if not matched:
            available = [b["beat"] for b in beats]
            log.error(f"Beat '{args.beat}' not found. Available: {available[:10]}...")
            sys.exit(1)
        aoi       = matched[0]["geometry"]
        aoi_label = f"Beat: {matched[0]['beat']} ({matched[0]['range']})"
        suffix    = args.beat.replace(" ", "_").lower()

    elif args.range:
        # Single range (dissolve all beats)
        beats = load_guna_beats(range_filter=args.range)
        geoms = [b["geometry"] for b in beats]
        aoi   = ee.Geometry.MultiPolygon(
            [list(g.coordinates().getInfo()) for g in geoms]
        ).dissolve()
        aoi_label = f"Range: {args.range}"
        suffix    = args.range.lower()

    else:
        # Full division
        aoi       = load_guna_division_aoi()
        aoi_label = "Full Guna Division (139 beats)"
        suffix    = "guna_division"

    log.info(f"\nExtracting NDVI phenology for: {aoi_label}")
    log.info("=" * 60)

    # ── Extract ───────────────────────────────────────────────────────────────
    records = extract_phenology(aoi, aoi_label, cfg)

    # ── Save JSON (so we can re-plot without GEE) ─────────────────────────────
    json_path = OUT_DIR / f"ndvi_records_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"  Records saved: {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(records, OUT_DIR, aoi_label)

    # ── Plot ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating plots...")
    plot_phenology(records, aoi_label, OUT_DIR)
    plot_seasonal_heatmap(records, aoi_label, OUT_DIR)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  NDVI PHENOLOGY SUMMARY — {aoi_label}")
    print("=" * 70)
    print(f"  {'Month':<8}  {'2022':>8}  {'2023':>8}  {'2024':>8}  {'2025':>8}")
    print(f"  {'-----':<8}  {'----':>8}  {'----':>8}  {'----':>8}  {'----':>8}")

    by_year_month = {}
    for r in records:
        key = (r["year"], r["month"])
        by_year_month[key] = r["ndvi_mean"]

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        vals = []
        for y in [2022, 2023, 2024, 2025]:
            v = by_year_month.get((y, m))
            vals.append(f"{v:.3f}" if v is not None else "  —  ")
        print(f"  {month_names[m-1]:<8}  {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}  {vals[3]:>8}")

    # ── Key phenology stats ───────────────────────────────────────────────────
    valid = [r for r in records if r["ndvi_mean"] is not None]
    if valid:
        all_ndvi = [r["ndvi_mean"] for r in valid]
        # Dry season (Feb–May)
        dry_ndvi = [r["ndvi_mean"] for r in valid if r["month"] in [2,3,4,5]]
        # Peak monsoon (Aug–Sep)
        peak_ndvi = [r["ndvi_mean"] for r in valid if r["month"] in [8,9]]

        print(f"\n  Overall range:     {min(all_ndvi):.3f} – {max(all_ndvi):.3f}")
        if dry_ndvi:
            print(f"  Dry season avg:    {np.mean(dry_ndvi):.3f}  (Feb–May)")
            print(f"  Dry season min:    {min(dry_ndvi):.3f}  (lowest leaf-drop baseline)")
        if peak_ndvi:
            print(f"  Monsoon peak avg:  {np.mean(peak_ndvi):.3f}  (Aug–Sep)")
            print(f"  Annual amplitude:  {np.mean(peak_ndvi) - np.mean(dry_ndvi):.3f}  " +
                  f"(natural phenological swing)")
        print(f"\n  >> To detect actual felling, same-season dNDVI must EXCEED the")
        print(f"     natural year-to-year variation (shown in the delta plot).")
        print(f"     Check ndvi_delta_analysis.png for calibrated thresholds.")

    print("=" * 70)
    print(f"\n  Outputs saved to: {OUT_DIR.resolve()}/")
    print(f"    ndvi_phenology.csv    — raw monthly values")
    print(f"    ndvi_phenology.png    — yearly curves + delta analysis")
    print(f"    ndvi_heatmap.png      — year × month heatmap")
    print(f"    ndvi_records_{suffix}.json  — full JSON (for re-plotting)")
    print("=" * 70)


if __name__ == "__main__":
    main()
