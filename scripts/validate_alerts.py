"""
Alert Validation Engine  (v2 — DW-native class schema)
========================================================
Intersects daily alert GeoJSONs with annual Dynamic World ground truth polygons.

Alert class comes directly from the `Source` field produced by ALERT_GUNA.ipynb:
    Trees         — tree probability dropped  (can be defor OR encroachment origin)
    Crops         — crop probability rose     (encroachment destination signal)
    Bare          — bare probability rose     (deforestation destination signal)
    Shrub and Scrub — scrub probability changed
    Built         — built-up probability rose (settlement expansion)
    Grass         — grass probability changed

Cross-referencing `Source` against GT class:
    GT=deforestation: expect Source ∈ {Trees, Bare, Shrub and Scrub}
    GT=encroachment:  expect Source ∈ {Trees, Crops, Shrub and Scrub}
    Class AGREE if alert Source aligns with GT class (using mapping below)
    Class MISMATCH if e.g. Crops alert matches a deforestation GT polygon

This answers:
  1. ACCURACY    — IoU match ≥ threshold with a GT polygon?
  2. CLASS MATCH — Does alert Source align with the GT class type?
  3. AREA DELTA  — Alert area vs confirmed GT polygon area (ha)
  4. TIME LEAD   — Days before Dec 31 ground truth the alert fired
  5. RECALL      — What % of GT polygons had at least one alert?

Input layout:
    data/alerts/            ← named alert_YYYY-MM-DD.geojson
                              each feature must have a `Source` attribute
    data/ground_truth/      ← dw_gt_{YEAR}_{deforestation|encroachment}.geojson
    (optional)
    data/validation/        ← output directory

Usage:
    python scripts/validate_alerts.py --year 2025
    python scripts/validate_alerts.py --year 2025 --iou-threshold 0.10
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALERTS_DIR = Path("data/alerts")
GT_DIR     = Path("data/ground_truth")
OUT_DIR    = Path("data/validation")

DEFAULT_IOU = 0.10

# ── DW Source → GT class alignment map ────────────────────────────────────────
# Which DW alert Sources are consistent with each GT class?
# An alert is "class-correct" if its Source is in the expected set.
DW_SOURCE_TO_GT = {
    "Trees":           ["deforestation", "encroachment"],  # tree loss — ambiguous alone
    "Bare":            ["deforestation"],
    "Crops":           ["encroachment"],
    "Shrub and Scrub": ["deforestation", "encroachment"],  # open-forest, ambiguous
    "Built":           [],          # settlement — neither class; always mismatch
    "Grass":           [],          # seasonal noise — always mismatch
}

# Canonical DW Source names (as exported by ALERT_GUNA notebook)
DW_SOURCES = list(DW_SOURCE_TO_GT.keys())


# ── Helpers ────────────────────────────────────────────────────────────────────

def iou(a, b) -> float:
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def area_ha(geom) -> float:
    return geom.area * (111_320 ** 2) * np.cos(np.radians(24)) / 10_000


def source_matches_gt(source: str, gt_class: str) -> bool:
    """True if the alert's DW Source is consistent with the GT class type."""
    return gt_class in DW_SOURCE_TO_GT.get(source, [])


# ── Data loading ───────────────────────────────────────────────────────────────

def load_alerts(year: int) -> gpd.GeoDataFrame:
    """
    Load all daily alert GeoJSONs, preserving the `Source` field.
    Expects files named: alert_YYYY-MM-DD.geojson
    Each feature should have attribute `Source` (from ALERT_GUNA notebook).
    """
    files = sorted(ALERTS_DIR.glob(f"alert_{year}-*.geojson"))
    if not files:
        log.warning(f"No alert files for {year} in {ALERTS_DIR}/")
        return gpd.GeoDataFrame()

    frames = []
    for f in files:
        try:
            gdf = gpd.read_file(f)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            date_str = f.stem.replace("alert_", "").split("_")[0]
            gdf["alert_date"] = pd.to_datetime(date_str)
            gdf["alert_file"] = f.name
            # Normalise Source field (notebook exports e.g. 'Shrub and Scrub')
            if "Source" not in gdf.columns:
                gdf["Source"] = "Unknown"
            frames.append(gdf)
        except Exception as e:
            log.warning(f"  Failed to load {f.name}: {e}")

    if not frames:
        return gpd.GeoDataFrame()

    combined = pd.concat(frames, ignore_index=True)
    alerts   = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    log.info(f"Loaded {len(alerts)} alerts from {len(files)} files ({year})")
    for src in DW_SOURCES:
        n = (alerts["Source"] == src).sum()
        if n > 0:
            log.info(f"  {src:<20}: {n} alerts")

    return alerts


def load_ground_truth(year: int) -> gpd.GeoDataFrame:
    frames = []
    for cls in ["deforestation", "encroachment"]:
        p = GT_DIR / f"dw_gt_{year}_{cls}.geojson"
        if p.exists():
            gdf = gpd.read_file(p)
            if "class" not in gdf.columns:
                gdf["class"] = cls
            frames.append(gdf)
        else:
            log.warning(f"  GT not found: {p}")

    if not frames:
        return gpd.GeoDataFrame()

    gt = pd.concat(frames, ignore_index=True)
    log.info(f"GT polygons: {len(gt)} "
             f"({(gt['class']=='deforestation').sum()} defor, "
             f"{(gt['class']=='encroachment').sum()} encr)")
    return gpd.GeoDataFrame(gt, crs="EPSG:4326")


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(alerts: gpd.GeoDataFrame, ground_truth: gpd.GeoDataFrame,
             year: int, iou_threshold: float) -> dict:

    dec_31     = pd.Timestamp(f"{year}-12-31")
    gt_sindex  = ground_truth.sindex
    gt_matched = set()
    records    = []

    for _, alert_row in alerts.iterrows():
        alert_geom  = alert_row.geometry
        alert_src   = alert_row.get("Source", "Unknown")
        alert_date  = alert_row.get("alert_date", pd.NaT)
        alert_area  = area_ha(alert_geom)

        candidates = list(gt_sindex.intersection(alert_geom.bounds))
        best_iou, best_gt, best_gt_id = 0.0, None, None

        for idx in candidates:
            score = iou(alert_geom, ground_truth.iloc[idx].geometry)
            if score > best_iou:
                best_iou   = score
                best_gt    = ground_truth.iloc[idx]
                best_gt_id = idx

        if best_iou >= iou_threshold and best_gt is not None:
            gt_matched.add(best_gt_id)
            gt_class    = best_gt.get("class", "unknown")
            gt_area     = area_ha(best_gt.geometry)
            days_early  = int((dec_31 - alert_date).days) if pd.notna(alert_date) else None
            class_match = source_matches_gt(alert_src, gt_class)

            records.append({
                "alert_date":    alert_date,
                "alert_file":    alert_row.get("alert_file", ""),
                "Source":        alert_src,          # DW class of alert
                "alert_area_ha": round(float(alert_area), 4),
                "matched":       True,
                "gt_class":      gt_class,
                "class_match":   class_match,        # True if Source consistent with GT
                "iou":           round(best_iou, 4),
                "area_delta_ha": round(float(alert_area - gt_area), 4),
                "days_early":    days_early,
                "geometry":      alert_geom,
            })
        else:
            records.append({
                "alert_date":    alert_date,
                "alert_file":    alert_row.get("alert_file", ""),
                "Source":        alert_src,
                "alert_area_ha": round(float(alert_area), 4),
                "matched":       False,
                "gt_class":      None,
                "class_match":   False,
                "iou":           round(best_iou, 4),
                "area_delta_ha": None,
                "days_early":    None,
                "geometry":      alert_geom,
            })

    missed_ids = [i for i in range(len(ground_truth)) if i not in gt_matched]
    missed_gt  = ground_truth.iloc[missed_ids].copy() if missed_ids else gpd.GeoDataFrame()
    return {"records": records, "missed_gt": missed_gt}


# ── Report ─────────────────────────────────────────────────────────────────────

def build_report(records: list, missed_gt: gpd.GeoDataFrame,
                 ground_truth: gpd.GeoDataFrame, year: int,
                 iou_threshold: float) -> str:

    total    = len(records)
    matched  = [r for r in records if r["matched"]]
    fp       = [r for r in records if not r["matched"]]
    n_gt     = len(ground_truth)
    n_caught = n_gt - len(missed_gt)

    precision = len(matched) / total   * 100 if total  > 0 else 0
    recall    = n_caught     / n_gt    * 100 if n_gt   > 0 else 0

    # Per-Source breakdown
    src_stats = {}
    for r in records:
        s = r["Source"]
        if s not in src_stats:
            src_stats[s] = {"total": 0, "matched": 0, "class_match": 0}
        src_stats[s]["total"]      += 1
        src_stats[s]["matched"]    += int(r["matched"])
        src_stats[s]["class_match"]+= int(r.get("class_match", False))

    deltas = [r["area_delta_ha"] for r in matched if r["area_delta_ha"] is not None]
    leads  = [r["days_early"]    for r in matched if r["days_early"]    is not None]

    lines = [
        f"{'='*62}",
        f"  eNetra Alert Validation  ·  {year}  ·  IoU≥{iou_threshold}",
        f"{'='*62}",
        f"",
        f"  OVERALL",
        f"    Total alerts:           {total:>6}",
        f"    Matched to GT:          {len(matched):>6}  ({precision:.1f}% precision)",
        f"    False positives:        {len(fp):>6}",
        f"    GT precision (recall):  {n_caught}/{n_gt}  ({recall:.1f}%)",
        f"",
        f"  PER DW SOURCE CLASS  (Source = field from ALERT_GUNA notebook)",
        f"    {'Source':<22} {'Alerts':>6} {'Matched':>8} {'ClassOK':>8}",
        f"    {'-'*48}",
    ]
    for src, st in sorted(src_stats.items()):
        m_pct = st["matched"]     / st["total"] * 100 if st["total"] > 0 else 0
        c_pct = st["class_match"] / st["total"] * 100 if st["total"] > 0 else 0
        lines.append(
            f"    {src:<22} {st['total']:>6}  {st['matched']:>5} ({m_pct:.0f}%)  "
            f"{st['class_match']:>5} ({c_pct:.0f}%)"
        )

    if deltas:
        lines += [
            f"",
            f"  AREA ACCURACY (matched alerts only)",
            f"    Mean area delta:   {np.mean(deltas):>+.2f} ha  (+ = alert over-reports)",
            f"    Std deviation:     {np.std(deltas):>.2f} ha",
        ]
    if leads:
        lines += [
            f"",
            f"  TIME LEAD (days before Dec 31)",
            f"    Mean:  {np.mean(leads):.0f} d  |  Min: {min(leads)} d  |  Max: {max(leads)} d",
        ]

    lines += [
        f"",
        f"  MISSED CHANGES (GT polygons with no alert)",
    ]
    if missed_gt.empty:
        lines.append("    None ✅")
    elif "class" in missed_gt.columns:
        for cls in missed_gt["class"].unique():
            sub = missed_gt[missed_gt["class"] == cls]
            ha  = sub["area_ha"].sum() if "area_ha" in sub else 0
            lines.append(f"    {cls:<22} {len(sub):>5} polygons  ({ha:.1f} ha total)")
    else:
        lines.append(f"    {len(missed_gt)} polygons")

    lines.append(f"\n{'='*62}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",          type=int,   required=True)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*55}")
    log.info(f"  eNetra Alert Validation  |  Year: {args.year}")
    log.info(f"{'='*55}\n")

    alerts       = load_alerts(args.year)
    ground_truth = load_ground_truth(args.year)

    if alerts.empty or ground_truth.empty:
        log.error("Need both alert GeoJSONs and GT GeoJSONs to validate.")
        return

    log.info(f"\nRunning intersection (IoU≥{args.iou_threshold}) ...")
    result = validate(alerts, ground_truth, args.year, args.iou_threshold)

    # Save matched alerts GeoJSON (with Source + class_match fields)
    gdf_out = gpd.GeoDataFrame(result["records"], crs="EPSG:4326")
    matched_path = OUT_DIR / f"alert_validation_{args.year}.geojson"
    gdf_out.to_file(matched_path, driver="GeoJSON")
    log.info(f"  Saved: {matched_path}")

    if not result["missed_gt"].empty:
        miss_path = OUT_DIR / f"missed_changes_{args.year}.geojson"
        result["missed_gt"].to_file(miss_path, driver="GeoJSON")
        log.info(f"  Saved: {miss_path}")

    report = build_report(result["records"], result["missed_gt"],
                          ground_truth, args.year, args.iou_threshold)
    print(f"\n{report}")
    (OUT_DIR / f"report_{args.year}.txt").write_text(report)


if __name__ == "__main__":
    main()
