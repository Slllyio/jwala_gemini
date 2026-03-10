"""
Vectorize Raster — Filtered TIF In, GeoJSON Out
================================================

Takes the filtered raster from filter_alert_raster.py and converts
the confirmed mask (Band 3) into GeoJSON polygons with change type
classification.

Usage:
    python scripts/vectorize_raster.py \
        --input-tif outputs/filtered/filtered_HAMEERPUR.tif \
        --alert-tif data/alerts/alert_delta_HAMEERPUR_2025-12-19_to_2025-12-24.tif \
        --out-geojson outputs/filtered/alerts.geojson

If --alert-tif is provided, change type is classified from the
original alert band deltas. Otherwise, polygons get "Unknown" type.
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import json
import logging
import os

import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import label as ndlabel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
N_BANDS = 8


def classify_change_type(mean_deltas: dict) -> str:
    """Classify change type from polygon-mean deltas."""
    trees = mean_deltas.get("trees", 0)
    crops = mean_deltas.get("crops", 0)
    bare  = mean_deltas.get("bare", 0)
    built = mean_deltas.get("built", 0)
    shrub = mean_deltas.get("shrub_scrub", 0)
    grass = mean_deltas.get("grass", 0)

    if trees > 0.05:
        return "Greening"

    if trees < -0.05:
        gains = {"crops": crops, "bare": bare, "built": built,
                 "shrub_scrub": shrub, "grass": grass}
        top_band = max(gains, key=gains.get)
        top_val  = gains[top_band]

        if top_band == "crops" and top_val > 0.03:
            return "Encroachment"
        elif top_band == "built" and top_val > 0.03:
            return "Built expansion"
        elif top_band == "bare" and top_val > 0.03:
            return "Clearing"
        elif top_band in ("shrub_scrub", "grass") and top_val > 0.03:
            return "Degradation"
        else:
            return "Tree loss (unclassified)"

    return "Other change"


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize filtered raster to GeoJSON"
    )
    parser.add_argument("--input-tif", required=True,
                        help="Filtered raster from filter_alert_raster.py "
                             "(3-band: score, n_fires, confirmed_mask)")
    parser.add_argument("--alert-tif", default=None,
                        help="Original alert delta TIF for change type "
                             "classification (optional, last window)")
    parser.add_argument("--out-geojson", required=True,
                        help="Output GeoJSON path")
    parser.add_argument("--min-pixels", type=int, default=5,
                        help="Min cluster size (default: 5)")
    args = parser.parse_args()

    # ── Load filtered raster ──────────────────────────────────────────
    with rasterio.open(args.input_tif) as ds:
        score_map = ds.read(1)     # Band 1: stacked_score
        n_fires_map = ds.read(2)   # Band 2: n_fires
        confirmed = ds.read(3)     # Band 3: confirmed_mask
        transform = ds.transform
        crs = str(ds.crs)
        H, W = score_map.shape

    confirmed_mask = confirmed > 0.5  # binary
    n_confirmed = int(confirmed_mask.sum())
    log.info(f"Filtered raster: {H}x{W}, {n_confirmed:,} confirmed pixels")

    if n_confirmed == 0:
        log.info("No confirmed pixels — writing empty GeoJSON")
        with open(args.out_geojson, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        return

    # ── Load alert deltas for change type (optional) ──────────────────
    alert_deltas = None
    if args.alert_tif and os.path.isfile(args.alert_tif):
        with rasterio.open(args.alert_tif) as ds:
            alert_data = ds.read()
        alert_deltas = alert_data[:N_BANDS].astype(np.float32)
        log.info(f"Alert deltas loaded from: {args.alert_tif}")

    # ── Label connected components ────────────────────────────────────
    labeled, n_components = ndlabel(confirmed_mask.astype(np.int32))
    log.info(f"Connected components: {n_components}")

    # ── Vectorize ─────────────────────────────────────────────────────
    shapes_gen = rasterio.features.shapes(
        labeled.astype(np.int32),
        mask=confirmed_mask,
        transform=transform,
    )

    features = []
    seen = set()

    for geom, value in shapes_gen:
        lid = int(value)
        if lid in seen or lid == 0:
            continue
        seen.add(lid)

        poly_mask = labeled == lid
        n_pix = int(poly_mask.sum())
        if n_pix < args.min_pixels:
            continue

        mean_score = float(np.mean(score_map[poly_mask]))
        mean_fires = float(np.mean(n_fires_map[poly_mask]))

        # Area (approximate at ~24N latitude)
        pixel_w_m = abs(transform.a) * 101300
        pixel_h_m = abs(transform.e) * 110600
        area_ha = n_pix * pixel_w_m * pixel_h_m / 10000

        # Centroid
        coords = np.array(geom["coordinates"][0])
        cx = float(np.mean(coords[:, 0]))
        cy = float(np.mean(coords[:, 1]))

        # Change type
        if alert_deltas is not None:
            mean_d = {}
            for b, name in enumerate(DW_BANDS):
                mean_d[name] = float(np.mean(alert_deltas[b][poly_mask]))
            change_type = classify_change_type(mean_d)
        else:
            change_type = "Unknown"

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "change_type": change_type,
                "stacked_score": round(mean_score, 3),
                "mean_fires": round(mean_fires, 1),
                "area_ha": round(area_ha, 4),
                "n_pixels": n_pix,
                "centroid": [round(cy, 6), round(cx, 6)],
            }
        })

    # ── Write GeoJSON ─────────────────────────────────────────────────
    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(args.out_geojson) or ".", exist_ok=True)
    with open(args.out_geojson, "w") as f:
        json.dump(geojson, f, indent=2)

    # Summary
    from collections import Counter
    type_counts = Counter(f["properties"]["change_type"] for f in features)
    total_area = sum(f["properties"]["area_ha"] for f in features)

    log.info(f"\nGenerated {len(features)} polygons")
    log.info(f"Total area: {total_area:.2f} ha")
    for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {ct}: {cnt}")
    log.info(f"GeoJSON: {args.out_geojson}")


if __name__ == "__main__":
    main()
