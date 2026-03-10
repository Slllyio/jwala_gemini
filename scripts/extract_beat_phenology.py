"""
scripts/extract_beat_phenology.py
===================================
Extract monthly NDVI + DW-tree time series for every RANGE in Guna Division.

KEY APPROACH - Server-side GEE batching:
  Instead of 72 sequential Python-side GEE API calls (one per month), we
  build the entire monthly time-series in a SINGLE server-side computation
  using ee.List.sequence + imageCollection.map(). One .getInfo() call per
  range pulls all 72 months at once.

  Time: ~5-10s per range  vs  ~3-4 min per range with sequential calls.
  Total wall time: ~1-2 min for all 8 ranges.

Granularity: Range-level (8 ranges, pooling 9-22 beats each for stable fits).

Outputs:
  data/phenology/range_ndvi_series.csv   <- raw monthly means per range
  data/phenology/range_models/           <- fitted per-range JSON models

Usage:
  python scripts/extract_beat_phenology.py               # extract + fit
  python scripts/extract_beat_phenology.py --extract-only
  python scripts/extract_beat_phenology.py --fit-only
  python scripts/extract_beat_phenology.py --limit 2     # test 2 ranges
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.phenology_model import PhenologyModel


# ── GEE helpers ───────────────────────────────────────────────────────────────
def _init_gee(project: str) -> None:
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def _load_config(path: Path) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ── geometry helpers ──────────────────────────────────────────────────────────
def _load_ranges(geojson_path: Path) -> list[dict]:
    """
    Dissolve guna_beats.geojson features by Range property.
    Returns [{range, label, features, n_beats}].
    """
    with open(geojson_path) as f:
        fc = json.load(f)

    by_range: dict[str, list] = defaultdict(list)
    for feat in fc.get("features", []):
        rng = feat["properties"].get("Range", "Unknown")
        by_range[rng].append(feat)

    ranges = []
    for rng, feats in sorted(by_range.items()):
        ranges.append({
            "range":    rng,
            "label":    rng,
            "features": feats,
            "n_beats":  len(feats),
        })
    return ranges


# ── SERVER-SIDE monthly extraction (single GEE call per range) ────────────────
def _extract_range_series_batched(
    rng: dict,
    start_year: int = 2020,
    end_year:   int = 2025,
) -> list[dict]:
    """
    Pull ALL monthly NDVI + DW stats for a range in ONE GEE server-side call.

    Strategy:
      1. Build ee.List of month-start timestamps via ee.List.sequence
      2. Map over list: for each month, build NDVI and DW composites,
         reduce over the range geometry, return a Feature with properties
      3. Call .getInfo() ONCE on the resulting FeatureCollection
      4. Unpack results in Python
    """
    import ee

    # Dissolve all beat polygons into the range geometry
    geom = ee.FeatureCollection(rng["features"]).geometry().dissolve(maxError=1)

    # Build list of month-start timestamps (milliseconds since epoch)
    start_ms = ee.Date(f"{start_year}-01-01").millis()
    end_ms   = ee.Date(f"{end_year}-12-31").millis()
    # ee.List.sequence(start, end, step) where step = ~30 days in ms
    # We use advance(1,'month') approach instead — more precise
    n_months = (end_year - start_year + 1) * 12
    month_starts = ee.List.sequence(0, n_months - 1).map(
        lambda i: ee.Date(start_ms).advance(i, "month").millis()
    )

    def monthly_stats(month_start_ms):
        """Server-side: one reduceRegion call per month covering NDVI + DW."""
        start = ee.Date(month_start_ms)
        end   = start.advance(1, "month")

        ndvi_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
                      .filterDate(start, end).filterBounds(geom).select("NDVI"))
        dw_col   = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                      .filterDate(start, end).filterBounds(geom).select("trees"))

        has_ndvi = ndvi_col.size().gt(0)
        has_dw   = dw_col.size().gt(0)

        # Build a 2-band composite image: ndvi | trees
        ndvi_img = ee.Image(ee.Algorithms.If(
            has_ndvi,
            ndvi_col.median().multiply(0.0001).rename("ndvi"),
            ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))))

        dw_img = ee.Image(ee.Algorithms.If(
            has_dw,
            dw_col.median().rename("dw"),
            ee.Image.constant(0).rename("dw").updateMask(ee.Image(0))))

        combined = ndvi_img.addBands(dw_img)

        # ONE reduceRegion call → dict with ndvi_mean, ndvi_stdDev, dw_mean, dw_stdDev
        stats = combined.reduceRegion(
            reducer   = ee.Reducer.mean().combine(
                            ee.Reducer.stdDev(), sharedInputs=True),
            geometry  = geom,
            scale     = 250,        # MODIS resolution; DW upsampled but fast
            maxPixels = 1e8,
            bestEffort= True,
        )

        return ee.Feature(None, {
            "year":       start.get("year"),
            "month":      start.get("month"),
            "date_label": start.format("YYYY-MM"),
            "ndvi_mean":  ee.Algorithms.If(has_ndvi, stats.get("ndvi_mean"),  None),
            "ndvi_std":   ee.Algorithms.If(has_ndvi, stats.get("ndvi_stdDev"),None),
            "dw_mean":    ee.Algorithms.If(has_dw,   stats.get("dw_mean"),    None),
            "dw_std":     ee.Algorithms.If(has_dw,   stats.get("dw_stdDev"),  None),
        })

    # Map over all months server-side → FeatureCollection → single getInfo()
    fc_result = ee.FeatureCollection(month_starts.map(monthly_stats))
    info      = fc_result.getInfo()   # ONE network round-trip

    records = []
    for feat in info["features"]:
        p = feat["properties"]
        records.append({
            "label":      rng["label"],
            "year":       int(p["year"])  if p.get("year")  is not None else None,
            "month":      int(p["month"]) if p.get("month") is not None else None,
            "date_label": p.get("date_label", ""),
            "ndvi_mean":  p.get("ndvi_mean"),
            "ndvi_std":   p.get("ndvi_std"),
            "dw_mean":    p.get("dw_mean"),
            "dw_std":     p.get("dw_std"),
        })
    return records


# ── model fitting ─────────────────────────────────────────────────────────────
def _fit_all_models(csv_path: Path, model_dir: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_label[row["label"]].append(row)

    results: dict = {}
    n_ok = n_skip = 0

    for label, rows in sorted(by_label.items()):
        records = [{
            "date_label": r["date_label"],
            "ndvi_mean":  float(r["ndvi_mean"]) if r.get("ndvi_mean") else None,
            "dw_mean":    float(r["dw_mean"])   if r.get("dw_mean")   else None,
        } for r in rows]

        beat_result: dict = {"label": label}

        for var in ("ndvi", "dw_trees"):
            safe  = label.replace("/", "__").replace(" ", "_")
            out_p = model_dir / f"{safe}__{var}.json"

            try:
                model = PhenologyModel.fit(records, label=label, variable=var)
                model.save(out_p)
                beat_result[var] = {
                    "r2": model.r2, "sigma": model.residual_std,
                    "n_obs": model.n_obs, "model_file": str(out_p),
                }
                flag = "OK"
            except ValueError as e:
                beat_result[var] = {"error": str(e)}
                flag = "SKIP"

            r2  = beat_result[var].get("r2", 0)
            sig = beat_result[var].get("sigma", 0)
            err = beat_result[var].get("error", "")
            print(f"  {flag}  {label:<25s}  [{var}]"
                  + (f"  R2={r2:.3f}  sigma={sig:.4f}" if flag == "OK"
                     else f"  SKIP: {err}"))

        n_ok   += int("error" not in beat_result.get("ndvi", {}))
        n_skip += int("error" in beat_result.get("ndvi", {}))
        results[label] = beat_result

    print(f"\n  Fitted: {n_ok}  Skipped: {n_skip}")
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only",  action="store_true")
    ap.add_argument("--fit-only",      action="store_true")
    ap.add_argument("--limit",         type=int, default=0)
    ap.add_argument("--start-year",    type=int, default=2020)
    ap.add_argument("--end-year",      type=int, default=2025)
    ap.add_argument("--beats-geojson", default="data/aoi/guna_beats.geojson")
    ap.add_argument("--out-csv",       default="data/phenology/range_ndvi_series.csv")
    ap.add_argument("--model-dir",     default="data/phenology/range_models")
    args = ap.parse_args()

    beats_geojson = ROOT / args.beats_geojson
    out_csv       = ROOT / args.out_csv
    model_dir     = ROOT / args.model_dir
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg         = _load_config(ROOT / "config.yaml")
    gee_project = cfg.get("gee_project", "van-suraksha-alert")

    if not args.fit_only:
        print(f"[phenology] GEE project : {gee_project}")
        print("[phenology] Initialising GEE...")
        _init_gee(project=gee_project)

        ranges = _load_ranges(beats_geojson)
        if args.limit:
            ranges = ranges[:args.limit]

        print(f"[phenology] Extracting {len(ranges)} ranges  "
              f"({args.start_year}-{args.end_year})  "
              f"[SERVER-SIDE batched, 1 GEE call per range]\n")

        fieldnames = ["label","year","month","date_label",
                      "ndvi_mean","ndvi_std","dw_mean","dw_std"]

        existing: set = set()
        mode = "a" if out_csv.exists() else "w"
        if mode == "a":
            with open(out_csv, newline="") as f:
                existing = {row["label"] for row in csv.DictReader(f)}

        with open(out_csv, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()

            for i, rng in enumerate(ranges):
                if rng["label"] in existing:
                    print(f"  SKIP {rng['label']} (already done)")
                    continue

                t0 = time.time()
                n_months = (args.end_year - args.start_year + 1) * 12
                print(f"  [{i+1}/{len(ranges)}] {rng['label']:<20s}"
                      f"({rng['n_beats']} beats, {n_months} months)...",
                      end=" ", flush=True)
                try:
                    records = _extract_range_series_batched(
                        rng, args.start_year, args.end_year)
                    writer.writerows(records)
                    f.flush()
                    n_valid = sum(1 for r in records if r["ndvi_mean"] is not None)
                    print(f"[OK]  {n_valid}/{len(records)} valid  "
                          f"({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"[ERR] {e}")

        print(f"\n  Saved -> {out_csv}")

    if not args.extract_only:
        print(f"\n[phenology] Fitting harmonic models from {out_csv.name}...")
        summary = _fit_all_models(out_csv, model_dir)
        out_s = model_dir / "summary.json"
        with open(out_s, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary -> {out_s}")


if __name__ == "__main__":
    main()
