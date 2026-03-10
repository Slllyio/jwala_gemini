"""
_export_10yr_vsn.py  --  Virtual Sensor Network: Local-First GEE Extractor
============================================================================
Architecture (double-batched to stay within GEE compute limits):
  1. Generate VSN point coordinates LOCALLY (rejection sampling in beat polygons)
  2. For each YEAR x SPATIAL_BATCH:
     - Filter DW collection to that year + bounding box
     - Call getRegion() with ~50 points -> returns flat table
  3. All pairing, cloud filtering, and scoring done LOCALLY with pandas

Key constraints solved:
  - Spatial Washout: 10m pixel sampling, NOT beat-level means
  - API Death Spiral: ~140 small getRegion() calls, not one massive export
  - Computation graph: tiny per call (~50 pts x 1 year of DW = fast)

Usage:
    python scripts/_export_10yr_vsn.py
    python scripts/_export_10yr_vsn.py --points-per-beat 5 --batch-size 50
"""
import sys, pathlib, argparse, time, json, random, math, os
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

parser = argparse.ArgumentParser(description="Extract 10-year DW pixel history")
parser.add_argument("--points-per-beat", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=50,
                    help="Points per getRegion call (default: 50)")
parser.add_argument("--start-year", type=int, default=2016)
parser.add_argument("--end-year", type=int, default=2026)
parser.add_argument("--output", default="")
parser.add_argument("--resume", action="store_true",
                    help="Resume from existing partial output")
args = parser.parse_args()

OUT_DIR = ROOT / "outputs" / "simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = pathlib.Path(args.output) if args.output else OUT_DIR / "Guna_10Yr_Pixel_History.csv"

# ── GEE Init ────────────────────────────────────────────────────────────────
try:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gee_project = cfg.get("gee", {}).get("gee_project", "van-suraksha-alert")
except Exception:
    gee_project = "van-suraksha-alert"

import ee
ee.Initialize(project=gee_project, opt_url="https://earthengine-highvolume.googleapis.com")
print("[OK] GEE initialized (%s)" % gee_project)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Generate VSN Points LOCALLY
# ═══════════════════════════════════════════════════════════════════════════════
BEATS_PATH = ROOT / "data" / "aoi" / "guna_beats.geojson"
with open(BEATS_PATH, "r", encoding="utf-8") as f:
    beats_gj = json.load(f)

PPB = args.points_per_beat

def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def random_points_in_polygon(coords, n, seed=42):
    rng = random.Random(seed)
    exterior = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
    lons = [p[0] for p in exterior]
    lats = [p[1] for p in exterior]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    points = []
    for _ in range(n * 500):
        if len(points) >= n:
            break
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        if point_in_polygon(lon, lat, exterior):
            points.append((lon, lat))
    return points

print("[..] Generating %d pts/beat from %d beats..." % (PPB, len(beats_gj["features"])))
vsn_rows = []
for feat in beats_gj["features"]:
    props = feat["properties"]
    beat_name = props.get("Beat", props.get("BEAT", "unknown"))
    range_name = props.get("Range", props.get("RANGE", "unknown"))
    geom = feat["geometry"]
    coords = geom["coordinates"]

    all_pts = []
    if geom["type"] == "Polygon":
        all_pts = random_points_in_polygon(coords, PPB)
    elif geom["type"] == "MultiPolygon":
        per_sub = max(1, PPB // len(coords))
        rem = PPB - per_sub * len(coords)
        for i, sub in enumerate(coords):
            n = per_sub + (1 if i < rem else 0)
            all_pts.extend(random_points_in_polygon(sub, n, seed=42 + i))
        all_pts = all_pts[:PPB]

    for j, (lon, lat) in enumerate(all_pts):
        vsn_rows.append({
            "point_id": "%s_pt%d" % (beat_name, j),
            "beat_name": beat_name,
            "range_name": range_name,
            "lon": round(lon, 6),
            "lat": round(lat, 6),
        })

pts_df = pd.DataFrame(vsn_rows)
print("[OK] %d VSN points, %d beats" % (len(pts_df), pts_df["beat_name"].nunique()))

# Save point locations for reference
pts_path = OUT_DIR / "vsn_points.csv"
pts_df.to_csv(pts_path, index=False)
print("[OK] Point coords saved: %s" % pts_path)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Fetch DW time series -- double-batched (year x spatial_batch)
# ═══════════════════════════════════════════════════════════════════════════════
BATCH = args.batch_size
n_spatial_batches = math.ceil(len(pts_df) / BATCH)
years = list(range(args.start_year, args.end_year))
total_calls = n_spatial_batches * len(years)

# Resume support: track completed (year, batch_idx) pairs
done_file = OUT_DIR / "_vsn_progress.json"
done_pairs = set()
if args.resume and done_file.exists():
    with open(done_file) as f:
        done_pairs = set(tuple(x) for x in json.load(f))
    print("[RESUME] %d of %d year-batch pairs already done" % (len(done_pairs), total_calls))

print("\n[..] %d spatial batches x %d years = %d total getRegion calls" % (
    n_spatial_batches, len(years), total_calls - len(done_pairs)))
print("     Expect ~15-30s per call, ~%.0f min total\n" % (
    (total_calls - len(done_pairs)) * 20 / 60))

call_count = 0
fail_count = 0
row_count = 0
t_global = time.time()

# Open output file in append mode
write_header = not (args.resume and OUT_PATH.exists())
out_cols = ["point_id", "beat_name", "range_name",
            "date", "year", "month", "day", "doy",
            "trees", "crops", "built"]

if write_header:
    with open(OUT_PATH, "w") as f:
        f.write(",".join(out_cols) + "\n")

for yr in years:
    yr_start = "%d-01-01" % yr
    yr_end = "%d-01-01" % (yr + 1)

    # Build DW collection for this year ONCE (lazy)
    dw_yr = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(yr_start, yr_end)
        .select(["trees", "crops", "built"])
    )

    for bi in range(n_spatial_batches):
        pair_key = (yr, bi)
        if pair_key in done_pairs:
            continue

        batch_start = bi * BATCH
        batch_end = min(batch_start + BATCH, len(pts_df))
        batch_pts = pts_df.iloc[batch_start:batch_end]
        call_count += 1

        label = "Y%d B%d/%d (%d pts)" % (yr, bi + 1, n_spatial_batches, len(batch_pts))
        print("  [%s] %s ..." % (time.strftime("%H:%M:%S"), label), end=" ", flush=True)

        # Build ee point collection for this batch
        ee_pts = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r["lon"], r["lat"]]))
            for _, r in batch_pts.iterrows()
        ])

        # Spatial filter: only DW tiles overlapping this batch
        lons, lats = batch_pts["lon"], batch_pts["lat"]
        bbox = ee.Geometry.Rectangle([
            lons.min() - 0.05, lats.min() - 0.05,
            lons.max() + 0.05, lats.max() + 0.05,
        ])
        dw_local = dw_yr.filterBounds(bbox)

        # Fetch with retries
        t0 = time.time()
        region_data = None
        for attempt in range(4):
            try:
                region_data = dw_local.getRegion(ee_pts, scale=10).getInfo()
                break
            except Exception as e:
                err = str(e)[:120]
                if attempt < 3:
                    wait = min(60, 5 * (2 ** (attempt + 1)))
                    print("retry(%d,%ds)..." % (attempt + 1, wait), end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print("FAIL: %s" % err)
                    fail_count += 1
                    region_data = None

        if region_data is None or len(region_data) < 2:
            print("(skip)")
            continue

        # Parse response
        header = region_data[0]
        try:
            ci = {h: header.index(h) for h in ["longitude", "latitude", "time", "trees", "crops", "built"]}
        except ValueError:
            print("(bad header)")
            continue

        # Build lookup: (round(lon,4), round(lat,4)) -> point_id, beat_name, range_name
        pt_lookup = {}
        for _, r in batch_pts.iterrows():
            key = (round(r["lon"], 4), round(r["lat"], 4))
            pt_lookup[key] = (r["point_id"], r["beat_name"], r["range_name"])

        # Also build KD-tree for fuzzy matching
        pt_coords = batch_pts[["lon", "lat"]].values
        pt_ids = batch_pts[["point_id", "beat_name", "range_name"]].values

        parsed_rows = []
        for row in region_data[1:]:
            trees_val = row[ci["trees"]]
            if trees_val is None:
                continue

            gee_lon = row[ci["longitude"]]
            gee_lat = row[ci["latitude"]]
            time_ms = row[ci["time"]]

            # Match to point_id
            key = (round(gee_lon, 4), round(gee_lat, 4))
            match = pt_lookup.get(key)
            if match is None:
                # Nearest neighbor fallback
                dists = np.sqrt((pt_coords[:, 0] - gee_lon) ** 2 + (pt_coords[:, 1] - gee_lat) ** 2)
                idx = np.argmin(dists)
                if dists[idx] < 0.01:  # ~1km tolerance
                    match = tuple(pt_ids[idx])
                    pt_lookup[key] = match  # cache
                else:
                    continue

            dt = pd.Timestamp(time_ms, unit="ms")
            parsed_rows.append({
                "point_id": match[0],
                "beat_name": match[1],
                "range_name": match[2],
                "date": dt.strftime("%Y-%m-%d"),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "doy": dt.dayofyear,
                "trees": trees_val,
                "crops": row[ci["crops"]],
                "built": row[ci["built"]],
            })

        if parsed_rows:
            chunk_df = pd.DataFrame(parsed_rows)
            chunk_df[out_cols].to_csv(OUT_PATH, mode="a", header=False, index=False)
            row_count += len(parsed_rows)

        elapsed = time.time() - t0
        n_rows = len(parsed_rows)
        print("%d rows (%.0fs)" % (n_rows, elapsed))

        # Mark done for resume
        done_pairs.add(pair_key)
        with open(done_file, "w") as f:
            json.dump([list(x) for x in done_pairs], f)

        # Rate limit
        time.sleep(0.5)

# ═══════════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t_global
print("\n" + "=" * 60)
print("  VSN EXTRACTION COMPLETE")
print("  %d rows fetched | %d calls | %d failures | %.1f min" % (
    row_count, call_count, fail_count, total_time / 60))
print("  Output: %s" % OUT_PATH)
print("=" * 60)
print("\n  Next: python scripts/_score_10yr_vsn.py")
