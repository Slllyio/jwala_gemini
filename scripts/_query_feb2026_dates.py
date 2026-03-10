"""
List every S2 and DW acquisition date over Mar Ki Mahu in Feb 2026.
Shows cloudiness and whether it passes the 25% cloud gate.
"""
import sys, json, yaml
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

with open(ROOT / "data/aoi/guna_beats.geojson") as f:
    fc = json.load(f)

beat_feat = next(
    ft for ft in fc["features"]
    if ft["properties"].get("Beat", "").lower() == "mar_ki_mahu"
)
aoi = ee.Geometry(beat_feat["geometry"])

START  = "2026-02-01"
END    = "2026-03-01"   # exclusive end
THRESH = 25             # % cloud gate

# -- S2 SR Harmonized ----------------------------------------------------------
s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(START, END)
            .sort("system:time_start"))

# Fetch timestamps and cloud% as separate arrays
s2_ts    = s2_col.aggregate_array("system:time_start").getInfo()
s2_cloud = s2_col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
s2_idx   = s2_col.aggregate_array("system:index").getInfo()

s2_rows = []
for ts, cloud, idx in zip(s2_ts, s2_cloud, s2_idx):
    date  = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    cloud = round(cloud, 1) if cloud is not None else -1.0
    # granule tile identifier (last 6 chars of index, e.g. 43RGQ)
    parts = str(idx).split("_")
    tile  = parts[-1] if parts else idx
    s2_rows.append({"date": date, "cloud": cloud, "tile": tile})

# Deduplicate same date, same tile — take lowest cloud per date×tile
seen = {}
for r in s2_rows:
    key = (r["date"], r["tile"])
    if key not in seen or r["cloud"] < seen[key]["cloud"]:
        seen[key] = r

s2_rows = sorted(seen.values(), key=lambda x: (x["date"], x["cloud"]))

# -- Dynamic World -------------------------------------------------------------
dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(aoi)
            .filterDate(START, END)
            .sort("system:time_start"))

dw_ts  = dw_col.aggregate_array("system:time_start").getInfo()
dw_idx = dw_col.aggregate_array("system:index").getInfo()

dw_rows = {}
for ts, idx in zip(dw_ts, dw_idx):
    date  = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    parts = str(idx).split("_")
    tile  = parts[-1] if parts else idx
    dw_rows.setdefault(date, set()).add(tile)

# -- Print table ---------------------------------------------------------------
print(f"\nS2 + DW availability — Mar Ki Mahu — Feb 2026")
print(f"Cloud gate threshold: < {THRESH}%\n")
print(f"{'Date':<12} {'Tile':<8} {'Cloud%':>7}  {'Gate':>6}  {'DW avail':>9}")
print("-" * 56)

all_cloud_ok = []
for r in s2_rows:
    date  = r["date"]
    cloud = r["cloud"]
    tile  = r["tile"]
    gate  = "v PASS" if cloud < THRESH else "x FAIL"
    dw_ok = "YES" if date in dw_rows else " no"
    print(f"{date:<12} {tile:<8} {cloud:>7.1f}%  {gate}  {dw_ok:>9}")
    if cloud < THRESH and date in dw_rows:
        all_cloud_ok.append(date)

print("-" * 56)
uniq_dates  = sorted({r["date"] for r in s2_rows})
pass_dates  = sorted({r["date"] for r in s2_rows if r["cloud"] < THRESH})
dw_dates    = sorted(dw_rows.keys())
clean_pairs = sorted(set(pass_dates) & set(dw_dates))

print(f"\nSummary")
print(f"  S2 unique dates         : {len(uniq_dates)}")
print(f"  S2 under {THRESH}% cloud gate : {len(pass_dates)}")
print(f"  DW dates available      : {len(dw_dates)}")
print(f"  Valid DW+S2 cloud-clean : {len(clean_pairs)}")
print(f"\n  Clean pairs for DW instant delta:")
for i, d in enumerate(clean_pairs):
    mark = " <- use as anchor"
    if   i == 0: mark = " <- oldest usable"
    elif i == len(clean_pairs)-1: mark = " <- most recent"
    print(f"    {i+1}. {d}{mark}")
