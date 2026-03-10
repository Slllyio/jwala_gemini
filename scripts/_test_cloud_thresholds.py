"""
Quick test: what dates have clean imagery for a specific beat
at different cloud thresholds? Uses GEE directly.
"""
import sys
sys.path.insert(0, "scripts")

import ee
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

# Init GEE
ee.Initialize(project="van-suraksha-alert",
              opt_url="https://earthengine-highvolume.googleapis.com")
print("GEE OK")

# Load a few test beats
beats_path = Path("data/aoi/guna_beats.geojson")
fc = json.loads(beats_path.read_text(encoding="utf-8"))

# Pick 5 beats from different ranges
test_beats = {}
for feat in fc["features"]:
    p = feat.get("properties", {})
    rng, beat = p.get("Range"), p.get("Beat")
    if rng and beat and rng not in test_beats:
        test_beats[rng] = (beat, feat)
    if len(test_beats) >= 6:
        break

print(f"\nTest beats: {[(r, b) for r, (b, _) in test_beats.items()]}\n")

anchor = "2026-02-24"
end    = ee.Date(anchor).advance(1, "day")
start  = end.advance(-60, "day")   # 60-day window

for rng, (beat_name, feat) in test_beats.items():
    geom_type = feat["geometry"]["type"]
    coords    = feat["geometry"]["coordinates"]
    if geom_type == "Polygon":
        beat_geom = ee.Geometry.Polygon(coords)
    elif geom_type == "MultiPolygon":
        beat_geom = ee.Geometry.MultiPolygon(coords)
    else:
        continue

    dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(beat_geom)
              .filterDate(start, end))
    s2_cp  = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterBounds(beat_geom)
              .filterDate(start, end))

    def tag_cloud(img):
        date_str = img.date().format("YYYY-MM-dd")
        cp_col_day = s2_cp.filterDate(img.date(), img.date().advance(1, "day"))
        cp_img = ee.Image(ee.Algorithms.If(
            cp_col_day.size().gt(0),
            cp_col_day.mosaic().select("probability"),
            ee.Image.constant(100).rename("probability"),
        ))
        mean_cp = cp_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=beat_geom,
            scale=100, maxPixels=1e6
        ).get("probability")
        return ee.Feature(None, {"date": date_str, "cloud_pct": mean_cp})

    try:
        tagged = dw_col.map(tag_cloud)
        results = tagged.aggregate_array("cloud_pct").zip(
                  tagged.aggregate_array("date")).getInfo()

        # Sort by date
        pairs = sorted([(d, c) for c, d in results if c is not None])
        # Find dates below each threshold
        for thresh in [20, 30, 40, 50]:
            clean = [d for d, c in pairs if c <= thresh]
            print(f"  {beat_name:20s} ({rng:15s}) | cloud ≤{thresh:2d}%: "
                  f"{len(clean)} dates  [{', '.join(clean[:4])}]")
        print()
    except Exception as e:
        print(f"  {beat_name}: ERROR — {e}\n")
