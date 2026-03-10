"""
10-beat DW cloud audit — February 2026.
Uses COPERNICUS/S2_SR_HARMONIZED SCL band (classes 8,9,10=cloud) from Sentinel-2
for accurate, temporally-aligned cloud measurements per DW pass.
"""
import sys, json, random
sys.path.insert(0, "scripts")

import ee
from pathlib import Path

ee.Initialize(project="van-suraksha-alert",
              opt_url="https://earthengine-highvolume.googleapis.com")
print("GEE OK — using S2_SR_HARMONIZED SCL for cloud detection\n")

beats_path = Path("data/aoi/guna_beats.geojson")
fc         = json.loads(beats_path.read_text(encoding="utf-8"))
all_feats  = [f for f in fc["features"] if f.get("properties", {}).get("Beat")]

random.seed(42)
sample = random.sample(all_feats, 10)

WIN_START = "2026-02-01"
WIN_END   = "2026-03-01"

# SCL cloud classes: 8=cloud medium, 9=cloud high, 10=thin cirrus
# Also useful: 3=cloud shadow
SCL_CLOUD_LOW  = 8
SCL_CLOUD_HIGH = 10

print(f"{'Beat':<22} {'Range':<18} {'Date':<12} {'SCL_cloud%':>11}  Status")
print("-" * 75)

for feat in sample:
    p         = feat["properties"]
    beat_name = p.get("Beat", "?")
    rng       = p.get("Range", "?")
    g         = feat["geometry"]

    if g["type"] == "Polygon":
        geom = ee.Geometry.Polygon(g["coordinates"])
    elif g["type"] == "MultiPolygon":
        geom = ee.Geometry.MultiPolygon(g["coordinates"])
    else:
        continue

    # DW collection — anchors the dates we care about
    dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(geom)
              .filterDate(WIN_START, WIN_END))

    # S2 SR with SCL — for cloud classification
    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(geom)
              .filterDate(WIN_START, WIN_END)
              .select("SCL"))

    def tag(dw_img):
        date_str = dw_img.date().format("YYYY-MM-dd")
        # S2 scenes on the same calendar day (±1 day for tile-edge timing diff)
        s2_day = s2_col.filterDate(
            dw_img.date().advance(-1, "day"),   # -1 day safety margin
            dw_img.date().advance(2,  "day"),   # +1 day safety margin
        )
        s2_scl = ee.Image(ee.Algorithms.If(
            s2_day.size().gt(0),
            s2_day.mosaic().select("SCL"),
            ee.Image.constant(9).rename("SCL"),   # no S2 on this day → assume cloudy
        ))
        # Fraction of pixels with SCL 8–10 (cloud classes)
        cloud_frac = (s2_scl.gte(SCL_CLOUD_LOW)
                            .And(s2_scl.lte(SCL_CLOUD_HIGH))
                            .reduceRegion(
                                reducer  = ee.Reducer.mean(),
                                geometry = geom,
                                scale    = 100,
                                maxPixels= 1e6,
                            ).get("SCL"))
        return ee.Feature(None, {"date": date_str, "cloud_pct": cloud_frac})

    try:
        tagged = dw_col.map(tag).filter(ee.Filter.notNull(["cloud_pct"]))
        dates  = tagged.aggregate_array("date").getInfo()
        clouds = tagged.aggregate_array("cloud_pct").getInfo()

        seen = {}
        for d, c in zip(dates, clouds):
            if d not in seen:
                seen[d] = c

        if not seen:
            print(f"{beat_name:<22} {rng:<18}  —  NO DW IMAGERY IN FEB")
            continue

        for d in sorted(seen):
            c     = seen[d]
            c_pct = (c or 0) * 100
            if   c_pct <= 10: status = "✓✓ EXCELLENT"
            elif c_pct <= 30: status = "✓  USABLE"
            elif c_pct <= 60: status = "~  MARGINAL"
            else:             status = "✗  CLOUDY"
            print(f"{beat_name:<22} {rng:<18} {d:<12} {c_pct:>10.1f}%  {status}")
        print()

    except Exception as e:
        print(f"{beat_name:<22} {rng:<18} ERROR: {e}\n")
