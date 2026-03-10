"""Quick breakdown of alerts by change type."""
import json, sys, os

geojson_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "models", "HAMEERPUR", "alerts_HAMEERPUR.geojson"
)

with open(geojson_path) as f:
    gj = json.load(f)

features = gj["features"]
print(f"Total polygons: {len(features)}\n")

by_type = {}
for feat in features:
    ct = feat["properties"]["change_type"]
    if ct not in by_type:
        by_type[ct] = {"count": 0, "area_ha": 0.0, "pixels": 0, "scores": [], "fires": []}
    by_type[ct]["count"] += 1
    by_type[ct]["area_ha"] += feat["properties"]["area_ha"]
    by_type[ct]["pixels"] += feat["properties"]["n_pixels"]
    by_type[ct]["scores"].append(feat["properties"]["stacked_score"])
    by_type[ct]["fires"].append(feat["properties"]["mean_fires"])

header = f"{'Change Type':<28s} {'Count':>6s} {'Area (ha)':>10s} {'Pixels':>9s} {'Avg Score':>10s} {'Avg Fires':>10s}"
print(header)
print("-" * len(header))

total_area = 0.0
total_pixels = 0
for ct in sorted(by_type, key=lambda x: -by_type[x]["area_ha"]):
    d = by_type[ct]
    avg_s = sum(d["scores"]) / len(d["scores"])
    avg_f = sum(d["fires"]) / len(d["fires"])
    total_area += d["area_ha"]
    total_pixels += d["pixels"]
    print(f"{ct:<28s} {d['count']:>6d} {d['area_ha']:>10.2f} {d['pixels']:>9,d} {avg_s:>10.3f} {avg_f:>10.1f}")

print("-" * len(header))
print(f"{'TOTAL':<28s} {len(features):>6d} {total_area:>10.2f} {total_pixels:>9,d}")

# Percentage breakdown
print(f"\n{'='*50}")
print("PERCENTAGE BREAKDOWN (by area)")
print(f"{'='*50}")
for ct in sorted(by_type, key=lambda x: -by_type[x]["area_ha"]):
    d = by_type[ct]
    pct = 100 * d["area_ha"] / total_area if total_area > 0 else 0
    bar = "█" * int(pct / 2)
    print(f"  {ct:<28s} {pct:5.1f}%  {bar}")
