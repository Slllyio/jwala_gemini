import json, sys
with open("outputs/alert_filter/stacked_v2/stacked_confirmed_alerts.geojson") as f:
    data = json.load(f)
features = data["features"]
print(f"Total: {len(features)} polygons\n")
hdr = f"{'#':>4} {'Change Type':30} {'Score':>6} {'Fires':>5} {'MaxP':>5} {'Area(ha)':>9} {'Pixels':>6} {'FT':>3} {'Centroid (lat,lon)':>22}"
print(hdr)
print("-" * len(hdr))
for i, feat in enumerate(features, 1):
    p = feat["properties"]
    ct = p.get("change_type", "?")
    sc = p.get("stacked_score", 0)
    fi = p.get("mean_fires", 0)
    mp = p.get("max_single_P", 0)
    ah = p.get("area_ha", 0)
    npx = p.get("n_pixels", 0)
    ft = "Y" if p.get("fast_tracked", False) else ""
    c = p.get("centroid", [0, 0])
    print(f"{i:>4} {ct:30} {sc:>6.3f} {fi:>5.1f} {mp:>5.3f} {ah:>9.4f} {npx:>6} {ft:>3} {c[0]:>10.5f},{c[1]:>10.5f}")
