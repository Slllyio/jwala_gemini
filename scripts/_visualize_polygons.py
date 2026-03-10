"""
Visualize filtered polygons on a map, colored by change type, with areas.
Reads the stacked GeoJSON and produces a plot.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

CHANGE_COLORS = {
    "Greening":              "#2ecc71",
    "Degradation":           "#e74c3c",
    "Encroachment":          "#e67e22",
    "Other change":          "#95a5a6",
    "Clearing":              "#c0392b",
    "Built expansion":       "#8e44ad",
    "Tree loss (unclassified)": "#d35400",
}

def main():
    geojson_path = "outputs/filtered/alerts_HAMEERPUR.geojson"
    if not os.path.exists(geojson_path):
        geojson_path = "outputs/alert_filter/stacked_v2/stacked_confirmed_alerts.geojson"

    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"]
    print(f"Loaded {len(features)} polygons from {geojson_path}\n")

    # ── Area summary table ────────────────────────────────────────────
    print(f"{'#':>4} {'Change Type':30} {'Area (ha)':>10} {'Pixels':>7} {'Score':>7}")
    print("-"*65)
    
    type_areas = {}
    all_areas = []
    for i, feat in enumerate(features, 1):
        p = feat["properties"]
        ct = p.get("change_type", "?")
        ah = p.get("area_ha", 0)
        np_ = p.get("n_pixels", 0)
        sc = p.get("stacked_score", p.get("confidence", 0))
        all_areas.append(ah)
        type_areas.setdefault(ct, []).append(ah)
        if i <= 30 or ah > 10:  # show first 30 + all large ones
            print(f"{i:>4} {ct:30} {ah:>10.4f} {np_:>7} {sc:>7.3f}")

    if len(features) > 30:
        print(f"  ... ({len(features)-30} more polygons)")

    print(f"\n{'='*65}")
    print(f"{'Change Type':30} {'Count':>6} {'Total ha':>10} {'Mean ha':>10} {'Max ha':>10}")
    print("-"*65)
    grand_total = 0
    for ct in sorted(type_areas.keys(), key=lambda x: -sum(type_areas[x])):
        areas = type_areas[ct]
        total = sum(areas)
        grand_total += total
        print(f"{ct:30} {len(areas):>6} {total:>10.2f} {np.mean(areas):>10.4f} {max(areas):>10.4f}")
    print("-"*65)
    print(f"{'TOTAL':30} {len(features):>6} {grand_total:>10.2f}")

    # ── Figure 1: Map of polygons colored by change type ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("HAMEERPUR Alert Filter Polygons", fontsize=16, fontweight="bold")

    # Collect all coordinates for bounds
    all_lons, all_lats = [], []
    patches_by_type = {}

    for feat in features:
        geom = feat["geometry"]
        ct = feat["properties"].get("change_type", "Other change")
        coords_list = []

        if geom["type"] == "Polygon":
            coords_list = [geom["coordinates"][0]]  # outer ring only
        elif geom["type"] == "MultiPolygon":
            for part in geom["coordinates"]:
                coords_list.append(part[0])

        for coords in coords_list:
            xy = np.array(coords)
            all_lons.extend(xy[:, 0])
            all_lats.extend(xy[:, 1])
            patch = MplPolygon(xy, closed=True)
            patches_by_type.setdefault(ct, []).append(patch)

    # Plot map
    ax = axes[0]
    ax.set_title("Polygons by Change Type", fontsize=13)
    for ct, patches in patches_by_type.items():
        color = CHANGE_COLORS.get(ct, "#7f8c8d")
        pc = PatchCollection(patches, facecolor=color, edgecolor="black",
                             linewidth=0.3, alpha=0.7, label=ct)
        ax.add_collection(pc)

    margin = 0.002
    ax.set_xlim(min(all_lons)-margin, max(all_lons)+margin)
    ax.set_ylim(min(all_lats)-margin, max(all_lats)+margin)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Figure 2: Area distribution ──────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Area Distribution by Change Type", fontsize=13)

    # Stacked histogram
    bins = np.logspace(np.log10(0.01), np.log10(max(all_areas)*1.1), 30)
    hist_data = []
    hist_labels = []
    hist_colors = []
    for ct in sorted(type_areas.keys(), key=lambda x: -sum(type_areas[x])):
        hist_data.append(type_areas[ct])
        hist_labels.append(ct)
        hist_colors.append(CHANGE_COLORS.get(ct, "#7f8c8d"))

    ax2.hist(hist_data, bins=bins, stacked=True, color=hist_colors,
             label=hist_labels, edgecolor="white", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Area (ha)")
    ax2.set_ylabel("Number of polygons")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = "outputs/filtered/polygon_visualization.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nMap saved: {out_path}")

    # ── Figure 3: Top 20 largest polygons bar chart ──────────────────
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    sorted_feats = sorted(features, key=lambda f: -f["properties"].get("area_ha", 0))
    top20 = sorted_feats[:20]

    labels = []
    areas_top = []
    colors_top = []
    for i, feat in enumerate(top20):
        p = feat["properties"]
        ct = p.get("change_type", "?")
        ah = p.get("area_ha", 0)
        labels.append(f"#{i+1}\n{ct[:12]}\n{ah:.1f}ha")
        areas_top.append(ah)
        colors_top.append(CHANGE_COLORS.get(ct, "#7f8c8d"))

    bars = ax3.bar(range(len(top20)), areas_top, color=colors_top,
                   edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(len(top20)))
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylabel("Area (hectares)")
    ax3.set_title("Top 20 Largest Alert Polygons", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    out2 = "outputs/filtered/top20_polygons.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Top 20 chart saved: {out2}")

    plt.close("all")

if __name__ == "__main__":
    main()
