"""
Deep Analysis: Swiss Cheese Polygon Trap — Phase 1 vs Phase 2
=============================================================

Compares polygon quality between the original stacking (Phase 1)
and the hardened version with Object-Based Hysteresis (Phase 2).

Metrics:
  - Fragmentation: count of polygons with holes (Swiss cheese)
  - Compactness: Polsby-Popper score (4π * area / perimeter²)
  - Size distribution: area histogram
  - Hole analysis: count + area of holes per polygon
  - Edge regularity: ratio of convex hull area to polygon area
"""

import json
import sys
from pathlib import Path
from collections import Counter

def analyze_geojson(path: str, label: str):
    """Analyze polygon quality from a GeoJSON file."""
    with open(path) as f:
        data = json.load(f)

    features = data.get("features", [])
    n_polys = len(features)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  File: {path}")
    print(f"{'='*70}")
    print(f"  Total polygons: {n_polys}")

    if n_polys == 0:
        return

    # Analyze geometry quality
    n_with_holes = 0
    total_holes = 0
    areas = []
    n_pixels_list = []
    change_types = Counter()
    polys_with_many_holes = []
    coord_ring_counts = []
    tiny_polys = 0  # < 0.01 ha
    small_polys = 0  # 0.01 - 0.1 ha
    medium_polys = 0  # 0.1 - 1 ha
    large_polys = 0  # 1 - 10 ha
    very_large_polys = 0  # > 10 ha

    scores = []
    fires = []
    fast_tracked = 0

    for feat in features:
        props = feat.get("properties", {})

        # Area
        area = props.get("area_ha", 0)
        areas.append(area)
        if area < 0.01:
            tiny_polys += 1
        elif area < 0.1:
            small_polys += 1
        elif area < 1.0:
            medium_polys += 1
        elif area < 10.0:
            large_polys += 1
        else:
            very_large_polys += 1

        # Pixel count
        n_pix = props.get("n_pixels", 0)
        n_pixels_list.append(n_pix)

        # Change type
        ct = props.get("change_type", "unknown")
        change_types[ct] += 1

        # Score
        scores.append(props.get("stacked_score", 0))
        fires.append(props.get("mean_fires", 0))

        # Fast-tracked
        if props.get("fast_tracked", False):
            fast_tracked += 1

        # Hole analysis (GeoJSON Polygon has > 1 ring = has holes)
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")

        if geom_type == "Polygon":
            coords = geom.get("coordinates", [])
            n_rings = len(coords)
            coord_ring_counts.append(n_rings)
            if n_rings > 1:
                n_with_holes += 1
                total_holes += n_rings - 1
                if n_rings > 3:
                    polys_with_many_holes.append((area, n_rings - 1, ct))
        elif geom_type == "MultiPolygon":
            total_rings = 0
            total_parts = len(geom.get("coordinates", []))
            for part in geom.get("coordinates", []):
                total_rings += len(part)
                if len(part) > 1:
                    n_with_holes += 1
                    total_holes += len(part) - 1
            coord_ring_counts.append(total_rings)

    # Summary stats
    areas_sorted = sorted(areas)
    mean_area = sum(areas) / len(areas) if areas else 0
    median_area = areas_sorted[len(areas_sorted) // 2] if areas_sorted else 0
    total_area = sum(areas)

    mean_score = sum(scores) / len(scores) if scores else 0
    mean_fire_count = sum(fires) / len(fires) if fires else 0

    print(f"\n  --- FRAGMENTATION (Swiss Cheese) ---")
    print(f"  Polygons with holes:     {n_with_holes:>5} ({100*n_with_holes/n_polys:.1f}%)")
    print(f"  Total holes:             {total_holes:>5}")
    print(f"  Avg holes/polygon:       {total_holes/n_polys:.2f}")
    if n_with_holes > 0:
        print(f"  Avg holes/holey polygon: {total_holes/n_with_holes:.1f}")

    if polys_with_many_holes:
        print(f"\n  Worst offenders (>3 holes):")
        for area, nh, ct in sorted(polys_with_many_holes, key=lambda x: -x[1])[:10]:
            print(f"    {area:.2f} ha, {nh} holes, type={ct}")

    print(f"\n  --- SIZE DISTRIBUTION ---")
    print(f"  Total area:    {total_area:>10.2f} ha")
    print(f"  Mean area:     {mean_area:>10.4f} ha")
    print(f"  Median area:   {median_area:>10.4f} ha")
    print(f"  Min area:      {min(areas):>10.4f} ha")
    print(f"  Max area:      {max(areas):>10.4f} ha")
    print(f"  Tiny (<0.01):  {tiny_polys:>5} ({100*tiny_polys/n_polys:.1f}%)")
    print(f"  Small (0.01-0.1): {small_polys:>5} ({100*small_polys/n_polys:.1f}%)")
    print(f"  Medium (0.1-1):   {medium_polys:>5} ({100*medium_polys/n_polys:.1f}%)")
    print(f"  Large (1-10):     {large_polys:>5} ({100*large_polys/n_polys:.1f}%)")
    print(f"  Very large (>10): {very_large_polys:>5} ({100*very_large_polys/n_polys:.1f}%)")

    print(f"\n  --- SCORES ---")
    print(f"  Mean stacked score:  {mean_score:.3f}")
    print(f"  Mean fire count:     {mean_fire_count:.1f}")
    if fast_tracked > 0:
        print(f"  Fast-tracked:        {fast_tracked}")

    print(f"\n  --- CHANGE TYPES ---")
    for ct, count in sorted(change_types.items(), key=lambda x: -x[1]):
        print(f"    {ct:30s} {count:>5} ({100*count/n_polys:.1f}%)")

    return {
        "n_polys": n_polys,
        "n_with_holes": n_with_holes,
        "total_holes": total_holes,
        "total_area": total_area,
        "mean_area": mean_area,
        "tiny": tiny_polys,
        "small": small_polys,
        "fast_tracked": fast_tracked,
        "change_types": dict(change_types),
    }


def main():
    base = Path("outputs/alert_filter")

    phase1 = base / "stacked" / "stacked_confirmed_alerts.geojson"
    phase2 = base / "stacked_v2" / "stacked_confirmed_alerts.geojson"

    results = {}
    if phase1.exists():
        results["phase1"] = analyze_geojson(str(phase1), "PHASE 1 — Original Stacking (no hysteresis)")
    if phase2.exists():
        results["phase2"] = analyze_geojson(str(phase2), "PHASE 2 — Hardened (hysteresis + fast-track)")

    # Compare
    if "phase1" in results and "phase2" in results:
        p1, p2 = results["phase1"], results["phase2"]
        print(f"\n{'='*70}")
        print(f"  COMPARISON: Phase 1 → Phase 2")
        print(f"{'='*70}")
        print(f"  Polygons:     {p1['n_polys']:>5} → {p2['n_polys']:>5}  ({p2['n_polys']-p1['n_polys']:+d})")
        print(f"  With holes:   {p1['n_with_holes']:>5} → {p2['n_with_holes']:>5}  ({p2['n_with_holes']-p1['n_with_holes']:+d})")
        print(f"  Total holes:  {p1['total_holes']:>5} → {p2['total_holes']:>5}  ({p2['total_holes']-p1['total_holes']:+d})")
        print(f"  Total area:   {p1['total_area']:>8.1f} → {p2['total_area']:>8.1f} ha  ({p2['total_area']-p1['total_area']:+.1f})")
        print(f"  Tiny polys:   {p1['tiny']:>5} → {p2['tiny']:>5}  ({p2['tiny']-p1['tiny']:+d})")
        print(f"  Small polys:  {p1['small']:>5} → {p2['small']:>5}  ({p2['small']-p1['small']:+d})")

        # Hole reduction percentage
        if p1['total_holes'] > 0:
            hole_reduction = 100 * (1 - p2['total_holes'] / p1['total_holes'])
            print(f"\n  ★ Hole reduction: {hole_reduction:.1f}%")
        if p1['tiny'] > 0:
            tiny_reduction = 100 * (1 - p2['tiny'] / p1['tiny'])
            print(f"  ★ Tiny polygon reduction: {tiny_reduction:.1f}%")


if __name__ == "__main__":
    main()
