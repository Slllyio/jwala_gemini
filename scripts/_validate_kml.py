"""
Spatial validation: check if the field-verified felling KML (P472, अवैध कटाई)
aligns with any of our Mar Ki Mahu alert patches (Feb 09-22).

Run: python scripts/_validate_kml.py
"""
import json, math, re, sys
from pathlib import Path

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── 1. Parse KML polygon (no external libs needed) ─────────────────────────
KML_FILE = Path(r"C:\Users\S.C.C\Downloads\Export 2026-02-23 1656.kml")
raw = KML_FILE.read_text(encoding="utf-8")
coords_str = re.search(r"<coordinates>(.*?)</coordinates>", raw, re.DOTALL).group(1)
kml_ring = []
for tok in coords_str.split():
    tok = tok.strip()
    if tok:
        parts = tok.split(",")
        lon, lat = float(parts[0]), float(parts[1])
        kml_ring.append((lon, lat))

def poly_centroid(ring):
    n = len(ring)
    cx = sum(p[0] for p in ring) / n
    cy = sum(p[1] for p in ring) / n
    return cx, cy

def poly_bbox(ring):
    lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
    return min(lons), min(lats), max(lons), max(lats)

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def point_in_poly(px, py, ring):
    """Ray-casting for point-in-polygon."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]; xj, yj = ring[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def bboxes_overlap(b1, b2, pad_deg=0.0):
    """Check if two bboxes overlap (with optional padding in degrees)."""
    lon1a, lat1a, lon1b, lat1b = b1
    lon2a, lat2a, lon2b, lat2b = b2
    return (lon1a - pad_deg <= lon2b + pad_deg and
            lon1b + pad_deg >= lon2a - pad_deg and
            lat1a - pad_deg <= lat2b + pad_deg and
            lat1b + pad_deg >= lat2a - pad_deg)

kml_cx, kml_cy = poly_centroid(kml_ring)
kml_bbox = poly_bbox(kml_ring)
kml_lon_span = (kml_bbox[2]-kml_bbox[0]) * 111_320 * math.cos(math.radians(kml_cy))
kml_lat_span = (kml_bbox[3]-kml_bbox[1]) * 111_320
kml_area_approx = kml_lon_span * kml_lat_span / 10_000  # rough ha

print("="*72)
print("  FIELD KML  — अवैध कटाई  (P472, Mar Ki Mahu)")
print("="*72)
print(f"  Centroid  : {kml_cx:.6f}°E  {kml_cy:.6f}°N")
print(f"  Bbox      : lon [{kml_bbox[0]:.5f} – {kml_bbox[2]:.5f}]")
print(f"              lat [{kml_bbox[1]:.5f} – {kml_bbox[3]:.5f}]")
print(f"  Size      : ~{kml_lon_span:.0f} m × {kml_lat_span:.0f} m  ≈ {kml_area_approx:.2f} ha")
print()

# ─── 2. Load all alert GeoJSONs ─────────────────────────────────────────────
ALERT_DIR = Path("outputs/alerts")
FILES = sorted(ALERT_DIR.glob("mar_ki_mahu_*.geojson"))

# 500 m pad: roughly how far a 10 m patch can be offset
PAD_DEG = 500 / 111_320  # ~0.0045°

print(f"{'DATE':<12} {'PATCH ID':<16} {'DIST_TO_KML':>12} {'OVERLAP':>8} {'BBOX_NEAR':>10} {'Δtrees':>8}  SCORE_CONTEXT")
print("─"*90)

all_hits = []

for fpath in FILES:
    date_str = fpath.stem.replace("mar_ki_mahu_", "")
    fc = json.loads(fpath.read_text())
    for feat in fc["features"]:
        fid = feat["id"]
        props = feat["properties"]
        delta = props.get("dw_trees_delta", 0.0)

        geom = feat["geometry"]
        gtype = geom["type"]

        # Collect all rings from Polygon or MultiPolygon
        rings = []
        if gtype == "Polygon":
            rings = [geom["coordinates"][0]]
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings.append(poly[0])

        for ring in rings:
            ring2d = [(c[0], c[1]) for c in ring]
            pcx, pcy = poly_centroid(ring2d)
            pbbox = poly_bbox(ring2d)

            dist = haversine_m(kml_cx, kml_cy, pcx, pcy)
            bbox_near = bboxes_overlap(kml_bbox, pbbox, pad_deg=PAD_DEG)
            # Check if patch centroid is inside KML polygon
            inside = point_in_poly(pcx, pcy, kml_ring)
            # Check if KML centroid is inside patch polygon
            kml_in_patch = point_in_poly(kml_cx, kml_cy, ring2d)
            overlap = inside or kml_in_patch

            flag = ""
            if overlap:
                flag = "  ← OVERLAP ✓"
            elif dist < 200:
                flag = "  ← WITHIN 200 m"
            elif dist < 500:
                flag = "  ← WITHIN 500 m"

            if dist < 2000 or bbox_near:  # only print nearby ones
                print(f"{date_str:<12} {fid:<16} {dist:>9.0f} m  {'YES' if overlap else 'no':>8}  {'YES' if bbox_near else 'no':>10}  {delta:>+8.4f}{flag}")
                all_hits.append((date_str, fid, dist, overlap, delta))

print()

# ─── 3. Summary ─────────────────────────────────────────────────────────────
print("="*72)
overlaps = [(d, fid, dist, delta) for d, fid, dist, ov, delta in all_hits if ov]
near500  = [(d, fid, dist, delta) for d, fid, dist, ov, delta in all_hits if dist < 500]

if overlaps:
    print(f"  ✅  OVERLAP CONFIRMED — field KML overlaps {len(overlaps)} alert patch(es):")
    for d, fid, dist, delta in overlaps:
        print(f"     {d}  patch {fid}  trees_Δ={delta:+.4f}")
elif near500:
    print(f"  🟡  NEAR-HIT — field KML within 500 m of {len(near500)} alert patch(es):")
    for d, fid, dist, delta in near500:
        print(f"     {d}  patch {fid}  dist={dist:.0f} m  trees_Δ={delta:+.4f}")
else:
    print("  ❌  NO spatial match within 2 km. Check if KML is in a different beat/range.")
print("="*72)
print()
print(f"  KML centroid: {kml_cx:.5f}°E  {kml_cy:.5f}°N")
print(f"  Nearest alert date: {min(all_hits, key=lambda x: x[2])[0] if all_hits else 'none'}")
