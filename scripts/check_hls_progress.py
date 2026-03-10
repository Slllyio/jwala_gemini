"""
HLS Download Progress Checker
==============================
Scans the data_lake/satellite_imagery directory and reports:
  - Number of scenes / bands per product, tile, and year
  - Estimated total vs expected coverage
  - Disk usage

Usage:
  python scripts/check_hls_progress.py
  python scripts/check_hls_progress.py --verbose
"""

import argparse
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake" / "satellite_imagery"

PRITHVI_BANDS_S30 = {"B02", "B03", "B04", "B8A", "B11", "B12"}
PRITHVI_BANDS_L30 = {"B02", "B03", "B04", "B05", "B06", "B07"}
BANDS_BY_PRODUCT = {"hls_s30": PRITHVI_BANDS_S30, "hls_l30": PRITHVI_BANDS_L30}

EXPECTED_GRANULES = {
    "hls_s30": {
        "2024-2025 fire seasons (Feb-May)": 456,
        "2017-2023 fire seasons (Feb-May)": 1819,
        "total S30 (2017-2025)":            2275,
    },
    "hls_l30": {
        "total L30 (2013-2025 fire seasons)": 1100,
    },
}


def scan_product(product_dir: Path, bands: set) -> dict:
    """
    Returns dict of {year: {tile: {date: {band: path}}}}
    and counts complete scenes (all 6 bands present).
    """
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    if not product_dir.exists():
        return stats, 0, 0, 0

    total_files = 0
    total_bytes = 0

    for tif in product_dir.rglob("*.tif"):
        # Expected layout: product_dir/<tile>/<year>/<month>/<day>/<filename>.tif
        parts = tif.parts
        # Find tile (6-char starting with T) in path
        tile = None
        year = None
        month = None
        day = None
        for i, p in enumerate(parts):
            if p.startswith("T") and len(p) == 6 and tile is None:
                tile = p
            elif tile and p.isdigit() and len(p) == 4 and year is None:
                year = p
            elif tile and year and p.isdigit() and len(p) == 2 and month is None:
                month = p
            elif tile and year and month and p.isdigit() and len(p) == 2 and day is None:
                day = p
                break

        if not (tile and year and month and day):
            continue

        # Detect band from filename
        name = tif.stem  # e.g. HLS.S30.T43RGH.2024094T052641.v2.0.B02
        detected_band = None
        for band in bands:
            if f".{band}" in name or name.endswith(f".{band}"):
                detected_band = band
                break
        if detected_band is None:
            # Try last segment
            parts_name = name.split(".")
            if parts_name[-1] in bands:
                detected_band = parts_name[-1]

        if detected_band is None:
            continue

        date_str = f"{year}-{month}-{day}"
        stats[year][tile][date_str][detected_band] = tif
        total_files += 1
        total_bytes += tif.stat().st_size

    # Count complete scenes (all 6 bands)
    complete_scenes = 0
    partial_scenes  = 0
    for yr in stats:
        for tile in stats[yr]:
            for date_str in stats[yr][tile]:
                n_bands = len(stats[yr][tile][date_str])
                if n_bands == len(bands):
                    complete_scenes += 1
                else:
                    partial_scenes += 1

    return stats, total_files, total_bytes, complete_scenes, partial_scenes


def fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.2f} GB"
    elif n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MB"
    else:
        return f"{n_bytes / (1 << 10):.0f} KB"


def main():
    parser = argparse.ArgumentParser(description="HLS download progress checker")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-year breakdown")
    args = parser.parse_args()

    grand_files = 0
    grand_bytes = 0
    grand_scenes = 0

    for product in ["hls_s30", "hls_l30"]:
        prod_dir = DATA_LAKE / product
        bands    = BANDS_BY_PRODUCT[product]

        if not prod_dir.exists():
            print(f"\n[{product.upper()}] directory not found — no data yet")
            continue

        stats, n_files, n_bytes, n_complete, n_partial = scan_product(prod_dir, bands)

        grand_files  += n_files
        grand_bytes  += n_bytes
        grand_scenes += n_complete

        print(f"\n{'='*60}")
        print(f"  {product.upper()}")
        print(f"{'='*60}")
        print(f"  Files on disk   : {n_files:,}  ({fmt_size(n_bytes)})")
        print(f"  Complete scenes : {n_complete:,}  (all {len(bands)} bands present)")
        print(f"  Partial scenes  : {n_partial:,}  (download in progress)")

        # Expected context
        if product in EXPECTED_GRANULES:
            print(f"\n  Expected granule counts (fire seasons, cc<=15%):")
            for desc, count in EXPECTED_GRANULES[product].items():
                print(f"    {desc}: ~{count:,} granules")

        if args.verbose and stats:
            print(f"\n  Year breakdown:")
            for yr in sorted(stats):
                yr_tiles = set()
                yr_scenes = 0
                yr_files  = 0
                yr_bytes  = 0
                for tile in stats[yr]:
                    yr_tiles.add(tile)
                    for date_str in stats[yr][tile]:
                        band_dict = stats[yr][tile][date_str]
                        yr_files  += len(band_dict)
                        yr_bytes  += sum(p.stat().st_size for p in band_dict.values())
                        if len(band_dict) == len(bands):
                            yr_scenes += 1
                print(f"    {yr}: {yr_scenes:3d} scenes | "
                      f"{len(yr_tiles)} tiles | "
                      f"{yr_files:4d} files | {fmt_size(yr_bytes)}")

        # Tile summary
        all_tiles = set()
        for yr in stats:
            all_tiles.update(stats[yr].keys())
        if all_tiles:
            print(f"\n  Tiles: {', '.join(sorted(all_tiles))}")

    print(f"\n{'='*60}")
    print(f"  GRAND TOTAL")
    print(f"{'='*60}")
    print(f"  Total files       : {grand_files:,}")
    print(f"  Total disk usage  : {fmt_size(grand_bytes)}")
    print(f"  Complete scenes   : {grand_scenes:,}  (across all products)")
    print(f"\n  Expected totals:")
    print(f"    S30 (2017-2025, fire seasons): ~2,275 granules")
    print(f"    L30 (2013-2025, fire seasons): ~1,100 granules")
    print(f"    Grand total: ~3,375 granules x 6 bands = ~20,250 files")
    print()


if __name__ == "__main__":
    main()
