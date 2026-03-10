"""
Download Missing HLS Temporal Frames for VanAagni Training
==========================================================
Scans fire_links.csv, identifies fire events missing HLS S30 imagery
for the 3-frame temporal stack (t_input, t_input-16d, t_input-32d),
and downloads only the needed scenes from NASA EarthData.

Usage:
  python scripts/download_missing_hls.py                    # Diagnose + download
  python scripts/download_missing_hls.py --diagnose-only    # Just report gaps
  python scripts/download_missing_hls.py --cloud 30         # Higher cloud tolerance
"""

import os, sys, logging, argparse
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict

# ── PROJ fix (must be before rasterio) ──────────────────────────────────────
def _fix_proj():
    try:
        import pyproj as _pp
        _proj_data = str(Path(_pp.datadir.get_data_dir()))
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"]  = _proj_data
        os.environ.pop("GDAL_DATA", None)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
_fix_proj()

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
HLS_DIR   = DATA_LAKE / "satellite_imagery" / "hls_s30"
LINKS_CSV = DATA_LAKE / "burn_scars" / "fire_links.csv"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Constants (must match build_training_dataset.py) ─────────────────────────
PRED_HORIZON     = 3      # days before fire_date for t_input
N_FRAMES         = 3      # temporal depth
TEMPORAL_CADENCE = 16     # days between frames
SEARCH_WINDOW    = 12     # +- days to search for cloud-free HLS scene
GUNA_BBOX        = (76.75, 23.85, 77.52, 25.15)  # west, south, east, north
PRITHVI_BANDS    = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_missing_hls.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("missing_hls")


def find_hls_scene(tile: str, target: datetime,
                   search_days: int = SEARCH_WINDOW) -> bool:
    """Check if an HLS scene exists near target date (same logic as build_training_dataset)."""
    for delta in range(0, search_days + 1):
        for sign in ([0] if delta == 0 else [+delta, -delta]):
            dt = target + timedelta(days=sign)
            scene_dir = HLS_DIR / tile / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
            if scene_dir.exists() and any(scene_dir.glob("*.B02.tif")):
                return True
    return False


def diagnose_missing_frames(df: pd.DataFrame) -> list:
    """
    For each fire event, check if the 3 temporal HLS frames exist.
    Returns list of dicts with event info and missing frame dates.
    """
    missing_events = []

    for idx, row in df.iterrows():
        tile      = str(row["tile"])
        fire_date = pd.to_datetime(row["fire_date"])
        t_input   = fire_date - timedelta(days=PRED_HORIZON)

        # The 3 target dates for the temporal stack
        target_dates = [
            t_input - timedelta(days=i * TEMPORAL_CADENCE)
            for i in range(N_FRAMES)
        ]

        missing_frames = []
        for i, td in enumerate(target_dates):
            if not find_hls_scene(tile, td):
                missing_frames.append({
                    "frame_idx": i,
                    "target_date": td,
                    "search_start": td - timedelta(days=SEARCH_WINDOW),
                    "search_end": td + timedelta(days=SEARCH_WINDOW),
                })

        if missing_frames:
            missing_events.append({
                "idx": idx,
                "tile": tile,
                "fire_date": fire_date,
                "t_input": t_input,
                "tier": str(row.get("tier", "UNKNOWN")),
                "missing_frames": missing_frames,
                "n_missing": len(missing_frames),
                "all_missing": len(missing_frames) == N_FRAMES,
            })

    return missing_events


def compute_download_windows(missing_events: list) -> list:
    """
    Merge missing frame date ranges into download windows per tile.
    Returns list of (tile, start_date, end_date) for earthaccess queries.
    """
    # Group by tile
    tile_ranges = defaultdict(list)
    for evt in missing_events:
        tile = evt["tile"]
        for frame in evt["missing_frames"]:
            tile_ranges[tile].append((
                frame["search_start"].date() if hasattr(frame["search_start"], 'date') else frame["search_start"],
                frame["search_end"].date() if hasattr(frame["search_end"], 'date') else frame["search_end"],
            ))

    # Merge overlapping ranges per tile
    download_windows = []
    for tile, ranges in tile_ranges.items():
        ranges.sort(key=lambda x: x[0])
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            if start <= merged[-1][1] + timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            download_windows.append((tile, start, end))

    return download_windows


def download_for_windows(windows: list, cloud_max: int = 30, workers: int = 3,
                         dry_run: bool = False) -> dict:
    """
    Download HLS S30 scenes for each (tile, start, end) window.
    Uses earthaccess search with tight temporal bounds.
    """
    import earthaccess
    import threading

    # Import from sibling script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_hls_imagery import (
        _login, _get_https_session, _out_dir, _crop_and_save,
        PRITHVI_BANDS_S30
    )

    _login()
    session = _get_https_session()
    bands = PRITHVI_BANDS_S30

    summary = {"granules": 0, "downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}

    for tile, start, end in windows:
        log.info(f"Searching HLS S30 for {tile}: {start} -> {end} (cloud<={cloud_max}%)")

        results = earthaccess.search_data(
            short_name="HLSS30",
            cloud_hosted=True,
            temporal=(str(start), str(end)),
            bounding_box=GUNA_BBOX,
            cloud_cover=(0, cloud_max),
        )

        # Filter to matching tile only
        tile_results = []
        for g in results:
            ur = g["umm"]["GranuleUR"]
            if tile in ur:
                tile_results.append(g)

        log.info(f"  Found {len(tile_results)} granules for tile {tile}")
        summary["granules"] += len(tile_results)

        if dry_run:
            for g in tile_results[:3]:
                log.info(f"    DRY-RUN: {g['umm']['GranuleUR']}")
            continue

        for granule in tile_results:
            ur = granule["umm"]["GranuleUR"]
            acq_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            acq_date = datetime.fromisoformat(acq_str[:10]).date()
            out_dir = _out_dir("S30", tile, acq_date)

            # Check if already exists
            existing = [b for b in bands if out_dir.exists() and any(out_dir.glob(f"*{b}*.tif"))]
            if len(existing) == len(bands):
                summary["skipped"] += len(bands)
                continue

            try:
                file_objs = earthaccess.open([granule])
            except Exception as e:
                log.warning(f"  [{ur}] open failed: {e}")
                summary["errors"] += 1
                continue

            for band in bands:
                fo = next((f for f in file_objs if f".{band}." in str(f)), None)
                if fo is None:
                    continue

                fname = str(fo).split("/")[-1].split("\\")[-1]
                if not fname.endswith(".tif"):
                    fname = f"{ur}.{band}.tif"
                out_path = out_dir / fname

                if out_path.exists():
                    summary["skipped"] += 1
                    continue

                try:
                    sz = _crop_and_save(session, fo, out_path)
                    summary["downloaded"] += 1
                    summary["bytes"] += sz
                    log.info(f"    {acq_date} {tile} {band}: {sz/1024:.0f} KB")
                except Exception as exc:
                    log.warning(f"    [{ur}] {band}: {exc}")
                    summary["errors"] += 1

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download missing HLS for VanAagni training")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Only report gaps, don't download")
    parser.add_argument("--cloud", type=int, default=30,
                        help="Max cloud cover %% (default 30, higher than normal to catch more scenes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search but don't download")
    parser.add_argument("--pred-horizon", type=int, default=PRED_HORIZON,
                        help="Days before fire for input imagery")
    args = parser.parse_args()

    # ── Load fire events ─────────────────────────────────────────────────
    df = pd.read_csv(LINKS_CSV)
    df["fire_date"] = pd.to_datetime(df["fire_date"])

    # Only process events that have fire_date
    df = df.dropna(subset=["fire_date"])
    log.info(f"Loaded {len(df)} fire events from fire_links.csv")

    # ── Diagnose missing frames ──────────────────────────────────────────
    log.info("Scanning for missing HLS temporal frames...")
    missing = diagnose_missing_frames(df)

    # ── Report ───────────────────────────────────────────────────────────
    n_total = len(df)
    n_missing = len(missing)
    n_all_missing = sum(1 for e in missing if e["all_missing"])
    n_partial = n_missing - n_all_missing
    n_ok = n_total - n_missing

    log.info("")
    log.info("=" * 60)
    log.info("HLS Temporal Frame Gap Analysis")
    log.info("=" * 60)
    log.info(f"Total fire events:           {n_total:4d}")
    log.info(f"Events with ALL frames OK:   {n_ok:4d}  ({n_ok/n_total*100:.0f}%)")
    log.info(f"Events missing SOME frames:  {n_partial:4d}  ({n_partial/n_total*100:.0f}%)")
    log.info(f"Events missing ALL frames:   {n_all_missing:4d}  ({n_all_missing/n_total*100:.0f}%)")
    log.info("")

    # Per-tile breakdown
    tile_missing = defaultdict(lambda: {"total": 0, "missing": 0})
    for _, row in df.iterrows():
        tile_missing[str(row["tile"])]["total"] += 1
    for evt in missing:
        tile_missing[evt["tile"]]["missing"] += 1

    log.info("Per-tile breakdown:")
    for tile in sorted(tile_missing.keys()):
        info = tile_missing[tile]
        log.info(f"  {tile}: {info['missing']}/{info['total']} events missing HLS")

    # Year breakdown
    year_missing = defaultdict(lambda: {"total": 0, "missing": 0})
    for _, row in df.iterrows():
        y = pd.to_datetime(row["fire_date"]).year
        year_missing[y]["total"] += 1
    for evt in missing:
        y = evt["fire_date"].year
        year_missing[y]["missing"] += 1

    log.info("")
    log.info("Per-year breakdown:")
    for year in sorted(year_missing.keys()):
        info = year_missing[year]
        log.info(f"  {year}: {info['missing']}/{info['total']} events missing HLS")

    # Show sample missing events
    log.info("")
    log.info("Sample missing events (first 15):")
    for evt in missing[:15]:
        frames_str = ", ".join(
            f"F{f['frame_idx']}={f['target_date'].strftime('%Y-%m-%d')}"
            for f in evt["missing_frames"]
        )
        log.info(
            f"  {evt['tile']} fire={evt['fire_date'].strftime('%Y-%m-%d')} "
            f"tier={evt['tier']:10s} missing=[{frames_str}]"
        )
    if len(missing) > 15:
        log.info(f"  ... and {len(missing) - 15} more")

    if args.diagnose_only:
        log.info("")
        log.info("Diagnose-only mode. Use without --diagnose-only to download.")
        return

    if not missing:
        log.info("No missing HLS frames! All fire events have complete temporal stacks.")
        return

    # ── Compute download windows ─────────────────────────────────────────
    windows = compute_download_windows(missing)
    log.info("")
    log.info(f"Download plan: {len(windows)} time windows across tiles:")
    for tile, start, end in windows:
        days = (end - start).days
        log.info(f"  {tile}: {start} -> {end} ({days} days)")

    # ── Download ─────────────────────────────────────────────────────────
    log.info("")
    log.info("Starting downloads...")
    summary = download_for_windows(windows, cloud_max=args.cloud, dry_run=args.dry_run)

    log.info("")
    log.info("=" * 60)
    log.info("Download Summary")
    log.info("=" * 60)
    log.info(f"Granules found:    {summary['granules']}")
    log.info(f"Files downloaded:  {summary['downloaded']}")
    log.info(f"Files skipped:     {summary['skipped']}")
    log.info(f"Errors:            {summary['errors']}")
    log.info(f"Total size:        {summary['bytes']/1024/1024:.0f} MB")

    # ── Re-diagnose ──────────────────────────────────────────────────────
    if not args.dry_run and summary["downloaded"] > 0:
        log.info("")
        log.info("Re-scanning after download...")
        missing_after = diagnose_missing_frames(df)
        n_fixed = n_missing - len(missing_after)
        log.info(f"Fixed {n_fixed} events. Still missing: {len(missing_after)}")


if __name__ == "__main__":
    main()
