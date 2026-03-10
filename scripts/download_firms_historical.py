"""
FIRMS Historical Fire Archive Download
======================================
Downloads VIIRS SNPP SP and MODIS SP fire detections for Guna Division, MP
covering 2013-01-01 to 2025-12-31 using 5-day API chunks.

Sources:
  - VIIRS_SNPP_SP : Jan 20 2012 – Dec 31 2025   (375m, high quality)
  - MODIS_SP      : Nov  1 2000 – Dec 31 2025   (1km, longer history)

API limit: 5 days per request for Standard Product (SP) archive sources.
           5000 transactions / 10 min  →  ~1924 total calls ≈ 4 min.

Output layout:
  data_lake/fire_detections/
      firms_viirs_snpp_sp/<YYYY>/viirs_snpp_sp_<YYYY>_<MM>_<DD>_5d.parquet
      firms_modis_sp/<YYYY>/modis_sp_<YYYY>_<MM>_<DD>_5d.parquet

Usage:
  python scripts/download_firms_historical.py
  python scripts/download_firms_historical.py --start 2020-01-01 --end 2020-12-31
  python scripts/download_firms_historical.py --dry-run
"""

import io, sys, os, time, argparse, logging
from pathlib import Path
from datetime import date, timedelta

import urllib.request, urllib.error
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"

# ── Constants ─────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("FIRMS_MAP_KEY", "")
if not API_KEY:
    raise SystemExit("FIRMS_MAP_KEY env var not set. See jwalaNetra/.env or register at https://firms.modaps.eosdis.nasa.gov/api/area/")
BBOX      = "76.75,23.85,77.52,25.15"        # Guna Division with buffer
BBOX_TUPLE = (76.75, 23.85, 77.52, 25.15)    # (west, south, east, north)
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CHUNK_DAYS = 5                                # max days per SP query
SLEEP_BETWEEN = 0.13                          # ~7.5 req/s → well under 500/min limit

SOURCES = {
    "viirs_snpp_sp": {
        "code"    : "VIIRS_SNPP_SP",
        "start"   : date(2012, 1, 20),
        "subdir"  : "firms_viirs_snpp_sp",
        "prefix"  : "viirs_snpp_sp",
    },
    "modis_sp": {
        "code"    : "MODIS_SP",
        "start"   : date(2000, 11, 1),
        "subdir"  : "firms_modis_sp",
        "prefix"  : "modis_sp",
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("firms_dl")


def _out_path(source_key: str, chunk_start: date) -> Path:
    """Return the parquet file path for a given source and chunk start date."""
    meta = SOURCES[source_key]
    subdir = DATA_LAKE / "fire_detections" / meta["subdir"] / str(chunk_start.year)
    fname  = f"{meta['prefix']}_{chunk_start.strftime('%Y_%m_%d')}_{CHUNK_DAYS}d.parquet"
    return subdir / fname


def _fetch_chunk(source_code: str, chunk_start: date, days: int = CHUNK_DAYS) -> pd.DataFrame:
    """Fetch a chunk from FIRMS API and return a DataFrame (empty if no fires)."""
    url = (
        f"{FIRMS_BASE}/{API_KEY}/{source_code}/{BBOX}"
        f"/{days}/{chunk_start.isoformat()}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "jwalaNetra_2/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")

    lines = [l for l in raw.strip().split("\n") if l.strip()]
    if len(lines) <= 1:
        return pd.DataFrame()               # header only → no fires

    df = pd.read_csv(io.StringIO(raw))

    # Filter strictly to Guna Division bbox (API bbox may have float rounding)
    w, s, e, n = BBOX_TUPLE
    mask = (
        (df["longitude"] >= w) & (df["longitude"] <= e) &
        (df["latitude"]  >= s) & (df["latitude"]  <= n)
    )
    return df[mask].reset_index(drop=True)


def download_historical(
    start: date = date(2013, 1, 1),
    end:   date = date(2025, 12, 31),
    dry_run: bool = False,
    sources: dict = None,
) -> dict:
    """
    Download all fire detections for the date range using 5-day chunks.
    Returns a summary dict with counts per source.
    """
    if sources is None:
        sources = SOURCES
    summary = {sk: {"chunks": 0, "fires": 0, "skipped": 0} for sk in sources}

    for sk, meta in sources.items():
        src_start = max(start, meta["start"])
        if src_start > end:
            log.info(f"Skipping {sk}: no coverage in requested range")
            continue

        chunk_date = src_start
        total_days  = (end - src_start).days + 1
        total_chunks = (total_days + CHUNK_DAYS - 1) // CHUNK_DAYS
        log.info(
            f"[{meta['code']}] {src_start} → {end}  |  "
            f"{total_chunks} chunks × {CHUNK_DAYS}d"
        )

        chunk_num = 0
        while chunk_date <= end:
            chunk_num += 1
            actual_days = min(CHUNK_DAYS, (end - chunk_date).days + 1)
            out = _out_path(sk, chunk_date)

            if out.exists():
                summary[sk]["skipped"] += 1
                chunk_date += timedelta(days=CHUNK_DAYS)
                continue

            if dry_run:
                log.info(f"  DRY-RUN  {meta['code']} {chunk_date} ({actual_days}d) → {out.name}")
                chunk_date += timedelta(days=CHUNK_DAYS)
                continue

            try:
                df = _fetch_chunk(meta["code"], chunk_date, actual_days)
                n  = len(df)

                out.parent.mkdir(parents=True, exist_ok=True)
                if n > 0:
                    df.to_parquet(out, index=False)
                else:
                    # Write empty sentinel so we don't re-query
                    pd.DataFrame().to_parquet(out, index=False)

                summary[sk]["chunks"] += 1
                summary[sk]["fires"]  += n

                if chunk_num % 50 == 0 or n > 0:
                    pct = chunk_num / total_chunks * 100
                    log.info(
                        f"  [{meta['code']}] {chunk_date} ({actual_days}d): "
                        f"{n:4d} fires  [{chunk_num}/{total_chunks} = {pct:.0f}%]"
                    )

                time.sleep(SLEEP_BETWEEN)

            except urllib.error.HTTPError as e:
                body = e.read(200).decode("utf-8", errors="replace")
                log.warning(f"  HTTP {e.code} on {chunk_date}: {body[:80]}")
                time.sleep(2)
            except Exception as e:
                log.warning(f"  Error on {chunk_date}: {e}")
                time.sleep(2)

            chunk_date += timedelta(days=CHUNK_DAYS)

        fires_total = summary[sk]["fires"]
        skipped     = summary[sk]["skipped"]
        done        = summary[sk]["chunks"]
        log.info(
            f"[{meta['code']}] Done — {done} new chunks, "
            f"{skipped} skipped (already cached), {fires_total} fire detections"
        )

    return summary


def combine_to_annual(
    start_year: int = 2013,
    end_year:   int = 2025,
    sources: dict = None,
) -> None:
    """
    (Optional) Merge all 5-day parquet chunks into annual GeoParquet files.
    Called after download_historical().
    """
    if sources is None:
        sources = SOURCES
    for sk, meta in sources.items():
        subdir = DATA_LAKE / "fire_detections" / meta["subdir"]
        for yr in range(start_year, end_year + 1):
            yr_dir = subdir / str(yr)
            if not yr_dir.exists():
                continue
            chunks = sorted(yr_dir.glob("*.parquet"))
            if not chunks:
                continue

            dfs = [pd.read_parquet(p) for p in chunks]
            combined = pd.concat(dfs, ignore_index=True)
            if combined.empty:
                continue

            out = subdir / f"{meta['prefix']}_{yr}_annual.parquet"
            combined.to_parquet(out, index=False)
            log.info(
                f"  Annual  {meta['code']} {yr}: "
                f"{len(combined)} detections → {out.name}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download FIRMS historical archive")
    parser.add_argument("--start",   default="2013-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--source",  choices=list(SOURCES.keys()) + ["all"],
                        default="all", help="Which source to download")
    parser.add_argument("--dry-run", action="store_true", help="List chunks without downloading")
    parser.add_argument("--annual",  action="store_true",
                        help="After download, merge chunks into annual parquet files")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    sources_to_use = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}

    log.info(f"FIRMS historical download: {start} → {end}  (dry_run={args.dry_run})")
    summary = download_historical(start, end, dry_run=args.dry_run, sources=sources_to_use)

    if args.annual and not args.dry_run:
        log.info("Building annual merged parquet files …")
        combine_to_annual(start.year, end.year, sources=sources_to_use)

    log.info("=== Summary ===")
    for sk, s in summary.items():
        log.info(f"  {sk:20s}: {s['chunks']:4d} new + {s['skipped']:4d} skipped  |  {s['fires']:6d} fire detections")


if __name__ == "__main__":
    main()
