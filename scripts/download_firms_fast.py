"""
FIRMS Historical Fire Archive - Parallel Downloader
====================================================
Multi-threaded version of download_firms_historical.py.
Uses ThreadPoolExecutor with a token-bucket rate limiter to stay
under the FIRMS API limit of 5000 transactions/10 min (≈ 8.33 req/s).

Already-downloaded chunks are skipped automatically.

Usage:
  python scripts/download_firms_fast.py
  python scripts/download_firms_fast.py --workers 4 --start 2013-01-01
"""

import io, sys, os, time, threading, argparse, logging
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request, urllib.error
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("FIRMS_MAP_KEY", "")
if not API_KEY:
    raise SystemExit("FIRMS_MAP_KEY env var not set. See jwalaNetra/.env or register at https://firms.modaps.eosdis.nasa.gov/api/area/")
BBOX       = "76.75,23.85,77.52,25.15"
BBOX_TUPLE = (76.75, 23.85, 77.52, 25.15)
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CHUNK_DAYS = 5
MAX_RPS    = 7.5          # stay under 8.33 req/s (5000/10min) with margin

SOURCES = {
    "viirs_snpp_sp": {
        "code"   : "VIIRS_SNPP_SP",
        "start"  : date(2012, 1, 20),
        "subdir" : "firms_viirs_snpp_sp",
        "prefix" : "viirs_snpp_sp",
    },
    "modis_sp": {
        "code"   : "MODIS_SP",
        "start"  : date(2000, 11, 1),
        "subdir" : "firms_modis_sp",
        "prefix" : "modis_sp",
    },
}

# ── Logging ────────────────────────────────────────────────────────────────────
fh = logging.FileHandler(LOG_DIR / "firms_historical_parallel.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[fh, ch],
)
log = logging.getLogger("firms_fast")

# ── Token-bucket rate limiter ──────────────────────────────────────────────────
class RateLimiter:
    """Token-bucket allowing at most `rate` calls per second."""
    def __init__(self, rate: float):
        self._rate      = rate
        self._tokens    = rate
        self._last      = time.monotonic()
        self._lock      = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last   = now
            if self._tokens >= 1:
                self._tokens -= 1
            else:
                wait = (1 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0
                self._last   = time.monotonic()

_limiter = RateLimiter(MAX_RPS)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _out_path(source_key: str, chunk_start: date) -> Path:
    meta  = SOURCES[source_key]
    subdir = DATA_LAKE / "fire_detections" / meta["subdir"] / str(chunk_start.year)
    fname  = f"{meta['prefix']}_{chunk_start.strftime('%Y_%m_%d')}_{CHUNK_DAYS}d.parquet"
    return subdir / fname


def _fetch_and_save(source_key: str, chunk_start: date, actual_days: int) -> int:
    """
    Fetch one chunk, save as parquet (empty sentinel if no fires).
    Returns number of fires in chunk, or -1 on error.
    """
    meta = SOURCES[source_key]
    out  = _out_path(source_key, chunk_start)

    if out.exists():
        return -2   # already cached

    url = (
        f"{FIRMS_BASE}/{API_KEY}/{meta['code']}/{BBOX}"
        f"/{actual_days}/{chunk_start.isoformat()}"
    )

    _limiter.acquire()          # respect rate limit

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jwalaNetra_2/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            break
        except Exception as exc:
            if attempt == 2:
                log.warning(f"  Failed {meta['code']} {chunk_start} after 3 attempts: {exc}")
                return -1
            time.sleep(2 ** attempt)

    lines = [l for l in raw.strip().split("\n") if l.strip()]
    if len(lines) <= 1:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(io.StringIO(raw))
        w, s, e, n = BBOX_TUPLE
        mask = (
            (df["longitude"] >= w) & (df["longitude"] <= e) &
            (df["latitude"]  >= s) & (df["latitude"]  <= n)
        )
        df = df[mask].reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


def _build_tasks(start: date, end: date) -> list:
    """Generate list of (source_key, chunk_start, actual_days) tuples."""
    tasks = []
    for sk, meta in SOURCES.items():
        src_start = max(start, meta["start"])
        if src_start > end:
            continue
        d = src_start
        while d <= end:
            actual = min(CHUNK_DAYS, (end - d).days + 1)
            tasks.append((sk, d, actual))
            d += timedelta(days=CHUNK_DAYS)
    return tasks


def download_parallel(
    start:   date = date(2013, 1, 1),
    end:     date = date(2025, 12, 31),
    workers: int  = 4,
) -> dict:
    tasks = _build_tasks(start, end)
    total = len(tasks)

    # Pre-count already-cached
    cached = sum(1 for sk, d, _ in tasks if _out_path(sk, d).exists())
    remaining = total - cached

    log.info(
        f"Tasks: {total} total | {cached} cached | {remaining} to download"
    )
    log.info(
        f"Workers: {workers} | Rate limit: {MAX_RPS:.1f} req/s | "
        f"Estimated time: {remaining / MAX_RPS / 60:.1f} min"
    )

    counter   = {"done": 0, "fires": 0, "errors": 0}
    lock      = threading.Lock()
    t0        = time.monotonic()

    def _progress(source_key, chunk_start, n_fires):
        with lock:
            counter["done"] += 1
            if n_fires > 0:
                counter["fires"] += n_fires
            if n_fires == -1:
                counter["errors"] += 1
            done = counter["done"]
            if done % 100 == 0 or n_fires > 10:
                elapsed = time.monotonic() - t0
                rps     = done / max(elapsed, 0.01)
                eta_min = (remaining - done) / max(rps, 0.1) / 60
                log.info(
                    f"  [{done:4d}/{remaining}] {source_key} {chunk_start} | "
                    f"{n_fires:3d} fires | "
                    f"{rps:.1f} req/s | ETA {eta_min:.1f} min"
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_fetch_and_save, sk, d, actual): (sk, d)
            for sk, d, actual in tasks
        }
        for fut in as_completed(futs):
            sk, d = futs[fut]
            try:
                n = fut.result()
                if n != -2:   # not cached
                    _progress(sk, d, n)
            except Exception as exc:
                log.warning(f"  Exception {sk} {d}: {exc}")

    elapsed = time.monotonic() - t0
    log.info(
        f"=== Done in {elapsed:.0f}s | "
        f"{counter['done']} downloaded | {counter['fires']} fires | "
        f"{counter['errors']} errors ==="
    )
    return counter


def combine_to_annual(start_year: int = 2013, end_year: int = 2025) -> None:
    """Merge 5-day parquet chunks into per-year files."""
    for sk, meta in SOURCES.items():
        subdir = DATA_LAKE / "fire_detections" / meta["subdir"]
        for yr in range(start_year, end_year + 1):
            yr_dir = subdir / str(yr)
            if not yr_dir.exists():
                continue
            chunks = sorted(yr_dir.glob("*.parquet"))
            if not chunks:
                continue
            dfs = []
            for p in chunks:
                try:
                    df = pd.read_parquet(p)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
            if not dfs:
                continue
            combined = pd.concat(dfs, ignore_index=True)
            if combined.empty:
                continue
            out = subdir / f"{meta['prefix']}_{yr}_annual.parquet"
            combined.to_parquet(out, index=False)
            log.info(
                f"  Annual {meta['code']} {yr}: "
                f"{len(combined)} detections → {out.name}"
            )


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="FIRMS parallel historical download")
    p.add_argument("--start",   default="2013-01-01")
    p.add_argument("--end",     default="2025-12-31")
    p.add_argument("--workers", type=int, default=4, help="Parallel threads (default 4)")
    p.add_argument("--annual",  action="store_true", help="Build annual merged parquet")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    counter = download_parallel(start, end, workers=args.workers)

    if args.annual:
        log.info("Building annual merged files …")
        combine_to_annual(start.year, end.year)

    log.info("=== Final Summary ===")
    log.info(f"  Chunks downloaded : {counter['done']}")
    log.info(f"  Total fires       : {counter['fires']}")
    log.info(f"  Errors            : {counter['errors']}")


if __name__ == "__main__":
    main()
