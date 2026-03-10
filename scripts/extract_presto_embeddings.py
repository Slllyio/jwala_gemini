"""
Extract Presto temporal embeddings for Guna Division compartments.
================================================================

Uses NASA Harvest Presto (3.5M param transformer) as a FROZEN feature
extractor on HLS pixel time series at compartment centroids.

Pipeline:
  1. Load 747 compartment centroids from GeoJSON
  2. For each HLS granule, extract 6-band pixel values at all centroids
  3. Group into 3-timestep windows per (compartment, date)
  4. Add weather + terrain context to the 17-band Presto input
  5. Run frozen Presto encoder -> 128-D embedding per (compartment, date)
  6. Save as parquet for Ignition v2.3 feature augmentation

Presto input format (17 bands):
  [0,1]     S1 (VV, VH)          -- masked (no SAR data)
  [2,3,4]   S2 RGB (B02,B03,B04) -- from HLS /10000
  [5,6,7]   S2 Red Edge          -- masked (not in HLS)
  [8]       S2 NIR 10m (B08)     -- masked (not in HLS subset)
  [9]       S2 NIR 20m (B8A)     -- from HLS /10000
  [10,11]   S2 SWIR (B11,B12)    -- from HLS /10000
  [12,13]   ERA5 (temp, precip)  -- from weather cache
  [14,15]   SRTM (elev, slope)   -- from terrain cache
  [16]      NDVI                  -- computed from B8A, B04

Output:
  data/training/presto_embeddings.parquet
  Columns: NEW_No_, date, presto_pc0 ... presto_pc15

Usage:
  py -3.12 scripts/extract_presto_embeddings.py
  py -3.12 scripts/extract_presto_embeddings.py --tile T43QFG
"""

import os, sys, time, logging, argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np

# ── Fix PROJ ────────────────────────────────────────────────────────────────
try:
    import pyproj as _pp
    _proj_data = str(Path(_pp.datadir.get_data_dir()))
    os.environ["PROJ_DATA"] = _proj_data
    os.environ["PROJ_LIB"]  = _proj_data
    os.environ.pop("GDAL_DATA", None)
except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

ROOT = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
HLS_BASE  = DATA_LAKE / "satellite_imagery" / "hls_s30"
PRESTO_DIR = ROOT / "models" / "presto"
PRESTO_WEIGHTS = PRESTO_DIR / "default_model.pt"
PRESTO_CODE    = PRESTO_DIR / "single_file_presto.py"

# GeoJSON for compartment centroids
GEOJSON_PATH = ROOT / "data" / "aoi" / "gunafinal.geojson"

# Output
OUT_DIR = ROOT / "data" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HLS bands we use
BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]
NO_DATA_DN = -9999

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("presto_extract")


# ── Load compartment centroids ──────────────────────────────────────────────
def load_centroids():
    """Load 747 compartment centroids from GeoJSON."""
    import json

    with open(GEOJSON_PATH, encoding="utf-8") as f:
        gj = json.load(f)

    centroids = []
    for feat in gj["features"]:
        props = feat["properties"]
        new_no = props.get("NEW_No_") or props.get("NEW_No")
        if new_no is None:
            continue

        # Compute centroid from geometry
        geom = feat["geometry"]
        coords = _flatten_coords(geom)
        if not coords:
            continue
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        clat = sum(lats) / len(lats)
        clon = sum(lons) / len(lons)
        centroids.append({"NEW_No_": new_no, "lat": clat, "lon": clon})

    log.info("Loaded %d compartment centroids from GeoJSON", len(centroids))
    return centroids


def _flatten_coords(geom):
    """Recursively flatten GeoJSON geometry coordinates."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Point":
        return [coords]
    elif gtype in ("Polygon", "MultiLineString"):
        return [c for ring in coords for c in ring]
    elif gtype in ("MultiPolygon",):
        return [c for poly in coords for ring in poly for c in ring]
    elif gtype == "LineString":
        return coords
    return []


# ── HLS pixel extraction ────────────────────────────────────────────────────
def build_hls_index(tile_filter=None):
    """Build index: {(tile, date_str) -> {band: path}} for all HLS granules."""
    index = {}
    for tile_dir in sorted(HLS_BASE.iterdir()):
        if not tile_dir.is_dir() or not tile_dir.name.startswith("T"):
            continue
        tile = tile_dir.name
        if tile_filter and tile != tile_filter:
            continue

        for yr_dir in sorted(tile_dir.iterdir()):
            if not yr_dir.is_dir() or not yr_dir.name.isdigit():
                continue
            for mo_dir in sorted(yr_dir.iterdir()):
                if not mo_dir.is_dir():
                    continue
                for day_dir in sorted(mo_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    tifs = list(day_dir.glob("*.tif"))
                    bands = {}
                    for tif in tifs:
                        for band in BAND_ORDER:
                            if f".{band}." in tif.name:
                                bands[band] = tif
                                break
                    if len(bands) >= 6:
                        try:
                            d = date(int(yr_dir.name), int(mo_dir.name),
                                     int(day_dir.name))
                        except ValueError:
                            continue
                        key = (tile, d)
                        if key not in index:
                            index[key] = bands
    log.info("HLS index: %d (tile, date) entries", len(index))
    return index


def extract_pixels_at_centroids(band_paths, centroids):
    """
    Extract 6-band HLS values at centroid locations from one granule.
    Returns dict: {NEW_No_: np.array([B02, B03, B04, B8A, B11, B12])}.
    """
    import rasterio
    from rasterio.transform import rowcol
    from pyproj import Transformer

    # Read first band to get transform/CRS
    first_band = list(band_paths.values())[0]
    with rasterio.open(first_band) as src:
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    # Convert all centroids to pixel coords
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    lons = [c["lon"] for c in centroids]
    lats = [c["lat"] for c in centroids]
    xs, ys = transformer.transform(lons, lats)

    # Find valid centroids (within raster bounds)
    valid_centroids = []
    pixel_coords = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        row, col = rowcol(transform, x, y)
        if 0 <= row < h and 0 <= col < w:
            valid_centroids.append(centroids[i])
            pixel_coords.append((int(row), int(col)))

    if not valid_centroids:
        return {}

    # Read all bands
    rows_arr = [rc[0] for rc in pixel_coords]
    cols_arr = [rc[1] for rc in pixel_coords]

    band_arrays = {}
    for band_name in BAND_ORDER:
        path = band_paths[band_name]
        with rasterio.open(path) as src:
            band_arrays[band_name] = src.read(1)

    results = {}
    for i, centroid in enumerate(valid_centroids):
        r, c = rows_arr[i], cols_arr[i]
        vals = []
        has_nodata = False
        for band_name in BAND_ORDER:
            v = float(band_arrays[band_name][r, c])
            if v == NO_DATA_DN or v == 0:
                has_nodata = True
                break
            vals.append(v)
        if not has_nodata and len(vals) == 6:
            results[centroid["NEW_No_"]] = np.array(vals, dtype=np.float32)

    return results


# ── Build time series table ─────────────────────────────────────────────────
def build_spectral_table(hls_index, centroids):
    """
    For all (tile, date) in HLS index, extract pixel values at centroids.
    Returns dict: {NEW_No_: [(date, 6-band array), ...]} sorted by date.
    """
    log.info("Extracting HLS pixel values at %d centroids...", len(centroids))
    table = defaultdict(list)
    n_entries = len(hls_index)

    for i, ((tile, d), band_paths) in enumerate(sorted(hls_index.items())):
        if i % 100 == 0:
            log.info("  [%d/%d] %s %s", i, n_entries, tile, d)
        pixels = extract_pixels_at_centroids(band_paths, centroids)
        for new_no, vals in pixels.items():
            table[new_no].append((d, vals))

    # Sort each compartment's entries by date
    for new_no in table:
        table[new_no].sort(key=lambda x: x[0])

    n_total = sum(len(v) for v in table.values())
    log.info("Spectral table: %d compartments, %d total observations",
             len(table), n_total)
    return table


# ── Weather and terrain context ─────────────────────────────────────────────
def load_terrain_cache():
    """Load terrain data (elevation, slope) per compartment."""
    import pandas as pd

    # Try grid_embeddings for centroid terrain
    grid_path = OUT_DIR / "grid_embeddings.parquet"
    if grid_path.exists():
        gdf = pd.read_parquet(grid_path)
        terrain = (gdf.groupby("NEW_No_")
                   .agg(elevation_m=("elevation_m", "mean"),
                        slope_deg=("slope_mean_deg", "mean"))
                   .to_dict("index"))
        log.info("Terrain cache: %d compartments from grid_embeddings",
                 len(terrain))
        return terrain

    # Fallback: samples.parquet
    samples_path = OUT_DIR / "samples.parquet"
    if samples_path.exists():
        df = pd.read_parquet(samples_path)
        terrain = (df.groupby("NEW_No_")
                   .agg(elevation_m=("elevation_m", "mean"),
                        slope_deg=("slope_mean_deg", "mean"))
                   .to_dict("index"))
        log.info("Terrain cache: %d compartments from samples", len(terrain))
        return terrain

    log.warning("No terrain cache found -- using defaults")
    return {}


def load_weather_cache():
    """Load weather data per (NEW_No_, date) from weather_cache.parquet."""
    import pandas as pd

    wc_path = OUT_DIR / "weather_cache.parquet"
    if not wc_path.exists():
        log.warning("No weather_cache.parquet found -- ERA5 bands will be masked")
        return {}

    df = pd.read_parquet(wc_path)
    if "acq_date" in df.columns:
        df["date"] = pd.to_datetime(df["acq_date"]).dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    weather = {}
    for _, row in df.iterrows():
        key = (row.get("NEW_No_"), row["date"])
        temp = row.get("max_temp_c")
        precip = row.get("rain_7d_mm", 0.0)
        if temp is not None:
            weather[key] = (float(temp), float(precip) if precip else 0.0)

    log.info("Weather cache: %d (NEW_No_, date) entries", len(weather))
    return weather


# ── Build Presto input tensors ──────────────────────────────────────────────
def build_presto_inputs(spectral_table, terrain_cache, weather_cache,
                        centroids, n_timesteps=3):
    """
    Build Presto-format inputs from spectral table.

    For each (compartment, date), build a 3-timestep window:
      T=0 (target date), T-16d, T-32d (nearest available observations).

    Returns:
      rows: list of (NEW_No_, date) tuples
      x: np.array [N, n_timesteps, 17]
      latlons: np.array [N, 2]
      masks: np.array [N, n_timesteps, 17]
      months: np.array [N] (int, 0-indexed)
    """
    centroid_lookup = {c["NEW_No_"]: (c["lat"], c["lon"]) for c in centroids}

    all_rows = []
    all_x = []
    all_masks = []
    all_latlons = []
    all_months = []

    for new_no, observations in spectral_table.items():
        if new_no not in centroid_lookup:
            continue
        lat, lon = centroid_lookup[new_no]

        # Get terrain for this compartment
        terrain = terrain_cache.get(new_no, {})
        elev = terrain.get("elevation_m", 500.0)
        slope = terrain.get("slope_deg", 5.0)

        # Build date->values lookup
        dates = [obs[0] for obs in observations]
        vals_by_date = {obs[0]: obs[1] for obs in observations}

        for target_date in dates:
            # Find nearest observations for T-16d and T-32d
            timestep_dates = [target_date]
            for offset_days in [16, 32]:
                target_prior = target_date - timedelta(days=offset_days)
                best = None
                best_dist = 999
                for d in dates:
                    dist = abs((d - target_prior).days)
                    if dist < best_dist:
                        best_dist = dist
                        best = d
                if best is not None and best_dist <= 12:
                    timestep_dates.append(best)
                else:
                    timestep_dates.append(None)

            # Build 17-band input for each timestep
            x_sample = np.zeros((n_timesteps, 17), dtype=np.float32)
            mask_sample = np.ones((n_timesteps, 17), dtype=np.float32)

            for t_idx, t_date in enumerate(timestep_dates):
                if t_date is None or t_date not in vals_by_date:
                    continue

                hls = vals_by_date[t_date]
                # [B02, B03, B04, B8A, B11, B12] raw DN

                # S2_RGB: B02, B03, B04 -> indices 2,3,4
                x_sample[t_idx, 2] = hls[0] / 10000.0
                x_sample[t_idx, 3] = hls[1] / 10000.0
                x_sample[t_idx, 4] = hls[2] / 10000.0
                mask_sample[t_idx, 2:5] = 0.0

                # S2_NIR_20m: B8A -> index 9
                x_sample[t_idx, 9] = hls[3] / 10000.0
                mask_sample[t_idx, 9] = 0.0

                # S2_SWIR: B11, B12 -> indices 10, 11
                x_sample[t_idx, 10] = hls[4] / 10000.0
                x_sample[t_idx, 11] = hls[5] / 10000.0
                mask_sample[t_idx, 10:12] = 0.0

                # NDVI -> index 16
                nir = hls[3] / 10000.0
                red = hls[2] / 10000.0
                ndvi = (nir - red) / (nir + red + 1e-8)
                x_sample[t_idx, 16] = ndvi
                mask_sample[t_idx, 16] = 0.0

                # ERA5: temp and precip -> indices 12, 13
                w_key = (new_no, t_date)
                if w_key in weather_cache:
                    temp_c, precip_mm = weather_cache[w_key]
                    x_sample[t_idx, 12] = (temp_c + 273.15) / 35.0
                    x_sample[t_idx, 13] = (precip_mm / 1000.0) / 0.03
                    mask_sample[t_idx, 12:14] = 0.0

            # SRTM: static across timesteps -> indices 14, 15
            x_sample[:, 14] = elev / 2000.0
            x_sample[:, 15] = slope / 50.0
            mask_sample[:, 14:16] = 0.0

            # S1 (0,1), Red Edge (5,6,7), NIR_10m (8) stay masked=1

            all_rows.append((new_no, target_date))
            all_x.append(x_sample)
            all_masks.append(mask_sample)
            all_latlons.append([lat, lon])
            all_months.append(target_date.month - 1)  # 0-indexed

    log.info("Built %d Presto input samples from %d compartments",
             len(all_rows), len(spectral_table))

    return (
        all_rows,
        np.array(all_x, dtype=np.float32),
        np.array(all_latlons, dtype=np.float32),
        np.array(all_masks, dtype=np.float32),
        np.array(all_months, dtype=np.int64),
    )


# ── Presto model loading ────────────────────────────────────────────────────
def load_presto():
    """Load frozen Presto encoder from single_file_presto.py + weights."""
    import importlib.util
    import torch

    # Import single_file_presto.py dynamically
    spec = importlib.util.spec_from_file_location(
        "single_file_presto", str(PRESTO_CODE))
    sfp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfp)

    # Build model and load weights
    model = sfp.Presto.construct()
    state = torch.load(str(PRESTO_WEIGHTS), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state)
    model.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    log.info("Presto loaded: %d params (%.1f MB)", n_params, size_mb)
    return model.encoder


# ── Run inference ────────────────────────────────────────────────────────────
def run_presto_inference(encoder, x, latlons, masks, months, batch_size=256):
    """
    Run frozen Presto encoder on all samples.
    Returns: np.array [N, 128] embeddings.
    """
    import torch

    N = x.shape[0]
    all_embs = []

    encoder.eval()
    t0 = time.monotonic()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        x_batch = torch.tensor(x[start:end], dtype=torch.float32)
        ll_batch = torch.tensor(latlons[start:end], dtype=torch.float32)
        mask_batch = torch.tensor(masks[start:end], dtype=torch.float32)

        # Dynamic World: 9 = unknown class
        dw_batch = torch.full(
            (x_batch.shape[0], x_batch.shape[1]), 9, dtype=torch.long)

        month_batch = torch.tensor(months[start:end], dtype=torch.long)

        with torch.no_grad():
            emb = encoder(
                x=x_batch,
                dynamic_world=dw_batch,
                latlons=ll_batch,
                mask=mask_batch,
                month=month_batch,
                eval_task=True,
            )
        all_embs.append(emb.cpu().numpy())

        if (start // batch_size) % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = (start + batch_size) / max(elapsed, 0.001)
            log.info("  [%d/%d] %.0f samples/s", min(end, N), N, rate)

    embeddings = np.concatenate(all_embs, axis=0)
    elapsed = time.monotonic() - t0
    log.info("Presto inference done: %d samples in %.1fs (%.0f/s)",
             N, elapsed, N / max(elapsed, 0.001))
    return embeddings


# ── PCA reduction ────────────────────────────────────────────────────────────
def apply_pca(embeddings, n_components=16):
    """Reduce 128-D Presto embeddings to n_components via PCA."""
    from sklearn.decomposition import PCA
    import joblib

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum()
    log.info("PCA: 128-D -> %d-D (%.1f%% variance explained)",
             n_components, 100 * var_explained)

    pca_path = OUT_DIR / "presto_pca.joblib"
    joblib.dump(pca, pca_path)
    log.info("PCA model saved: %s", pca_path)

    return reduced


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Extract Presto embeddings for Guna Division")
    p.add_argument("--tile", type=str, default=None,
                   help="Filter to single MGRS tile")
    p.add_argument("--pca-dims", type=int, default=16,
                   help="PCA output dimensions (default 16)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Inference batch size")
    args = p.parse_args()

    t_start = time.monotonic()

    # 1. Load centroids
    centroids = load_centroids()

    # 2. Build HLS index
    hls_index = build_hls_index(tile_filter=args.tile)

    # 3. Extract spectral values at centroids
    spectral_table = build_spectral_table(hls_index, centroids)

    # 4. Load context data
    terrain_cache = load_terrain_cache()
    weather_cache = load_weather_cache()

    # 5. Build Presto input tensors
    rows, x, latlons, masks, months = build_presto_inputs(
        spectral_table, terrain_cache, weather_cache, centroids)

    if len(rows) == 0:
        log.error("No valid Presto inputs. Check HLS data and centroids.")
        sys.exit(1)

    # 6. Run Presto
    encoder = load_presto()
    embeddings = run_presto_inference(
        encoder, x, latlons, masks, months, batch_size=args.batch_size)

    # 7. Save raw 128-D embeddings
    import pandas as pd

    emb_cols = [f"presto_{i:02d}" for i in range(128)]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df["NEW_No_"] = [r[0] for r in rows]
    df["date"] = [r[1] for r in rows]

    raw_path = OUT_DIR / "presto_embeddings_raw.parquet"
    df.to_parquet(raw_path, index=False)
    log.info("Raw embeddings: %s (%d rows)", raw_path, len(df))

    # 8. Apply PCA
    reduced = apply_pca(embeddings, n_components=args.pca_dims)
    pca_cols = [f"presto_pc{i}" for i in range(args.pca_dims)]
    df_pca = pd.DataFrame(reduced, columns=pca_cols)
    df_pca["NEW_No_"] = [r[0] for r in rows]
    df_pca["date"] = [r[1] for r in rows]

    pca_path = OUT_DIR / "presto_embeddings.parquet"
    df_pca.to_parquet(pca_path, index=False)
    log.info("PCA embeddings: %s (%d rows, %d dims)",
             pca_path, len(df_pca), args.pca_dims)

    elapsed = time.monotonic() - t_start
    log.info("=== Done in %.1f min ===", elapsed / 60)


if __name__ == "__main__":
    main()
