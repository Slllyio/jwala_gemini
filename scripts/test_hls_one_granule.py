"""Quick test: download one HLS S30 granule (6 bands) with streaming AOI crop."""
import os, sys, time
from pathlib import Path

# Fix PROJ before everything
import pyproj
proj_data = str(Path(pyproj.datadir.get_data_dir()))
os.environ["PROJ_DATA"] = proj_data
os.environ["PROJ_LIB"]  = proj_data
os.environ.pop("GDAL_DATA", None)

os.environ.setdefault("EARTHDATA_USERNAME", "akshayr11@gmail.com")
os.environ.setdefault("EARTHDATA_PASSWORD", "40983#Akki")

import earthaccess, rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds as wfb

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
GUNA  = (76.75, 23.85, 77.52, 25.15)

earthaccess.login(strategy="netrc")

# Search for a granule with confirmed fire activity
results = earthaccess.search_data(
    short_name="HLSS30", cloud_hosted=True,
    temporal=("2024-04-01", "2024-04-30"),
    bounding_box=GUNA, cloud_cover=(0, 15),
)
print(f"Found {len(results)} granules for April 2024 cc<=15%")
if not results:
    print("No results — exiting")
    sys.exit(1)

# Pick the first one
g = results[0]
ur = g["umm"]["GranuleUR"]
acq = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
tile = next((p for p in ur.split(".") if p.startswith("T") and len(p) == 6), "UNK")
print(f"Using: {ur}  tile={tile}  date={acq}")

out_base = (
    Path("data_lake/satellite_imagery/hls_s30")
    / tile / acq[:4] / acq[5:7] / acq[8:10]
)
out_base.mkdir(parents=True, exist_ok=True)

t0 = time.time()
file_objs = earthaccess.open([g])
print(f"Opened {len(file_objs)} file objects in {time.time()-t0:.1f}s")

total_bytes = 0
for band in BANDS:
    fo = next((f for f in file_objs if f".{band}." in str(f)), None)
    if fo is None:
        print(f"  {band}: NOT FOUND")
        continue

    # Derive filename
    fo_str = str(fo)
    fname = fo_str.replace("\\", "/").split("/")[-1]
    if not fname.endswith(".tif"):
        fname = f"{ur}.{band}.tif"
    out_path = out_base / fname

    if out_path.exists():
        print(f"  {band}: already exists ({out_path.stat().st_size // 1024} KB)")
        total_bytes += out_path.stat().st_size
        continue

    t1 = time.time()
    try:
        with rasterio.open(fo) as src:
            tb = src.bounds
            tr = Transformer.from_crs("EPSG:4326", src.crs.to_wkt(), always_xy=True)
            x0, y0 = tr.transform(GUNA[0], GUNA[1])
            x1, y1 = tr.transform(GUNA[2], GUNA[3])
            cx0 = max(x0, tb.left);  cx1 = min(x1, tb.right)
            cy0 = max(y0, tb.bottom); cy1 = min(y1, tb.top)

            if cx1 <= cx0 or cy1 <= cy0:
                print(f"  {band}: no overlap with Guna bbox")
                continue

            win  = wfb(cx0, cy0, cx1, cy1, src.transform)
            data = src.read(1, window=win)
            prof = src.profile.copy()
            prof.update({
                "height"    : data.shape[0],
                "width"     : data.shape[1],
                "transform" : src.window_transform(win),
                "compress"  : "lzw",
                "tiled"     : True,
                "blockxsize": 256,
                "blockysize": 256,
                "count"     : 1,
            })

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data, 1)

        sz = out_path.stat().st_size
        total_bytes += sz
        print(f"  {band}: {data.shape[1]}x{data.shape[0]}px  {sz//1024} KB  ({time.time()-t1:.1f}s)")

    except Exception as exc:
        print(f"  {band}: ERROR — {exc}")

print()
print(f"=== 6-band scene total: {total_bytes/1024/1024:.2f} MB in {time.time()-t0:.1f}s ===")
print(f"Output directory: {out_base}")
for f in sorted(out_base.iterdir()):
    print(f"  {f.name}  {f.stat().st_size//1024} KB")
