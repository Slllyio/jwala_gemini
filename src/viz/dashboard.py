"""
Interactive Visualization Dashboard
=====================================
Creates a Folium-based interactive HTML map showing:
  1. Fatehgarh AOI boundary
  2. Year-by-year forest change detection layers
  3. Future change risk heatmap

Usage:
    python src/viz/dashboard.py \
        --config config.yaml \
        --change_maps outputs/change_map_2022_2023.tif \
        --risk_map outputs/risk_map.tif \
        --output outputs/dashboard.html
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import yaml
import logging
import argparse
import glob
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from pathlib import Path
from typing import Optional, List
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
from folium import LayerControl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import io

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def raster_to_png_overlay(tif_path: str, colormap_name: str = "Reds",
                            alpha: float = 0.7) -> tuple:
    """
    Convert a GeoTIFF to a base64 PNG for Folium image overlay.

    Returns:
        (png_base64_str, bounds_latlon) where bounds = [[south, west], [north, east]]
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        bounds = src.bounds
        crs = src.crs

    # Reproject bounds to WGS84 if needed
    if str(crs) != "EPSG:4326":
        from rasterio.crs import CRS
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Normalize data
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-6:
        norm_data = np.zeros_like(data)
    else:
        norm_data = (data - vmin) / (vmax - vmin)

    # Apply colormap
    colormap = cm.get_cmap(colormap_name)
    rgba = colormap(norm_data)
    rgba[..., 3] = np.where(data > 0, alpha, 0)  # Transparent where no data

    # Save to in-memory PNG
    buf = io.BytesIO()
    plt.figure(figsize=(data.shape[1] / 100, data.shape[0] / 100), dpi=100)
    plt.imshow(rgba, aspect="auto")
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("utf-8")

    bounds_latlon = [[south, west], [north, east]]
    return f"data:image/png;base64,{png_b64}", bounds_latlon


def build_dashboard(
    cfg: dict,
    change_map_paths: List[str],
    risk_map_path: Optional[str],
    output_path: str,
):
    """
    Build and save the interactive Folium dashboard.

    Args:
        cfg: Config dict
        change_map_paths: List of change map GeoTIF paths (one per year)
        risk_map_path: Future risk map GeoTIFF path (optional)
        output_path: Output HTML file path
    """
    viz = cfg["visualization"]
    center_lat = viz["center_lat"]
    center_lon = viz["center_lon"]
    zoom_start = viz["zoom_start"]

    # ── Create base map ──────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
    )

    # Add satellite basemap
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # ── AOI boundary ─────────────────────────────────────────────────────────
    # Try to add the AOI polygon if we have geemap/ee available
    try:
        import ee
        ee.Initialize(project="ee-akshayr1")
        fc = ee.FeatureCollection(cfg["gee"]["aoi_asset"])
        aoi_geojson = fc.getInfo()

        folium.GeoJson(
            aoi_geojson,
            name="AOI: Fatehgarh Sahib",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "#00FF88",
                "weight": 3,
                "dashArray": "8 5",
            },
            tooltip=folium.GeoJsonTooltip(fields=[], aliases=[]),
        ).add_to(m)
        log.info("✅ AOI boundary added from GEE")
    except Exception as e:
        log.warning(f"Could not load AOI from GEE: {e}. Using center marker.")
        folium.Circle(
            location=[center_lat, center_lon],
            radius=15000,
            color="#00FF88",
            fill=False,
            weight=3,
            popup="Fatehgarh Sahib AOI (approximate)",
        ).add_to(m)

    # ── Change detection layers ───────────────────────────────────────────────
    log.info(f"Adding {len(change_map_paths)} change detection layers...")
    for tif_path in sorted(change_map_paths):
        year_label = Path(tif_path).stem.replace("change_map_", "").replace("_", " → ")
        log.info(f"  Processing: {year_label}")

        try:
            png_url, bounds = raster_to_png_overlay(tif_path,
                                                      colormap_name=viz["change_colormap"],
                                                      alpha=0.65)
            folium.raster_layers.ImageOverlay(
                image=png_url,
                bounds=bounds,
                opacity=0.7,
                name=f"🌳 Change: {year_label}",
                show=False,
            ).add_to(m)
        except Exception as e:
            log.warning(f"  Could not add layer for {tif_path}: {e}")

    # ── Risk prediction heatmap ───────────────────────────────────────────────
    if risk_map_path and Path(risk_map_path).exists():
        log.info(f"Adding risk prediction layer: {risk_map_path}")
        try:
            # Sample risk map for heatmap (downsampled for performance)
            with rasterio.open(risk_map_path) as src:
                risk_data = src.read(1).astype(np.float32)
                transform = src.transform
                height, width = risk_data.shape

            # Downsample for heatmap rendering
            step = max(1, min(height, width) // 200)
            heat_data = []
            for r in range(0, height, step):
                for c in range(0, width, step):
                    risk_val = float(risk_data[r, c])
                    if risk_val > 0.1:  # Only show significant risk
                        # Convert pixel to lat/lon
                        lon, lat = transform * (c + 0.5, r + 0.5)
                        heat_data.append([lat, lon, risk_val])

            if heat_data:
                HeatMap(
                    heat_data,
                    name="🚨 Future Risk Heatmap (T+1)",
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=15,
                    blur=20,
                    gradient={
                        "0.0": "#00FF00",
                        "0.4": "#FFFF00",
                        "0.7": "#FF8C00",
                        "1.0": "#FF0000",
                    },
                    show=True,
                ).add_to(m)
                log.info(f"  ✅ Risk heatmap added ({len(heat_data)} datapoints)")

            # Also add as image overlay
            png_url, bounds = raster_to_png_overlay(risk_map_path,
                                                      colormap_name=viz["risk_colormap"],
                                                      alpha=0.7)
            folium.raster_layers.ImageOverlay(
                image=png_url,
                bounds=bounds,
                opacity=0.6,
                name=f"📊 Risk Map (probability overlay)",
                show=False,
            ).add_to(m)

        except Exception as e:
            log.warning(f"Could not add risk layer: {e}")

    # ── Info panel ───────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; width: 260px;
        background: rgba(10, 10, 10, 0.85);
        border: 2px solid #00FF88;
        border-radius: 12px;
        padding: 16px;
        z-index: 9999;
        font-family: 'Segoe UI', sans-serif;
        color: #FFFFFF;
        backdrop-filter: blur(8px);
    ">
        <div style="font-size: 14px; font-weight: 700; color: #00FF88; margin-bottom: 12px;">
            🌍 Prithvi Forest Monitor
        </div>
        <div style="font-size: 11px; color: #CCCCCC; margin-bottom: 10px;">
            AOI: Fatehgarh Sahib, Punjab, India
        </div>
        <hr style="border-color: #333; margin: 8px 0;">
        <div style="font-size: 11px; line-height: 1.8;">
            <span style="color: #FF4444;">■</span> Forest Loss Detected<br>
            <span style="color: #FF8C00;">■</span> High Risk (Predicted)<br>
            <span style="color: #FFFF00;">■</span> Medium Risk<br>
            <span style="color: #00FF00;">■</span> Low Risk<br>
            <span style="color: #00FF88;">⬜</span> AOI Boundary<br>
        </div>
        <hr style="border-color: #333; margin: 8px 0;">
        <div style="font-size: 10px; color: #888;">
            Model: IBM/NASA Prithvi-100M<br>
            Data: HLS (Harmonized Landsat Sentinel-2)<br>
            Labels: Hansen GFC 2023<br>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ─────────────────────────────────────────────────────────
    LayerControl(collapsed=False).add_to(m)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    log.info(f"\n✅ Dashboard saved: {output_path}")
    log.info(f"   Open in browser: file:///{Path(output_path).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--change_maps", nargs="*", default=None,
                        help="Change map GeoTIFF paths (auto-detect if not specified)")
    parser.add_argument("--risk_map", default=None, help="Risk map GeoTIFF path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["output_dir"]

    # Auto-detect change maps
    change_map_paths = args.change_maps
    if not change_map_paths:
        change_map_paths = sorted(glob.glob(os.path.join(out_dir, "change_map_*.tif")))
        # Exclude probability maps
        change_map_paths = [p for p in change_map_paths if "_prob" not in p]
        log.info(f"Auto-detected {len(change_map_paths)} change maps")

    # Auto-detect risk map
    risk_map_path = args.risk_map
    if not risk_map_path:
        candidates = glob.glob(os.path.join(out_dir, "risk_map*.tif"))
        candidates = [p for p in candidates if "_binary" not in p]
        risk_map_path = candidates[0] if candidates else None

    output_path = args.output or cfg["visualization"]["dashboard_out"]

    build_dashboard(cfg, change_map_paths, risk_map_path, output_path)
