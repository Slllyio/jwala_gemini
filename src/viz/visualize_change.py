"""
Change Detection Visualization
================================
Produces a publication-quality PNG showing:
  Left:   True-color composite (2023 HLS imagery)
  Center: Change probability heatmap
  Right:  Binary change mask overlaid on RGB

Usage:
    python src/viz/visualize_change.py \
        --rgb   data/raw/hls_2023.tif \
        --mask  outputs/change_map_2022_2023.tif \
        --prob  outputs/change_map_2022_2023_prob.tif \
        --out   outputs/change_viz_2022_2023.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")   # headless — no GUI required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────

def read_band(path: str, band_idx: int) -> np.ndarray:
    """Read a single raster band, returned as float32."""
    with rasterio.open(path) as src:
        return src.read(band_idx).astype(np.float32)


def read_rgb(path: str, red_idx=3, green_idx=2, blue_idx=1) -> np.ndarray:
    """
    Read RGB from HLS tif (Prithvi band order: B02,B03,B04,B8A,B11,B12).
    Bands: 1=B02(Blue), 2=B03(Green), 3=B04(Red), 4=B8A(NIR), 5=B11, 6=B12
    """
    with rasterio.open(path) as src:
        r = src.read(red_idx).astype(np.float32)
        g = src.read(green_idx).astype(np.float32)
        b = src.read(blue_idx).astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)   # (H, W, 3)

    # Percentile stretch for display (2–98%)
    for c in range(3):
        p2, p98 = np.percentile(rgb[..., c], (2, 98))
        rgb[..., c] = np.clip((rgb[..., c] - p2) / (p98 - p2 + 1e-6), 0, 1)

    return rgb


def make_overlay(rgb: np.ndarray, mask: np.ndarray,
                 color=(1.0, 0.15, 0.0), alpha=0.55) -> np.ndarray:
    """Blend red change pixels onto the RGB image."""
    overlay = rgb.copy()
    changed = mask > 0
    for c, v in enumerate(color):
        channel = overlay[..., c]
        channel[changed] = alpha * v + (1 - alpha) * channel[changed]
        overlay[..., c] = channel
    return overlay


# ── colormaps ──────────────────────────────────────────────────────────────

PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob",
    ["#0a0a0a", "#1a3a1a", "#2d7a2d", "#ffdd00", "#ff6600", "#ff0000"],
    N=256,
)


# ── main ───────────────────────────────────────────────────────────────────

def visualize(rgb_path: str, mask_path: str, prob_path: str, out_path: str):
    print(f"[INFO] Loading imagery: {rgb_path}")
    rgb = read_rgb(rgb_path)

    print(f"[INFO] Loading change mask: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    print(f"[INFO] Loading probability map: {prob_path}")
    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32) / 255.0   # normalise 0-1

    overlay = make_overlay(rgb, mask)

    # ── figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#0d1117")

    titles = [
        "Satellite (2023 HLS true-colour)",
        "Change Probability Heatmap",
        "Detected Forest Loss (red overlay)",
    ]
    panels = [rgb, prob, overlay]
    cmaps  = [None, PROB_CMAP, None]

    for ax, img, title, cmap in zip(axes, panels, titles, cmaps):
        ax.set_facecolor("#0d1117")
        if cmap:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        else:
            ax.imshow(np.clip(img, 0, 1), interpolation="bilinear")
        ax.set_title(title, color="#e6edf3", fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    # Colorbar for probability panel
    cbar_ax = fig.add_axes([0.385, 0.08, 0.23, 0.025])
    sm = plt.cm.ScalarMappable(cmap=PROB_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Change probability", color="#e6edf3", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="#e6edf3")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="#e6edf3", fontsize=8)

    # Legend for overlay panel
    patch = mpatches.Patch(color=(1.0, 0.15, 0.0), label="Forest loss detected")
    axes[2].legend(handles=[patch], loc="lower left",
                   facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=9)

    # Stats banner
    total = mask.size
    changed = int(mask.sum())
    area_ha = changed * 900 / 10_000
    pct = 100 * changed / total
    fig.text(0.5, 0.97,
             f"Fatehgarh Sahib AOI  ·  2022 → 2023  ·  "
             f"Changed pixels: {changed:,} ({pct:.1f}%)  ·  "
             f"Estimated loss: {area_ha:.0f} ha",
             ha="center", va="top", color="#58a6ff",
             fontsize=11, fontweight="bold")

    fig.text(0.5, 0.005,
             "Model: IBM/NASA Prithvi-100M  ·  Data: HLS (Harmonized Landsat Sentinel-2)  ·  Labels: Hansen GFC 2023",
             ha="center", va="bottom", color="#8b949e", fontsize=8)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.04)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] ✅ Saved visualization: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb",  default="data/raw/hls_2023.tif")
    parser.add_argument("--mask", default="outputs/change_map_2022_2023.tif")
    parser.add_argument("--prob", default="outputs/change_map_2022_2023_prob.tif")
    parser.add_argument("--out",  default="outputs/change_viz_2022_2023.png")
    args = parser.parse_args()
    visualize(args.rgb, args.mask, args.prob, args.out)
