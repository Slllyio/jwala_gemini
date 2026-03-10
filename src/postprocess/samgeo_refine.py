"""
src/postprocess/samgeo_refine.py
================================
SamGeo Fire Mask Refinement Pipeline

Takes VanAagni fire probability rasters (GeoTIFF) and refines them into
clean, vectorized burn-scar polygons using Meta's Segment Anything Model (SAM).

Pipeline:
    1. Load VanAagni fire probability raster (30m GeoTIFF)
    2. Threshold → binary fire mask
    3. SAM refines boundaries to natural edges (rivers, roads, ridgelines)
    4. Export as GeoJSON / Shapefile / GeoPackage for:
       - Dashboard overlay
       - dNBR severity computation (pre/post-fire NBR differencing)
       - Patrol route planning

Usage:
    # Single raster
    python src/postprocess/samgeo_refine.py \
        --input outputs/wfire/predictions/fire_prob_20260305.tif \
        --output outputs/wfire/polygons/fire_polygons_20260305.geojson

    # Batch mode (all rasters in a directory)
    python src/postprocess/samgeo_refine.py \
        --input-dir outputs/wfire/predictions/ \
        --output-dir outputs/wfire/polygons/ \
        --threshold 0.5

    # As Python API
    from src.postprocess.samgeo_refine import SamGeoFireRefiner
    refiner = SamGeoFireRefiner(model_type="vit_h", threshold=0.5)
    polygons_gdf = refiner.refine("fire_prob.tif", "fire_polygons.geojson")

Requirements:
    pip install "segment-geospatial[samgeo]"
"""

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

log = logging.getLogger(__name__)


class SamGeoFireRefiner:
    """
    Refine VanAagni fire probability rasters into clean polygons using SAM.

    Parameters
    ----------
    model_type : str
        SAM model variant: 'vit_h' (best), 'vit_l' (fast), 'vit_b' (fastest).
    checkpoint : str or None
        Path to SAM checkpoint. If None, auto-downloads from Meta.
    threshold : float
        Fire probability threshold for binary mask (default=0.5).
    min_area_ha : float
        Minimum polygon area in hectares to keep (filters noise).
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance in meters.
    device : str
        Torch device ('cuda', 'cpu', or 'auto').
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: Optional[str] = None,
        threshold: float = 0.5,
        min_area_ha: float = 0.1,
        simplify_tolerance: float = 15.0,
        device: str = "auto",
    ):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.min_area_ha = min_area_ha
        self.simplify_tolerance = simplify_tolerance
        self.device = device
        self._sam = None

    def _init_sam(self):
        """Lazy-initialize SAM model on first use."""
        if self._sam is not None:
            return

        from samgeo import SamGeo

        log.info(f"Initializing SAM model: {self.model_type}")

        kwargs = {"model_type": self.model_type}
        if self.checkpoint:
            kwargs["checkpoint"] = self.checkpoint
        if self.device != "auto":
            kwargs["device"] = self.device

        self._sam = SamGeo(**kwargs)
        log.info("SAM model loaded successfully")

    def refine(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        return_gdf: bool = True,
    ):
        """
        Refine a fire probability raster into clean polygons.

        Parameters
        ----------
        input_path : str or Path
            Path to VanAagni fire probability GeoTIFF.
        output_path : str or Path, optional
            Output path for vector file (.geojson, .shp, .gpkg).
            If None, replaces .tif extension with .geojson.
        return_gdf : bool
            If True, return a GeoDataFrame of the polygons.

        Returns
        -------
        geopandas.GeoDataFrame or None
            Polygons with columns: [geometry, fire_prob_mean, area_ha, confidence]
        """
        import rasterio
        from rasterio.features import shapes
        from shapely.geometry import shape

        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".geojson")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Processing: {input_path}")

        # -- Step 1: Load fire probability raster ----------------------------
        with rasterio.open(input_path) as src:
            fire_prob = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()

        log.info(
            f"  Raster shape: {fire_prob.shape}, "
            f"range: [{fire_prob.min():.3f}, {fire_prob.max():.3f}]"
        )

        # -- Step 2: Binary threshold ----------------------------------------
        fire_mask = (fire_prob >= self.threshold).astype(np.uint8)
        n_fire_pixels = fire_mask.sum()
        pixel_area_ha = abs(transform.a * transform.e) / 10_000  # m² to ha
        total_fire_ha = n_fire_pixels * pixel_area_ha

        log.info(
            f"  Fire pixels: {n_fire_pixels:,} "
            f"({total_fire_ha:.1f} ha at threshold={self.threshold})"
        )

        if n_fire_pixels == 0:
            log.warning("  No fire detected — skipping SAM refinement")
            return self._empty_gdf(crs) if return_gdf else None

        # -- Step 3: SAM refinement ------------------------------------------
        try:
            self._init_sam()
            sam_polygons = self._sam_segment(
                input_path, fire_mask, transform, crs, profile
            )
        except Exception as e:
            log.warning(f"  SAM refinement failed ({e}), falling back to rasterio vectorization")
            sam_polygons = None

        # -- Step 4: Vectorize (SAM or fallback) ----------------------------
        if sam_polygons is not None:
            gdf = sam_polygons
            log.info(f"  SAM produced {len(gdf)} polygons")
        else:
            gdf = self._rasterio_vectorize(
                fire_mask, fire_prob, transform, crs
            )
            log.info(f"  Rasterio produced {len(gdf)} polygons")

        # -- Step 5: Filter and enrich --------------------------------------
        if len(gdf) > 0:
            gdf = self._filter_and_enrich(gdf, fire_prob, transform, crs)

        # -- Step 6: Save ---------------------------------------------------
        if len(gdf) > 0:
            ext = output_path.suffix.lower()
            if ext == ".geojson":
                gdf.to_file(output_path, driver="GeoJSON")
            elif ext == ".shp":
                gdf.to_file(output_path, driver="ESRI Shapefile")
            elif ext == ".gpkg":
                gdf.to_file(output_path, driver="GPKG", layer="fire_polygons")
            else:
                gdf.to_file(output_path, driver="GeoJSON")
            log.info(f"  Saved {len(gdf)} polygons to {output_path}")
        else:
            log.warning("  No polygons after filtering — output empty")

        return gdf if return_gdf else None

    def _sam_segment(self, input_path, fire_mask, transform, crs, profile):
        """
        Use SAM to segment the fire raster, guided by the binary mask.

        SAM operates on the raw raster image (it uses visual features to
        find natural boundaries). The fire_mask provides foreground points
        as prompts to guide SAM toward fire regions.
        """
        import geopandas as gpd
        import tempfile

        # Save the binary mask as a temporary GeoTIFF for SAM
        mask_path = Path(tempfile.mktemp(suffix="_fire_mask.tif"))
        with __import__('rasterio').open(mask_path, 'w', **{**profile, 'count': 1, 'dtype': 'uint8'}) as dst:
            dst.write(fire_mask, 1)

        try:
            # Generate SAM masks using the fire raster as image
            # and fire pixels as foreground prompts
            self._sam.set_image(str(input_path))

            # Use automatic mask generation on fire regions
            output_vector = str(mask_path.with_suffix(".geojson"))
            self._sam.generate(str(input_path), output=output_vector)

            if os.path.exists(output_vector):
                gdf = gpd.read_file(output_vector)
                # Keep only SAM segments that overlap with fire mask
                gdf = self._filter_by_fire_overlap(gdf, fire_mask, transform, crs)
                return gdf if len(gdf) > 0 else None
        except Exception as e:
            log.debug(f"SAM segmentation error: {e}")
            raise
        finally:
            # Cleanup temp files
            for f in [mask_path, mask_path.with_suffix(".geojson")]:
                if f.exists():
                    f.unlink()

        return None

    def _filter_by_fire_overlap(self, gdf, fire_mask, transform, crs):
        """Keep SAM segments that have >50% overlap with the fire mask."""
        from rasterio.features import geometry_mask
        import geopandas as gpd

        keep = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            try:
                mask = geometry_mask(
                    [geom], out_shape=fire_mask.shape,
                    transform=transform, invert=True
                )
                overlap = (mask & fire_mask.astype(bool)).sum()
                total = mask.sum()
                if total > 0 and overlap / total > 0.5:
                    keep.append(idx)
            except Exception:
                continue

        return gdf.loc[keep].copy() if keep else gpd.GeoDataFrame()

    def _rasterio_vectorize(self, fire_mask, fire_prob, transform, crs):
        """Fallback: vectorize fire mask using rasterio.features.shapes."""
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape

        results = list(shapes(fire_mask, mask=fire_mask > 0, transform=transform))

        if not results:
            return self._empty_gdf(crs)

        geometries = [shape(geom) for geom, val in results if val > 0]

        gdf = gpd.GeoDataFrame(
            {"geometry": geometries},
            crs=crs
        )
        return gdf

    def _filter_and_enrich(self, gdf, fire_prob, transform, crs):
        """Filter small polygons and add attributes."""
        import geopandas as gpd
        from rasterio.features import geometry_mask

        # Compute area in hectares
        if gdf.crs and gdf.crs.is_geographic:
            # Project to UTM for accurate area calculation
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf["area_ha"] = gdf_utm.geometry.area / 10_000
        else:
            gdf["area_ha"] = gdf.geometry.area / 10_000

        # Filter by minimum area
        before = len(gdf)
        gdf = gdf[gdf["area_ha"] >= self.min_area_ha].copy()
        log.info(f"  Filtered: {before} → {len(gdf)} polygons (min_area={self.min_area_ha} ha)")

        if len(gdf) == 0:
            return gdf

        # Compute mean fire probability per polygon
        mean_probs = []
        for _, row in gdf.iterrows():
            try:
                mask = geometry_mask(
                    [row.geometry], out_shape=fire_prob.shape,
                    transform=transform, invert=True
                )
                vals = fire_prob[mask]
                mean_probs.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            except Exception:
                mean_probs.append(0.0)

        gdf["fire_prob_mean"] = mean_probs

        # Confidence from mean probability
        gdf["confidence"] = gdf["fire_prob_mean"].apply(
            lambda p: "HIGH" if p > 0.75 else ("MEDIUM" if p > 0.5 else "LOW")
        )

        # Simplify geometry for smaller file size
        if self.simplify_tolerance > 0 and gdf.crs and not gdf.crs.is_geographic:
            gdf["geometry"] = gdf.geometry.simplify(self.simplify_tolerance)

        # Round numeric columns
        gdf["area_ha"] = gdf["area_ha"].round(2)
        gdf["fire_prob_mean"] = gdf["fire_prob_mean"].round(3)

        return gdf.reset_index(drop=True)

    @staticmethod
    def _empty_gdf(crs):
        """Return an empty GeoDataFrame with the expected schema."""
        import geopandas as gpd
        return gpd.GeoDataFrame(
            columns=["geometry", "fire_prob_mean", "area_ha", "confidence"],
            crs=crs
        )

    def batch_refine(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.tif",
    ):
        """
        Process all fire probability rasters in a directory.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing VanAagni fire probability GeoTIFFs.
        output_dir : str or Path
            Directory for output polygon files.
        pattern : str
            Glob pattern for input files.

        Returns
        -------
        dict
            {filename: GeoDataFrame} for each processed file.
        """
        import geopandas as gpd

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(input_dir.glob(pattern))
        log.info(f"Batch processing {len(tif_files)} rasters from {input_dir}")

        results = {}
        for tif in tif_files:
            out_name = tif.stem + ".geojson"
            out_path = output_dir / out_name
            try:
                gdf = self.refine(tif, out_path, return_gdf=True)
                results[tif.name] = gdf
                log.info(f"  ✓ {tif.name} → {len(gdf) if gdf is not None else 0} polygons")
            except Exception as e:
                log.error(f"  ✗ {tif.name} failed: {e}")
                results[tif.name] = None

        total = sum(len(g) for g in results.values() if g is not None)
        log.info(f"Batch complete: {len(results)} files, {total} total polygons")
        return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Refine VanAagni fire masks into clean polygons using SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python src/postprocess/samgeo_refine.py \\
      --input fire_prob.tif --output fire_polygons.geojson

  # Batch mode
  python src/postprocess/samgeo_refine.py \\
      --input-dir outputs/wfire/predictions/ \\
      --output-dir outputs/wfire/polygons/

  # Custom threshold + smaller SAM model
  python src/postprocess/samgeo_refine.py \\
      --input fire_prob.tif --threshold 0.3 --model vit_l
        """
    )

    parser.add_argument("--input", type=str, help="Input fire probability GeoTIFF")
    parser.add_argument("--output", type=str, help="Output vector file (.geojson/.shp/.gpkg)")
    parser.add_argument("--input-dir", type=str, help="Input directory for batch mode")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch mode")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fire probability threshold (default: 0.5)")
    parser.add_argument("--min-area", type=float, default=0.1,
                        help="Minimum polygon area in hectares (default: 0.1)")
    parser.add_argument("--model", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model variant (default: vit_h)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint (default: auto-download)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: cuda/cpu/auto")
    parser.add_argument("--simplify", type=float, default=15.0,
                        help="Polygon simplification tolerance in meters (default: 15)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    refiner = SamGeoFireRefiner(
        model_type=args.model,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        min_area_ha=args.min_area,
        simplify_tolerance=args.simplify,
        device=args.device,
    )

    if args.input_dir:
        # Batch mode
        output_dir = args.output_dir or str(Path(args.input_dir) / "polygons")
        refiner.batch_refine(args.input_dir, output_dir)
    elif args.input:
        # Single file
        gdf = refiner.refine(args.input, args.output)
        if gdf is not None and len(gdf) > 0:
            print(f"\n{'='*60}")
            print(f"  Fire Polygons Summary")
            print(f"{'='*60}")
            print(f"  Total polygons:     {len(gdf)}")
            print(f"  Total area:         {gdf['area_ha'].sum():.1f} ha")
            print(f"  Avg fire prob:      {gdf['fire_prob_mean'].mean():.3f}")
            print(f"  HIGH confidence:    {(gdf['confidence']=='HIGH').sum()}")
            print(f"  MEDIUM confidence:  {(gdf['confidence']=='MEDIUM').sum()}")
            print(f"  LOW confidence:     {(gdf['confidence']=='LOW').sum()}")
            print(f"{'='*60}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
