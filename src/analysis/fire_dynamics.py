"""
Fire Dynamics and Spread Statistics (FEDS methodology)
=======================================================
Derives quantitative fire behaviour metrics from sequential VIIRS point
detections — replicating the Fire Event Data Suite (FEDS) approach described
in the PDF §"Analysis of Fire Dynamics and Spread Statistics".

Key outputs
-----------
FireEvent
  Clusters VIIRS points into discrete fire objects using DBSCAN spatial
  search, then delineates each object's boundary via an alpha hull
  (concave hull approximation).  Sequential perimeters are compared to
  derive:

    CROS — Coarse Rate of Spread (km² / km / h)
        CROS = [Area(P_{k+1}) - Area(P_k)] / [fperim(P_k) × Δt]

    spread_direction — bearing from ignition centroid to new-pixel centroid

    level_set_evolution — fire front propagation using the WRF-Fire
        level-set PDE:  ∂φ/∂t + R_f |∇φ| = 0

Usage::

    from src.analysis.fire_dynamics import FireEventTracker

    tracker = FireEventTracker(alpha=0.015)  # alpha hull tightness
    tracker.add_snapshot(gdf_viirs_t1, timestamp=datetime(2025,3,1,1,30))
    tracker.add_snapshot(gdf_viirs_t2, timestamp=datetime(2025,3,1,13,30))

    events = tracker.cluster_events()
    stats  = tracker.compute_spread_stats()
    perims = tracker.export_perimeters_geojson("outputs/fire_perimeters.geojson")
"""

from __future__ import annotations

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Hull (concave hull approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_hull(points: np.ndarray, alpha: float) -> "shapely.geometry.MultiPolygon":
    """
    Compute an alpha-hull of *points* (N×2 array of [lon, lat]).

    Alpha hull is preferred over convex hull because it can represent the
    concave "fingers" and unburned islands characteristic of wildfire fronts.

    Parameters
    ----------
    points : (N, 2) array of [longitude, latitude] in WGS-84 decimal degrees.
    alpha  : Controls concavity (smaller = more concave).  ~0.01–0.05 works
             well for fire perimeters at 375 m VIIRS resolution.

    Returns
    -------
    Shapely MultiPolygon (in EPSG:4326).
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiLineString, MultiPolygon, Polygon
    from shapely.ops import cascaded_union, polygonize

    if len(points) < 4:
        # Fall back to convex hull for very small clusters
        from shapely.geometry import MultiPoint
        return MultiPoint([tuple(p) for p in points]).convex_hull

    tri = Delaunay(points)
    edge_set: set = set()

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        # Circumradius of triangle
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2.0
        area = max(math.sqrt(abs(s * (s-a) * (s-b) * (s-c))), 1e-10)
        circumR = a * b * c / (4.0 * area)

        if circumR < 1.0 / alpha:
            for edge in [(ia, ib), (ib, ic), (ic, ia)]:
                key = tuple(sorted(edge))
                if key in edge_set:
                    edge_set.discard(key)
                else:
                    edge_set.add(key)

    edges = [
        (points[i], points[j])
        for i, j in edge_set
    ]
    m = MultiLineString([(tuple(a), tuple(b)) for a, b in edges])
    polys = list(polygonize(m))
    if not polys:
        from shapely.geometry import MultiPoint
        return MultiPoint([tuple(p) for p in points]).convex_hull
    return cascaded_union(polys)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FirePerimeter:
    """Single time-stamped perimeter of a fire event."""
    timestamp: datetime
    geometry: "shapely.geometry.base.BaseGeometry"
    farea_km2: float        # total area within perimeter (km²)
    fperim_km: float        # perimeter length (km)
    n_pixels: int           # number of VIIRS detections in this snapshot
    mean_frp_mw: float      # mean Fire Radiative Power (MW)
    centroid_lon: float     # centroid longitude (WGS-84)
    centroid_lat: float     # centroid latitude  (WGS-84)


@dataclass
class SpreadStats:
    """
    Derived fire behaviour statistics between two consecutive perimeters.

    CROS formula (PDF §"Mathematical Derivation of Spread Statistics"):
        CROS = [Area(P_{k+1}) - Area(P_k)] / [fperim(P_k) × Δt]

    where Δt is the inter-overpass interval in hours (nominally 12 h for
    the 01:30 / 13:30 VIIRS overpasses).
    """
    dt_hours: float                # time between perimeters
    area_change_km2: float         # Area(P_{k+1}) - Area(P_k)
    cros_km_h: float               # Coarse Rate of Spread (km/h)
    spread_bearing_deg: float      # bearing from ignition to new centroid (°N)
    new_pixels: int                # n_newpixels in the later perimeter
    flinelen_km: float             # estimated active fire-line length (km)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class FireEventTracker:
    """
    Tracks a single fire event across multiple satellite overpasses and
    computes FEDS-derived spread statistics.

    Parameters
    ----------
    alpha      : Alpha-hull parameter (tightness); ~0.015 typical for VIIRS.
    eps_deg    : DBSCAN spatial search radius in decimal degrees (~0.01° ≈ 1 km).
    min_pixels : Minimum VIIRS detections to form a cluster.
    """

    def __init__(
        self,
        alpha: float = 0.015,
        eps_deg: float = 0.01,
        min_pixels: int = 3,
    ) -> None:
        self.alpha = alpha
        self.eps_deg = eps_deg
        self.min_pixels = min_pixels
        self._snapshots: List[Tuple[datetime, "geopandas.GeoDataFrame"]] = []
        self.perimeters: List[FirePerimeter] = []
        self.spread_stats: List[SpreadStats] = []

    # ── data ingestion ────────────────────────────────────────────────────────

    def add_snapshot(
        self,
        gdf: "geopandas.GeoDataFrame",
        timestamp: datetime,
    ) -> None:
        """
        Register a VIIRS detection GeoDataFrame for a specific overpass time.

        Parameters
        ----------
        gdf       : GeoDataFrame with 'geometry' (Point), 'frp' columns.
        timestamp : Satellite overpass UTC datetime.
        """
        self._snapshots.append((timestamp, gdf))
        log.info(
            "Snapshot added: %s  (%d detections)", timestamp.isoformat(), len(gdf)
        )

    # ── perimeter computation ─────────────────────────────────────────────────

    def _compute_perimeter(
        self,
        gdf: "geopandas.GeoDataFrame",
        timestamp: datetime,
    ) -> Optional[FirePerimeter]:
        """
        Build an alpha-hull perimeter from VIIRS point detections and
        compute FEDS vector attributes.
        """
        import geopandas as gpd

        if len(gdf) < self.min_pixels:
            log.debug(
                "Snapshot %s: only %d pixels — below min_pixels=%d.",
                timestamp, len(gdf), self.min_pixels,
            )
            return None

        coords = np.array([[g.x, g.y] for g in gdf.geometry])
        hull = _alpha_hull(coords, self.alpha)

        # Project to UTM zone 44N (EPSG:32644) for metric calculations
        perim_gdf = gpd.GeoDataFrame({"geometry": [hull]}, crs="EPSG:4326")
        perim_utm = perim_gdf.to_crs("EPSG:32644")
        geom_utm  = perim_utm.geometry.iloc[0]

        farea_km2  = geom_utm.area / 1e6
        fperim_km  = geom_utm.length / 1e3

        centroid = hull.centroid
        mean_frp = float(gdf["frp"].mean()) if "frp" in gdf.columns else 0.0

        return FirePerimeter(
            timestamp=timestamp,
            geometry=hull,
            farea_km2=farea_km2,
            fperim_km=fperim_km,
            n_pixels=len(gdf),
            mean_frp_mw=mean_frp,
            centroid_lon=centroid.x,
            centroid_lat=centroid.y,
        )

    def build_perimeters(self) -> List[FirePerimeter]:
        """
        Compute perimeters for all registered snapshots (sorted by time).
        """
        self.perimeters = []
        for ts, gdf in sorted(self._snapshots, key=lambda x: x[0]):
            p = self._compute_perimeter(gdf, ts)
            if p is not None:
                self.perimeters.append(p)
        log.info("Built %d perimeters.", len(self.perimeters))
        return self.perimeters

    # ── spread statistics ─────────────────────────────────────────────────────

    def compute_spread_stats(self) -> List[SpreadStats]:
        """
        Derive CROS and directional spread statistics for each consecutive
        perimeter pair.

        CROS = [Area(P_{k+1}) - Area(P_k)] / [fperim(P_k) × Δt]

        where Δt is in hours (nominal 12 h for VIIRS 01:30/13:30 overpasses).
        """
        if len(self.perimeters) < 2:
            self.build_perimeters()

        self.spread_stats = []

        for i in range(len(self.perimeters) - 1):
            pk  = self.perimeters[i]
            pk1 = self.perimeters[i + 1]

            dt_sec  = (pk1.timestamp - pk.timestamp).total_seconds()
            dt_h    = dt_sec / 3600.0
            if dt_h <= 0:
                continue

            area_delta = pk1.farea_km2 - pk.farea_km2

            # Coarse Rate of Spread (perimeter-normalised, km/h)
            cros = (
                area_delta / (pk.fperim_km * dt_h)
                if pk.fperim_km > 0
                else 0.0
            )

            # Spread direction: bearing from previous centroid to new centroid
            bearing = _bearing(
                pk.centroid_lon, pk.centroid_lat,
                pk1.centroid_lon, pk1.centroid_lat,
            )

            # Active fire-line length proxy: perimeter change normalised by 2
            # (new perimeter adds new front length on both sides of growth)
            flinelen = abs(pk1.fperim_km - pk.fperim_km) / 2.0

            stats = SpreadStats(
                dt_hours=round(dt_h, 2),
                area_change_km2=round(area_delta, 4),
                cros_km_h=round(cros, 5),
                spread_bearing_deg=round(bearing, 1),
                new_pixels=pk1.n_pixels - pk.n_pixels,
                flinelen_km=round(flinelen, 3),
            )
            self.spread_stats.append(stats)
            log.info(
                "  Step %d→%d: Δt=%.1f h, CROS=%.4f km/h, bearing=%.0f°",
                i, i+1, dt_h, cros, bearing,
            )

        return self.spread_stats

    # ── level-set evolution (WRF-Fire PDE) ───────────────────────────────────

    @staticmethod
    def level_set_advance(
        phi: np.ndarray,
        R_f: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        """
        Advance the fire front by one time step using the WRF-Fire level-set
        upwind finite-difference scheme.

        PDE (PDF §"Mathematical Derivation"):
            ∂φ/∂t + R_f |∇φ| = 0

        where φ(x,y,t) is the level-set function (φ < 0 = burned,
        φ > 0 = unburned), R_f is the local rate of spread (m/s).

        Parameters
        ----------
        phi  : (H, W) ndarray — current level-set field.
        R_f  : (H, W) ndarray — local fire rate of spread (m/s).
        dt   : Time step (seconds).
        dx   : Grid spacing in x-direction (metres).
        dy   : Grid spacing in y-direction (metres).

        Returns
        -------
        Updated φ field after one time step.
        """
        # Upwind gradients (Godunov scheme)
        phi_x_p = np.roll(phi, -1, axis=1) - phi   # forward  x
        phi_x_m = phi - np.roll(phi,  1, axis=1)   # backward x
        phi_y_p = np.roll(phi, -1, axis=0) - phi   # forward  y
        phi_y_m = phi - np.roll(phi,  1, axis=0)   # backward y

        # Godunov Hamiltonian: ∂φ/∂t = -R_f |∇φ|
        grad_phi = np.sqrt(
            np.maximum(np.maximum(phi_x_m / dx, 0) ** 2,
                       np.minimum(phi_x_p / dx, 0) ** 2)
            + np.maximum(np.maximum(phi_y_m / dy, 0) ** 2,
                         np.minimum(phi_y_p / dy, 0) ** 2)
        )
        return phi - dt * R_f * grad_phi

    # ── export ────────────────────────────────────────────────────────────────

    def export_perimeters_geojson(self, out_path: Path) -> Path:
        """
        Write all perimeters to a GeoJSON FeatureCollection.
        Each feature carries FEDS vector attributes as properties.
        """
        import geopandas as gpd

        if not self.perimeters:
            self.build_perimeters()

        records = []
        for p in self.perimeters:
            records.append({
                "timestamp": p.timestamp.isoformat(),
                "farea_km2": p.farea_km2,
                "fperim_km": p.fperim_km,
                "n_pixels": p.n_pixels,
                "mean_frp_mw": p.mean_frp_mw,
                "centroid_lon": p.centroid_lon,
                "centroid_lat": p.centroid_lat,
                "geometry": p.geometry,
            })

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON")
        log.info("Perimeters exported → %s", out_path)
        return out_path

    def export_spread_stats_json(self, out_path: Path) -> Path:
        """Write spread statistics to JSON."""
        if not self.spread_stats:
            self.compute_spread_stats()

        data = [
            {
                "step": i,
                "dt_hours": s.dt_hours,
                "area_change_km2": s.area_change_km2,
                "cros_km_h": s.cros_km_h,
                "spread_bearing_deg": s.spread_bearing_deg,
                "new_pixels": s.new_pixels,
                "flinelen_km": s.flinelen_km,
            }
            for i, s in enumerate(self.spread_stats)
        ]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        log.info("Spread stats exported → %s", out_path)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Compute the forward azimuth (meteorological bearing, degrees clockwise
    from North) between two WGS-84 coordinate pairs.
    """
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def cluster_viirs_to_events(
    gdf: "geopandas.GeoDataFrame",
    eps_deg: float = 0.01,
    min_pixels: int = 3,
) -> "geopandas.GeoDataFrame":
    """
    Spatially cluster VIIRS point detections into discrete fire events using
    DBSCAN, mirroring the FEDS "spatial and temporal search criteria".

    Adds a 'fireid' column (−1 = noise / isolated pixels).

    Parameters
    ----------
    gdf        : GeoDataFrame of VIIRS point detections (EPSG:4326).
    eps_deg    : DBSCAN neighbourhood radius in decimal degrees.
    min_pixels : Minimum cluster size.

    Returns
    -------
    GeoDataFrame with 'fireid' column.
    """
    from sklearn.cluster import DBSCAN

    coords = np.array([[g.x, g.y] for g in gdf.geometry])
    labels = DBSCAN(eps=eps_deg, min_samples=min_pixels).fit_predict(coords)
    gdf = gdf.copy()
    gdf["fireid"] = labels
    n_events = (labels >= 0).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log.info(
        "Clustered %d detections → %d events (%d noise pixels).",
        len(gdf), n_clusters, (labels == -1).sum(),
    )
    return gdf
