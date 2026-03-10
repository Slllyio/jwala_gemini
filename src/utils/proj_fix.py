"""
Centralised PROJ / GDAL environment fix.
=========================================
PostgreSQL (PostGIS) ships its own ``PROJ.db`` that shadows the newer one
bundled with the Python venv's rasterio / pyproj wheels on Windows.  This
module **must** be imported before any geospatial library (rasterio, pyproj,
geopandas, osgeo) to pin the correct data directory.

Usage — add this ONE line at the very top of any geo-aware script::

    import src.utils.proj_fix  # noqa: F401  (side-effect import)

Or, for scripts that manipulate ``sys.path`` themselves::

    from src.utils.proj_fix import setup_proj_env; setup_proj_env()
"""

from __future__ import annotations

import os
import warnings

_APPLIED = False


def setup_proj_env() -> str | None:
    """Pin PROJ_DATA / PROJ_LIB to the venv-bundled directory.

    Returns the directory path that was set, or ``None`` if neither
    rasterio nor pyproj could provide one.
    """
    global _APPLIED
    if _APPLIED:
        return os.environ.get("PROJ_DATA")

    # 1. Remove stale vars that PostgreSQL / other installers may have set
    os.environ.pop("GDAL_DATA", None)

    # 2. Prefer rasterio's bundled proj_data (version-matched to its GDAL)
    proj_dir: str | None = None
    try:
        import rasterio as _rio

        candidate = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
        if os.path.isdir(candidate):
            proj_dir = candidate
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    # 3. Fall back to pyproj's data directory
    if proj_dir is None:
        try:
            import pyproj as _pp

            proj_dir = _pp.datadir.get_data_dir()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    # 4. Apply
    if proj_dir:
        os.environ["PROJ_DATA"] = proj_dir  # PROJ >= 9
        os.environ["PROJ_LIB"] = proj_dir   # PROJ < 9 compat alias

    # 5. Suppress noisy PROJ version-mismatch warnings
    os.environ.setdefault("PROJ_DEBUG", "0")
    warnings.filterwarnings("ignore", message=".*PROJ.*")

    _APPLIED = True
    return proj_dir


# Auto-apply on import (side-effect import pattern)
setup_proj_env()
