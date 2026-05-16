"""Preprocess STORM dataset cyclone return-period wind speeds into a SQLite database.

Source: STORM dataset (Bloemendaal et al. 2020)
        https://www.science.org/doi/10.1126/sciadv.aba8275
Variable: 100-year return period maximum sustained wind speed (km/h)
Grid resolution: 0.1 degrees (native STORM resolution)
Output: data/storm_cyclone.db (SQLite, INTEGER keys via grid_key())

The STORM dataset ships as 4 NetCDF files, one per ocean basin:
  EP, NA, NI, SI, SP, WP (or combined).
Where basins overlap, take max wind speed. Cells not covered by any basin → 0.0
at runtime (absent from DB → _db_lookup returns 0.0).

Usage:
    # Place STORM NetCDF files at data/storm/
    python3 scripts/preprocess_storm_cyclone.py --src data/storm/ --out data/storm_cyclone.db

Requires: netCDF4 or xarray (pip install netCDF4 or xarray)

CRITICAL: This script imports grid_key from bor_risk.utils.  Do NOT use Python's
built-in round() for coordinate quantization.  round() uses banker's rounding
(round-half-to-even) which disagrees with grid_key() at .5 boundaries, causing
silent location drift between preprocessing and runtime lookup.
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bor_risk.utils import grid_key  # noqa: E402 — must use grid_key, not round()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grid (
    lat_key INTEGER NOT NULL,
    lon_key INTEGER NOT NULL,
    value   REAL    NOT NULL,
    PRIMARY KEY (lat_key, lon_key)
);
"""


def _is_valid(v_kmh: float) -> bool:
    return math.isfinite(v_kmh) and v_kmh >= 0.0


def _grid_key_array(values: np.ndarray, decimals: int) -> np.ndarray:
    factor = 10 ** decimals
    return (
        np.floor(np.abs(values) * factor + 0.5).astype(np.int32)
        * np.where(values >= 0, 1, -1).astype(np.int32)
    )


def _load_netcdf_basin(nc_path: Path) -> dict[tuple[int, int], float]:
    """Load one STORM NetCDF basin file. Returns {(lat_key, lon_key): v100_kmh}."""
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        try:
            import xarray as xr  # type: ignore
            ds = xr.open_dataset(str(nc_path))
            lats = ds.latitude.values
            lons = ds.longitude.values
            # Expect variable named 'wind' or similar; adapt to actual STORM var name
            var_name = next(v for v in ds.data_vars if "wind" in v.lower() or "vmax" in v.lower())
            values = ds[var_name].values  # shape: (lat, lon) or similar
            cells: dict[tuple[int, int], float] = {}
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    v_ms = float(values[i, j]) if values.ndim == 2 else float(values[0, i, j])
                    if not _is_valid(v_ms):
                        continue
                    v_kmh = v_ms * 3.6
                    key = (grid_key(float(lat), 1), grid_key(float(lon), 1))
                    cells[key] = max(cells.get(key, 0.0), v_kmh)
            return cells
        except ImportError:
            raise RuntimeError("Install netCDF4 or xarray: pip install netCDF4")

    dataset = nc.Dataset(str(nc_path))
    lats = dataset.variables["latitude"][:]
    lons = dataset.variables["longitude"][:]
    var_name = next(v for v in dataset.variables if "wind" in v.lower() or "vmax" in v.lower())
    values = dataset.variables[var_name][:]
    cells: dict[tuple[int, int], float] = {}
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            v_ms = float(values[i, j])
            if not _is_valid(v_ms):
                continue
            v_kmh = v_ms * 3.6
            key = (grid_key(float(lat), 1), grid_key(float(lon), 1))
            cells[key] = max(cells.get(key, 0.0), v_kmh)
    dataset.close()
    return cells


def _load_tif_basin(tif_path: Path) -> dict[tuple[int, int], float]:
    """Load one STORM fixed-return-period GeoTIFF.

    TIFF values are 100-year wind speeds in m/s; runtime stores km/h. Longitudes
    above 180 degrees are wrapped into the conventional -180..180 range used by
    supplier coordinates.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise RuntimeError("Install rasterio to preprocess GeoTIFF inputs") from e

    cells: dict[tuple[int, int], float] = {}
    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        transform = ds.transform
        for row_off in range(0, ds.height, 256):
            height = min(256, ds.height - row_off)
            window = Window(0, row_off, ds.width, height)
            arr = ds.read(1, window=window, masked=True)
            data = np.asarray(arr.filled(np.nan), dtype=np.float32)
            valid = np.isfinite(data) & (data > 0.0)
            if nodata is not None and math.isfinite(nodata):
                valid &= data != nodata
            if not np.any(valid):
                continue

            r_idx, c_idx = np.nonzero(valid)
            vals_kmh = data[r_idx, c_idx].astype(np.float64) * 3.6
            cols = c_idx.astype(np.float64) + window.col_off + 0.5
            rows = r_idx.astype(np.float64) + window.row_off + 0.5
            lon = transform.c + transform.a * cols + transform.b * rows
            lat = transform.f + transform.d * cols + transform.e * rows
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            lat_keys = _grid_key_array(lat, 1)
            lon_keys = _grid_key_array(lon, 1)

            for la, lo, v in zip(lat_keys, lon_keys, vals_kmh, strict=False):
                key = (int(la), int(lo))
                cells[key] = max(cells.get(key, 0.0), float(v))
    return cells


def main(src: Path, out: Path) -> None:
    nc_files = sorted(src.glob("*.nc"))
    tif_files = sorted(src.glob("*_100_YR_RP.tif")) + sorted(src.glob("*_100_YR_RP.tiff"))
    if not nc_files and not tif_files:
        raise FileNotFoundError(f"No .nc files or *_100_YR_RP.tif files found in {src}")

    logger.info("Found %d NetCDF basin files and %d 100-year GeoTIFF basin files", len(nc_files), len(tif_files))
    merged: dict[tuple[int, int], float] = {}

    skipped_files = 0
    for basin_path in [*nc_files, *tif_files]:
        logger.info("Loading %s ...", basin_path.name)
        try:
            if basin_path.suffix.lower() == ".nc":
                basin = _load_netcdf_basin(basin_path)
            else:
                basin = _load_tif_basin(basin_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", basin_path.name, e)
            skipped_files += 1
            continue
        for key, v in basin.items():
            merged[key] = max(merged.get(key, 0.0), v)
        logger.info("  %d cells loaded (merged total: %d)", len(basin), len(merged))

    if not merged:
        raise RuntimeError("No cells loaded from any basin file.")

    rows = [(lk, lok, v) for (lk, lok), v in merged.items()]
    logger.info("Writing %d cells to %s", len(rows), out)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    conn.execute(_SCHEMA)
    conn.executemany("INSERT OR REPLACE INTO grid (lat_key, lon_key, value) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    logger.info("Done. %s contains %d cells.", out.name, len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build STORM cyclone SQLite grid")
    parser.add_argument("--src", default="data/storm/", help="Directory containing STORM NetCDF files")
    parser.add_argument("--out", default="data/storm_cyclone.db", help="Output SQLite path")
    args = parser.parse_args()
    main(Path(args.src), Path(args.out))
