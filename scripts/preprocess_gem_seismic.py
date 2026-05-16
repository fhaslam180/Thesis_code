"""Preprocess GEM 2023 Global Seismic Hazard data into a SQLite database.

Source: GEM Global Seismic Hazard Model 2023 (Pagani et al. 2018)
        https://github.com/GEMScienceTools/oq-mbtk / Zenodo DOI
Variable: PGA at 10% probability of exceedance in 50 years (~475-yr return period), in g
Grid resolution: 0.1 degrees
Output: data/gem_seismic.db (SQLite, INTEGER keys via grid_key())

Usage:
    # Place the GEM CSV (columns: lon, lat, PGA) at data/gem_seismic_raw.csv
    python3 scripts/preprocess_gem_seismic.py --src data/gem_seismic_raw.csv

CRITICAL: This script imports grid_key from bor_risk.utils.  Do NOT use Python's
built-in round() for coordinate quantization.  round() uses banker's rounding
(round-half-to-even) which disagrees with grid_key() at .5 boundaries, causing
silent location drift between preprocessing and runtime lookup.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
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


def _is_valid(pga_g: float) -> bool:
    """Return True iff the PGA value is physically valid."""
    return math.isfinite(pga_g) and pga_g >= 0.0


def _grid_key_array(values: np.ndarray, decimals: int) -> np.ndarray:
    """Vectorised equivalent of bor_risk.utils.grid_key."""
    factor = 10 ** decimals
    return (
        np.floor(np.abs(values) * factor + 0.5).astype(np.int32)
        * np.where(values >= 0, 1, -1).astype(np.int32)
    )


def _insert_rows(conn: sqlite3.Connection, rows: list[tuple[int, int, float]]) -> None:
    conn.executemany(
        """
        INSERT INTO grid (lat_key, lon_key, value) VALUES (?, ?, ?)
        ON CONFLICT(lat_key, lon_key) DO UPDATE SET
            value = max(grid.value, excluded.value)
        """,
        rows,
    )


def main_tif(src: Path, out: Path, chunk_rows: int = 256) -> None:
    """Build the SQLite grid directly from a GEM GeoTIFF.

    The GEM raster is finer than the runtime 0.1-degree lookup grid. When
    several raster pixels fall in the same runtime cell, keep the max PGA as a
    conservative exposure screen.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise RuntimeError("Install rasterio to preprocess GeoTIFF inputs") from e

    logger.info("Reading GEM seismic GeoTIFF from %s", src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(str(out))
    conn.execute(_SCHEMA)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")

    rows_written = 0
    skipped = 0
    total = 0
    batch: list[tuple[int, int, float]] = []
    batch_size = 250_000

    with rasterio.open(src) as ds:
        nodata = ds.nodata
        transform = ds.transform
        for row_off in range(0, ds.height, chunk_rows):
            height = min(chunk_rows, ds.height - row_off)
            window = Window(0, row_off, ds.width, height)
            arr = ds.read(1, window=window, masked=True)
            data = np.asarray(arr.filled(np.nan), dtype=np.float64)

            valid = np.isfinite(data) & (data >= 0.0)
            if nodata is not None and math.isfinite(nodata):
                valid &= data != nodata
            total += data.size
            skipped += int(data.size - np.count_nonzero(valid))
            if not np.any(valid):
                continue

            r_idx, c_idx = np.nonzero(valid)
            vals = data[r_idx, c_idx]
            cols = c_idx.astype(np.float64) + window.col_off + 0.5
            rows = r_idx.astype(np.float64) + window.row_off + 0.5
            lon = transform.c + transform.a * cols + transform.b * rows
            lat = transform.f + transform.d * cols + transform.e * rows
            lat_keys = _grid_key_array(lat, 1)
            lon_keys = _grid_key_array(lon, 1)

            batch.extend(
                (int(la), int(lo), float(v))
                for la, lo, v in zip(lat_keys, lon_keys, vals, strict=False)
            )
            if len(batch) >= batch_size:
                _insert_rows(conn, batch)
                rows_written += len(batch)
                batch.clear()
                logger.info("  processed %.1f%%", 100 * (row_off + height) / ds.height)

    if batch:
        _insert_rows(conn, batch)
        rows_written += len(batch)
    conn.commit()
    cell_count = conn.execute("SELECT COUNT(*) FROM grid").fetchone()[0]
    conn.close()
    logger.info(
        "Done. %s contains %d cells from %d valid raster pixels (skipped %d/%d).",
        out.name, cell_count, rows_written, skipped, total,
    )


def main(src: Path, out: Path) -> None:
    if src.suffix.lower() in {".tif", ".tiff"}:
        main_tif(src, out)
        return

    logger.info("Reading GEM seismic CSV from %s", src)
    rows: list[tuple[int, int, float]] = []
    skipped = 0
    total = 0

    with src.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                pga_g = float(row["PGA"])
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed row %d: %s", total, e)
                skipped += 1
                continue

            if not _is_valid(pga_g):
                logger.warning("Skipping non-finite/negative PGA at (%.3f, %.3f): %s", lat, lon, pga_g)
                skipped += 1
                continue

            rows.append((grid_key(lat, 1), grid_key(lon, 1), pga_g))

    if total > 0 and skipped / total > 0.10:
        raise RuntimeError(
            f"Too many invalid cells: {skipped}/{total} ({100*skipped/total:.1f}%). "
            "Check source file — this indicates a data ingestion problem, not sparse data."
        )

    logger.info("Writing %d cells to %s (skipped %d)", len(rows), out, skipped)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    conn.execute(_SCHEMA)
    conn.executemany("INSERT OR REPLACE INTO grid (lat_key, lon_key, value) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    logger.info("Done. %s contains %d cells.", out.name, len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GEM seismic SQLite grid")
    parser.add_argument("--src", default="data/gem_seismic_raw.csv", help="Input CSV path")
    parser.add_argument("--out", default="data/gem_seismic.db", help="Output SQLite path")
    args = parser.parse_args()
    main(Path(args.src), Path(args.out))
