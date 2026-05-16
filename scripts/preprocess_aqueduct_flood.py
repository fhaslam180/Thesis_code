"""Preprocess WRI Aqueduct Floods data into a SQLite database.

Source: WRI Aqueduct Floods (Winsemius et al. 2018)
        https://www.wri.org/data/aqueduct-floods-hazard-maps
Variable: 100-year return period river flood inundation depth (m)
Grid resolution: 0.01 degrees (nearest-cell lookup; absent cells → 0.0)
Output: data/aqueduct_flood.db (SQLite, INTEGER keys via grid_key())

STORAGE NOTE: Global 0.01-degree coverage is large (~50-200 MB SQLite).
Track with git-lfs; do NOT commit to normal git history.

Usage:
    # Place the Aqueduct CSV (columns: lon, lat, depth_m) at data/aqueduct_flood_raw.csv
    python3 scripts/preprocess_aqueduct_flood.py --src data/aqueduct_flood_raw.csv

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


def _is_valid(depth_m: float) -> bool:
    return math.isfinite(depth_m) and depth_m >= 0.0


def _grid_key_array(values: np.ndarray, decimals: int) -> np.ndarray:
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


def _ingest_tif(conn: sqlite3.Connection, src: Path, chunk_rows: int = 256) -> tuple[int, int, int]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise RuntimeError("Install rasterio to preprocess GeoTIFF inputs") from e

    logger.info("Reading Aqueduct flood GeoTIFF from %s", src)
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
            data = np.asarray(arr.filled(np.nan), dtype=np.float32)

            # Store only positive inundation depths. Runtime lookup treats
            # absent cells as 0.0, which is equivalent and keeps the DB compact.
            valid = np.isfinite(data) & (data > 0.0)
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
            lat_keys = _grid_key_array(lat, 2)
            lon_keys = _grid_key_array(lon, 2)

            batch.extend(
                (int(la), int(lo), float(v))
                for la, lo, v in zip(lat_keys, lon_keys, vals, strict=False)
            )
            if len(batch) >= batch_size:
                _insert_rows(conn, batch)
                rows_written += len(batch)
                batch.clear()
                logger.info("  %s processed %.1f%%", src.name, 100 * (row_off + height) / ds.height)

    if batch:
        _insert_rows(conn, batch)
        rows_written += len(batch)
    return rows_written, skipped, total


def main_tifs(srcs: list[Path], out: Path) -> None:
    """Build flood DB from one or more Aqueduct GeoTIFFs.

    If both riverine and coastal layers are supplied, the DB stores the max
    depth per runtime cell so a single flood hazard covers both exposure modes.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(str(out))
    conn.execute(_SCHEMA)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")

    total_written = 0
    total_skipped = 0
    total_cells = 0
    for src in srcs:
        written, skipped, cells = _ingest_tif(conn, src)
        total_written += written
        total_skipped += skipped
        total_cells += cells
        conn.commit()

    cell_count = conn.execute("SELECT COUNT(*) FROM grid").fetchone()[0]
    conn.close()
    logger.info(
        "Done. %s contains %d cells from %d positive raster pixels (skipped %d/%d).",
        out.name, cell_count, total_written, total_skipped, total_cells,
    )


def main(src: Path, out: Path, extra_srcs: list[Path] | None = None) -> None:
    srcs = [src, *(extra_srcs or [])]
    if all(p.suffix.lower() in {".tif", ".tiff"} for p in srcs):
        main_tifs(srcs, out)
        return

    logger.info("Reading Aqueduct flood CSV from %s", src)
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
                depth_m = float(row["depth_m"])
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed row %d: %s", total, e)
                skipped += 1
                continue

            if not _is_valid(depth_m):
                logger.warning("Skipping non-finite/negative depth at (%.4f, %.4f): %s", lat, lon, depth_m)
                skipped += 1
                continue

            rows.append((grid_key(lat, 2), grid_key(lon, 2), depth_m))

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
    parser = argparse.ArgumentParser(description="Build WRI Aqueduct flood SQLite grid")
    parser.add_argument("--src", default="data/aqueduct_flood_raw.csv", help="Input CSV path")
    parser.add_argument(
        "--extra-src",
        action="append",
        default=[],
        help="Additional input GeoTIFF/CSV path. For flood rasters, cells are merged by max depth.",
    )
    parser.add_argument("--out", default="data/aqueduct_flood.db", help="Output SQLite path")
    args = parser.parse_args()
    main(Path(args.src), Path(args.out), [Path(p) for p in args.extra_src])
