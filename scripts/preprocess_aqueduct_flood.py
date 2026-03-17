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


def main(src: Path, out: Path) -> None:
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
    parser.add_argument("--out", default="data/aqueduct_flood.db", help="Output SQLite path")
    args = parser.parse_args()
    main(Path(args.src), Path(args.out))
