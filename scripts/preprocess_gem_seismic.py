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


def main(src: Path, out: Path) -> None:
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
