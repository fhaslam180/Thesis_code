#!/usr/bin/env python3
"""Preprocess NOAA IBTrACS data into a 1-degree cyclone exposure grid.

Usage:
    1. Download the IBTrACS since-1980 CSV from NOAA NCEI:
       https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv

    2. Run this script:
       python3 scripts/preprocess_cyclone.py /path/to/ibtracs.since1980.list.v04r01.csv

    3. Output is written to data/cyclone_grid.json

The output maps "lat_bin,lon_bin" -> {"storm_count": N, "max_wind_kt": W},
where lat_bin and lon_bin are rounded to the nearest integer degree.
Only track points with wind >= 34 kt (tropical storm strength) are included.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(csv_path: str) -> None:
    # Track unique storm IDs and max wind per grid cell
    cell_storms: dict[str, set[str]] = defaultdict(set)
    cell_max_wind: dict[str, float] = defaultdict(float)
    rows_processed = 0
    rows_skipped = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        # IBTrACS CSVs have a header row followed by a units row; skip units
        reader = csv.DictReader(f)
        for row in reader:
            # Skip the units row and any non-data rows
            sid = row.get("SID", "").strip()
            if not sid or sid.startswith("SID") or sid.startswith("_"):
                continue

            try:
                lat = float(row.get("LAT", "").strip())
                lon = float(row.get("LON", "").strip())
            except (ValueError, TypeError):
                rows_skipped += 1
                continue

            # Parse wind speed — prefer USA_WIND, fall back to WMO_WIND
            wind_str = row.get("USA_WIND", "").strip()
            if not wind_str:
                wind_str = row.get("WMO_WIND", "").strip()
            try:
                wind = float(wind_str)
            except (ValueError, TypeError):
                rows_skipped += 1
                continue

            # Filter: only tropical storm strength and above
            if wind < 34:
                continue

            lat_bin = round(lat)
            lon_bin = round(lon)
            key = f"{lat_bin},{lon_bin}"

            cell_storms[key].add(sid)
            if wind > cell_max_wind[key]:
                cell_max_wind[key] = wind
            rows_processed += 1

    # Build output
    grid = {}
    for key in cell_storms:
        grid[key] = {
            "storm_count": len(cell_storms[key]),
            "max_wind_kt": int(cell_max_wind[key]),
        }

    out_path = Path(__file__).resolve().parent.parent / "data" / "cyclone_grid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grid, indent=2))
    print(f"Wrote {len(grid)} grid cells to {out_path}")
    print(f"  ({rows_processed} track points processed, {rows_skipped} skipped)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ibtracs.since1980.list.v04r01.csv>")
        sys.exit(1)
    main(sys.argv[1])
