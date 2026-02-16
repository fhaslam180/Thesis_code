#!/usr/bin/env python3
"""Preprocess NASA FIRMS VIIRS fire data into a 0.5-degree grid lookup.

Usage:
    1. Download the annual VIIRS fire detection CSV from NASA FIRMS:
       https://firms.modaps.eosdis.nasa.gov/active_fire/
       (e.g., fire_nrt_SV-C2_*.csv or DL_FIRE_SV-C2_*.csv)

    2. Run this script:
       python3 scripts/preprocess_wildfire.py /path/to/fire_data.csv

    3. Output is written to data/wildfire_grid.json

The output maps "lat_bin,lon_bin" -> annual_fire_count, where lat_bin and
lon_bin are rounded to the nearest 0.5 degrees.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main(csv_path: str) -> None:
    grid: Counter[str] = Counter()
    row_count = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if row_count % 2_000_000 == 0:
                print(f"  ... {row_count:,} rows processed", flush=True)
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                skipped += 1
                continue

            # Bin to 0.5-degree grid
            lat_bin = round(lat * 2) / 2
            lon_bin = round(lon * 2) / 2
            key = f"{lat_bin},{lon_bin}"
            grid[key] += 1

    out_path = Path(__file__).resolve().parent.parent / "data" / "wildfire_grid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(grid), indent=2))
    print(f"Wrote {len(grid)} grid cells to {out_path}")
    print(f"({row_count:,} fire detections processed, {skipped:,} skipped)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <firms_fire_data.csv>")
        sys.exit(1)
    main(sys.argv[1])
