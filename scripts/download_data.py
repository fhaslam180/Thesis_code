#!/usr/bin/env python3
"""Download and preprocess hazard datasets for wildfire and cyclone grids.

Usage:
    python3 scripts/download_data.py

This script:
1. Downloads the IBTrACS since-1980 CSV from NOAA (141 MB, no auth)
2. Runs preprocess_cyclone.py to build data/cyclone_grid.json
3. Checks for a FIRMS VIIRS CSV in data/raw/ and preprocesses if found
4. Prints instructions for obtaining FIRMS data if not found
"""

from __future__ import annotations

import glob
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)
IBTRACS_PATH = _RAW_DIR / "ibtracs.since1980.csv"


def _download_ibtracs() -> Path:
    """Download IBTrACS since-1980 CSV (~142 MB) from NOAA NCEI."""
    if IBTRACS_PATH.exists():
        size_mb = IBTRACS_PATH.stat().st_size / (1024 * 1024)
        print(f"  IBTrACS already downloaded ({size_mb:.1f} MB): {IBTRACS_PATH}")
        return IBTRACS_PATH

    print(f"  Downloading IBTrACS from NOAA (~142 MB)...")
    print(f"  URL: {IBTRACS_URL}")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            print(f"\r  Progress: {pct}% ({mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(IBTRACS_URL, IBTRACS_PATH, reporthook=_progress)
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("\n  SSL cert issue on macOS — retrying with unverified context...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(IBTRACS_URL, IBTRACS_PATH, reporthook=_progress)
        else:
            raise
    print()  # newline after progress
    size_mb = IBTRACS_PATH.stat().st_size / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB to {IBTRACS_PATH}")
    return IBTRACS_PATH


def _find_firms_csv() -> Path | None:
    """Look for a FIRMS VIIRS CSV in data/raw/ or its subdirectories.

    FIRMS downloads unzip into a subfolder like DL_FIRE_J1V-C2_716184/
    containing fire_archive_*.csv (historical) and fire_nrt_*.csv (NRT).
    We prefer the larger archive file over the NRT file.
    """
    # Search both data/raw/ and one level of subdirectories
    search_dirs = [_RAW_DIR] + [
        p for p in _RAW_DIR.iterdir() if p.is_dir()
    ] if _RAW_DIR.exists() else []

    # Prefer fire_archive (historical, larger) over fire_nrt (near-real-time)
    filename_patterns = [
        "fire_archive*.csv",
        "fire_nrt*.csv",
        "DL_FIRE*.csv",
        "firms_viirs*.csv",
        "VIIRS*.csv",
        "viirs*.csv",
    ]

    for pattern in filename_patterns:
        for search_dir in search_dirs:
            matches = glob.glob(str(search_dir / pattern))
            if matches:
                # Return the largest file if multiple matches
                return max((Path(m) for m in matches), key=lambda p: p.stat().st_size)
    return None


def _run_preprocess(script_name: str, csv_path: Path) -> bool:
    """Run a preprocessing script and return True on success."""
    script = _SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script), str(csv_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
        return True
    else:
        print(f"  ERROR: {result.stderr.strip()}")
        return False


def main() -> None:
    print("=" * 60)
    print("Hazard Data Download & Preprocessing")
    print("=" * 60)

    # --- Cyclone (IBTrACS) ---
    print("\n[1/2] Cyclone data (NOAA IBTrACS)")
    ibtracs = _download_ibtracs()
    if _run_preprocess("preprocess_cyclone.py", ibtracs):
        grid_path = _PROJECT_ROOT / "data" / "cyclone_grid.json"
        import json
        grid = json.loads(grid_path.read_text())
        print(f"  cyclone_grid.json: {len(grid)} grid cells")
    else:
        print("  WARNING: Cyclone preprocessing failed")

    # --- Wildfire (FIRMS) ---
    print("\n[2/2] Wildfire data (NASA FIRMS VIIRS)")
    firms_csv = _find_firms_csv()
    if firms_csv:
        print(f"  Found FIRMS CSV: {firms_csv}")
        if _run_preprocess("preprocess_wildfire.py", firms_csv):
            grid_path = _PROJECT_ROOT / "data" / "wildfire_grid.json"
            import json
            grid = json.loads(grid_path.read_text())
            print(f"  wildfire_grid.json: {len(grid)} grid cells")
        else:
            print("  WARNING: Wildfire preprocessing failed")
    else:
        print("  No FIRMS CSV found in data/raw/")
        print()
        print("  To download FIRMS wildfire data:")
        print("  1. Go to https://firms.modaps.eosdis.nasa.gov/download/")
        print("  2. Click 'Create New Request'")
        print("  3. Authenticate with email code (no account needed)")
        print("  4. Select: Source = VIIRS NOAA-20, Date Range = past year,")
        print("     Area = World, Format = CSV")
        print("  5. Submit request and wait for email with download link")
        print(f"  6. Save the CSV to: {_RAW_DIR}/")
        print("  7. Re-run this script: python3 scripts/download_data.py")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
