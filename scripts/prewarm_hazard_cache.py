"""Prewarm the Open-Meteo API hazard cache for a set of reference locations.

Reads a frozen reference_locations.json, deduplicates suppliers to 0.1° cells
using grid_key, then calls the three Open-Meteo fetchers (heat_stress, drought,
wildfire) which handle cache read/write and rate limiting internally.

Usage:
    python3 scripts/prewarm_hazard_cache.py \\
        --locations data/reference_locations.json \\
        [--company Apple]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path when run as a script.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from bor_risk.tools import (  # noqa: E402
    _fetch_approx_fwi_q95,
    _fetch_heat_fraction,
    _fetch_spei12_fraction,
)
from bor_risk.utils import grid_key  # noqa: E402


def deduplicate_locations(locations: list[dict]) -> list[dict]:
    """Return one representative location per 0.1° (lat, lon) cell."""
    seen: dict[tuple[int, int], dict] = {}
    for loc in locations:
        cell = (grid_key(loc["lat"], 1), grid_key(loc["lon"], 1))
        if cell not in seen:
            seen[cell] = loc
    return list(seen.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prewarm Open-Meteo hazard cache.")
    parser.add_argument(
        "--locations",
        required=True,
        help="Path to reference_locations.json",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Optional: only prewarm locations for this company.",
    )
    args = parser.parse_args(argv)

    loc_path = Path(args.locations)
    if not loc_path.exists():
        print(f"[ERROR] Locations file not found: {loc_path}")
        sys.exit(1)

    raw_data = json.loads(loc_path.read_text())
    all_locations: list[dict] = []
    for company_name, locs in raw_data.get("companies", {}).items():
        for loc in locs:
            entry = dict(loc)
            entry.setdefault("company", company_name)
            all_locations.append(entry)

    if not all_locations:
        print("[ERROR] No locations found. Is the file in company-grouped format "
              '({"companies": {"Name": [...]}})? ')
        sys.exit(1)

    if args.company:
        all_locations = [
            loc for loc in all_locations
            if loc.get("company", "").lower() == args.company.lower()
        ]
        if not all_locations:
            print(f"[WARNING] No locations found for company: {args.company}")

    unique = deduplicate_locations(all_locations)
    total = len(unique)
    print(f"Prewarming {total} unique 0.1° cells (from {len(all_locations)} total locations)...")

    api_calls = 0
    cache_hits = 0
    errors = 0

    fetchers = [
        ("heat_stress", _fetch_heat_fraction),
        ("drought", _fetch_spei12_fraction),
        ("wildfire", _fetch_approx_fwi_q95),
    ]

    for i, loc in enumerate(unique, 1):
        lat = loc["lat"]
        lon = loc["lon"]
        name = loc.get("name", loc.get("company", f"loc_{i}"))
        print(f"[{i}/{total}] {name}  lat={lat:.4f}  lon={lon:.4f}")

        for hazard, fetcher in fetchers:
            try:
                from bor_risk.tools import _api_cache_lookup
                from bor_risk.tools import _HEAT_METHOD_VERSION, _DROUGHT_METHOD_VERSION, _WILDFIRE_METHOD_VERSION
                method_map = {
                    "heat_stress": _HEAT_METHOD_VERSION,
                    "drought": _DROUGHT_METHOD_VERSION,
                    "wildfire": _WILDFIRE_METHOD_VERSION,
                }
                if _api_cache_lookup(hazard, lat, lon, method_map[hazard]) is not None:
                    cache_hits += 1
                    print(f"  {hazard}: cache hit")
                else:
                    fetcher(lat, lon)
                    api_calls += 1
                    print(f"  {hazard}: fetched")
            except Exception as e:
                errors += 1
                print(f"  {hazard}: ERROR — {e}")

    print()
    print(f"Done. cells={total}, api_calls={api_calls}, cache_hits={cache_hits}, errors={errors}")


if __name__ == "__main__":
    main()
