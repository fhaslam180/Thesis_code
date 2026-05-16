"""Prewarm the Open-Meteo API hazard cache for a set of supplier locations.

Reads either a frozen reference_locations.json or a pipeline graph JSON, deduplicates
suppliers to 0.1° cells using grid_key, then calls the three Open-Meteo fetchers
(heat_stress, drought, wildfire) which handle cache read/write and rate limiting.

Usage:
    # From reference corpus:
    python3 scripts/prewarm_hazard_cache.py \\
        --locations data/reference_locations.json [--company Apple]

    # From a previous pipeline run (prefers strict-filtered suppliers):
    python3 scripts/prewarm_hazard_cache.py \\
        --from-graph outputs/apple_strict_fixed/apple_graph.json \\
        --sleep 1.5 --retries 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))


def deduplicate_locations(locations: list[dict]) -> list[dict]:
    """Return one representative location per 0.1° (lat, lon) cell."""
    from bor_risk.utils import grid_key
    seen: dict[tuple[int, int], dict] = {}
    for loc in locations:
        cell = (grid_key(loc["lat"], 1), grid_key(loc["lon"], 1))
        if cell not in seen:
            seen[cell] = loc
    return list(seen.values())


def load_from_locations(path: Path, company: str | None) -> list[dict]:
    """Read company-grouped reference_locations.json."""
    raw_data = json.loads(path.read_text())
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
    if company:
        all_locations = [
            loc for loc in all_locations
            if loc.get("company", "").lower() == company.lower()
        ]
        if not all_locations:
            print(f"[WARNING] No locations found for company: {company}")
    return all_locations


def load_from_graph(path: Path) -> list[dict]:
    """Read supplier locations from a pipeline graph JSON output."""
    data = json.loads(path.read_text())
    suppliers = data.get("suppliers", [])
    locations = [
        {"lat": s["lat"], "lon": s["lon"], "name": s.get("name", "unknown")}
        for s in suppliers
        if s.get("lat") is not None and s.get("lon") is not None
        and not (s.get("lat") == 0.0 and s.get("lon") == 0.0)
    ]
    if not locations:
        print("[WARNING] No usable supplier locations found in graph JSON.")
    return locations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prewarm Open-Meteo hazard cache.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--locations",
        metavar="PATH",
        help="Path to reference_locations.json (company-grouped format).",
    )
    group.add_argument(
        "--from-graph",
        metavar="PATH",
        help="Path to a pipeline graph JSON (e.g. apple_graph.json); reads supplier locations.",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="(--locations only) Filter to locations for this company.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Minimum seconds between Open-Meteo API calls (default: 1.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Max HTTP attempts per API call, including 429 retries (default: 4).",
    )
    args = parser.parse_args(argv)

    # Set env vars BEFORE importing bor_risk.tools so module-level constants pick them up.
    os.environ["OPEN_METEO_MIN_INTERVAL"] = str(args.sleep)
    os.environ["OPEN_METEO_MAX_ATTEMPTS"] = str(args.retries)

    from bor_risk.tools import (  # noqa: E402
        _DROUGHT_METHOD_VERSION,
        _HEAT_METHOD_VERSION,
        _WILDFIRE_METHOD_VERSION,
        _api_cache_lookup,
        _fetch_approx_fwi_q95,
        _fetch_heat_fraction,
        _fetch_spei12_fraction,
    )

    if args.locations:
        loc_path = Path(args.locations)
        if not loc_path.exists():
            print(f"[ERROR] Locations file not found: {loc_path}")
            sys.exit(1)
        all_locations = load_from_locations(loc_path, args.company)
    else:
        graph_path = Path(args.from_graph)
        if not graph_path.exists():
            print(f"[ERROR] Graph file not found: {graph_path}")
            sys.exit(1)
        all_locations = load_from_graph(graph_path)

    unique = deduplicate_locations(all_locations)
    total = len(unique)
    print(f"Prewarming {total} unique 0.1° cells (from {len(all_locations)} total locations)...")
    print(f"  sleep={args.sleep}s  retries={args.retries}")

    api_calls = 0
    cache_hits = 0
    errors = 0

    fetchers = [
        ("heat_stress", _fetch_heat_fraction, _HEAT_METHOD_VERSION),
        ("drought", _fetch_spei12_fraction, _DROUGHT_METHOD_VERSION),
        ("wildfire", _fetch_approx_fwi_q95, _WILDFIRE_METHOD_VERSION),
    ]

    for i, loc in enumerate(unique, 1):
        lat = loc["lat"]
        lon = loc["lon"]
        name = loc.get("name", loc.get("company", f"loc_{i}"))
        print(f"[{i}/{total}] {name}  lat={lat:.4f}  lon={lon:.4f}")

        for hazard, fetcher, version in fetchers:
            try:
                if _api_cache_lookup(hazard, lat, lon, version) is not None:
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

    # Cache coverage summary
    print("\nCache coverage:")
    for hazard, _, version in fetchers:
        cached_count = sum(
            1 for loc in unique
            if _api_cache_lookup(hazard, loc["lat"], loc["lon"], version) is not None
        )
        print(f"  {hazard}: {cached_count}/{total} cells cached")


if __name__ == "__main__":
    main()
