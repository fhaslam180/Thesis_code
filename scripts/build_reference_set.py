"""Build the study-specific reference corpus and reference set.

Two strictly separate phases — NEVER combined:

Phase 1: Corpus collection (run ONCE, commit result, never rerun)
    python3 scripts/build_reference_set.py --collect-locations \
        --out data/reference_locations.json [--companies "Apple,Toyota,..."]

    Runs llm_only pipeline for each company, collects all discovered supplier
    lat/lon, deduplicates.  Fails if output file already exists (pass --force
    to override).  Commit reference_locations.json immediately after.

Phase 2: Reference-set build (run once after corpus is frozen)
    python3 scripts/build_reference_set.py \
        --locations data/reference_locations.json \
        --out data/reference_set.json

    Reads frozen reference_locations.json — does NOT re-query LLM.  Computes
    raw hazard indicators for each location, sorts per hazard, stores sorted
    arrays, computes empirical tertile thresholds.  Commit reference_set.json
    immediately after first successful build.

CORPUS FREEZE INVARIANT:
    No evaluation script, pipeline run, or test may regenerate
    reference_locations.json.  The --collect-locations guard (fails if file
    exists without --force) enforces this at the script level.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bor_risk.utils import grid_key, load_reference_set  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 30 benchmark companies for corpus collection
EVAL_COMPANIES = [
    "Apple", "Toyota", "Samsung", "Microsoft", "Amazon",
    "Tesla", "NVIDIA", "Intel", "TSMC", "Foxconn",
    "BMW", "Volkswagen", "Boeing", "Airbus", "General Electric",
    "Siemens", "Caterpillar", "3M", "Honeywell", "Johnson & Johnson",
    "Pfizer", "BASF", "Shell", "ExxonMobil", "Rio Tinto",
    "Walmart", "Nestlé", "Unilever", "Nike", "Adidas",
]

_HAZARD_ORDER = (
    "earthquake",
    "flood",
    "wildfire",
    "cyclone",
    "heat_stress",
    "drought",
)


# ---------------------------------------------------------------------------
# Phase 1: Corpus collection
# ---------------------------------------------------------------------------


def _collect_locations(companies: list[str], out: Path, force: bool) -> None:
    """Run llm_only pipeline for each company and collect supplier locations."""
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists. The reference corpus is frozen — "
            "do not regenerate it. Pass --force to override (destructive)."
        )

    # Import here to avoid circular dependency issues at module level
    try:
        from bor_risk.graph import run_vcg_graph
    except ImportError as e:
        raise ImportError(
            "Cannot import bor_risk.graph — ensure package is installed: "
            f"pip install -e . ({e})"
        )

    all_companies: dict[str, list[dict]] = {}
    total_locations = 0

    for company in companies:
        logger.info("Collecting suppliers for '%s' ...", company)
        try:
            result = run_vcg_graph(
                company=company,
                enable_web=False,
                no_verify=True,
                tier_depth=2,
            )
        except Exception as e:
            logger.warning("Pipeline failed for '%s': %s — skipping", company, e)
            continue

        suppliers = result.get("suppliers", [])
        locations: list[dict] = []
        seen: set[tuple[float, float]] = set()

        for sup in suppliers:
            # Supplier objects or dicts
            if hasattr(sup, "lat"):
                lat, lon, name = sup.lat, sup.lon, sup.name
            else:
                lat = float(sup.get("lat", 0.0))
                lon = float(sup.get("lon", 0.0))
                name = sup.get("name", "unknown")

            if not (math.isfinite(lat) and math.isfinite(lon)):
                logger.warning("Non-finite coordinates for '%s'; skipping", name)
                continue

            key = (round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            locations.append({"name": name, "lat": lat, "lon": lon})

        all_companies[company] = locations
        total_locations += len(locations)
        logger.info("  %d unique locations collected (total so far: %d)", len(locations), total_locations)

    payload = {
        "metadata": {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "n_companies": len(all_companies),
            "corpus_note": (
                "study-specific: supplier locations for benchmark companies. "
                "Normalization basis — do NOT regenerate mid-study."
            ),
        },
        "companies": all_companies,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    logger.info(
        "Saved %d locations across %d companies to %s",
        total_locations, len(all_companies), out,
    )
    logger.info("IMPORTANT: commit %s immediately and do not regenerate it.", out.name)


# ---------------------------------------------------------------------------
# Phase 2: Reference-set build
# ---------------------------------------------------------------------------


def _compute_raw_hazards(lat: float, lon: float) -> dict[str, float]:
    """Compute raw hazard indicators for a single location.

    Returns {hazard_name: raw_value} for all 6 hazards.
    Raises ValueError if any value is NaN or inf (fail-fast — indicates
    a data pipeline problem, not sparse data).
    """
    from bor_risk.tools import (
        _db_lookup,
        _fetch_approx_fwi_q95,
        _fetch_heat_fraction,
        _fetch_spei12_fraction,
        _CYCLONE_DB,
        _FLOOD_DB,
        _GEM_DB,
        _guard_raw,
    )

    raw: dict[str, float] = {}

    # Earthquake — GEM 2023, 0.1° grid
    if not _DATA_DIR.joinpath("gem_seismic.db").exists():
        logger.warning("gem_seismic.db not found; earthquake=0.0 for (%.4f, %.4f)", lat, lon)
        raw["earthquake"] = 0.0
    else:
        raw["earthquake"] = _db_lookup(_GEM_DB, grid_key(lat, 1), grid_key(lon, 1))

    # Flood — WRI Aqueduct, 0.01° grid
    if not _DATA_DIR.joinpath("aqueduct_flood.db").exists():
        logger.warning("aqueduct_flood.db not found; flood=0.0 for (%.4f, %.4f)", lat, lon)
        raw["flood"] = 0.0
    else:
        raw["flood"] = _db_lookup(_FLOOD_DB, grid_key(lat, 2), grid_key(lon, 2))

    # Cyclone — STORM, 0.1° grid
    if not _DATA_DIR.joinpath("storm_cyclone.db").exists():
        logger.warning("storm_cyclone.db not found; cyclone=0.0 for (%.4f, %.4f)", lat, lon)
        raw["cyclone"] = 0.0
    else:
        raw["cyclone"] = _db_lookup(_CYCLONE_DB, grid_key(lat, 1), grid_key(lon, 1))

    # Wildfire — approx FWI Q95, Open-Meteo ERA5
    # _strict=True: API failures raise RuntimeError so they are not silently
    # accepted as valid zero-exposure values in the reference corpus.
    raw["wildfire"] = _fetch_approx_fwi_q95(lat, lon, _strict=True)

    # Heat stress — fraction days apparent_temperature_max >= 32°C, Open-Meteo ERA5
    raw["heat_stress"] = _fetch_heat_fraction(lat, lon, _strict=True)

    # Drought — fraction months SPEI-12 <= -1, Open-Meteo ERA5
    raw["drought"] = _fetch_spei12_fraction(lat, lon, _strict=True)

    # Fail-fast on non-finite values
    for hazard, value in raw.items():
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite raw value for hazard '{hazard}' at ({lat}, {lon}): {value}. "
                "This indicates a data pipeline problem."
            )
        if value < 0.0:
            raise ValueError(
                f"Negative raw value for hazard '{hazard}' at ({lat}, {lon}): {value}."
            )

    return raw


def _compute_supplier_exposure_indices(
    supplier_raws: list[dict[str, float]],
    hazard_raws: dict[str, list[float]],
    non_informative: list[str],
) -> tuple[list[float], list[str]]:
    """Compute per-location exposure indices using informative hazards only.

    Non-informative hazards are excluded entirely from the composite. Treating
    them as neutral 0.5 values collapses the exposure distribution and produces
    degenerate tertile thresholds when several hazards are constant.
    """
    from bor_risk.utils import percentile_rank

    informative_hazards = [h for h in _HAZARD_ORDER if h not in set(non_informative)]
    if not informative_hazards:
        raise ValueError(
            "All hazards are non-informative in the reference corpus; "
            "cannot compute exposure thresholds."
        )

    supplier_Es: list[float] = []
    for raw in supplier_raws:
        Hs = [
            percentile_rank(raw.get(hazard, 0.0), hazard_raws[hazard])
            for hazard in informative_hazards
        ]
        supplier_Es.append(sum(Hs) / len(Hs))

    return supplier_Es, informative_hazards


def _build_reference_set(locations_path: Path, out: Path) -> None:
    """Build reference_set.json from frozen reference_locations.json."""
    payload = json.loads(locations_path.read_text())
    companies: dict[str, list[dict]] = payload.get("companies", {})

    all_locations: list[dict] = []
    for locs in companies.values():
        all_locations.extend(locs)

    n_total = sum(len(locs) for locs in companies.values())
    logger.info("Processing %d locations across %d companies", n_total, len(companies))

    # Collect raw values per hazard
    hazard_raws: dict[str, list[float]] = {
        "earthquake": [], "flood": [], "wildfire": [],
        "cyclone": [], "heat_stress": [], "drought": [],
    }
    supplier_raws: list[dict[str, float]] = []  # flat list for percentile ranking
    company_supplier_raws: list[list[dict]] = []  # per-company successful raws

    flat_idx = 0
    for company_name, company_locs in companies.items():
        per_company_raws: list[dict] = []
        for loc in company_locs:
            flat_idx += 1
            lat = float(loc["lat"])
            lon = float(loc["lon"])
            name = loc.get("name", f"loc_{flat_idx}")
            logger.info("[%d/%d] %s (%.4f, %.4f)", flat_idx, n_total, name, lat, lon)
            try:
                raw = _compute_raw_hazards(lat, lon)
            except (ValueError, Exception) as e:
                logger.error("Failed for '%s' at (%.4f, %.4f): %s — skipping", name, lat, lon, e)
                continue
            for hazard, value in raw.items():
                hazard_raws[hazard].append(value)
            supplier_raws.append(raw)
            per_company_raws.append(raw)
        company_supplier_raws.append(per_company_raws)

    # Sort each hazard array
    for hazard in hazard_raws:
        hazard_raws[hazard].sort()

    # Detect non-informative hazards (constant reference)
    non_informative: list[str] = []
    for hazard, arr in hazard_raws.items():
        if not arr:
            non_informative.append(hazard)
            logger.warning(
                "WARNING: hazard '%s' has no values in the reference corpus "
                "and will be treated as non-informative",
                hazard,
            )
        elif arr[0] == arr[-1]:
            non_informative.append(hazard)
            logger.warning(
                "WARNING: hazard '%s' is constant across reference corpus "
                "(all values = %.4f) — will be treated as non-informative (score=0.5)",
                hazard, arr[0],
            )

    supplier_Es, informative_hazards = _compute_supplier_exposure_indices(
        supplier_raws, hazard_raws, non_informative
    )

    supplier_Es_sorted = sorted(supplier_Es)

    def _empirical_tertile(sorted_vals: list[float], fraction: float) -> float:
        if not sorted_vals:
            return 0.5
        idx = int(fraction * len(sorted_vals))
        idx = max(0, min(len(sorted_vals) - 1, idx))
        return round(sorted_vals[idx], 6)

    supplier_med = _empirical_tertile(supplier_Es_sorted, 1.0 / 3.0)
    supplier_high = _empirical_tertile(supplier_Es_sorted, 2.0 / 3.0)

    # Company-level E_s: mean(E_s) per company, sliced by actual success count
    company_Es: list[float] = []
    idx = 0
    for per_company_raws in company_supplier_raws:
        n = len(per_company_raws)   # actual successes, not attempted locations
        chunk = supplier_Es[idx:idx + n]
        if chunk:
            company_Es.append(sum(chunk) / len(chunk))
        idx += n

    if idx != len(supplier_Es):
        raise RuntimeError(
            f"Company grouping drift: consumed {idx} of {len(supplier_Es)} supplier "
            "exposure values. Per-company success counts do not sum to total successes "
            "— output would be silently wrong."
        )

    company_Es_sorted = sorted(company_Es)
    company_med = _empirical_tertile(company_Es_sorted, 1.0 / 3.0)
    company_high = _empirical_tertile(company_Es_sorted, 2.0 / 3.0)

    reference_set = {
        "metadata": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_suppliers": len(supplier_raws),
            "source": "build_reference_set.py",
            "corpus": (
                "study-specific: supplier locations for benchmark companies. "
                "Percentile ranks are relative to this corpus, not a universal baseline."
            ),
            "non_informative_hazards": non_informative,
            "informative_hazards": informative_hazards,
        },
        "hazards": hazard_raws,
        "supplier_exposure_thresholds": {
            "medium": supplier_med,
            "high": supplier_high,
        },
        "company_exposure_thresholds": {
            "medium": company_med,
            "high": company_high,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference_set, indent=2))
    logger.info(
        "Saved reference_set.json: N=%d suppliers; "
        "supplier thresholds medium=%.4f high=%.4f; "
        "company thresholds medium=%.4f high=%.4f",
        len(supplier_raws),
        supplier_med, supplier_high,
        company_med, company_high,
    )
    if non_informative:
        logger.warning(
            "Non-informative hazards (constant reference, assigned score=0.5): %s",
            non_informative,
        )
    logger.info("IMPORTANT: commit %s immediately.", out.name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SMHEI reference corpus and reference set (two-phase)"
    )
    parser.add_argument(
        "--collect-locations", action="store_true",
        help="Phase 1: collect supplier locations via llm_only pipeline",
    )
    parser.add_argument(
        "--locations", default=None,
        help="Phase 2: path to frozen reference_locations.json",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path (reference_locations.json or reference_set.json)",
    )
    parser.add_argument(
        "--companies", default=None,
        help="Comma-separated list of companies (Phase 1 only; default: all 30 benchmark companies)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output file (Phase 1 only; destructive)",
    )
    args = parser.parse_args()

    if args.collect_locations:
        companies = (
            [c.strip() for c in args.companies.split(",")]
            if args.companies
            else EVAL_COMPANIES
        )
        _collect_locations(companies, Path(args.out), force=args.force)
    elif args.locations:
        _build_reference_set(Path(args.locations), Path(args.out))
    else:
        parser.error("Specify --collect-locations (Phase 1) or --locations <path> (Phase 2).")


if __name__ == "__main__":
    main()
