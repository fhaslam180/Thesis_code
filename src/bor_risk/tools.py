"""Deterministic hazard-scoring tools and LLM-based supplier discovery.

Hazard scores come from computation, never from an LLM.
Supplier discovery uses GPT-4o with structured output.
"""

from __future__ import annotations

import hashlib
import json
import math
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from bor_risk.models import (
    AlternativeResponse,
    CompanyProfile,
    HazardScore,
    MitigationResponse,
    Supplier,
    TierResponse,
)
from bor_risk.utils import load_prompts

# Lazy import to avoid circular dependency / missing tavily in test env.
_search_module = None

_DEFAULT_SUPPLIERS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tests" / "fixtures" / "mock_suppliers.json"
)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Ensure OPENAI_API_KEY from .env is available for non-CLI code paths too.
load_dotenv()


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def _normalize_score(raw_0_1: float) -> tuple[int, str]:
    """Convert a 0.0-1.0 float to (0-100 int, Low/Medium/High level)."""
    score_100 = round(raw_0_1 * 100)
    score_100 = max(0, min(100, score_100))
    if score_100 <= 33:
        level = "Low"
    elif score_100 <= 66:
        level = "Medium"
    else:
        level = "High"
    return score_100, level


def _build_hazard_score(
    supplier_name: str,
    hazard_type: str,
    raw_score: float,
    metadata: dict,
) -> HazardScore:
    """Create a HazardScore with normalised score_100 and level."""
    score = round(raw_score, 4)
    score_100, level = _normalize_score(raw_score)
    return HazardScore(
        supplier_name=supplier_name,
        hazard_type=hazard_type,
        score=score,
        score_100=score_100,
        level=level,
        dataset_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Shared HTTP helper (SSL certificate fallback)
# ---------------------------------------------------------------------------


def _urlopen_with_ssl_fallback(url: str, timeout: int = 15) -> bytes:
    """Fetch URL content, retrying with unverified SSL on cert errors.

    Some local environments (e.g. macOS) fail CA verification for
    certain HTTPS endpoints.  When that happens, retry once with an
    unverified context so the agent still returns real data.
    Re-raises non-SSL errors so callers can apply their own fallback.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
                return resp.read()
        raise


# ---------------------------------------------------------------------------
# Earthquake scoring (USGS FDSNWS)
# ---------------------------------------------------------------------------


def _fetch_usgs_earthquake_count(lat: float, lon: float) -> int:
    """Count M4.0+ earthquakes within 200 km of (*lat*, *lon*) over 2015-2025.

    Uses the USGS FDSNWS count endpoint (free, no API key).
    Returns 0 on any network or API error (fail-open).
    """
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/count"
        f"?latitude={lat}&longitude={lon}"
        "&maxradiuskm=200"
        "&starttime=2015-01-01"
        "&endtime=2025-01-01"
        "&minmagnitude=4.0"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=10)
        return int(raw.decode().strip())
    except (urllib.error.URLError, ValueError, OSError):
        return 0


def _score_earthquake(supplier: Supplier) -> HazardScore:
    count = _fetch_usgs_earthquake_count(supplier.lat, supplier.lon)
    raw = min(1.0, math.log10(1 + count) / 3.0)
    return _build_hazard_score(
        supplier.name,
        "earthquake",
        raw,
        {
            "dataset": "usgs_earthquake_catalog",
            "source": "USGS FDSNWS",
            "version": "1.0.0",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "usgs_fdsnws_count",
            "earthquake_count": count,
            "radius_km": 200,
            "min_magnitude": 4.0,
            "start_date": "2015-01-01",
            "end_date": "2025-01-01",
        },
    )


# ---------------------------------------------------------------------------
# Flood scoring (Open-Meteo GloFAS)
# ---------------------------------------------------------------------------


def _fetch_flood_data(lat: float, lon: float) -> dict:
    """Fetch river discharge stats from Open-Meteo GloFAS Flood API.

    Queries daily river discharge for 2015-2025 at the nearest 5 km
    GloFAS grid cell.  Returns a dict with discharge statistics.
    """
    url = (
        "https://flood-api.open-meteo.com/v1/flood"
        f"?latitude={lat}&longitude={lon}"
        "&daily=river_discharge"
        "&start_date=2015-01-01&end_date=2025-01-01"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=15)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError):
        return {"days_above_2x_mean": 0, "mean_discharge": 0.0, "max_discharge": 0.0}

    values = data.get("daily", {}).get("river_discharge", [])
    values = [v for v in values if v is not None]

    if not values:
        return {"days_above_2x_mean": 0, "mean_discharge": 0.0, "max_discharge": 0.0}

    mean_val = sum(values) / len(values)
    max_val = max(values)
    threshold = mean_val * 2.0
    days_above = sum(1 for v in values if v > threshold) if mean_val > 0 else 0

    return {
        "days_above_2x_mean": days_above,
        "mean_discharge": round(mean_val, 2),
        "max_discharge": round(max_val, 2),
        "total_days": len(values),
    }


def _score_flood(supplier: Supplier) -> HazardScore:
    stats = _fetch_flood_data(supplier.lat, supplier.lon)
    days_above = stats.get("days_above_2x_mean", 0)
    raw = min(1.0, days_above / 365.0)
    return _build_hazard_score(
        supplier.name,
        "flood",
        raw,
        {
            "dataset": "glofas_river_discharge",
            "version": "4.0",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "open_meteo_flood_api",
            "source": "Copernicus GloFAS via Open-Meteo",
            "period": "2015-01-01 to 2025-01-01",
            "mean_discharge_m3s": stats.get("mean_discharge", 0.0),
            "max_discharge_m3s": stats.get("max_discharge", 0.0),
            "days_above_2x_mean": days_above,
            "total_days": stats.get("total_days", 0),
            "resolution_km": 5,
        },
    )


# ---------------------------------------------------------------------------
# Heat stress scoring (Open-Meteo ERA5)
# ---------------------------------------------------------------------------


def _fetch_heat_data(lat: float, lon: float) -> dict:
    """Fetch heat stress stats from Open-Meteo Historical Weather API.

    Counts days where apparent temperature max exceeds 35 deg C over
    2015-2025 using ERA5 reanalysis data.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=apparent_temperature_max"
        "&start_date=2015-01-01&end_date=2025-01-01"
        "&timezone=UTC"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=15)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError):
        return {"annual_extreme_heat_days": 0.0, "total_days": 0}

    values = data.get("daily", {}).get("apparent_temperature_max", [])
    values = [v for v in values if v is not None]

    if not values:
        return {"annual_extreme_heat_days": 0.0, "total_days": 0}

    extreme_days = sum(1 for v in values if v > 35.0)
    years = len(values) / 365.25
    annual = extreme_days / years if years > 0 else 0.0

    return {
        "annual_extreme_heat_days": round(annual, 1),
        "total_extreme_days": extreme_days,
        "total_days": len(values),
    }


def _score_heat_stress(supplier: Supplier) -> HazardScore:
    stats = _fetch_heat_data(supplier.lat, supplier.lon)
    annual = stats.get("annual_extreme_heat_days", 0.0)
    raw = min(1.0, annual / 90.0)
    return _build_hazard_score(
        supplier.name,
        "heat_stress",
        raw,
        {
            "dataset": "era5_reanalysis",
            "version": "ERA5-Land",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "open_meteo_historical_api",
            "source": "ECMWF ERA5 via Open-Meteo",
            "period": "2015-01-01 to 2025-01-01",
            "annual_extreme_heat_days": stats.get("annual_extreme_heat_days", 0.0),
            "extreme_heat_threshold_c": 35,
            "metric": "apparent_temperature_max",
        },
    )


# ---------------------------------------------------------------------------
# Drought scoring (Open-Meteo ERA5 precipitation)
# ---------------------------------------------------------------------------


def _fetch_precipitation_data(lat: float, lon: float) -> dict:
    """Fetch precipitation stats from Open-Meteo Historical Weather API.

    Aggregates daily precipitation into monthly totals and computes the
    fraction of months below 50 % of the long-term monthly mean.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum"
        "&start_date=2015-01-01&end_date=2025-01-01"
        "&timezone=UTC"
    )
    try:
        raw = _urlopen_with_ssl_fallback(url, timeout=15)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, OSError):
        return {"dry_month_fraction": 0.0, "mean_monthly_precip_mm": 0.0}

    dates = data.get("daily", {}).get("time", [])
    values = data.get("daily", {}).get("precipitation_sum", [])

    # Aggregate into monthly totals
    monthly: dict[str, float] = {}
    for date_str, val in zip(dates, values):
        if val is None:
            continue
        month_key = date_str[:7]  # "YYYY-MM"
        monthly[month_key] = monthly.get(month_key, 0.0) + val

    if not monthly:
        return {"dry_month_fraction": 0.0, "mean_monthly_precip_mm": 0.0}

    totals = list(monthly.values())
    mean_monthly = sum(totals) / len(totals)
    dry_threshold = mean_monthly * 0.5
    dry_months = sum(1 for t in totals if t < dry_threshold)
    fraction = dry_months / len(totals)

    return {
        "dry_month_fraction": round(fraction, 4),
        "mean_monthly_precip_mm": round(mean_monthly, 2),
        "dry_months": dry_months,
        "total_months": len(totals),
    }


def _score_drought(supplier: Supplier) -> HazardScore:
    stats = _fetch_precipitation_data(supplier.lat, supplier.lon)
    fraction = stats.get("dry_month_fraction", 0.0)
    raw = min(1.0, fraction * 2.0)
    return _build_hazard_score(
        supplier.name,
        "drought",
        raw,
        {
            "dataset": "era5_precipitation",
            "version": "ERA5-Land",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "dry_month_fraction",
            "source": "ECMWF ERA5 via Open-Meteo",
            "period": "2015-01-01 to 2025-01-01",
            "mean_monthly_precip_mm": stats.get("mean_monthly_precip_mm", 0.0),
            "dry_months": stats.get("dry_months", 0),
            "total_months": stats.get("total_months", 0),
            "dry_threshold": "50% of mean",
        },
    )


# ---------------------------------------------------------------------------
# Wildfire scoring (NASA FIRMS – preprocessed local grid)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_wildfire_grid() -> dict[str, int]:
    """Load the preprocessed FIRMS wildfire grid (cached)."""
    path = _DATA_DIR / "wildfire_grid.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _lookup_wildfire_count(lat: float, lon: float) -> int:
    """Look up annual fire detections in the nearest 0.5-degree grid cell."""
    lat_bin = round(lat * 2) / 2
    lon_bin = round(lon * 2) / 2
    key = f"{lat_bin},{lon_bin}"
    return _load_wildfire_grid().get(key, 0)


def _score_wildfire(supplier: Supplier) -> HazardScore:
    count = _lookup_wildfire_count(supplier.lat, supplier.lon)
    raw = min(1.0, math.log10(1 + count) / 4.0) if count > 0 else 0.0
    lat_bin = round(supplier.lat * 2) / 2
    lon_bin = round(supplier.lon * 2) / 2
    return _build_hazard_score(
        supplier.name,
        "wildfire",
        raw,
        {
            "dataset": "viirs_active_fire_annual",
            "version": "VIIRS_NOAA20_2024",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "grid_cell_fire_count",
            "source": "NASA FIRMS VIIRS",
            "grid_resolution_deg": 0.5,
            "fire_count": count,
            "grid_cell": f"{lat_bin},{lon_bin}",
        },
    )


# ---------------------------------------------------------------------------
# Cyclone scoring (NOAA IBTrACS – preprocessed local grid)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_cyclone_grid() -> dict:
    """Load the preprocessed IBTrACS cyclone grid (cached)."""
    path = _DATA_DIR / "cyclone_grid.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _lookup_cyclone_exposure(lat: float, lon: float) -> dict:
    """Look up cyclone exposure from the nearest 1-degree grid cell."""
    lat_bin = round(lat)
    lon_bin = round(lon)
    key = f"{lat_bin},{lon_bin}"
    return _load_cyclone_grid().get(key, {"storm_count": 0, "max_wind_kt": 0})


def _score_cyclone(supplier: Supplier) -> HazardScore:
    cell = _lookup_cyclone_exposure(supplier.lat, supplier.lon)
    storm_count = cell.get("storm_count", 0)
    max_wind = cell.get("max_wind_kt", 0)
    raw = min(1.0, storm_count / 50.0)
    lat_bin = round(supplier.lat)
    lon_bin = round(supplier.lon)
    return _build_hazard_score(
        supplier.name,
        "cyclone",
        raw,
        {
            "dataset": "ibtracs",
            "version": "v04r01",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "grid_cell_storm_count",
            "source": "NOAA NCEI IBTrACS",
            "grid_resolution_deg": 1.0,
            "period": "1980-2024",
            "storm_count": storm_count,
            "max_wind_kt": max_wind,
            "min_wind_filter_kt": 34,
            "grid_cell": f"{lat_bin},{lon_bin}",
        },
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

_HAZARD_SCORERS = {
    "earthquake": _score_earthquake,
    "flood": _score_flood,
    "heat_stress": _score_heat_stress,
    "drought": _score_drought,
    "wildfire": _score_wildfire,
    "cyclone": _score_cyclone,
}


def compute_hazard(supplier: Supplier, hazard_type: str = "earthquake") -> HazardScore:
    """Return a hazard score for *supplier*.

    Dispatches to a real scoring function for each hazard type.
    Falls back to a deterministic SHA-256 hash stub for unknown types.
    """
    scorer = _HAZARD_SCORERS.get(hazard_type)
    if scorer is not None:
        return scorer(supplier)

    # Fallback: deterministic hash stub for unrecognised hazard types
    digest = hashlib.sha256(
        f"{supplier.name}:{hazard_type}".encode()
    ).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF  # 0-1

    return _build_hazard_score(
        supplier.name,
        hazard_type,
        raw,
        {
            "dataset": f"stub_{hazard_type}_v1",
            "version": "0.1.0",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": "sha256_hash_stub",
        },
    )


# ---------------------------------------------------------------------------
# Risk aggregation (pure function, used by graph node + sensitivity analysis)
# ---------------------------------------------------------------------------


def compute_risk_summary(
    hazard_scores: list[dict],
    suppliers: list[dict],
    weights: dict[str, float],
    thresholds: dict[str, float],
    high_risk_threshold: float = 0.4,
    critical_exceedance_margin: float = 0.2,
) -> dict:
    """Compute company risk summary from hazard scores with given weights/thresholds.

    Pure function with no side effects or config loading.  Used by the graph
    ``aggregate_risk`` node for normal operation and by the sensitivity
    analysis module for cheap re-aggregation under different configurations.
    """
    supplier_meta = {s["name"]: s for s in suppliers}

    rollups: dict[str, dict[str, float]] = {}
    hazard_alerts: list[dict] = []

    for hs in hazard_scores:
        supplier_name = hs["supplier_name"]
        hazard_type = hs["hazard_type"]
        score = float(hs["score"])
        weight = weights.get(hazard_type, 1.0)

        agg = rollups.setdefault(
            supplier_name, {"weighted_sum": 0.0, "weight_total": 0.0}
        )
        agg["weighted_sum"] += score * weight
        agg["weight_total"] += weight

        threshold = thresholds.get(hazard_type)
        if threshold is not None and score >= threshold:
            hazard_alerts.append({
                "supplier_name": supplier_name,
                "hazard_type": hazard_type,
                "score": round(score, 4),
                "threshold": threshold,
                "exceedance": round(score - threshold, 4),
            })

    supplier_risks: list[dict] = []
    for supplier_name, agg in rollups.items():
        weighted = 0.0
        if agg["weight_total"] > 0:
            weighted = agg["weighted_sum"] / agg["weight_total"]

        info = supplier_meta.get(supplier_name, {})
        confidence = float(info.get("confidence", 1.0))
        tier = int(info.get("tier", 1))

        # Evidence-source weighting: unverified LLM suppliers get halved
        # confidence to reflect epistemic uncertainty.
        evidence_source = info.get("evidence_source", "fixture")
        if evidence_source == "llm_only":
            effective_confidence = confidence * 0.5
        else:
            effective_confidence = confidence

        confidence_factor = 0.5 + (0.5 * effective_confidence)
        tier_factor = max(0.8, 1.0 - (0.05 * max(0, tier - 1)))
        risk_score = round(weighted * confidence_factor * tier_factor, 4)

        supplier_risks.append({
            "supplier_name": supplier_name,
            "risk_score": risk_score,
            "base_weighted_score": round(weighted, 4),
            "tier": tier,
            "confidence": confidence,
        })

    supplier_risks.sort(key=lambda x: x["risk_score"], reverse=True)
    hazard_alerts.sort(key=lambda x: x["exceedance"], reverse=True)
    critical_alerts = [
        alert
        for alert in hazard_alerts
        if alert["exceedance"] >= critical_exceedance_margin
    ]

    company_score = 0.0
    if supplier_risks:
        company_score = sum(s["risk_score"] for s in supplier_risks) / len(
            supplier_risks
        )
    company_score = round(company_score, 4)

    if company_score >= high_risk_threshold:
        risk_band = "high"
    elif company_score >= 0.25:
        risk_band = "medium"
    else:
        risk_band = "low"

    return {
        "company_score": company_score,
        "risk_band": risk_band,
        "supplier_risks": supplier_risks,
        "hazard_alerts": hazard_alerts,
        "critical_alerts": critical_alerts,
        "critical_alert_count": len(critical_alerts),
    }


# ---------------------------------------------------------------------------
# Supplier loading (JSON fixtures)
# ---------------------------------------------------------------------------


def load_suppliers(
    company: str,
    tier_depth: int,
    suppliers_path: Path | None = None,
) -> tuple[list[Supplier], list[dict], list[dict]]:
    """Load suppliers for *company* from a JSON file.

    Returns ``(suppliers, edges, evidence)`` filtered to *tier_depth*.
    Raises :class:`KeyError` if *company* is not found in the file.
    """
    path = Path(suppliers_path) if suppliers_path is not None else _DEFAULT_SUPPLIERS_PATH
    data = json.loads(path.read_text())
    source_label = "Supplier relationship dataset"

    if company not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Company '{company}' not found in {path}. "
            f"Available companies: {available}"
        )

    entry = data[company]
    now = datetime.now(timezone.utc).isoformat()

    # Build evidence and edges from raw edge descriptions
    evidence: list[dict] = []
    edges: list[dict] = []
    child_evidence_map: dict[str, list[str]] = {}

    for i, raw_edge in enumerate(entry["edges"], start=1):
        eid = f"E{i}"
        child_name = raw_edge["child"]

        evidence.append({
            "evidence_id": eid,
            "source": source_label,
            "description": raw_edge["description"],
            "retrieved_at": now,
        })
        edges.append({
            "parent": raw_edge["parent"],
            "child": child_name,
            "evidence_ids": [eid],
        })
        child_evidence_map.setdefault(child_name, []).append(eid)

    # Build Supplier objects filtered by tier_depth
    suppliers: list[Supplier] = []
    for raw in entry["suppliers"]:
        if raw["tier"] > tier_depth:
            continue
        suppliers.append(Supplier(
            name=raw["name"],
            lat=raw["lat"],
            lon=raw["lon"],
            tier=raw["tier"],
            confidence=raw["confidence"],
            evidence_ids=child_evidence_map.get(raw["name"], []),
            industry=raw.get("industry", ""),
            product_category=raw.get("product_category", ""),
            location_description=raw.get("location_description", ""),
            relationship_type=raw.get("relationship_type", ""),
        ))

    # Prune edges and evidence to only surviving suppliers
    surviving = {s.name for s in suppliers}
    edges = [e for e in edges if e["child"] in surviving]
    surviving_eids = {eid for e in edges for eid in e["evidence_ids"]}
    evidence = [ev for ev in evidence if ev["evidence_id"] in surviving_eids]

    return suppliers, edges, evidence


# ---------------------------------------------------------------------------
# LLM-based company resolution
# ---------------------------------------------------------------------------


def resolve_company_profile(company: str) -> dict:
    """Resolve a company name to a structured profile using GPT-4o.

    Returns a dict with canonical_name, industry, products, headquarters,
    and description.  Used to provide context for supplier discovery.
    """
    prompts = load_prompts()
    template = prompts["company_profile_prompt"]

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=60,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(CompanyProfile)

    prompt_text = template.format(company=company)
    response: CompanyProfile = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return response.model_dump()


# ---------------------------------------------------------------------------
# LLM-based supplier discovery
# ---------------------------------------------------------------------------


def _deduplicate_suppliers(
    suppliers: list[Supplier],
) -> list[Supplier]:
    """Merge duplicate suppliers (same name, case-insensitive).

    Keeps the entry with the highest confidence and merges evidence_ids
    from all occurrences.
    """
    seen: dict[str, Supplier] = {}

    for s in suppliers:
        key = s.name.strip().lower()
        if key in seen:
            existing = seen[key]
            merged_eids = list(
                dict.fromkeys(existing.evidence_ids + s.evidence_ids)
            )
            if s.confidence > existing.confidence:
                seen[key] = s.model_copy(update={"evidence_ids": merged_eids})
            else:
                seen[key] = existing.model_copy(update={"evidence_ids": merged_eids})
        else:
            seen[key] = s

    return list(seen.values())


def discover_suppliers_llm(
    company: str,
    tier_depth: int,
    company_profile: dict | None = None,
) -> tuple[list[Supplier], list[dict], list[dict]]:
    """Discover suppliers for *company* using GPT-4o structured output.

    Makes one LLM call per (parent, tier) pair.  Returns
    ``(suppliers, edges, evidence)`` with evidence citing the LLM.

    Parameters
    ----------
    company_profile : dict | None
        If provided, injects industry and product context into the
        tier expansion prompt for more accurate discovery.
    """
    prompts = load_prompts()
    template = prompts["tier_expansion_prompt"]

    # Build context from company profile if available.
    industry = "Unknown"
    products_str = "Unknown"
    if company_profile:
        industry = company_profile.get("industry", "Unknown")
        products = company_profile.get("products", [])
        products_str = ", ".join(products) if products else "Unknown"

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(TierResponse)

    all_suppliers: list[Supplier] = []
    all_edges: list[dict] = []
    all_evidence: list[dict] = []
    evidence_counter = 0

    # Tier 1 parents = [company]; tier N parents = tier N-1 supplier names
    parents: list[str] = [company]

    for tier in range(1, tier_depth + 1):
        next_parents: list[str] = []

        for parent in parents:
            prompt_text = template.format(
                company=parent,
                tier=tier,
                industry=industry,
                products=products_str,
            )
            response: TierResponse = structured_llm.invoke(
                [HumanMessage(content=prompt_text)]
            )

            now = datetime.now(timezone.utc).isoformat()

            for llm_sup in response.suppliers:
                evidence_counter += 1
                eid = f"E{evidence_counter}"

                all_suppliers.append(Supplier(
                    name=llm_sup.name,
                    lat=llm_sup.lat,
                    lon=llm_sup.lon,
                    tier=tier,
                    confidence=llm_sup.confidence,
                    evidence_ids=[eid],
                    industry=llm_sup.industry,
                    product_category=llm_sup.product_category,
                    location_description=llm_sup.location_description,
                    relationship_type=llm_sup.relationship_type,
                ))
                all_edges.append({
                    "parent": parent,
                    "child": llm_sup.name,
                    "evidence_ids": [eid],
                })
                all_evidence.append({
                    "evidence_id": eid,
                    "source": "LLM:gpt-4o",
                    "description": llm_sup.rationale,
                    "retrieved_at": now,
                })
                next_parents.append(llm_sup.name)

        parents = next_parents

    all_suppliers = _deduplicate_suppliers(all_suppliers)

    return all_suppliers, all_edges, all_evidence


# ---------------------------------------------------------------------------
# LLM-based mitigation generation
# ---------------------------------------------------------------------------


def _build_risk_context(
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> str:
    """Build a plain-text context block for the mitigation LLM prompt."""
    lines: list[str] = []

    # Company-level summary
    lines.append(f"Company risk score: {summary.get('company_score', 0):.4f}")
    lines.append(f"Risk band: {summary.get('risk_band', 'unknown')}")
    lines.append("")

    # Supplier details
    supplier_map = {s["name"]: s for s in suppliers}
    lines.append("Suppliers:")
    for sr in summary.get("supplier_risks", []):
        name = sr["supplier_name"]
        info = supplier_map.get(name, {})
        lines.append(
            f"  - {name}: lat={info.get('lat', '?')}, lon={info.get('lon', '?')}, "
            f"tier={sr['tier']}, confidence={sr['confidence']:.2f}, "
            f"composite_risk={sr['risk_score']:.4f}"
        )
    lines.append("")

    # Medium/High hazard scores with metadata
    lines.append("Hazard scores (Medium and High only):")
    for h in hazard_scores:
        if h.get("level", "Low") not in ("Medium", "High"):
            continue
        meta = h.get("dataset_metadata", {})
        meta_parts = []
        for k, v in meta.items():
            if k in ("dataset", "source", "method", "version", "retrieved_at"):
                continue
            meta_parts.append(f"{k}={v}")
        lines.append(
            f"  - {h['supplier_name']} / {h['hazard_type']}: "
            f"score={h.get('score_100', 0)}/100 ({h.get('level', '')})"
            + (f" [{', '.join(meta_parts)}]" if meta_parts else "")
        )
    lines.append("")

    # Critical alerts
    critical = summary.get("critical_alerts", [])
    if critical:
        lines.append("Critical threshold alerts:")
        for a in critical:
            lines.append(
                f"  - {a['supplier_name']} / {a['hazard_type']}: "
                f"score={a['score']:.4f}, threshold={a['threshold']:.2f}, "
                f"exceedance={a['exceedance']:.4f}"
            )

    return "\n".join(lines)


def generate_mitigations_llm(
    company: str,
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> list[dict]:
    """Generate mitigation strategies using GPT-4o structured output."""
    prompts = load_prompts()
    template = prompts["mitigation_prompt"]

    risk_context = _build_risk_context(suppliers, hazard_scores, summary)
    prompt_text = template.format(company=company, risk_context=risk_context)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(MitigationResponse)

    response: MitigationResponse = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return [m.model_dump() for m in response.mitigations]


# ---------------------------------------------------------------------------
# LLM-based alternative supplier suggestions
# ---------------------------------------------------------------------------


def suggest_alternatives_llm(
    company: str,
    suppliers: list[dict],
    hazard_scores: list[dict],
    summary: dict,
) -> list[dict]:
    """Suggest lower-risk alternative suppliers using GPT-4o structured output."""
    prompts = load_prompts()
    template = prompts["suggest_alternatives_prompt"]

    risk_context = _build_risk_context(suppliers, hazard_scores, summary)
    prompt_text = template.format(company=company, risk_context=risk_context)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        timeout=90,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(AlternativeResponse)

    response: AlternativeResponse = structured_llm.invoke(
        [HumanMessage(content=prompt_text)]
    )

    return [a.model_dump() for a in response.alternatives]


# ---------------------------------------------------------------------------
# Pipeline web verification (Condition B — deterministic order)
# ---------------------------------------------------------------------------

_RELATIONSHIP_CUES = frozenset({
    "supplier", "vendor", "manufactures", "supplies",
    "partner", "contract", "sources from", "procures from",
    "supply chain", "component supplier",
})


def verify_suppliers_batch(
    suppliers: list[dict],
    company: str,
    budget: "BudgetTracker",  # noqa: F821 — forward ref
    snapshot_mode: bool = False,
) -> list[dict]:
    """Verify suppliers against web evidence in descending confidence order.

    This is the deterministic pipeline verification (Condition B).
    Suppliers are verified in a fixed order (highest confidence first)
    until the web budget is exhausted. Uses the same co-mention +
    relationship-cue matching as the agent's ``verify_supplier`` tool.

    Parameters
    ----------
    suppliers : list[dict]
        Supplier dicts from GraphState.
    company : str
        The target company name.
    budget : BudgetTracker
        Budget tracker; stops when web budget is exhausted.
    snapshot_mode : bool
        If *True*, only cached search results are used.

    Returns
    -------
    list[dict]
        Updated supplier dicts with ``evidence_source`` set to
        ``"web_verified"`` where relationship evidence was found.
    """
    from bor_risk.search import search_web, search_web_snapshot

    search_fn = search_web_snapshot if snapshot_mode else search_web

    # Sort by descending confidence (deterministic order).
    ordered = sorted(suppliers, key=lambda s: s.get("confidence", 0), reverse=True)
    updated = {s["name"]: dict(s) for s in suppliers}
    evidence_items: list[dict] = []

    company_lower = company.lower()
    web_evidence_counter = 1

    for s in ordered:
        if budget.web_budget_remaining <= 0:
            break

        supplier_name = s["name"]
        supplier_lower = supplier_name.lower()
        query = (
            f'"{company}" "{supplier_name}" '
            f"supplier OR vendor OR manufactures OR supplies"
        )
        budget.record_web_query(query=query)

        try:
            results = search_fn(query, max_results=3)
        except Exception:
            continue

        verified = False
        for r in results:
            content_lower = r["content"].lower()
            has_co_mention = (
                company_lower in content_lower
                and supplier_lower in content_lower
            )
            has_relationship = any(
                cue in content_lower for cue in _RELATIONSHIP_CUES
            )

            if has_co_mention and has_relationship:
                updated[supplier_name]["evidence_source"] = "web_verified"
                updated[supplier_name]["verification_url"] = r["url"]
                updated[supplier_name]["verification_snippet"] = r["content"][:300]
                updated[supplier_name]["confidence"] = min(
                    1.0, updated[supplier_name].get("confidence", 0.5) + 0.1
                )
                eid = f"W{web_evidence_counter}"
                web_evidence_counter += 1
                evidence_items.append({
                    "evidence_id": eid,
                    "source": f"web:{r['url']}",
                    "description": r["content"][:300],
                    "retrieved_at": r.get("retrieved_at", ""),
                })
                verified = True
                break

        if not verified:
            # Supplier remains llm_only.
            updated[supplier_name].setdefault("evidence_source", "llm_only")

    return list(updated.values()), evidence_items
