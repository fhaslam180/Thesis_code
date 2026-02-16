"""Report formatting — enhanced Phase 8 report."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from bor_risk.models import GraphState
from bor_risk.utils import load_hazards

_DATASET_URLS: dict[str, tuple[str, str]] = {
    "usgs_earthquake_catalog": (
        "USGS",
        "https://earthquake.usgs.gov/fdsnws/event/1/count",
    ),
    "glofas_river_discharge": (
        "Copernicus / Open-Meteo",
        "https://flood-api.open-meteo.com/v1/flood",
    ),
    "era5_reanalysis": (
        "ECMWF / Open-Meteo",
        "https://archive-api.open-meteo.com/v1/archive",
    ),
    "era5_precipitation": (
        "ECMWF / Open-Meteo",
        "https://archive-api.open-meteo.com/v1/archive",
    ),
    "viirs_active_fire_annual": (
        "NASA FIRMS",
        "https://firms.modaps.eosdis.nasa.gov/",
    ),
    "ibtracs": (
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/products/international-best-track-archive",
    ),
}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_executive_summary(
    company: str,
    suppliers: list[dict],
    summary: dict,
) -> list[str]:
    """Deterministic executive summary paragraph."""
    score = summary.get("company_score", 0.0)
    band = summary.get("risk_band", "unknown")
    critical_count = summary.get("critical_alert_count", 0)
    supplier_risks = summary.get("supplier_risks", [])

    top_name = supplier_risks[0]["supplier_name"] if supplier_risks else "N/A"
    top_score = supplier_risks[0]["risk_score"] if supplier_risks else 0.0

    lines = [
        "--- Executive Summary ---",
        "",
        f"  This report assesses the supply-chain risk exposure of {company} across "
        f"{len(suppliers)} analysed supplier(s). The overall company risk band is "
        f"{band.upper()} (score: {score:.4f}). The highest-risk supplier is "
        f"{top_name} with a composite risk score of {top_score:.4f}. "
        f"{critical_count} critical threshold alert(s) were identified"
        + (" requiring immediate attention." if critical_count > 0 else "."),
        "",
    ]
    return lines


def _build_risk_register_matrix(
    suppliers: list[dict],
    hazards: list[dict],
) -> list[str]:
    """Fixed-width text table: supplier rows x hazard columns."""
    hazard_names = [h["name"] for h in load_hazards()]

    lookup: dict[tuple[str, str], tuple[int, str]] = {}
    for h in hazards:
        key = (h["supplier_name"], h["hazard_type"])
        lookup[key] = (h.get("score_100", round(h["score"] * 100)), h.get("level", ""))

    supplier_names = []
    seen = set()
    for s in suppliers:
        if s["name"] not in seen:
            supplier_names.append(s["name"])
            seen.add(s["name"])

    name_col_w = max((len(n) for n in supplier_names), default=10) + 2
    name_col_w = max(name_col_w, 12)
    haz_col_w = 12

    lines = ["", "--- Risk Register Matrix ---"]

    header = "  " + "Supplier".ljust(name_col_w) + "|"
    sep = "  " + "-" * name_col_w + "|"
    for hz in hazard_names:
        header += hz.center(haz_col_w) + "|"
        sep += "-" * haz_col_w + "|"
    lines.append(header)
    lines.append(sep)

    for sname in supplier_names:
        row = "  " + sname.ljust(name_col_w) + "|"
        for hz in hazard_names:
            s100, lvl = lookup.get((sname, hz), (0, "Low"))
            cell = f"{s100:>3} {lvl[:3]}"
            row += cell.center(haz_col_w) + "|"
        lines.append(row)

    return lines


def _build_mitigations(state: GraphState) -> list[str]:
    """Build mitigation section from LLM output."""
    llm_mitigations = state.get("llm_mitigations", [])
    lines = ["", "--- Mitigations ---"]

    # LLM-generated mitigations with structured rationale/actions.
    for item in llm_mitigations:
        actions = item.get("actions", [])
        if not actions:
            continue
        supplier = item.get("supplier_name", "Unknown")
        hazard_type = item.get("hazard_type", "unknown")
        level = item.get("level", "Medium")
        priority = item.get("priority", "P2")
        rationale = item.get("rationale", "")
        lines.append(f"  {supplier} / {hazard_type} ({level}) [{priority}]:")
        if rationale:
            lines.append(f"    Rationale: {rationale}")
        for action in actions:
            lines.append(f"    - {action}")

    if not any(item.get("actions") for item in llm_mitigations):
        lines.append("  No mitigations generated.")

    return lines


def _build_suggested_alternatives(state: GraphState) -> list[str]:
    """Build the suggested alternative suppliers section."""
    alternatives = state.get("suggested_alternatives", [])
    lines = ["", "--- Suggested Alternative Suppliers ---"]

    if not alternatives:
        lines.append("  No alternatives suggested.")
        return lines

    for suggestion in alternatives:
        supplier = suggestion.get("alternative_for", "Unknown")
        hazard = suggestion.get("hazard_type", "unknown")
        rationale = suggestion.get("rationale", "")
        candidates = suggestion.get("candidates", [])

        lines.append(f"  Replace {supplier} (dominant hazard: {hazard}):")
        if rationale:
            lines.append(f"    Rationale: {rationale}")
        for i, c in enumerate(candidates, 1):
            conf = c.get("confidence", 0)
            lines.append(
                f"    {i}. {c.get('name', 'Unknown')} "
                f"(lat={c.get('lat', '?')}, lon={c.get('lon', '?')}, "
                f"confidence={conf:.2f})"
            )
            if c.get("rationale"):
                lines.append(f"       {c['rationale']}")

    return lines


def _build_evidence_appendix(
    evidence: list[dict],
    hazards: list[dict],
) -> list[str]:
    """Evidence items followed by dataset metadata grouped by supplier."""
    lines = ["", "--- Evidence Appendix ---"]

    for e in evidence:
        lines.append(f"  [{e['evidence_id']}] {e['source']}: {e['description']}")

    by_supplier: dict[str, list[dict]] = defaultdict(list)
    for h in hazards:
        by_supplier[h["supplier_name"]].append(h)

    if by_supplier:
        lines.append("")
        lines.append("  Dataset Details:")
        for sname, scores in by_supplier.items():
            lines.append(f"    {sname}:")
            for h in scores:
                meta = h.get("dataset_metadata", {})
                ds = meta.get("dataset", "unknown")
                parts = [f"dataset={ds}"]
                for k, v in meta.items():
                    if k in ("dataset", "source", "method", "version"):
                        continue
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.2f}")
                    else:
                        parts.append(f"{k}={v}")
                lines.append(f"      {h['hazard_type']}: {', '.join(parts)}")

    return lines


def _build_ieee_references(hazards: list[dict]) -> list[str]:
    """Deduplicated IEEE-style references from hazard score metadata."""
    seen: dict[str, dict] = {}
    for h in hazards:
        meta = h.get("dataset_metadata", {})
        ds = meta.get("dataset", "")
        if ds and ds not in seen:
            seen[ds] = meta

    lines = ["", "--- IEEE References ---"]
    today = date.today().isoformat()

    for i, (ds, meta) in enumerate(seen.items(), 1):
        source = meta.get("source", "Unknown")
        version = meta.get("version", "")
        org, url = _DATASET_URLS.get(ds, ("Unknown", ""))

        ref = f'  [{i}] {org}, "{source},"'
        if version:
            ref += f" {ds} {version},"
        else:
            ref += f" {ds},"
        if url:
            ref += f" {url},"
        ref += f" accessed {today}."
        lines.append(ref)

    return lines


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------


def format_report(state: GraphState) -> str:
    """Produce a plain-text report from the graph state."""
    company = state.get("company", "Unknown")
    suppliers = state.get("suppliers", [])
    edges = state.get("edges", [])
    hazards = state.get("hazard_scores", [])
    evidence = state.get("evidence", [])
    summary = state.get("company_risk_summary", {})
    decision = state.get("workflow_decision", {})
    actions = state.get("workflow_actions", [])
    trace = state.get("workflow_trace", [])

    lines: list[str] = [
        f"Supply-Chain Risk Report: {company}",
        "=" * 50,
        "",
    ]

    # Executive Summary
    lines += _build_executive_summary(company, suppliers, summary)

    # Company Profile (from LLM entity resolution)
    company_profile = state.get("company_profile", {})
    if company_profile:
        lines.append("--- Company Profile ---")
        lines.append(f"  Name: {company_profile.get('canonical_name', company)}")
        lines.append(f"  Industry: {company_profile.get('industry', 'N/A')}")
        products = company_profile.get("products", [])
        if products:
            lines.append(f"  Products: {', '.join(products)}")
        lines.append(f"  Headquarters: {company_profile.get('headquarters', 'N/A')}")
        desc = company_profile.get("description", "")
        if desc:
            lines.append(f"  Description: {desc}")
        lines.append("")

    # Overview counts
    lines.append(f"Suppliers analysed: {len(suppliers)}")
    lines.append(f"Edges mapped: {len(edges)}")
    lines.append(f"Evidence items: {len(evidence)}")

    # Agent Workflow (unchanged markers)
    lines += [
        "",
        "--- Agent Workflow ---",
    ]

    if trace:
        lines.append(f"  Trace: {' -> '.join(trace)}")
    if summary:
        lines.append(f"  Company risk score: {summary.get('company_score', 0.0):.4f}")
        lines.append(f"  Risk band: {summary.get('risk_band', 'unknown')}")
    if decision:
        lines.append(f"  Decision route: {decision.get('route', 'unknown')}")
        lines.append(f"  Decision reason: {decision.get('reason', 'n/a')}")

    if actions:
        lines.append("  Planned actions:")
        for action in actions:
            lines.append(f"    - [{action['priority']}] {action['action']}")

    # Supplier Risk Ranking (enriched with industry/location)
    supplier_risks = summary.get("supplier_risks", [])
    supplier_map = {s["name"]: s for s in suppliers}
    if supplier_risks:
        lines.append("")
        lines.append("--- Supplier Risk Ranking ---")
        for item in supplier_risks:
            name = item["supplier_name"]
            info = supplier_map.get(name, {})
            line = (
                f"  {name}: risk={item['risk_score']:.4f}, "
                f"tier={item['tier']}, confidence={item['confidence']:.2f}"
            )
            industry = info.get("industry", "")
            product_cat = info.get("product_category", "")
            location_desc = info.get("location_description", "")
            extras = []
            if industry:
                extras.append(f"industry={industry}")
            if product_cat:
                extras.append(f"supplies={product_cat}")
            if location_desc:
                extras.append(f"location={location_desc}")
            if extras:
                line += f", {', '.join(extras)}"
            lines.append(line)

    # Threshold Alerts (unchanged)
    hazard_alerts = summary.get("hazard_alerts", [])
    if hazard_alerts:
        lines.append("")
        lines.append("--- Threshold Alerts ---")
        for alert in hazard_alerts:
            lines.append(
                "  "
                f"{alert['supplier_name']} / {alert['hazard_type']}: "
                f"{alert['score']:.4f} (threshold {alert['threshold']:.2f}, "
                f"exceedance {alert['exceedance']:.4f})"
            )

    # Risk Register Matrix (new)
    lines += _build_risk_register_matrix(suppliers, hazards)

    # Hazard Summary (unchanged)
    lines += [
        "",
        "--- Hazard Summary ---",
    ]
    for h in hazards:
        s100 = h.get("score_100", round(h["score"] * 100))
        lvl = h.get("level", "")
        lines.append(
            f"  {h['supplier_name']}: {h['hazard_type']} = {s100}/100 ({lvl})"
        )

    # Mitigations (LLM first, deterministic fallback)
    lines += _build_mitigations(state)

    # Suggested Alternative Suppliers
    lines += _build_suggested_alternatives(state)

    # Evidence Appendix (enhanced)
    lines += _build_evidence_appendix(evidence, hazards)

    # IEEE References (enhanced)
    lines += _build_ieee_references(hazards)

    lines.append("")
    return "\n".join(lines)
