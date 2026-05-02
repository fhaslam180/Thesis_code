"""Report formatting — enhanced Phase 8 report."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from datetime import date

from bor_risk.models import GraphState
from bor_risk.utils import load_hazards

_DATASET_URLS: dict[str, tuple[str, str]] = {
    "GEM_2023": (
        "GEM Foundation",
        "https://github.com/GEMScienceTools/oq-mbtk",
    ),
    "Aqueduct_Floods": (
        "WRI Aqueduct",
        "https://www.wri.org/data/aqueduct-floods-hazard-maps",
    ),
    "STORM_2020": (
        "STORM dataset",
        "https://www.science.org/doi/10.1126/sciadv.aba8275",
    ),
    "Open-Meteo ERA5": (
        "Open-Meteo / ECMWF ERA5",
        "https://archive-api.open-meteo.com/v1/archive",
    ),
}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _humanize_hazard(hazard: str) -> str:
    """Render machine hazard names as readable text."""
    return hazard.replace("_", " ")


def _join_with_and(items: list[str]) -> str:
    """Join text fragments using natural language rules."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _build_executive_summary(
    company: str,
    suppliers: list[dict],
    summary: dict,
    reference_set: dict | None = None,
) -> list[str]:
    """Deterministic executive summary with human-readable narrative."""
    score = summary.get("company_score", 0.0)
    band = summary.get("company_band", summary.get("risk_band", "unknown"))
    supplier_risks = summary.get("supplier_risks", [])

    top_name = supplier_risks[0]["supplier_name"] if supplier_risks else "N/A"
    top_score = supplier_risks[0].get("exposure_index", supplier_risks[0].get("risk_score", 0.0)) if supplier_risks else 0.0
    top_three = [
        f"{item['supplier_name']} ({item.get('exposure_index', item.get('risk_score', 0.0)):.3f})"
        for item in supplier_risks[:3]
    ]

    # Dominant hazard across all suppliers
    dominant_counts: dict[str, int] = {}
    for sr in supplier_risks:
        dh = sr.get("dominant_hazard")
        if dh:
            dominant_counts[dh] = dominant_counts.get(dh, 0) + 1
    top_dominant = sorted(dominant_counts, key=dominant_counts.get, reverse=True)[:2]  # type: ignore[arg-type]
    dominant_hazards = [
        f"{_humanize_hazard(name)} ({dominant_counts[name]} suppliers)"
        for name in top_dominant
    ]

    supplier_phrase = ""
    if top_three:
        supplier_phrase = (
            f" The top {len(top_three)} supplier(s) by exposure index are "
            f"{_join_with_and(top_three)}."
        )
    non_inform = []
    informative = []
    if reference_set:
        metadata = reference_set.get("metadata", {})
        non_inform = metadata.get("non_informative_hazards", [])
        informative = metadata.get("informative_hazards", [])
    smhei_definition = (
        f"SMHEI = equal-weight arithmetic mean of {len(informative)} informative "
        "hazard percentile ranks."
        if informative
        else "SMHEI = equal-weight arithmetic mean of hazard percentile ranks."
    )
    overview_paragraph = (
        f"  Overall company exposure is {band.upper()} "
        f"(Supplier Multi-Hazard Exposure Index mean: {score:.4f}). "
        f"The highest-exposure supplier is {top_name} (E_s = {top_score:.4f})."
        f"{supplier_phrase}"
    )

    dominant_phrase = (
        f" Dominant hazard(s) across suppliers: {_join_with_and(dominant_hazards)}."
        if dominant_hazards else ""
    )

    lines = [
        "--- Executive Summary ---",
        "",
        f"  This report provides a Supplier Multi-Hazard Exposure Index (SMHEI) "
        f"assessment for {company}'s supply chain across {len(suppliers)} supplier(s). "
        f"{smhei_definition}",
        "",
        overview_paragraph,
        "",
        f"  Exposure bands are defined by empirical tertiles of the study-specific "
        f"reference corpus.{dominant_phrase}",
        "",
    ]

    # Non-informative hazard footnote
    if non_inform:
        lines.append(
            f"  NOTE: The following hazard(s) were non-informative across the "
            f"reference corpus and assigned a neutral score of 0.5: "
            f"{', '.join(non_inform)}. They are excluded from SMHEI, severity "
            f"rankings, and mitigation generation."
        )
        lines.append("")

    return lines


def _build_narrative_assessment(
    hazards: list[dict],
    summary: dict,
    decision: dict,
    actions: list[dict],
    reference_set: dict | None = None,
) -> list[str]:
    """Narrative interpretation section before detailed data tables."""
    supplier_risks = summary.get("supplier_risks", [])
    top_suppliers = [
        f"{item['supplier_name']} ({item.get('exposure_index', item.get('risk_score', 0.0)):.3f})"
        for item in supplier_risks[:5]
    ]

    non_informative = set()
    if reference_set:
        non_informative = set(
            reference_set.get("metadata", {}).get("non_informative_hazards", [])
        )

    by_hazard: dict[str, list[float]] = defaultdict(list)
    high_count = 0
    for hazard in hazards:
        h_type = hazard.get("hazard_type", "unknown")
        if h_type in non_informative:
            continue
        score = float(hazard.get("score", 0.0))
        by_hazard[h_type].append(score)
        if hazard.get("level") == "High":
            high_count += 1

    hazard_ranked = sorted(
        (
            (hazard, sum(scores) / len(scores), len(scores))
            for hazard, scores in by_hazard.items()
            if scores
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_hazards = [
        f"{_humanize_hazard(hazard)} (avg {avg * 100:.0f}/100)"
        for hazard, avg, _ in hazard_ranked[:3]
    ]

    route = decision.get("route", "n/a")
    reason = decision.get("reason", "n/a")
    action_text = _join_with_and([
        f"[{a.get('priority', 'P2')}] {a.get('action', '').rstrip('.')}"
        for a in actions[:3]
        if a.get("action")
    ])

    lines = [
        "--- Narrative Assessment ---",
        "",
        (
            f"  Exposure is concentrated in a small subset of suppliers. "
            f"The upper end of the risk distribution is led by "
            f"{_join_with_and(top_suppliers[:3])}."
            if top_suppliers
            else "  Supplier-level concentration cannot be ranked because no supplier risk scores are available."
        ),
        "",
        (
            f"  Across all supplier-hazard combinations, {high_count} high-severity "
            f"score(s) were recorded. The strongest hazard signals are "
            f"{_join_with_and(top_hazards)}."
            if top_hazards
            else "  Hazard severity patterns cannot be profiled because no hazard scores are available."
        ),
        "",
        (
            f"  Workflow routing selected '{route}' ({reason}). "
            f"Planned near-term actions are {action_text}."
            if action_text
            else f"  Workflow routing selected '{route}' ({reason}). No explicit planned actions were recorded."
        ),
        "",
    ]
    return lines


def _build_risk_register_matrix(
    suppliers: list[dict],
    hazards: list[dict],
    reference_set: dict | None = None,
) -> list[str]:
    """Fixed-width text table: supplier rows x hazard columns."""
    hazard_names = [h["name"] for h in load_hazards()]
    non_informative = set()
    if reference_set:
        non_informative = set(
            reference_set.get("metadata", {}).get("non_informative_hazards", [])
        )

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
            if hz in non_informative:
                cell = "  N/A"
            else:
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
# Evidence source & budget sections
# ---------------------------------------------------------------------------


def _build_evidence_source_summary(suppliers: list[dict]) -> list[str]:
    """Breakdown of supplier evidence sources."""
    total = len(suppliers)
    if total == 0:
        return []

    web_verified = sum(1 for s in suppliers if s.get("evidence_source") == "web_verified")
    llm_only = sum(1 for s in suppliers if s.get("evidence_source") == "llm_only")
    fixture = sum(1 for s in suppliers if s.get("evidence_source") == "fixture")

    lines = ["", "--- Evidence Source Summary ---"]
    lines.append(f"  Web-verified suppliers: {web_verified} / {total} "
                 f"({100 * web_verified / total:.1f}%)")
    lines.append(f"  LLM-only suppliers:    {llm_only} / {total} "
                 f"({100 * llm_only / total:.1f}%)")
    lines.append(f"  Fixture suppliers:     {fixture} / {total} "
                 f"({100 * fixture / total:.1f}%)")
    return lines


def _build_claim_summary(state: GraphState) -> list[str]:
    """Verified Claim Graph summary: verdict breakdown + supporting quotes."""
    claims = state.get("claims", [])
    if not claims:
        return []

    total = len(claims)
    verdict_counts: dict[str, int] = {}
    for c in claims:
        v = c.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    lines = ["", "--- Claim Summary (Verified Claim Graph) ---"]
    lines.append(f"  Total claims: {total}")
    for verdict in ("SUPPORTED", "WEAK", "DISPUTED", "UNKNOWN"):
        count = verdict_counts.get(verdict, 0)
        pct = 100 * count / total if total > 0 else 0.0
        lines.append(f"  {verdict}: {count} ({pct:.0f}%)")

    strict_mode = state.get("strict_mode", False)
    shown = 0
    for c in claims:
        verdict = c.get("verdict", "UNKNOWN")
        if strict_mode and verdict not in ("SUPPORTED", "WEAK"):
            continue
        if verdict not in ("SUPPORTED", "WEAK"):
            continue
        spans = c.get("supporting_spans", [])
        quote = spans[0].get("quote", "") if spans else ""
        display = c.get("display_id", "?")
        obj = c.get("object_entity_id", "?")
        direction = ""
        expl = c.get("verdict_explanation", "")
        if "direction=" in expl:
            direction = expl.split("direction=")[-1].rstrip("]")
        line = f"  [{display}] {obj}: {verdict}"
        if direction and direction != "unclear":
            line += f" ({direction})"
        if quote:
            short_quote = quote[:150] + ("..." if len(quote) > 150 else "")
            line += f'\n    "{short_quote}"'
        lines.append(line)
        shown += 1
        if shown >= 10:
            break

    return lines


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------


def format_report(state: GraphState) -> str:
    """Produce a plain-text report from the graph state."""
    from bor_risk.utils import load_reference_set

    company = state.get("company", "Unknown")
    suppliers = state.get("suppliers", [])
    edges = state.get("edges", [])
    hazards = state.get("hazard_scores", [])
    evidence = state.get("evidence", [])
    summary = state.get("company_risk_summary", {})
    decision = state.get("workflow_decision", {})
    actions = state.get("workflow_actions", [])
    trace = state.get("workflow_trace", [])

    try:
        reference_set = load_reference_set()
    except Exception:
        reference_set = None

    lines: list[str] = [
        f"Supplier Multi-Hazard Exposure Index (SMHEI) Report: {company}",
        "=" * 60,
        "",
    ]

    # Executive Summary
    lines += _build_executive_summary(company, suppliers, summary, reference_set=reference_set)

    # Company Profile (from LLM entity resolution)
    company_profile = state.get("company_profile", {})
    if company_profile:
        lines.append("--- Company Profile ---")
        cname = company_profile.get("canonical_name", company)
        industry = company_profile.get("industry", "N/A")
        hq = company_profile.get("headquarters", "N/A")
        products = company_profile.get("products", [])
        product_text = ", ".join(products[:5]) if products else "n/a"
        lines.append(
            f"  {cname} operates in the {industry} sector and is headquartered in {hq}. "
            f"Key products and services include {product_text}."
        )
        desc = company_profile.get("description", "")
        if desc:
            lines.append("")
            lines.append(f"  {desc}")
        lines.append("")

    # Analysis scope in prose form.
    lines += [
        "--- Introduction ---",
        "",
        f"  The analysis covers {len(suppliers)} supplier node(s), {len(edges)} mapped "
        f"relationship edge(s), and {len(evidence)} supporting evidence item(s). "
        "Hazard scores are deterministic and derived from external geospatial "
        "or climate datasets rather than LLM-estimated risk values.",
        "",
    ]

    # Narrative interpretation up front (before data-heavy tables).
    lines += _build_narrative_assessment(
        hazards=hazards,
        summary=summary,
        decision=decision,
        actions=actions,
        reference_set=reference_set,
    )

    # Evidence Source Summary
    lines += _build_evidence_source_summary(suppliers)

    # Claim Summary (VCG)
    lines += _build_claim_summary(state)

    # Pipeline Workflow
    lines += [
        "",
        "--- Pipeline Workflow ---",
    ]

    if trace:
        lines.append(f"  Trace: {' -> '.join(trace)}")
    if summary:
        lines.append(f"  Company exposure index (mean E_s): {summary.get('company_score', 0.0):.4f}")
        lines.append(f"  Company exposure band: {summary.get('company_band', summary.get('risk_band', 'unknown'))}")
    if decision:
        lines.append(f"  Decision route: {decision.get('route', 'unknown')}")
        lines.append(f"  Decision reason: {decision.get('reason', 'n/a')}")

    if actions:
        lines.append("  Planned actions:")
        for action in actions:
            lines.append(f"    - [{action['priority']}] {action['action']}")

    # Keep detailed sections grouped at the end as an appendix-style block.
    lines += [
        "",
        "--- Detailed Data Appendix ---",
        (
            "  The sections below provide supplier-by-supplier and hazard-by-hazard "
            "supporting data for auditability and follow-up analysis."
        ),
    ]

    # Supplier SMHEI Ranking
    supplier_risks = summary.get("supplier_risks", [])
    supplier_map = {s["name"]: s for s in suppliers}
    if supplier_risks:
        lines.append("")
        lines.append("--- Supplier Multi-Hazard Exposure Index (SMHEI) Ranking ---")
        lines.append("  (E_s = exposure_index, M_s = exposure_max, dominant = dominant_hazard)")
        for item in supplier_risks:
            name = item["supplier_name"]
            info = supplier_map.get(name, {})
            ev_source = info.get("evidence_source", "")
            E_s = item.get("exposure_index", item.get("risk_score", 0.0))
            M_s = item.get("exposure_max", 0.0)
            dominant = item.get("dominant_hazard", "?")
            band = item.get("exposure_band", item.get("risk_band", "?"))
            line = (
                f"  {name}: E_s={E_s:.4f}, M_s={M_s:.4f}, "
                f"dominant={dominant}, band={band}, "
                f"tier={item['tier']}, conf={item['confidence']:.2f}"
            )
            if ev_source:
                line += f", source={ev_source}"
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
            verification_url = info.get("verification_url", "")
            if verification_url:
                lines.append(f"    Verified: {verification_url}")

    # Risk Register Matrix
    lines += _build_risk_register_matrix(suppliers, hazards, reference_set=reference_set)

    # Hazard Summary
    _HAZARD_LABELS = {
        "earthquake": "Seismic (GEM 2023 PGA, 475-yr RP)",
        "flood": "River Flood (WRI Aqueduct 100-yr depth, 0.01° grid)",
        "wildfire": "Approx. FWI Q95 (daily ERA5 stats proxy; Van Wagner 1987)",
        "cyclone": "Cyclone V_100 km/h (STORM 100-yr RP; Bloemendaal et al. 2020)",
        "heat_stress": "Apparent Heat Stress (apparent_temperature_max >= 32°C, ERA5 1991-2020)",
        "drought": "Drought (SPEI-12 fraction months <= -1; Vicente-Serrano et al. 2010)",
    }
    lines += [
        "",
        "--- Hazard Summary ---",
        "  Raw value = physical indicator; H = percentile rank (0-1 relative to reference corpus)",
    ]
    non_informative = set()
    if reference_set:
        non_informative = set(
            reference_set.get("metadata", {}).get("non_informative_hazards", [])
        )
    for h in hazards:
        s100 = h.get("score_100", round(h["score"] * 100))
        lvl = h.get("level", "")
        raw = h.get("raw_value", "?")
        raw_str = f"{raw:.4f}" if isinstance(raw, float) else str(raw)
        label = _HAZARD_LABELS.get(h["hazard_type"], h["hazard_type"])
        if h["hazard_type"] in non_informative:
            lvl = "Neutral/non-informative"
        lines.append(
            f"  {h['supplier_name']}: {label} | raw={raw_str}, H={h['score']:.4f}, {s100}/100 ({lvl})"
        )

    # Heat note (canonical sentence)
    lines.append("")
    lines.append(
        "  Heat note: Heat exposure is defined as the fraction of days with "
        "apparent_temperature_max >= 32°C over 1991-2020, derived from Open-Meteo ERA5. "
        "The 32°C value is used as an operational screening threshold in this study; "
        "it is not presented as a universal biological cutoff. "
        "Gasparrini et al. (2015) is cited for general heat-health context only."
    )
    lines.append(
        "  Wildfire note: FWI is computed from ERA5 daily statistics "
        "(temperature_2m_max, relative_humidity_2m_min, wind_speed_10m_max, "
        "precipitation_sum) via the Van Wagner (1987) recurrence formulas. "
        "This is an approximation of the standard Canadian FWI workflow, which "
        "requires noon-condition observations. Strict FWI benchmarks are not "
        "applicable to this computation."
    )

    # Mitigations (LLM first, deterministic fallback)
    lines += _build_mitigations(state)

    # Suggested Alternative Suppliers
    lines += _build_suggested_alternatives(state)

    # Evidence Appendix (enhanced)
    lines += _build_evidence_appendix(evidence, hazards)

    # IEEE References
    lines += _build_ieee_references(hazards)

    lines.append("")
    return "\n".join(lines)
