"""Agent tools for the ReAct supply-chain risk agent.

Each tool is a LangChain ``@tool`` closure over a shared accumulator dict
and :class:`BudgetTracker`.  The accumulator collects suppliers, edges,
evidence, and hazard scores as the agent works.

The ``verify_supplier`` tool searches for **relationship evidence** — a
web source that co-mentions the company and supplier with relationship cues.
This prevents false positives from name-only matching.
"""

from __future__ import annotations

from langchain_core.tools import tool

from bor_risk.budget import BudgetTracker
from bor_risk.models import HazardScore, Supplier
from bor_risk.search import search_web, search_web_snapshot
from bor_risk.tools import (
    _build_risk_context,
    compute_hazard,
    compute_risk_summary,
    discover_suppliers_llm,
    generate_mitigations_llm,
    resolve_company_profile,
    suggest_alternatives_llm,
)
from bor_risk.utils import load_hazards

# Relationship cue words for co-mention verification.
_RELATIONSHIP_CUES = frozenset({
    "supplier",
    "vendor",
    "manufactures",
    "supplies",
    "partner",
    "contract",
    "sources from",
    "procures from",
    "supply chain",
    "component supplier",
})


def build_agent_tools(
    acc: dict,
    budget: BudgetTracker,
    enable_web: bool = True,
    snapshot_mode: bool = False,
) -> list:
    """Build the tool list for the ReAct agent.

    Parameters
    ----------
    acc : dict
        Shared mutable accumulator that collects state (suppliers, edges,
        evidence, hazard_scores, etc.) across tool calls.
    budget : BudgetTracker
        Tracks LLM calls and web queries against caps.
    enable_web : bool
        If *False*, web search and verification tools are excluded
        (Condition C: agent without web).
    snapshot_mode : bool
        If *True*, web searches use cache only (no live queries).
    """

    # Choose the search function based on snapshot mode.
    _search_fn = search_web_snapshot if snapshot_mode else search_web

    # ------------------------------------------------------------------
    # Tool 1: Profile a company
    # ------------------------------------------------------------------
    @tool
    def profile_company(company: str) -> str:
        """Look up a company's industry, products, and headquarters.

        Uses GPT-4o to resolve company identity. Costs 1 LLM call.
        """
        if budget.llm_budget_remaining <= 0:
            return "LLM budget exhausted. Cannot profile company."
        budget.record_llm_call(purpose="profile_company", company=company)
        try:
            profile = resolve_company_profile(company)
            acc["company_profile"] = profile
            return (
                f"Company: {profile.get('canonical_name', company)}\n"
                f"Industry: {profile.get('industry', 'Unknown')}\n"
                f"Products: {', '.join(profile.get('products', []))}\n"
                f"HQ: {profile.get('headquarters', 'Unknown')}\n"
                f"Description: {profile.get('description', '')}"
            )
        except Exception as e:
            return f"Failed to profile company: {e}"

    # ------------------------------------------------------------------
    # Tool 2: Discover suppliers
    # ------------------------------------------------------------------
    @tool
    def discover_suppliers(company: str, tier: int = 1) -> str:
        """Discover candidate suppliers for a company using GPT-4o.

        These are HYPOTHESES that should be verified with verify_supplier.
        Costs 1 LLM call. Returns supplier names and locations.
        """
        if budget.llm_budget_remaining <= 0:
            return "LLM budget exhausted. Cannot discover suppliers."
        budget.record_llm_call(
            purpose="discover_suppliers", company=company, tier=tier
        )
        try:
            profile = acc.get("company_profile")
            suppliers, edges, evidence = discover_suppliers_llm(
                company, tier, company_profile=profile
            )
            # Merge into accumulator
            existing_names = {s["name"].lower() for s in acc["suppliers"]}
            for s in suppliers:
                if s.name.lower() not in existing_names:
                    d = s.model_dump()
                    d["evidence_source"] = "llm_only"
                    acc["suppliers"].append(d)
                    existing_names.add(s.name.lower())
            acc["edges"].extend(edges)
            acc["evidence"].extend(evidence)

            names = [s.name for s in suppliers]
            return (
                f"Discovered {len(suppliers)} tier-{tier} suppliers: "
                f"{', '.join(names)}. "
                f"These are LLM hypotheses (evidence_source=llm_only). "
                f"Use verify_supplier to check key candidates."
            )
        except Exception as e:
            return f"Failed to discover suppliers: {e}"

    # ------------------------------------------------------------------
    # Tool 3: Web search
    # ------------------------------------------------------------------
    @tool
    def web_search(query: str) -> str:
        """Search the web for supply chain information.

        Costs 1 web query. Returns titles, URLs, and content excerpts.
        """
        if budget.web_budget_remaining <= 0:
            return "Web budget exhausted. Cannot search."
        budget.record_web_query(query=query)
        try:
            results = _search_fn(query, max_results=5)
            if not results:
                return "No results found."
            lines = []
            for r in results[:5]:
                lines.append(
                    f"- {r['title']}\n  URL: {r['url']}\n"
                    f"  {r['content'][:200]}..."
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Search failed: {e}"

    # ------------------------------------------------------------------
    # Tool 4: Verify supplier relationship
    # ------------------------------------------------------------------
    @tool
    def verify_supplier(supplier_name: str, parent_company: str) -> str:
        """Verify that supplier_name has a supply relationship with parent_company.

        Searches the web for co-mentions of both names with relationship
        cues (supplier, vendor, manufactures, etc.). If verified, the
        supplier's evidence_source is upgraded to 'web_verified' and
        confidence gets a boost. Costs 1 web query.
        """
        if budget.web_budget_remaining <= 0:
            return "Web budget exhausted. Cannot verify."
        query = (
            f'"{parent_company}" "{supplier_name}" '
            f"supplier OR vendor OR manufactures OR supplies"
        )
        budget.record_web_query(query=query)
        try:
            results = _search_fn(query, max_results=3)
        except Exception as e:
            return f"Verification search failed: {e}"

        company_lower = parent_company.lower()
        supplier_lower = supplier_name.lower()

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
                # Upgrade supplier in accumulator.
                for s in acc["suppliers"]:
                    if s["name"].lower() == supplier_lower:
                        s["evidence_source"] = "web_verified"
                        s["verification_url"] = r["url"]
                        s["verification_snippet"] = r["content"][:300]
                        s["confidence"] = min(1.0, s["confidence"] + 0.1)

                # Record web evidence.
                counter = acc.get("_web_evidence_counter", 1)
                eid = f"W{counter}"
                acc["_web_evidence_counter"] = counter + 1
                acc["evidence"].append({
                    "evidence_id": eid,
                    "source": f"web:{r['url']}",
                    "description": r["content"][:300],
                    "retrieved_at": r.get("retrieved_at", ""),
                })
                return (
                    f"VERIFIED: Found relationship evidence for "
                    f"{supplier_name} as supplier to {parent_company} "
                    f"at {r['url']}"
                )

        return (
            f"UNVERIFIED: No relationship evidence found for "
            f"{supplier_name} as supplier to {parent_company}"
        )

    # ------------------------------------------------------------------
    # Tool 5: Score a hazard
    # ------------------------------------------------------------------
    @tool
    def score_hazard(
        supplier_name: str,
        lat: float,
        lon: float,
        hazard_type: str,
    ) -> str:
        """Score one supplier for one hazard type.

        This is FREE (deterministic, no budget cost). Available hazard
        types: earthquake, flood, heat_stress, drought, wildfire, cyclone.
        """
        supplier = Supplier(
            name=supplier_name, lat=lat, lon=lon, tier=1, confidence=1.0
        )
        budget.record_hazard_score(supplier=supplier_name, hazard=hazard_type)
        hs: HazardScore = compute_hazard(supplier, hazard_type)
        acc["hazard_scores"].append(hs.model_dump())
        return (
            f"{supplier_name} / {hazard_type}: "
            f"score={hs.score_100}/100 ({hs.level}). "
            f"Details: {hs.dataset_metadata}"
        )

    # ------------------------------------------------------------------
    # Tool 6: Aggregate risk
    # ------------------------------------------------------------------
    @tool
    def aggregate_risk() -> str:
        """Compute the overall company risk summary from all hazard scores.

        This is FREE (deterministic). Call after scoring hazards.
        """
        hazard_defs = load_hazards()
        weights = {h["name"]: float(h.get("weight", 1.0)) for h in hazard_defs}
        thresholds = {
            h["name"]: float(h.get("threshold", 1.0)) for h in hazard_defs
        }
        summary = compute_risk_summary(
            acc.get("hazard_scores", []),
            acc.get("suppliers", []),
            weights,
            thresholds,
        )
        acc["company_risk_summary"] = summary
        band = summary.get("risk_band", "unknown")
        score = summary.get("company_score", 0)
        alerts = summary.get("critical_alert_count", 0)
        top = summary.get("supplier_risks", [])[:3]
        top_str = ", ".join(
            f"{s['supplier_name']} ({s['risk_score']:.2f})" for s in top
        )
        return (
            f"Company risk: {score:.4f} ({band}). "
            f"Critical alerts: {alerts}. "
            f"Top-risk suppliers: {top_str}"
        )

    # ------------------------------------------------------------------
    # Tool 7: Generate mitigations
    # ------------------------------------------------------------------
    @tool
    def generate_mitigations(company: str) -> str:
        """Generate mitigation strategies for high-risk supplier-hazard pairs.

        Uses GPT-4o. Costs 1 LLM call.
        """
        if budget.llm_budget_remaining <= 0:
            return "LLM budget exhausted. Cannot generate mitigations."
        budget.record_llm_call(purpose="generate_mitigations", company=company)
        try:
            mitigations = generate_mitigations_llm(
                company=company,
                suppliers=acc.get("suppliers", []),
                hazard_scores=acc.get("hazard_scores", []),
                summary=acc.get("company_risk_summary", {}),
            )
            acc["llm_mitigations"] = mitigations
            return (
                f"Generated {len(mitigations)} mitigation strategies. "
                f"Covering: {', '.join(m['supplier_name'] + '/' + m['hazard_type'] for m in mitigations[:5])}"
            )
        except Exception as e:
            return f"Failed to generate mitigations: {e}"

    # ------------------------------------------------------------------
    # Tool 8: Suggest alternatives
    # ------------------------------------------------------------------
    @tool
    def suggest_alternatives(company: str) -> str:
        """Suggest lower-risk alternative suppliers for high-risk nodes.

        Uses GPT-4o. Costs 1 LLM call.
        """
        if budget.llm_budget_remaining <= 0:
            return "LLM budget exhausted. Cannot suggest alternatives."
        budget.record_llm_call(
            purpose="suggest_alternatives", company=company
        )
        try:
            alts = suggest_alternatives_llm(
                company=company,
                suppliers=acc.get("suppliers", []),
                hazard_scores=acc.get("hazard_scores", []),
                summary=acc.get("company_risk_summary", {}),
            )
            acc["suggested_alternatives"] = alts
            return (
                f"Suggested alternatives for {len(alts)} high-risk suppliers."
            )
        except Exception as e:
            return f"Failed to suggest alternatives: {e}"

    # Build final tool list.
    tools = [
        profile_company,
        discover_suppliers,
        score_hazard,
        aggregate_risk,
        generate_mitigations,
        suggest_alternatives,
    ]

    if enable_web:
        # Insert web tools after discover_suppliers.
        tools.insert(2, web_search)
        tools.insert(3, verify_supplier)

    return tools
