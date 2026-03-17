"""LangGraph orchestration — Verified Claim Graph (VCG) pipeline.

Single 11-node topology (conditions controlled by state flags):

  resolve_entity
    → discover_claims
    → [retrieve_evidence]       (skipped when enable_web=False)
    → [extract_mentions]        (skipped when enable_web=False)
    → [link_claims]             (skipped when enable_web=False)
    → [verify_claims]           (skipped when no_verify=True)
    → resolve_locations
    → [score_earthquake | score_flood | score_heat_stress |
       score_drought | score_wildfire | score_cyclone]   (parallel fan-out)
    → aggregate_risk
    → generate_report
    → export_artifacts
    → END

Supplier discovery uses GPT-4o (or mock JSON fixtures).
Hazard scoring is deterministic and evidence-backed.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bor_risk.budget import BudgetTracker
from bor_risk.evidence_store import EvidenceStore
from bor_risk.models import (
    Claim,
    EvidencePacket,
    EvidenceSpan,
    GraphState,
    LocationCandidate,
    Supplier,
)
from bor_risk.provenance import ProvenanceExporter
from bor_risk.report import format_report
from bor_risk.tools import (
    compute_hazard,
    compute_risk_summary,
    discover_suppliers_llm,
    extract_mentions_from_doc_llm,
    fetch_url_content,
    generate_mitigations_llm,
    link_mentions_to_claims,
    load_suppliers,
    resolve_company_profile,
    resolve_locations_for_claims,
    suggest_alternatives_llm,
    verify_claim_against_evidence,
)
from bor_risk.utils import load_hazards

HIGH_RISK_THRESHOLD = 0.4
CRITICAL_EXCEEDANCE_MARGIN = 0.2

# Hazard types to score (read from config, used for parallel node creation).
HAZARD_NAMES: list[str] = [h["name"] for h in load_hazards()]

_STORE = EvidenceStore()

# Verdict ranking for best-verdict selection (lower = better)
_VERDICT_RANK: dict[str, int] = {"SUPPORTED": 0, "DISPUTED": 1, "WEAK": 2, "UNKNOWN": 3}


# ---------------------------------------------------------------------------
# Helper: Supplier → Claim conversion
# ---------------------------------------------------------------------------


def _supplier_to_claim(supplier: Supplier, company: str, display_num: int) -> Claim:
    """Convert a discovered Supplier into a PROPOSED Claim."""
    norm = supplier.name.strip().lower()
    cid = hashlib.sha256(
        f"{company}|{supplier.name}|SUPPLIES_TO|{supplier.tier}".encode()
    ).hexdigest()[:12]
    loc = LocationCandidate(
        lat=supplier.lat,
        lon=supplier.lon,
        confidence=0.5,
        source="llm",
        place_name=supplier.location_description,
    )
    return Claim(
        claim_id=cid,
        display_id=f"C{display_num:03d}",
        subject_entity_id=company,
        object_entity_id=supplier.name,
        normalized_name=norm,
        claim_type="SUPPLIES_TO",
        status="PROPOSED",
        verdict="UNKNOWN",
        confidence=supplier.confidence,
        tier=supplier.tier,
        rationale=supplier.verification_snippet or "",
        location_candidates=[loc],
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def resolve_entity_node(state: GraphState) -> dict:
    """Node 1: Resolve company profile via GPT-4o."""
    company = state["company"]
    use_llm = state.get("use_llm", True)
    budget = state.get("_budget_tracker")

    if use_llm:
        try:
            profile = resolve_company_profile(company)
            if budget:
                budget.record_llm_call(purpose="profile_company")
        except Exception:
            profile = {}
    else:
        profile = {}

    return {
        "company_profile": profile,
        "workflow_trace": ["resolve_entity"],
    }


def discover_claims_node(state: GraphState) -> dict:
    """Node 2: Discover supplier candidates; map to PROPOSED Claims + Suppliers."""
    company = state["company"]
    tier_depth = state.get("tier_depth", 2)
    use_llm = state.get("use_llm", True)
    budget = state.get("_budget_tracker")
    profile = state.get("company_profile", {})

    if use_llm:
        suppliers, edges, evidence = discover_suppliers_llm(
            company, tier_depth, company_profile=profile or None
        )
        if budget:
            parents: list[str] = [company]
            for tier in range(1, tier_depth + 1):
                for parent in parents:
                    budget.record_llm_call(purpose=f"discover_tier{tier}_{parent}")
                parents = [s.name for s in suppliers if s.tier == tier]
    else:
        suppliers_path = state.get("suppliers_path")
        suppliers, edges, evidence = load_suppliers(
            company, tier_depth, suppliers_path
        )

    claims = [
        _supplier_to_claim(s, company, i + 1).model_dump()
        for i, s in enumerate(suppliers)
    ]

    return {
        "suppliers": [s.model_dump() for s in suppliers],
        "edges": edges,
        "evidence": evidence,
        "claims": claims,
        "workflow_trace": ["discover_claims"],
    }


def retrieve_evidence_node(state: GraphState) -> dict:
    """Node 3: Fetch web pages for each claim; build EvidencePackets.

    Skipped when ``enable_web=False``.
    """
    if not state.get("enable_web", False):
        return {"workflow_trace": ["retrieve_evidence(skipped)"]}

    import os
    if not os.environ.get("TAVILY_API_KEY"):
        print(
            "\n  [ERROR] TAVILY_API_KEY is not set — web retrieval is disabled.\n"
            "  Add TAVILY_API_KEY=<key> to your .env file and re-run.\n",
            flush=True,
        )
        return {"workflow_trace": ["retrieve_evidence(no_tavily_key)"]}

    try:
        from bor_risk.search import search_web, search_web_snapshot
    except ImportError:
        return {"workflow_trace": ["retrieve_evidence(search_unavailable)"]}

    company = state["company"]
    budget = state.get("_budget_tracker")
    snapshot_mode = state.get("snapshot_mode", False)
    search_fn = search_web_snapshot if snapshot_mode else search_web

    claims = [Claim(**c) for c in state.get("claims", [])]
    existing_packets = state.get("evidence_packets", [])
    # Build content_hash → evidence_id map so duplicate fetches link to
    # the right existing packet rather than creating orphaned records.
    existing_hash_to_eid: dict[str, str] = {
        p["content_hash"]: p["evidence_id"]
        for p in existing_packets
        if p.get("content_hash")
    }
    seen_hashes: set[str] = set(existing_hash_to_eid.keys())

    # In snapshot mode, pre-load the persistent URL index so live fetches
    # are skipped for URLs already cached on disk across previous runs.
    url_index: dict[str, dict] = _STORE.load_url_index() if snapshot_mode else {}

    now = datetime.now(timezone.utc).isoformat()
    new_packets: list[dict] = []
    claim_to_eids: dict[str, list[str]] = {}
    _search_warned = False

    for claim in claims:
        if budget and budget.web_budget_remaining <= 0:
            break

        query = (
            f'"{company}" "{claim.object_entity_id}" '
            f"supplier OR vendor OR manufacturer"
        )
        if budget:
            budget.record_web_query(query=query)

        try:
            results = search_fn(query, max_results=3)
        except Exception as exc:
            if not _search_warned:
                print(f"  [WARNING] Web search failed: {exc}", flush=True)
                _search_warned = True
            continue

        for r in results:
            url = r.get("url", "")
            if not url:
                continue

            # Snapshot mode: reuse cached page content from the persistent
            # URL index without making a live network request.
            if snapshot_mode and url in url_index:
                cached_eid = url_index[url]["evidence_id"]
                claim_to_eids.setdefault(claim.claim_id, []).append(cached_eid)
                # Ensure the packet appears in existing_hash_to_eid for dedup.
                cached_hash = url_index[url].get("content_hash", "")
                if cached_hash and cached_hash not in existing_hash_to_eid:
                    existing_hash_to_eid[cached_hash] = cached_eid
                    seen_hashes.add(cached_hash)
                continue

            plain_text, content_hash, mime_type, final_url, http_status = (
                fetch_url_content(url)
            )
            if not plain_text:
                continue

            if content_hash in seen_hashes:
                # Content already stored — link existing packet to this claim.
                eid = existing_hash_to_eid.get(content_hash)
                if eid:
                    claim_to_eids.setdefault(claim.claim_id, []).append(eid)
                continue

            seen_hashes.add(content_hash)
            packet = _STORE.store(
                url=url,
                final_url=final_url,
                content=plain_text,
                content_hash=content_hash,
                mime_type=mime_type,
                retrieved_at=now,
                http_status=http_status,
                title=r.get("title", ""),
            )
            eid = packet.evidence_id
            existing_hash_to_eid[content_hash] = eid
            new_packets.append(packet.model_dump())
            claim_to_eids.setdefault(claim.claim_id, []).append(eid)

    # Write evidence_refs back onto each claim, advancing status to RETRIEVED.
    updated_claims = [
        claim.model_copy(update={
            "evidence_refs": list(dict.fromkeys(
                list(claim.evidence_refs) + claim_to_eids.get(claim.claim_id, [])
            )),
            "status": "RETRIEVED" if claim_to_eids.get(claim.claim_id) else claim.status,
        }).model_dump()
        for claim in claims
    ]
    linked_count = sum(1 for eids in claim_to_eids.values() if eids)
    return {
        "claims": updated_claims,
        "evidence_packets": list(existing_packets) + new_packets,
        "workflow_trace": [f"retrieve_evidence({len(new_packets)} new, {linked_count} claims linked)"],
    }


def extract_mentions_node(state: GraphState) -> dict:
    """Node 4: Extract supplier mentions from each EvidencePacket via LLM.

    Skipped when ``enable_web=False``.
    """
    if not state.get("enable_web", False):
        return {"workflow_trace": ["extract_mentions(skipped)"]}

    company = state["company"]
    budget = state.get("_budget_tracker")
    packets = state.get("evidence_packets", [])

    all_mentions: list[dict] = []

    for packet_dict in packets:
        if budget and budget.llm_budget_remaining <= 0:
            break

        evidence_id = packet_dict.get("evidence_id", "")
        snapshot_path = packet_dict.get("snapshot_path")

        doc_text = ""
        if snapshot_path:
            p = Path(snapshot_path)
            if p.exists():
                doc_text = p.read_text(encoding="utf-8")
        # Defensive fallback: read via content-hash if snapshot_path is missing/stale
        if not doc_text and packet_dict.get("content_hash"):
            doc_text = _STORE.read_snapshot(EvidencePacket(**packet_dict)) or ""

        if not doc_text:
            continue

        mentions = extract_mentions_from_doc_llm(doc_text, company, evidence_id)
        if budget and mentions:
            budget.record_llm_call(purpose=f"extract_mentions_{evidence_id[:8]}")

        for m in mentions:
            m = dict(m)
            m["company"] = company
            all_mentions.append(m)

    return {
        "mentions": all_mentions,
        "workflow_trace": [f"extract_mentions({len(all_mentions)})"],
    }


def link_claims_node(state: GraphState) -> dict:
    """Node 5: Link extracted mentions to PROPOSED Claims.

    Skipped when ``enable_web=False``.
    """
    if not state.get("enable_web", False):
        return {"workflow_trace": ["link_claims(skipped)"]}

    claims = [Claim(**c) for c in state.get("claims", [])]
    mentions = state.get("mentions", [])

    if not mentions:
        return {"workflow_trace": ["link_claims(0 mentions)"]}

    updated = link_mentions_to_claims(mentions, claims)

    return {
        "claims": [c.model_dump() for c in updated],
        "workflow_trace": [f"link_claims({len(mentions)} mentions)"],
    }


def verify_claims_node(state: GraphState) -> dict:
    """Node 6: Two-stage verification of each claim against evidence.

    Skipped when ``no_verify=True``.
    """
    if state.get("no_verify", False):
        return {"workflow_trace": ["verify_claims(skipped)"]}

    claims = [Claim(**c) for c in state.get("claims", [])]
    packets = {p["evidence_id"]: p for p in state.get("evidence_packets", [])}
    budget = state.get("_budget_tracker")
    company = state.get("company", "")
    company_lower = company.lower()

    verified_claims: list[dict] = []
    verified_count = 0
    supplier_updates: dict[str, str] = {}  # normalized supplier name → evidence_source

    for claim in claims:
        # Step 2: Safety-net fallback — scan all packets for co-mention when
        # evidence_refs is empty (e.g. retrieval returned nothing for this claim).
        if not claim.evidence_refs:
            supplier_lower = claim.object_entity_id.lower()
            for eid, packet_dict in packets.items():
                text = _STORE.read_snapshot(EvidencePacket(**packet_dict)) or ""
                if company_lower in text.lower() and supplier_lower in text.lower():
                    claim = claim.model_copy(
                        update={"evidence_refs": [eid], "status": "RETRIEVED"}
                    )
                    break

        if not claim.evidence_refs:
            verified_claims.append(claim.model_dump())
            continue

        if budget and budget.llm_budget_remaining <= 0:
            verified_claims.append(claim.model_dump())
            continue

        # Step 3: Iterate all evidence refs, keep best verdict by
        # SUPPORTED > DISPUTED > WEAK > UNKNOWN.
        best_verdict_obj = None
        best_used_eid = ""

        for eid in claim.evidence_refs:
            if budget and budget.llm_budget_remaining <= 0:
                break
            packet_dict = packets.get(eid)
            if not packet_dict:
                continue
            evidence_text = _STORE.read_snapshot(EvidencePacket(**packet_dict)) or ""
            if not evidence_text:
                continue

            verdict_obj = verify_claim_against_evidence(claim, evidence_text, budget)

            if best_verdict_obj is None or (
                _VERDICT_RANK[verdict_obj.verdict] < _VERDICT_RANK[best_verdict_obj.verdict]
            ):
                best_verdict_obj = verdict_obj
                best_used_eid = eid

            if best_verdict_obj.verdict == "SUPPORTED":
                break  # Cannot improve further

        if best_verdict_obj is None:
            verified_claims.append(claim.model_dump())
            continue

        update: dict = {
            "status": "VERIFIED",
            "verdict": best_verdict_obj.verdict,
            "verdict_explanation": (
                f"{best_verdict_obj.rationale} [direction={best_verdict_obj.direction}]"
            ),
        }

        if best_verdict_obj.supporting_quote and best_used_eid:
            span = EvidenceSpan(
                evidence_id=best_used_eid,
                quote=best_verdict_obj.supporting_quote,
            )
            existing_spans = claim.supporting_spans[:]
            if not any(s.quote == best_verdict_obj.supporting_quote for s in existing_spans):
                existing_spans.append(span)
            update["supporting_spans"] = [s.model_dump() for s in existing_spans]

        updated_claim = claim.model_copy(update=update)
        verified_claims.append(updated_claim.model_dump())
        if best_verdict_obj.verdict in ("SUPPORTED", "WEAK"):
            verified_count += 1
            # Step 5: Mark this supplier as web_verified for hazard-score weighting.
            supplier_updates[updated_claim.object_entity_id.strip().lower()] = "web_verified"

    result: dict = {
        "claims": verified_claims,
        "workflow_trace": [f"verify_claims({verified_count} supported/weak)"],
    }

    # Step 5: Propagate evidence_source updates back to the suppliers list.
    if supplier_updates:
        result["suppliers"] = [
            {
                **s,
                "evidence_source": supplier_updates.get(
                    s["name"].strip().lower(), s.get("evidence_source", "llm_only")
                ),
            }
            for s in state.get("suppliers", [])
        ]

    return result


def resolve_locations_node(state: GraphState) -> dict:
    """Node 7: Enrich location_candidates from evidence; update suppliers lat/lon."""
    claims = [Claim(**c) for c in state.get("claims", [])]
    packets = {p["evidence_id"]: p for p in state.get("evidence_packets", [])}

    evidence_texts: dict[str, str] = {}
    for eid, packet_dict in packets.items():
        sp = packet_dict.get("snapshot_path")
        if sp:
            p = Path(sp)
            if p.exists():
                evidence_texts[eid] = p.read_text(encoding="utf-8")

    updated_claims = resolve_locations_for_claims(
        claims, evidence_texts, company=state.get("company", "")
    )

    # Update suppliers lat/lon with best location_candidate
    suppliers = [dict(s) for s in state.get("suppliers", [])]
    supplier_map = {s["name"]: s for s in suppliers}

    for claim in updated_claims:
        if not claim.location_candidates:
            continue
        best = max(claim.location_candidates, key=lambda lc: lc.confidence)
        s = supplier_map.get(claim.object_entity_id)
        if s:
            s["lat"] = best.lat
            s["lon"] = best.lon

    return {
        "claims": [c.model_dump() for c in updated_claims],
        "suppliers": list(supplier_map.values()),
        "workflow_trace": ["resolve_locations"],
    }


# -- Parallel hazard scoring (fan-out / fan-in) ----------------------------


def _make_hazard_scorer(hazard_name: str):
    """Factory: create a node that scores all suppliers for one hazard type."""
    def _score_node(state: GraphState) -> dict:
        scores: list[dict] = []
        for s_dict in state.get("suppliers", []):
            supplier = Supplier(**s_dict)
            hs = compute_hazard(supplier, hazard_name)
            scores.append(hs.model_dump())
        return {
            "hazard_scores": scores,
            "workflow_trace": [f"score_{hazard_name}"],
        }

    _score_node.__name__ = f"score_{hazard_name}"
    _score_node.__qualname__ = f"score_{hazard_name}"
    return _score_node


_HAZARD_SCORER_NODES = {name: _make_hazard_scorer(name) for name in HAZARD_NAMES}


def aggregate_risk_node(state: GraphState) -> dict:
    """Node 9: Roll up supplier and company SMHEI from hazard scores.

    Uses default equal-weight arithmetic mean over N_HAZARDS=6 hazards.
    Bands are read from reference_set.json (supplier_exposure_thresholds,
    company_exposure_thresholds), not from hard-coded constants.
    """
    summary = compute_risk_summary(
        state.get("hazard_scores", []),
        state.get("suppliers", []),
        # weights=None → equal 1/N_HAZARDS (primary SMHEI)
        # aggregation="arithmetic" → default
    )

    return {
        "company_risk_summary": summary,
        "workflow_trace": ["aggregate_risk"],
    }


def generate_report_node(state: GraphState) -> dict:
    """Node 10: Generate mitigations, alternatives, and the full report."""
    use_llm = state.get("use_llm", True)
    budget = state.get("_budget_tracker")
    mitigations: list[dict] = []
    alternatives: list[dict] = []

    if use_llm:
        try:
            mitigations = generate_mitigations_llm(
                company=state.get("company", "Unknown"),
                suppliers=state.get("suppliers", []),
                hazard_scores=state.get("hazard_scores", []),
                summary=state.get("company_risk_summary", {}),
            )
            if budget:
                budget.record_llm_call(purpose="generate_mitigations")
        except Exception:
            mitigations = []

        try:
            alternatives = suggest_alternatives_llm(
                company=state.get("company", "Unknown"),
                suppliers=state.get("suppliers", []),
                hazard_scores=state.get("hazard_scores", []),
                summary=state.get("company_risk_summary", {}),
            )
            if budget:
                budget.record_llm_call(purpose="suggest_alternatives")
        except Exception:
            alternatives = []

    budget_summary = budget.summary() if budget else state.get("budget_summary", {})

    full_trace = [*state.get("workflow_trace", []), "generate_report"]
    report_state = {
        **state,
        "llm_mitigations": mitigations,
        "suggested_alternatives": alternatives,
        "workflow_trace": full_trace,
        "budget_summary": budget_summary,
    }
    report_text = format_report(report_state)

    result: dict = {
        "llm_mitigations": mitigations,
        "suggested_alternatives": alternatives,
        "report_text": report_text,
        "workflow_trace": ["generate_report"],
    }
    if budget_summary:
        result["budget_summary"] = budget_summary
    return result


def export_artifacts_node(state: GraphState) -> dict:
    """Node 11: Signal completion of artifact export (CLI handles file writes)."""
    return {"workflow_trace": ["export_artifacts"]}


# ---------------------------------------------------------------------------
# Build and compile
# ---------------------------------------------------------------------------


def build_vcg_graph() -> StateGraph:
    """Construct the VCG LangGraph StateGraph (not yet compiled).

    Topology::

        START
          → resolve_entity → discover_claims
          → retrieve_evidence → extract_mentions → link_claims
          → verify_claims → resolve_locations
          → [score_* fan-out] → aggregate_risk
          → generate_report → export_artifacts → END
    """
    g = StateGraph(GraphState)

    g.add_node("resolve_entity", resolve_entity_node)
    g.add_node("discover_claims", discover_claims_node)
    g.add_node("retrieve_evidence", retrieve_evidence_node)
    g.add_node("extract_mentions", extract_mentions_node)
    g.add_node("link_claims", link_claims_node)
    g.add_node("verify_claims", verify_claims_node)
    g.add_node("resolve_locations", resolve_locations_node)
    for name, scorer_fn in _HAZARD_SCORER_NODES.items():
        g.add_node(f"score_{name}", scorer_fn)
    g.add_node("aggregate_risk", aggregate_risk_node)
    g.add_node("generate_report", generate_report_node)
    g.add_node("export_artifacts", export_artifacts_node)

    g.add_edge(START, "resolve_entity")
    g.add_edge("resolve_entity", "discover_claims")
    g.add_edge("discover_claims", "retrieve_evidence")
    g.add_edge("retrieve_evidence", "extract_mentions")
    g.add_edge("extract_mentions", "link_claims")
    g.add_edge("link_claims", "verify_claims")
    g.add_edge("verify_claims", "resolve_locations")

    for name in HAZARD_NAMES:
        g.add_edge("resolve_locations", f"score_{name}")

    for name in HAZARD_NAMES:
        g.add_edge(f"score_{name}", "aggregate_risk")

    g.add_edge("aggregate_risk", "generate_report")
    g.add_edge("generate_report", "export_artifacts")
    g.add_edge("export_artifacts", END)

    return g


def run_vcg_graph(
    company: str,
    tier_depth: int = 2,
    suppliers_path: Path | str | None = None,
    use_llm: bool = True,
    enable_web: bool = False,
    no_verify: bool = False,
    strict_mode: bool = False,
    max_web_queries: int = 30,
    snapshot_mode: bool = False,
    budget: BudgetTracker | None = None,
) -> dict:
    """Compile and invoke the VCG graph; return the final state.

    Condition mapping
    -----------------
    llm_only    : enable_web=False, no_verify=True  (default)
    web_retrieve: enable_web=True,  no_verify=True
    web_verify  : enable_web=True,  no_verify=False
    strict      : enable_web=True,  no_verify=False, strict_mode=True
    """
    checkpointer = MemorySaver()
    graph = build_vcg_graph().compile(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if budget is None:
        budget = BudgetTracker(max_web_queries=max_web_queries)

    init_state: dict = {
        "company": company,
        "tier_depth": tier_depth,
        "use_llm": use_llm,
        "enable_web": enable_web,
        "no_verify": no_verify,
        "strict_mode": strict_mode,
        "snapshot_mode": snapshot_mode,
        "hazard_scores": [],
        "workflow_trace": [],
        "claims": [],
        "evidence_packets": [],
        "_budget_tracker": budget,
        "_max_web_queries": max_web_queries,
    }
    if suppliers_path is not None:
        init_state["suppliers_path"] = str(suppliers_path)

    return graph.invoke(init_state, config)
