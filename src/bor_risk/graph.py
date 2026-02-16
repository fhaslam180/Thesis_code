"""LangGraph orchestration.

Workflow (parallel fan-out / fan-in):
  discover_suppliers
    -> [score_earthquake | score_flood | score_wildfire |
        score_cyclone | score_heat_stress | score_drought]   (parallel)
    -> aggregate_risk
    -> decide_workflow  (+ human-in-the-loop interrupt when interactive)
    -> [high_risk_response | monitoring_response]
    -> generate_mitigations
    -> suggest_alternatives
    -> format_report

Supplier discovery uses GPT-4o (or mock JSON).
Hazard scoring is deterministic and evidence-backed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from bor_risk.models import GraphState, Supplier
from bor_risk.report import format_report
from bor_risk.tools import (
    compute_hazard,
    compute_risk_summary,
    discover_suppliers_llm,
    generate_mitigations_llm,
    load_suppliers,
    resolve_company_profile,
    suggest_alternatives_llm,
)
from bor_risk.utils import load_hazards

HIGH_RISK_THRESHOLD = 0.4
CRITICAL_EXCEEDANCE_MARGIN = 0.2

# Hazard types to score (read from config, used for parallel node creation).
HAZARD_NAMES: list[str] = [h["name"] for h in load_hazards()]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def discover_suppliers_node(state: GraphState) -> GraphState:
    """Node 1: Look up or discover suppliers for the target company.

    When ``use_llm=True``, performs two-step contextual discovery:
    1. Resolve company profile (industry, products, HQ) via GPT-4o
    2. Discover suppliers with industry context injected into the prompt
    """
    company = state["company"]
    tier_depth = state.get("tier_depth", 2)
    use_llm = state.get("use_llm", False)

    company_profile: dict = {}

    if use_llm:
        try:
            company_profile = resolve_company_profile(company)
        except Exception:
            company_profile = {}
        suppliers, edges, evidence = discover_suppliers_llm(
            company, tier_depth, company_profile=company_profile or None,
        )
    else:
        suppliers_path = state.get("suppliers_path")
        suppliers, edges, evidence = load_suppliers(
            company, tier_depth, suppliers_path
        )

    result: dict = {
        "suppliers": [s.model_dump() for s in suppliers],
        "edges": edges,
        "evidence": evidence,
        "workflow_trace": ["discover_suppliers"],
    }
    if company_profile:
        result["company_profile"] = company_profile

    return result


# -- Parallel hazard scoring (fan-out / fan-in) ----------------------------

def _make_hazard_scorer(hazard_name: str):
    """Factory: create a LangGraph node that scores all suppliers for one hazard type.

    Each node runs in parallel after ``discover_suppliers`` and feeds into
    ``aggregate_risk``.  The ``operator.add`` reducer on ``hazard_scores``
    merges the partial lists from all six scorers.
    """
    def _score_node(state: GraphState) -> GraphState:
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


# Pre-build scorer node functions (one per hazard type from config).
_HAZARD_SCORER_NODES = {name: _make_hazard_scorer(name) for name in HAZARD_NAMES}


# -- Remaining sequential nodes --------------------------------------------

def aggregate_risk_node(state: GraphState) -> GraphState:
    """Node 3: Roll up supplier and company risk from hazard scores."""
    hazard_defs = load_hazards()
    weights = {h["name"]: float(h.get("weight", 1.0)) for h in hazard_defs}
    thresholds = {h["name"]: float(h.get("threshold", 1.0)) for h in hazard_defs}

    summary = compute_risk_summary(
        state.get("hazard_scores", []),
        state.get("suppliers", []),
        weights,
        thresholds,
        high_risk_threshold=HIGH_RISK_THRESHOLD,
        critical_exceedance_margin=CRITICAL_EXCEEDANCE_MARGIN,
    )

    return {
        "company_risk_summary": summary,
        "workflow_trace": ["aggregate_risk"],
    }


def decide_workflow_node(state: GraphState) -> GraphState:
    """Node 4: Choose the downstream workflow branch based on risk.

    When ``interactive=True`` and the route is ``high_risk_response``, the
    node calls ``interrupt()`` to pause for human approval.  The caller
    can resume with ``Command(resume="approved")`` or
    ``Command(resume="downgrade")`` to override the decision.
    """
    summary = state.get("company_risk_summary", {})
    company_score = float(summary.get("company_score", 0.0))
    critical_alert_count = int(summary.get("critical_alert_count", 0))

    if company_score >= HIGH_RISK_THRESHOLD or critical_alert_count > 0:
        route = "high_risk_response"
        reason = (
            f"Escalated because company_score={company_score:.4f} "
            f"or critical_alert_count={critical_alert_count}."
        )

        # Human-in-the-loop: pause for review when interactive mode is on.
        if state.get("interactive", False):
            human_response = interrupt({
                "question": "HIGH RISK DETECTED - Approve escalation?",
                "company_score": company_score,
                "critical_alert_count": critical_alert_count,
                "critical_alerts": summary.get("critical_alerts", [])[:5],
            })
            if str(human_response).lower() in ("n", "no", "downgrade"):
                route = "monitoring_response"
                reason = (
                    f"Downgraded by human operator from high_risk_response. "
                    f"Original score={company_score:.4f}."
                )
    else:
        route = "monitoring_response"
        reason = (
            f"Monitoring because company_score={company_score:.4f} "
            f"and critical_alert_count={critical_alert_count}."
        )

    return {
        "workflow_decision": {
            "route": route,
            "reason": reason,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "critical_exceedance_margin": CRITICAL_EXCEEDANCE_MARGIN,
        },
        "workflow_trace": ["decide_workflow"],
    }


def high_risk_response_node(state: GraphState) -> GraphState:
    """Node 5A: Build escalation actions for high-risk cases."""
    summary = state.get("company_risk_summary", {})
    top_suppliers = summary.get("supplier_risks", [])[:3]
    critical_alerts = summary.get("critical_alerts", [])[:3]

    top_names = ", ".join(
        f"{s['supplier_name']} ({s['risk_score']:.2f})"
        for s in top_suppliers
    ) or "None"
    critical_names = ", ".join(
        f"{a['supplier_name']}:{a['hazard_type']}"
        for a in critical_alerts
    ) or "None"

    actions = [
        {
            "priority": "P1",
            "action": f"Launch due diligence on top-risk suppliers: {top_names}.",
        },
        {
            "priority": "P1",
            "action": (
                "Request contingency plans for critical hazard signals: "
                f"{critical_names}."
            ),
        },
        {
            "priority": "P2",
            "action": "Prepare backup sourcing options and re-score within 7 days.",
        },
    ]

    return {
        "workflow_actions": actions,
        "workflow_trace": ["high_risk_response"],
    }


def monitoring_response_node(state: GraphState) -> GraphState:
    """Node 5B: Build monitoring actions for lower-risk cases."""
    summary = state.get("company_risk_summary", {})
    top_suppliers = summary.get("supplier_risks", [])[:2]
    top_names = ", ".join(
        f"{s['supplier_name']} ({s['risk_score']:.2f})"
        for s in top_suppliers
    ) or "None"

    actions = [
        {
            "priority": "P3",
            "action": (
                f"Maintain monthly monitoring for the current risk leaders: {top_names}."
            ),
        },
        {
            "priority": "P3",
            "action": "No immediate escalation; continue evidence collection for supplier changes.",
        },
    ]

    return {
        "workflow_actions": actions,
        "workflow_trace": ["monitoring_response"],
    }


def generate_mitigations_node(state: GraphState) -> GraphState:
    """Node 6: Generate LLM-powered mitigations (skipped when use_llm=False)."""
    use_llm = state.get("use_llm", False)
    mitigations: list[dict] = []

    if use_llm:
        try:
            mitigations = generate_mitigations_llm(
                company=state.get("company", "Unknown"),
                suppliers=state.get("suppliers", []),
                hazard_scores=state.get("hazard_scores", []),
                summary=state.get("company_risk_summary", {}),
            )
        except Exception:
            mitigations = []

    return {
        "llm_mitigations": mitigations,
        "workflow_trace": ["generate_mitigations"],
    }


def suggest_alternatives_node(state: GraphState) -> GraphState:
    """Node 7: Suggest lower-risk alternative suppliers (skipped when use_llm=False)."""
    use_llm = state.get("use_llm", False)
    alternatives: list[dict] = []

    if use_llm:
        try:
            alternatives = suggest_alternatives_llm(
                company=state.get("company", "Unknown"),
                suppliers=state.get("suppliers", []),
                hazard_scores=state.get("hazard_scores", []),
                summary=state.get("company_risk_summary", {}),
            )
        except Exception:
            alternatives = []

    return {
        "suggested_alternatives": alternatives,
        "workflow_trace": ["suggest_alternatives"],
    }


def format_report_node(state: GraphState) -> GraphState:
    """Node 7: Generate the plain-text report.

    Builds the full trace locally (including ``format_report``) so the
    report renderer can display it, but returns only ``["format_report"]``
    for the ``operator.add`` reducer.
    """
    full_trace = [*state.get("workflow_trace", []), "format_report"]
    report_text = format_report({**state, "workflow_trace": full_trace})
    return {"report_text": report_text, "workflow_trace": ["format_report"]}


def route_workflow(state: GraphState) -> str:
    """Route to either escalation or monitoring branch."""
    decision = state.get("workflow_decision", {})
    return decision.get("route", "monitoring_response")


# ---------------------------------------------------------------------------
# Build and compile
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph (not yet compiled).

    Topology (parallel fan-out / fan-in)::

        START -> discover_suppliers
              -> [score_earthquake | score_flood | score_wildfire |
                  score_cyclone | score_heat_stress | score_drought]
              -> aggregate_risk -> decide_workflow
              -> [high_risk_response | monitoring_response]
              -> generate_mitigations -> suggest_alternatives
              -> format_report -> END
    """
    g = StateGraph(GraphState)

    # -- Nodes --
    g.add_node("discover_suppliers", discover_suppliers_node)
    for name, scorer_fn in _HAZARD_SCORER_NODES.items():
        g.add_node(f"score_{name}", scorer_fn)
    g.add_node("aggregate_risk", aggregate_risk_node)
    g.add_node("decide_workflow", decide_workflow_node)
    g.add_node("high_risk_response", high_risk_response_node)
    g.add_node("monitoring_response", monitoring_response_node)
    g.add_node("generate_mitigations", generate_mitigations_node)
    g.add_node("suggest_alternatives", suggest_alternatives_node)
    g.add_node("format_report", format_report_node)

    # -- Edges --
    g.add_edge(START, "discover_suppliers")

    # Fan-out: discover -> 6 parallel hazard scorers
    for name in HAZARD_NAMES:
        g.add_edge("discover_suppliers", f"score_{name}")

    # Fan-in: all 6 hazard scorers -> aggregate_risk
    for name in HAZARD_NAMES:
        g.add_edge(f"score_{name}", "aggregate_risk")

    # Sequential: aggregate -> decide -> conditional branch
    g.add_edge("aggregate_risk", "decide_workflow")
    g.add_conditional_edges(
        "decide_workflow",
        route_workflow,
        {
            "high_risk_response": "high_risk_response",
            "monitoring_response": "monitoring_response",
        },
    )
    g.add_edge("high_risk_response", "generate_mitigations")
    g.add_edge("monitoring_response", "generate_mitigations")
    g.add_edge("generate_mitigations", "suggest_alternatives")
    g.add_edge("suggest_alternatives", "format_report")
    g.add_edge("format_report", END)

    return g


def run_graph(
    company: str,
    tier_depth: int,
    suppliers_path: Path | str | None = None,
    use_llm: bool = False,
    interactive: bool = False,
) -> dict:
    """Compile the graph, invoke it, and return the final state.

    Parameters
    ----------
    interactive : bool
        When *True* and the risk is high, the graph pauses at
        ``decide_workflow`` for human approval via ``interrupt()``.
        Default *False* for backward compatibility and testing.
    """
    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    init_state: dict = {
        "company": company,
        "tier_depth": tier_depth,
        "use_llm": use_llm,
        "interactive": interactive,
        "hazard_scores": [],
        "workflow_trace": [],
    }
    if suppliers_path is not None:
        init_state["suppliers_path"] = str(suppliers_path)
    result = graph.invoke(init_state, config)
    return result
