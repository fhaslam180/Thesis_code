"""Autonomous ReAct agent for supply-chain risk assessment.

Uses ``create_react_agent`` from LangGraph to build a tool-using agent
that dynamically chooses which suppliers to verify, which hazards to
score, and how to spend its budget.

Experimental conditions:
- Condition C (agent, no web): ``enable_web=False``
- Condition D (agent + web): ``enable_web=True``
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from bor_risk.agent_tools import build_agent_tools
from bor_risk.budget import BudgetTracker
from bor_risk.tools import compute_risk_summary
from bor_risk.utils import load_hazards, load_prompts


def _auto_aggregate(acc: dict) -> dict:
    """Compute risk summary if the agent didn't call aggregate_risk."""
    hazard_defs = load_hazards()
    weights = {h["name"]: float(h.get("weight", 1.0)) for h in hazard_defs}
    thresholds = {h["name"]: float(h.get("threshold", 1.0)) for h in hazard_defs}
    return compute_risk_summary(
        acc.get("hazard_scores", []),
        acc.get("suppliers", []),
        weights,
        thresholds,
    )


def _extract_trace(messages: list) -> list[dict]:
    """Extract a simplified trace from the agent's message history."""
    trace: list[dict] = []
    for msg in messages:
        entry: dict = {"role": getattr(msg, "type", "unknown")}
        content = getattr(msg, "content", "")
        if content:
            entry["content"] = content[:500]
        # Tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc["name"], "args": tc.get("args", {})}
                for tc in tool_calls
            ]
        # Tool result
        if hasattr(msg, "name") and msg.name:
            entry["tool_name"] = msg.name
        trace.append(entry)
    return trace


def run_agent(
    company: str,
    tier_depth: int = 2,
    enable_web: bool = True,
    budget: BudgetTracker | None = None,
    snapshot_mode: bool = False,
) -> dict:
    """Run the autonomous ReAct agent.

    Returns a dict compatible with ``GraphState`` so the same
    ``format_report()`` function can render the output.

    Parameters
    ----------
    company : str
        Target company name.
    tier_depth : int
        How many supplier tiers to explore.
    enable_web : bool
        If *False*, web search and verification tools are excluded
        (Condition C). If *True*, full agent + web (Condition D).
    budget : BudgetTracker | None
        Shared budget tracker. Created with defaults if not provided.
    snapshot_mode : bool
        If *True*, web searches use cached results only.
    """
    budget = budget or BudgetTracker()

    acc: dict = {
        "company": company,
        "tier_depth": tier_depth,
        "suppliers": [],
        "edges": [],
        "evidence": [],
        "hazard_scores": [],
        "company_profile": {},
        "company_risk_summary": {},
        "llm_mitigations": [],
        "suggested_alternatives": [],
        "_web_evidence_counter": 1,
    }

    tools = build_agent_tools(
        acc,
        budget,
        enable_web=enable_web,
        snapshot_mode=snapshot_mode,
    )

    prompts = load_prompts()
    system_prompt = prompts.get("agent_system_prompt", "")

    model = ChatOpenAI(model="gpt-4o", temperature=0, timeout=120)

    agent = create_react_agent(model, tools, prompt=system_prompt)

    # Build the initial user message with budget information.
    web_note = ""
    if enable_web:
        web_note = (
            f"Web queries available: {budget.max_web_queries}. "
            f"Use verify_supplier to check key suppliers. "
        )
    else:
        web_note = "Web search is DISABLED for this run. "

    user_msg = (
        f"Assess supply chain risk for '{company}'. "
        f"Explore up to {tier_depth} tiers of suppliers. "
        f"Budget: {budget.max_llm_calls} LLM calls remaining. "
        f"{web_note}"
        f"Hazard scoring is free. "
        f"Prioritise: verify high-risk/low-confidence suppliers first, "
        f"score geographically relevant hazards, generate mitigations "
        f"for high-risk findings."
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})

    # Post-process: ensure risk summary exists.
    if not acc.get("company_risk_summary"):
        acc["company_risk_summary"] = _auto_aggregate(acc)

    # Build a workflow_trace from the agent's actions.
    trace_steps = []
    if acc.get("company_profile"):
        trace_steps.append("profile_company")
    if acc.get("suppliers"):
        trace_steps.append("discover_suppliers")
    verified = sum(
        1 for s in acc["suppliers"] if s.get("evidence_source") == "web_verified"
    )
    if verified:
        trace_steps.append(f"verify_suppliers({verified})")
    if acc.get("hazard_scores"):
        trace_steps.append(f"score_hazards({len(acc['hazard_scores'])})")
    if acc.get("company_risk_summary"):
        trace_steps.append("aggregate_risk")
    if acc.get("llm_mitigations"):
        trace_steps.append("generate_mitigations")
    if acc.get("suggested_alternatives"):
        trace_steps.append("suggest_alternatives")
    trace_steps.append("format_report")

    acc["workflow_trace"] = trace_steps
    acc["agent_trace"] = _extract_trace(result["messages"])
    acc["budget_summary"] = budget.summary()
    acc["use_llm"] = True

    return acc
