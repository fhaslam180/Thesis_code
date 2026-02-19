"""CLI entry-point for bor-risk-agent."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from bor_risk.budget import BudgetTracker
from bor_risk.graph import build_graph, run_graph
from bor_risk.report import format_report
from bor_risk.sensitivity import format_sensitivity_report, run_sensitivity
from bor_risk.utils import ensure_dir, sanitize_company_name


# Node-name -> human-readable label for streaming output.
_NODE_LABELS: dict[str, str] = {
    "discover_suppliers": "Discovering suppliers",
    "verify_suppliers": "Verifying suppliers against web evidence",
    "score_earthquake": "Scoring earthquake hazard",
    "score_flood": "Scoring flood hazard",
    "score_wildfire": "Scoring wildfire hazard",
    "score_cyclone": "Scoring cyclone hazard",
    "score_heat_stress": "Scoring heat_stress hazard",
    "score_drought": "Scoring drought hazard",
    "aggregate_risk": "Aggregating risk",
    "decide_workflow": "Evaluating workflow decision",
    "high_risk_response": "Generating escalation actions",
    "monitoring_response": "Generating monitoring actions",
    "generate_mitigations": "Generating mitigations",
    "suggest_alternatives": "Suggesting alternative suppliers",
    "format_report": "Formatting report",
}


def _print_node_complete(node_name: str, state_update: dict) -> None:
    """Print a status line when a node completes."""
    label = _NODE_LABELS.get(node_name, node_name)

    detail = ""
    if node_name == "discover_suppliers":
        suppliers = state_update.get("suppliers", [])
        detail = f" ({len(suppliers)} found)"
    elif node_name == "verify_suppliers":
        suppliers = state_update.get("suppliers", [])
        verified = sum(
            1 for s in suppliers if s.get("evidence_source") == "web_verified"
        )
        detail = f" ({verified} verified)"
    elif node_name == "aggregate_risk":
        summary = state_update.get("company_risk_summary", {})
        score = summary.get("company_score", 0)
        band = summary.get("risk_band", "unknown")
        detail = f" (score: {score:.2f}, band: {band})"
    elif node_name == "generate_mitigations":
        mitigations = state_update.get("llm_mitigations", [])
        detail = f" ({len(mitigations)} items)"

    print(f"  \u2713 {label}{detail}")


def _write_outputs(state: dict, out_dir: Path, prefix: str) -> None:
    """Write report, graph JSON, and evidence JSONL files."""
    report_path = out_dir / f"{prefix}_report.txt"
    report_path.write_text(state["report_text"], encoding="utf-8")

    graph_data: dict = {
        "suppliers": state["suppliers"],
        "edges": state["edges"],
        "company_risk_summary": state.get("company_risk_summary", {}),
        "workflow_decision": state.get("workflow_decision", {}),
        "workflow_actions": state.get("workflow_actions", []),
        "workflow_trace": state.get("workflow_trace", []),
    }
    if state.get("budget_summary"):
        graph_data["budget_summary"] = state["budget_summary"]

    graph_path = out_dir / f"{prefix}_graph.json"
    graph_path.write_text(
        json.dumps(graph_data, indent=2),
        encoding="utf-8",
    )

    evidence_path = out_dir / f"{prefix}_evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8") as f:
        for item in state["evidence"]:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {report_path}")
    print(f"Wrote {graph_path}")
    print(f"Wrote {evidence_path}")


def _run_interactive(
    company: str,
    tier_depth: int,
    suppliers_path: str | None,
    use_llm: bool,
    enable_web: bool = False,
) -> dict:
    """Run the graph with streaming output and human-in-the-loop."""
    print(f"Analysing supply-chain risk for {company}...\n")

    checkpointer = MemorySaver()
    graph = build_graph(enable_web=enable_web).compile(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    init_state: dict = {
        "company": company,
        "tier_depth": tier_depth,
        "use_llm": use_llm,
        "interactive": True,
        "enable_web": enable_web,
        "hazard_scores": [],
        "workflow_trace": [],
    }
    if suppliers_path is not None:
        init_state["suppliers_path"] = str(suppliers_path)

    # Stream node completions with real-time progress.
    for event in graph.stream(init_state, config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name != "__interrupt__":
                _print_node_complete(node_name, update)

    # Check for human-in-the-loop interrupt.
    snapshot = graph.get_state(config)
    while snapshot.next:
        # Extract interrupt data from the paused state.
        interrupt_data = {}
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_data = snapshot.tasks[0].interrupts[0].value

        score = interrupt_data.get("company_score", 0)
        alerts = interrupt_data.get("critical_alert_count", 0)
        critical = interrupt_data.get("critical_alerts", [])

        print(f"\n  \u26a0  HIGH RISK DETECTED (score={score:.2f}, "
              f"critical alerts={alerts})")
        for alert in critical[:3]:
            print(f"     - {alert['supplier_name']}: {alert['hazard_type']} "
                  f"(exceedance={alert['exceedance']:.2f})")

        try:
            answer = input("\n     Approve escalation? [Y/n]: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "Y"

        if answer.lower() in ("n", "no"):
            resume_value = "downgrade"
            print("     Downgrading to monitoring response.\n")
        else:
            resume_value = "approved"
            print()

        # Resume the graph and continue streaming.
        for event in graph.stream(
            Command(resume=resume_value), config, stream_mode="updates"
        ):
            for node_name, update in event.items():
                if node_name != "__interrupt__":
                    _print_node_complete(node_name, update)

        snapshot = graph.get_state(config)

    return graph.get_state(config).values


def _run_agent_mode(
    company: str,
    tier_depth: int,
    enable_web: bool,
    budget_llm: int,
    budget_web: int,
    snapshot: bool,
) -> dict:
    """Run the autonomous ReAct agent (Conditions C and D)."""
    from bor_risk.agent import run_agent

    print(f"Running agent mode for {company}...")
    print(f"  Budget: {budget_llm} LLM calls, {budget_web} web queries")
    if not enable_web:
        print("  Web search: DISABLED")
    if snapshot:
        print("  Snapshot mode: ON (cache only)")
    print()

    budget = BudgetTracker(
        max_llm_calls=budget_llm,
        max_web_queries=budget_web,
    )
    state = run_agent(
        company=company,
        tier_depth=tier_depth,
        enable_web=enable_web,
        budget=budget,
        snapshot_mode=snapshot,
    )

    # Generate report text using the same formatter.
    state["report_text"] = format_report(state)

    # Print budget summary.
    bs = state.get("budget_summary", {})
    print(f"\n  Budget used: {bs.get('llm_calls', 0)}/{budget_llm} LLM, "
          f"{bs.get('web_queries', 0)}/{budget_web} web, "
          f"{bs.get('hazard_scores', 0)} hazard scores")
    print(f"  Wall clock: {bs.get('wall_clock_seconds', 0):.1f}s")

    return state


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run supply-chain risk analysis for a company.",
    )
    parser.add_argument("--company", required=True, help="Target company name")
    parser.add_argument(
        "--tier-depth",
        type=int,
        default=2,
        help="How many supplier tiers to expand (default: 2)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path (directory derived from this, e.g. outputs/acme.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "pipeline-web", "agent", "agent-web"],
        default="pipeline",
        help=(
            "Experimental condition: "
            "pipeline (A), pipeline-web (B), agent (C), agent-web (D). "
            "Default: pipeline."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable GPT-based supplier discovery and use deterministic supplier input.",
    )
    parser.add_argument(
        "--suppliers-path",
        default=None,
        help="Optional JSON path for deterministic supplier input.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable streaming output and human-in-the-loop approval for high-risk decisions.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis on weight/threshold variations after scoring.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print a Mermaid diagram of the graph and exit.",
    )
    parser.add_argument(
        "--budget-llm",
        type=int,
        default=20,
        help="Max LLM calls for agent/pipeline-web modes (default: 20).",
    )
    parser.add_argument(
        "--budget-web",
        type=int,
        default=30,
        help="Max web queries for agent-web/pipeline-web modes (default: 30).",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use cached web results only (for reproducible evaluation).",
    )
    args = parser.parse_args(argv)

    # --visualize: print Mermaid diagram and exit.
    if args.visualize:
        enable_web = args.mode in ("pipeline-web", "agent-web")
        graph = build_graph(enable_web=enable_web).compile()
        print(graph.get_graph().draw_mermaid())
        return

    # Derive output directory and file prefix.
    out_path = Path(args.out)
    out_dir = ensure_dir(out_path.parent)
    prefix = sanitize_company_name(args.company)

    use_llm = not args.no_llm

    if args.mode in ("agent", "agent-web"):
        # Agent modes (Conditions C and D).
        enable_web = args.mode == "agent-web"
        state = _run_agent_mode(
            args.company,
            args.tier_depth,
            enable_web=enable_web,
            budget_llm=args.budget_llm,
            budget_web=args.budget_web,
            snapshot=args.snapshot,
        )
    elif args.interactive:
        # Interactive pipeline path: streaming + human-in-the-loop.
        enable_web = args.mode == "pipeline-web"
        state = _run_interactive(
            args.company, args.tier_depth, args.suppliers_path, use_llm,
            enable_web=enable_web,
        )
    else:
        # Pipeline modes (Conditions A and B).
        enable_web = args.mode == "pipeline-web"
        state = run_graph(
            args.company,
            args.tier_depth,
            suppliers_path=args.suppliers_path,
            use_llm=use_llm,
            enable_web=enable_web,
            max_web_queries=args.budget_web,
            snapshot_mode=args.snapshot,
        )

    # Optional sensitivity analysis (appended to report + separate JSON).
    if args.sensitivity:
        sensitivity_results = run_sensitivity(state)
        sensitivity_text = format_sensitivity_report(sensitivity_results)

        # Append sensitivity section to report.
        state["report_text"] = state["report_text"].rstrip() + "\n\n" + sensitivity_text

        # Write raw sensitivity JSON.
        sensitivity_path = out_dir / f"{prefix}_sensitivity.json"
        sensitivity_path.write_text(
            json.dumps(sensitivity_results, indent=2), encoding="utf-8",
        )
        print(f"Wrote {sensitivity_path}")

    _write_outputs(state, out_dir, prefix)


if __name__ == "__main__":
    main()
