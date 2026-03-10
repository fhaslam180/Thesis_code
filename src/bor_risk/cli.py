"""CLI entry-point for bor-risk-agent (Verified Claim Graph pipeline)."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

from bor_risk.budget import BudgetTracker
from bor_risk.evidence_store import EvidenceStore
from bor_risk.graph import HAZARD_NAMES, build_vcg_graph, run_vcg_graph
from bor_risk.provenance import ProvenanceExporter
from bor_risk.sensitivity import format_sensitivity_report, run_sensitivity
from bor_risk.utils import ensure_dir, sanitize_company_name


_NODE_LABELS: dict[str, str] = {
    "resolve_entity": "Resolving company profile",
    "discover_claims": "Discovering supplier claims",
    "retrieve_evidence": "Retrieving web evidence",
    "extract_mentions": "Extracting supplier mentions",
    "link_claims": "Linking mentions to claims",
    "verify_claims": "Verifying claims against evidence",
    "resolve_locations": "Resolving facility locations",
    "score_earthquake": "Scoring earthquake hazard",
    "score_flood": "Scoring flood hazard",
    "score_wildfire": "Scoring wildfire hazard",
    "score_cyclone": "Scoring cyclone hazard",
    "score_heat_stress": "Scoring heat_stress hazard",
    "score_drought": "Scoring drought hazard",
    "aggregate_risk": "Aggregating risk",
    "generate_report": "Generating report",
    "export_artifacts": "Exporting artifacts",
}


def _print_node_complete(node_name: str, state_update: dict) -> None:
    """Print a status line when a node completes."""
    label = _NODE_LABELS.get(node_name, node_name)
    detail = ""

    if node_name == "discover_claims":
        n = len(state_update.get("claims", []))
        detail = f" ({n} claims)"
    elif node_name == "retrieve_evidence":
        n = len(state_update.get("evidence_packets", []))
        detail = f" ({n} packets)"
    elif node_name == "verify_claims":
        claims = state_update.get("claims", [])
        supported = sum(1 for c in claims if c.get("verdict") in ("SUPPORTED", "WEAK"))
        detail = f" ({supported} supported/weak)"
    elif node_name == "aggregate_risk":
        summary = state_update.get("company_risk_summary", {})
        score = summary.get("company_score", 0)
        band = summary.get("risk_band", "unknown")
        detail = f" (score: {score:.2f}, band: {band})"

    print(f"  \u2713 {label}{detail}")


def _write_outputs(
    state: dict,
    out_dir: Path,
    prefix: str,
    export_prov: bool = False,
) -> None:
    """Write report, graph JSON, evidence JSONL, and optional provenance."""
    # Report
    report_path = out_dir / f"{prefix}_report.txt"
    report_path.write_text(state.get("report_text", ""), encoding="utf-8")

    # Graph JSON (includes claims + evidence_packets)
    graph_data: dict = {
        "suppliers": state.get("suppliers", []),
        "edges": state.get("edges", []),
        "claims": state.get("claims", []),
        "evidence_packets": state.get("evidence_packets", []),
        "company_risk_summary": state.get("company_risk_summary", {}),
        "workflow_trace": state.get("workflow_trace", []),
    }
    if state.get("budget_summary"):
        graph_data["budget_summary"] = state["budget_summary"]

    graph_path = out_dir / f"{prefix}_graph.json"
    graph_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")

    # Evidence JSONL (legacy evidence items from LLM discovery)
    evidence_path = out_dir / f"{prefix}_evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8") as f:
        for item in state.get("evidence", []):
            f.write(json.dumps(item) + "\n")

    # Evidence index (fetched web packets)
    packets = state.get("evidence_packets", [])
    if packets:
        store = EvidenceStore()
        index_path = out_dir / f"{prefix}_evidence_index.jsonl"
        store.export_index(packets, index_path)
        print(f"Wrote {index_path}")

    # Provenance JSON-LD
    if export_prov:
        company = state.get("company", "unknown")
        prov_path = out_dir / f"{prefix}_provenance.jsonld"
        prov = ProvenanceExporter(
            company=company,
            claims=state.get("claims", []),
            evidence_packets=packets,
            condition=_detect_condition(state),
        )
        prov.export(prov_path)
        print(f"Wrote {prov_path}")

    print(f"Wrote {report_path}")
    print(f"Wrote {graph_path}")
    print(f"Wrote {evidence_path}")


def _detect_condition(state: dict) -> str:
    """Determine the evaluation condition from state flags."""
    if state.get("strict_mode"):
        return "strict"
    if state.get("no_verify"):
        return "web_retrieve" if state.get("enable_web") else "llm_only"
    if state.get("enable_web"):
        return "web_verify"
    return "llm_only"


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run supply-chain risk analysis (Verified Claim Graph pipeline).",
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
        help="Output path (e.g. outputs/acme.txt — directory is derived from this)",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Condition A (llm_only): LLM-only, no web retrieval.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Condition B (web_retrieve): retrieve evidence, skip LLM verification.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Condition D (strict): exclude DISPUTED/UNKNOWN claims from report.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable GPT-based supplier discovery (use fixture suppliers instead).",
    )
    parser.add_argument(
        "--suppliers-path",
        default=None,
        help="Optional JSON fixture path for deterministic supplier input.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis on weight/threshold variations.",
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
        help="Max LLM calls (default: 20).",
    )
    parser.add_argument(
        "--budget-web",
        type=int,
        default=30,
        help="Max web queries (default: 30).",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use cached web results only (for reproducible evaluation).",
    )
    parser.add_argument(
        "--export-prov",
        action="store_true",
        help="Export PROV-O JSON-LD provenance file.",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help=(
            "Generate interactive HTML map and static network PNG "
            "(requires folium, networkx, matplotlib — install with: pip install -e '.[viz]')."
        ),
    )
    args = parser.parse_args(argv)

    if args.visualize:
        graph = build_vcg_graph().compile()
        print(graph.get_graph().draw_mermaid())
        return

    out_path = Path(args.out)
    out_dir = ensure_dir(out_path.parent)
    prefix = sanitize_company_name(args.company)

    # Determine condition flags
    enable_web = not args.no_web
    no_verify = args.no_verify
    strict_mode = args.strict
    use_llm = not args.no_llm

    budget = BudgetTracker(
        max_llm_calls=args.budget_llm,
        max_web_queries=args.budget_web,
    )

    import os
    if enable_web and not os.environ.get("TAVILY_API_KEY"):
        print(
            "\n  [ERROR] Web retrieval is enabled but TAVILY_API_KEY is not set.\n"
            "  Add TAVILY_API_KEY=<key> to your .env file, or pass --no-web.\n"
        )
        raise SystemExit(1)

    print(f"Analysing supply-chain risk for {args.company}...")
    condition = _detect_condition({
        "enable_web": enable_web,
        "no_verify": no_verify,
        "strict_mode": strict_mode,
    })
    print(f"  Condition: {condition}")
    if not enable_web:
        print("  Web retrieval: DISABLED")
    if no_verify:
        print("  Claim verification: DISABLED")
    if strict_mode:
        print("  Strict mode: ON (DISPUTED/UNKNOWN excluded)")
    if args.snapshot:
        print("  Snapshot mode: ON (cache only)")
    print()

    state = run_vcg_graph(
        company=args.company,
        tier_depth=args.tier_depth,
        suppliers_path=args.suppliers_path,
        use_llm=use_llm,
        enable_web=enable_web,
        no_verify=no_verify,
        strict_mode=strict_mode,
        max_web_queries=args.budget_web,
        snapshot_mode=args.snapshot,
        budget=budget,
    )

    # Print budget summary
    bs = state.get("budget_summary", budget.summary())
    print(
        f"\n  Budget used: {bs.get('llm_calls', 0)}/{args.budget_llm} LLM, "
        f"{bs.get('web_queries', 0)}/{args.budget_web} web, "
        f"{bs.get('hazard_scores', 0)} hazard scores"
    )
    print(f"  Wall clock: {bs.get('wall_clock_seconds', 0):.1f}s")

    # Claim summary
    claims = state.get("claims", [])
    if claims:
        supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
        weak = sum(1 for c in claims if c.get("verdict") == "WEAK")
        print(
            f"  Claims: {len(claims)} total, {supported} SUPPORTED, "
            f"{weak} WEAK"
        )

    if args.sensitivity:
        sensitivity_results = run_sensitivity(state)
        sensitivity_text = format_sensitivity_report(sensitivity_results)
        state["report_text"] = state.get("report_text", "").rstrip() + "\n\n" + sensitivity_text

        sensitivity_path = out_dir / f"{prefix}_sensitivity.json"
        sensitivity_path.write_text(
            json.dumps(sensitivity_results, indent=2), encoding="utf-8"
        )
        print(f"Wrote {sensitivity_path}")

    _write_outputs(state, out_dir, prefix, export_prov=args.export_prov)

    if args.map:
        from bor_risk.visualize import render_network_graph, render_supplier_map

        render_supplier_map(state, args.company, out_dir / f"{prefix}_map.html")
        render_network_graph(state, args.company, out_dir / f"{prefix}_network.png")


if __name__ == "__main__":
    main()
