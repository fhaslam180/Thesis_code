"""Tests for the VCG LangGraph pipeline: topology, parallelism, and integration."""

from __future__ import annotations

from bor_risk.graph import HAZARD_NAMES, build_vcg_graph, run_vcg_graph


# ---------------------------------------------------------------------------
# 1. Graph topology tests
# ---------------------------------------------------------------------------


class TestGraphTopology:
    """Verify the VCG graph has the expected 11-node structure."""

    def test_graph_has_vcg_core_nodes(self) -> None:
        g = build_vcg_graph().compile()
        node_names = set(g.get_graph().nodes)
        for expected in [
            "resolve_entity",
            "discover_claims",
            "retrieve_evidence",
            "extract_mentions",
            "link_claims",
            "verify_claims",
            "resolve_locations",
            "aggregate_risk",
            "generate_report",
            "export_artifacts",
        ]:
            assert expected in node_names, f"Missing node: {expected}"

    def test_graph_has_six_hazard_scorer_nodes(self) -> None:
        g = build_vcg_graph().compile()
        node_names = set(g.get_graph().nodes)
        for name in HAZARD_NAMES:
            assert f"score_{name}" in node_names, f"Missing scorer node: score_{name}"

    def test_graph_does_not_have_old_nodes(self) -> None:
        g = build_vcg_graph().compile()
        node_names = set(g.get_graph().nodes)
        for old in ["discover_suppliers", "decide_workflow", "format_report",
                    "high_risk_response", "monitoring_response"]:
            assert old not in node_names, f"Old node still present: {old}"

    def test_resolve_locations_fans_out_to_six_scorers(self) -> None:
        g = build_vcg_graph().compile()
        fan_out_targets = {
            edge.target
            for edge in g.get_graph().edges
            if edge.source == "resolve_locations"
        }
        for name in HAZARD_NAMES:
            assert f"score_{name}" in fan_out_targets, (
                f"resolve_locations doesn't fan out to score_{name}"
            )

    def test_six_scorers_fan_into_aggregate(self) -> None:
        g = build_vcg_graph().compile()
        aggregate_sources = {
            edge.source
            for edge in g.get_graph().edges
            if edge.target == "aggregate_risk"
        }
        for name in HAZARD_NAMES:
            assert f"score_{name}" in aggregate_sources

    def test_mermaid_contains_vcg_nodes(self) -> None:
        g = build_vcg_graph().compile()
        mermaid = g.get_graph().draw_mermaid()
        for node in ["resolve_entity", "discover_claims", "resolve_locations",
                     "aggregate_risk", "score_earthquake", "generate_report"]:
            assert node in mermaid, f"Mermaid missing node: {node}"


# ---------------------------------------------------------------------------
# 2. Parallel scoring tests
# ---------------------------------------------------------------------------


class TestParallelScoring:
    """Verify parallel hazard scoring produces correct merged results."""

    def test_all_six_hazard_types_scored(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        hazard_types = {hs["hazard_type"] for hs in state["hazard_scores"]}
        for name in HAZARD_NAMES:
            assert name in hazard_types, f"Missing hazard type: {name}"

    def test_score_count_matches_suppliers_times_hazards(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        n_suppliers = len(state["suppliers"])
        n_hazards = len(HAZARD_NAMES)
        assert len(state["hazard_scores"]) == n_suppliers * n_hazards

    def test_trace_contains_all_scorer_entries(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        for name in HAZARD_NAMES:
            assert f"score_{name}" in trace, f"Missing scorer trace: score_{name}"

    def test_trace_order_discover_before_scorers(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        discover_idx = trace.index("discover_claims")
        for name in HAZARD_NAMES:
            scorer_idx = trace.index(f"score_{name}")
            assert scorer_idx > discover_idx

    def test_trace_order_scorers_before_aggregate(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        aggregate_idx = trace.index("aggregate_risk")
        for name in HAZARD_NAMES:
            scorer_idx = trace.index(f"score_{name}")
            assert scorer_idx < aggregate_idx


# ---------------------------------------------------------------------------
# 3. VCG integration tests (replaces HITL / backward-compat / streaming)
# ---------------------------------------------------------------------------


class TestVCGIntegration:
    """End-to-end VCG pipeline integration checks."""

    def test_run_produces_report_text(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert "report_text" in state
        assert len(state["report_text"]) > 0

    def test_run_produces_risk_summary(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert "company_risk_summary" in state
        summary = state["company_risk_summary"]
        assert 0.0 <= summary["company_score"] <= 1.0
        assert summary["company_band"] in ("High", "Medium", "Low")

    def test_run_trace_contains_core_nodes(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        for expected in ["resolve_entity", "discover_claims",
                         "resolve_locations", "aggregate_risk",
                         "generate_report", "export_artifacts"]:
            assert expected in trace, f"Trace missing: {expected}"

    def test_web_nodes_skipped_when_enable_web_false(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False, enable_web=False)
        trace = state["workflow_trace"]
        assert "retrieve_evidence(skipped)" in trace
        assert "extract_mentions(skipped)" in trace
        assert "link_claims(skipped)" in trace

    def test_verify_skipped_when_no_verify_true(self) -> None:
        state = run_vcg_graph(
            "ACME", tier_depth=2, use_llm=False,
            enable_web=False, no_verify=True
        )
        trace = state["workflow_trace"]
        assert "verify_claims(skipped)" in trace

    def test_claims_initialised_from_suppliers(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert "claims" in state
        assert len(state["claims"]) == len(state["suppliers"])

    def test_claims_have_deterministic_ids(self) -> None:
        state1 = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        state2 = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        ids1 = {c["claim_id"] for c in state1["claims"]}
        ids2 = {c["claim_id"] for c in state2["claims"]}
        assert ids1 == ids2, "Claim IDs should be deterministic across runs"

    def test_report_contains_pipeline_workflow_section(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert "--- Pipeline Workflow ---" in state["report_text"]

    def test_report_contains_trace(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert "Trace:" in state["report_text"]
