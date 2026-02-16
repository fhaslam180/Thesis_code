"""Tests for LangGraph advanced features: parallelism, HITL, streaming, visualization."""

from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from bor_risk.graph import HAZARD_NAMES, build_graph, run_graph


# ---------------------------------------------------------------------------
# 1. Graph topology tests
# ---------------------------------------------------------------------------


class TestGraphTopology:
    """Verify the graph has the expected fan-out / fan-in structure."""

    def test_graph_has_six_hazard_scorer_nodes(self) -> None:
        g = build_graph().compile()
        node_names = set(g.get_graph().nodes)
        for name in HAZARD_NAMES:
            assert f"score_{name}" in node_names, f"Missing node: score_{name}"

    def test_graph_does_not_have_old_score_hazards_node(self) -> None:
        g = build_graph().compile()
        node_names = set(g.get_graph().nodes)
        assert "score_hazards" not in node_names

    def test_discover_fans_out_to_six_scorers(self) -> None:
        g = build_graph().compile()
        discover_targets = set()
        for edge in g.get_graph().edges:
            if edge.source == "discover_suppliers":
                discover_targets.add(edge.target)
        for name in HAZARD_NAMES:
            assert f"score_{name}" in discover_targets

    def test_six_scorers_fan_into_aggregate(self) -> None:
        g = build_graph().compile()
        aggregate_sources = set()
        for edge in g.get_graph().edges:
            if edge.target == "aggregate_risk":
                aggregate_sources.add(edge.source)
        for name in HAZARD_NAMES:
            assert f"score_{name}" in aggregate_sources

    def test_mermaid_output_contains_fan_out(self) -> None:
        g = build_graph().compile()
        mermaid = g.get_graph().draw_mermaid()
        assert "discover_suppliers" in mermaid
        assert "score_earthquake" in mermaid
        assert "aggregate_risk" in mermaid


# ---------------------------------------------------------------------------
# 2. Parallel scoring tests
# ---------------------------------------------------------------------------


class TestParallelScoring:
    """Verify parallel hazard scoring produces correct merged results."""

    def test_all_six_hazard_types_scored(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        hazard_types = {hs["hazard_type"] for hs in state["hazard_scores"]}
        for name in HAZARD_NAMES:
            assert name in hazard_types, f"Missing hazard type: {name}"

    def test_score_count_matches_suppliers_times_hazards(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        n_suppliers = len(state["suppliers"])
        n_hazards = len(HAZARD_NAMES)
        assert len(state["hazard_scores"]) == n_suppliers * n_hazards

    def test_trace_contains_all_scorer_entries(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        for name in HAZARD_NAMES:
            assert f"score_{name}" in trace

    def test_trace_order_discover_before_scorers(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        discover_idx = trace.index("discover_suppliers")
        for name in HAZARD_NAMES:
            scorer_idx = trace.index(f"score_{name}")
            assert scorer_idx > discover_idx

    def test_trace_order_scorers_before_aggregate(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        aggregate_idx = trace.index("aggregate_risk")
        for name in HAZARD_NAMES:
            scorer_idx = trace.index(f"score_{name}")
            assert scorer_idx < aggregate_idx


# ---------------------------------------------------------------------------
# 3. Human-in-the-loop tests
# ---------------------------------------------------------------------------


def _invoke_interactive(company: str, tier_depth: int = 2) -> tuple:
    """Invoke the graph in interactive mode; return (graph, config, result)."""
    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {
            "company": company,
            "tier_depth": tier_depth,
            "use_llm": False,
            "interactive": True,
            "hazard_scores": [],
            "workflow_trace": [],
        },
        config,
    )
    return graph, config, result


class TestHumanInTheLoop:
    """Verify interrupt behavior in decide_workflow_node."""

    def test_non_interactive_runs_straight_through(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False, interactive=False)
        assert "workflow_decision" in state
        assert "report_text" in state

    def test_interactive_high_risk_triggers_interrupt(self) -> None:
        """ACME is high-risk; with interactive=True, graph should pause."""
        graph, config, result = _invoke_interactive("ACME")
        snapshot = graph.get_state(config)
        assert snapshot.next, "Graph should be paused at an interrupt"

    def test_interactive_approve_continues_high_risk(self) -> None:
        graph, config, _ = _invoke_interactive("ACME")
        final = graph.invoke(Command(resume="approved"), config)
        assert final["workflow_decision"]["route"] == "high_risk_response"
        assert "report_text" in final

    def test_interactive_downgrade_switches_to_monitoring(self) -> None:
        graph, config, _ = _invoke_interactive("ACME")
        final = graph.invoke(Command(resume="downgrade"), config)
        assert final["workflow_decision"]["route"] == "monitoring_response"
        assert "Downgraded by human" in final["workflow_decision"]["reason"]

    def test_interactive_low_risk_no_interrupt(self) -> None:
        """GlobalMfg is low-risk; no interrupt even when interactive."""
        state = run_graph("GlobalMfg", tier_depth=1, use_llm=False, interactive=True)
        assert state["workflow_decision"]["route"] == "monitoring_response"
        assert "report_text" in state


# ---------------------------------------------------------------------------
# 4. Streaming tests
# ---------------------------------------------------------------------------


class TestStreaming:
    """Verify streaming yields correct events for each node."""

    def test_stream_yields_all_node_events(self) -> None:
        checkpointer = MemorySaver()
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        node_names_seen: set[str] = set()
        for event in graph.stream(
            {
                "company": "ACME",
                "tier_depth": 2,
                "use_llm": False,
                "interactive": False,
                "hazard_scores": [],
                "workflow_trace": [],
            },
            config,
            stream_mode="updates",
        ):
            for node_name in event:
                node_names_seen.add(node_name)

        assert "discover_suppliers" in node_names_seen
        for name in HAZARD_NAMES:
            assert f"score_{name}" in node_names_seen
        assert "aggregate_risk" in node_names_seen
        assert "decide_workflow" in node_names_seen
        assert "format_report" in node_names_seen

    def test_stream_parallel_nodes_yield_separately(self) -> None:
        """Each parallel hazard scorer should yield its own event."""
        checkpointer = MemorySaver()
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        scorer_events: list[str] = []
        for event in graph.stream(
            {
                "company": "ACME",
                "tier_depth": 1,
                "use_llm": False,
                "interactive": False,
                "hazard_scores": [],
                "workflow_trace": [],
            },
            config,
            stream_mode="updates",
        ):
            for node_name in event:
                if node_name.startswith("score_"):
                    scorer_events.append(node_name)

        assert len(scorer_events) == 6


# ---------------------------------------------------------------------------
# 5. Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure run_graph() still works identically for all existing usage."""

    def test_run_graph_default_args(self) -> None:
        state = run_graph("ACME", tier_depth=2)
        assert "report_text" in state
        assert len(state["hazard_scores"]) > 0
        assert "discover_suppliers" in state["workflow_trace"]

    def test_run_graph_with_use_llm_false(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        assert state["workflow_decision"]["route"] == "high_risk_response"

    def test_workflow_trace_has_expected_entries(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        trace = state["workflow_trace"]
        assert "discover_suppliers" in trace
        assert "aggregate_risk" in trace
        assert "decide_workflow" in trace
        assert "format_report" in trace

    def test_report_contains_trace_line(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "Trace:" in report
        assert "discover_suppliers" in report
        assert "format_report" in report
