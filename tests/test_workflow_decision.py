"""Tests for LangGraph workflow branching and risk rollup output."""

from __future__ import annotations

from bor_risk.graph import run_graph


class TestWorkflowDecision:
    """Validate branch routing and risk summary payloads."""

    def test_acme_routes_to_high_risk_response(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        assert state["workflow_decision"]["route"] == "high_risk_response"
        assert "high_risk_response" in state["workflow_trace"]
        assert state["company_risk_summary"]["critical_alert_count"] >= 1

    def test_global_routes_to_monitoring_response(self) -> None:
        state = run_graph("GlobalMfg", tier_depth=1, use_llm=False)
        assert state["workflow_decision"]["route"] == "monitoring_response"
        assert "monitoring_response" in state["workflow_trace"]
        assert state["company_risk_summary"]["critical_alert_count"] == 0

    def test_company_score_is_bounded(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        score = state["company_risk_summary"]["company_score"]
        assert 0.0 <= score <= 1.0
        assert len(state["company_risk_summary"]["supplier_risks"]) > 0

    def test_report_contains_workflow_section(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "--- Agent Workflow ---" in report
        assert "Decision route" in report
        assert "Trace:" in report
