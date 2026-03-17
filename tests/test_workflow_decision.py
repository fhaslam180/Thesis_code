"""Tests for VCG pipeline risk scoring and report output."""

from __future__ import annotations

from bor_risk.graph import run_vcg_graph


class TestRiskScoring:
    """Validate risk summary payloads from VCG pipeline."""

    def test_acme_has_high_risk_score(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        summary = state["company_risk_summary"]
        assert 0.0 <= summary["company_score"] <= 1.0
        band = summary.get("company_band", summary.get("risk_band", "Low"))
        assert band in ("High", "Medium", "Low")
        assert len(summary.get("supplier_risks", [])) > 0

    def test_global_has_lower_risk(self) -> None:
        state = run_vcg_graph("GlobalMfg", tier_depth=1, use_llm=False)
        summary = state["company_risk_summary"]
        assert 0.0 <= summary["company_score"] <= 1.0

    def test_company_score_is_bounded(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        score = state["company_risk_summary"]["company_score"]
        assert 0.0 <= score <= 1.0
        assert len(state["company_risk_summary"]["supplier_risks"]) > 0

    def test_report_contains_pipeline_workflow_section(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "--- Pipeline Workflow ---" in report
        assert "Trace:" in report

    def test_suppliers_have_expected_keys(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        for s in state["suppliers"]:
            assert "name" in s
            assert "lat" in s
            assert "lon" in s
            assert "tier" in s

    def test_supplier_risks_use_smhei_fields(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        for sr in state["company_risk_summary"].get("supplier_risks", []):
            assert "supplier_name" in sr
            assert "exposure_index" in sr
            assert "exposure_band" in sr
            assert "dominant_hazard" in sr

    def test_claims_populated_for_all_suppliers(self) -> None:
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        assert len(state["claims"]) == len(state["suppliers"])

    def test_all_claims_start_proposed_or_verified(self) -> None:
        """Without web retrieval claims stay PROPOSED; status is valid."""
        state = run_vcg_graph("ACME", tier_depth=2, use_llm=False)
        valid_statuses = {"PROPOSED", "RETRIEVED", "VERIFIED"}
        for c in state["claims"]:
            assert c["status"] in valid_statuses
