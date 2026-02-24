"""Tests for Phase 8 enhanced report sections."""

from __future__ import annotations

from bor_risk.graph import run_graph


def _acme_state() -> dict:
    return run_graph("ACME", tier_depth=2, use_llm=False)


def _global_state() -> dict:
    return run_graph("GlobalMfg", tier_depth=1, use_llm=False)


class TestExecutiveSummary:
    """Tests for the executive summary section."""

    def test_present_in_report(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Executive Summary ---" in report

    def test_contains_company_name(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Executive Summary ---")
        section = report[idx : idx + 500]
        assert "ACME" in section

    def test_contains_risk_band(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Executive Summary ---")
        section = report[idx : idx + 500].lower()
        assert any(band in section for band in ["high", "medium", "low"])

    def test_contains_supplier_count(self) -> None:
        state = _acme_state()
        report = state["report_text"]
        idx = report.index("--- Executive Summary ---")
        section = report[idx : idx + 500]
        assert str(len(state["suppliers"])) in section

    def test_contains_critical_mention(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Executive Summary ---")
        section = report[idx : idx + 500].lower()
        assert "critical" in section


class TestNarrativeSections:
    """Tests for prose-first narrative report sections."""

    def test_introduction_present(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Introduction ---" in report

    def test_narrative_assessment_present(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Narrative Assessment ---" in report

    def test_narrative_assessment_has_multiple_paragraphs(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Narrative Assessment ---")
        section = report[idx:]
        # Expect blank-line paragraph breaks in the narrative section.
        assert section.count("\n\n") >= 3

    def test_detailed_data_appendix_marker_present(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Detailed Data Appendix ---" in report


class TestRiskRegisterMatrix:
    """Tests for the risk register matrix section."""

    def test_present_in_report(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Risk Register Matrix ---" in report

    def test_contains_all_supplier_names(self) -> None:
        state = _acme_state()
        report = state["report_text"]
        idx = report.index("--- Risk Register Matrix ---")
        section = report[idx:]
        for s in state["suppliers"]:
            assert s["name"] in section

    def test_contains_all_hazard_types(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Risk Register Matrix ---")
        section = report[idx : idx + 500]
        for hazard in ["earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought"]:
            assert hazard in section

    def test_contains_level_abbreviations(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Risk Register Matrix ---")
        section = report[idx:]
        assert any(lvl in section for lvl in ["Low", "Med", "Hig"])


class TestMitigations:
    """Tests for the LLM-driven mitigations section."""

    def test_present_in_report(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Mitigations ---" in report

    def test_has_content_or_none_message(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Mitigations ---")
        section = report[idx:]
        assert (
            "No mitigations generated." in section
            or "Rationale:" in section
            or "    - " in section
        )

    def test_mitigation_format_when_present(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Mitigations ---")
        section = report[idx:]
        if "No mitigations generated." in section:
            return
        assert " / " in section
        assert any(f"({lvl})" in section for lvl in ["Medium", "High"])

    def test_global_has_section(self) -> None:
        """GlobalMfg should still have the section header."""
        report = _global_state()["report_text"]
        assert "--- Mitigations ---" in report


class TestEvidenceAppendix:
    """Tests for the enhanced evidence appendix."""

    def test_present_in_report(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Evidence Appendix ---" in report

    def test_contains_evidence_ids(self) -> None:
        state = _acme_state()
        report = state["report_text"]
        for ev in state["evidence"]:
            assert ev["evidence_id"] in report

    def test_contains_dataset_details(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Evidence Appendix ---")
        section = report[idx:]
        assert "Dataset Details:" in section

    def test_contains_dataset_metadata(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- Evidence Appendix ---")
        section = report[idx:]
        assert "usgs_earthquake_catalog" in section


class TestIEEEReferences:
    """Tests for IEEE-style references."""

    def test_present_in_report(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- IEEE References ---" in report

    def test_contains_numbered_references(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- IEEE References ---")
        section = report[idx:]
        assert "[1]" in section

    def test_contains_urls(self) -> None:
        report = _acme_state()["report_text"]
        idx = report.index("--- IEEE References ---")
        section = report[idx:]
        assert "https://" in section

    def test_datasets_deduplicated(self) -> None:
        """3 suppliers x 6 hazards = 18 scores, but <= 6 unique datasets."""
        report = _acme_state()["report_text"]
        idx = report.index("--- IEEE References ---")
        section = report[idx:]
        # Count reference numbers — should be at most 6
        import re
        refs = re.findall(r"\[\d+\]", section)
        assert len(refs) <= 6


class TestExistingMarkersPreserved:
    """Verify that all existing test assertions still pass with enhanced report."""

    def test_supply_chain_risk_report_marker(self) -> None:
        report = _acme_state()["report_text"]
        assert "Supply-Chain Risk Report" in report

    def test_agent_workflow_marker(self) -> None:
        report = _acme_state()["report_text"]
        assert "--- Agent Workflow ---" in report
        assert "Decision route" in report
        assert "Trace:" in report

    def test_report_text_nonempty(self) -> None:
        assert len(_acme_state()["report_text"]) > 0
