"""Tests for LLM-powered mitigation generation and report rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from bor_risk.graph import run_graph
from bor_risk.models import CompanyProfile, MitigationResponse, TierResponse
from bor_risk.tools import generate_mitigations_llm

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm_responses"


def _load_tier_response(filename: str) -> TierResponse:
    return TierResponse.model_validate_json((FIXTURES_DIR / filename).read_text())


def _load_mitigation_response(filename: str) -> MitigationResponse:
    return MitigationResponse.model_validate_json((FIXTURES_DIR / filename).read_text())


def _load_profile(filename: str) -> CompanyProfile:
    return CompanyProfile.model_validate_json((FIXTURES_DIR / filename).read_text())


def _make_mock_chain(*responses: object) -> MagicMock:
    """Mock ChatOpenAI.with_structured_output().invoke() call sequence."""
    mock_llm_cls = MagicMock()
    mock_chain = mock_llm_cls.return_value.with_structured_output.return_value
    mock_chain.invoke.side_effect = list(responses)
    return mock_llm_cls


class TestGenerateMitigationsLLM:
    """Unit tests for generate_mitigations_llm with mocked LLM."""

    def test_parses_structured_output(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        mitigation_response = _load_mitigation_response("acme_mitigations.json")
        mock_cls = _make_mock_chain(mitigation_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            mitigations = generate_mitigations_llm(
                company="ACME",
                suppliers=state["suppliers"],
                hazard_scores=state["hazard_scores"],
                summary=state["company_risk_summary"],
            )

        assert len(mitigations) == len(mitigation_response.mitigations)
        assert mitigations[0]["supplier_name"] == "ChipWorks"
        assert mitigations[0]["priority"] == "P1"
        assert len(mitigations[0]["actions"]) >= 2

    def test_prompt_includes_risk_context(self) -> None:
        state = run_graph("ACME", tier_depth=1, use_llm=False)
        mitigation_response = _load_mitigation_response("acme_mitigations.json")
        mock_cls = _make_mock_chain(mitigation_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            generate_mitigations_llm(
                company="ACME",
                suppliers=state["suppliers"],
                hazard_scores=state["hazard_scores"],
                summary=state["company_risk_summary"],
            )

        invoke_args = (
            mock_cls.return_value.with_structured_output.return_value.invoke.call_args[0][0]
        )
        prompt_text = invoke_args[0].content
        assert "Company risk score:" in prompt_text
        assert "Suppliers:" in prompt_text
        assert "Hazard scores (Medium and High only):" in prompt_text


class TestMitigationNodeMockMode:
    """Graph behavior with use_llm=False skips LLM mitigation generation."""

    def test_use_llm_false_produces_empty_mitigations(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "--- Mitigations ---" in report
        assert "llm_mitigations" in state
        assert len(state["llm_mitigations"]) == 0

    def test_use_llm_false_shows_no_mitigations_message(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "No mitigations generated" in report

    def test_workflow_trace_includes_mitigation_node(self) -> None:
        state = run_graph("GlobalMfg", tier_depth=1, use_llm=False)
        assert "generate_mitigations" in state["workflow_trace"]


class TestMitigationInReport:
    """Integration tests with mocked LLM discovery + mitigation generation."""

    def test_run_graph_llm_mode_includes_structured_mitigations(self) -> None:
        profile = _load_profile("acme_profile.json")
        tier1 = _load_tier_response("acme_tier1.json")
        mitigation_response = _load_mitigation_response("acme_mitigations.json")
        mock_cls = _make_mock_chain(profile, tier1, mitigation_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        report = state["report_text"]
        assert "--- Mitigations ---" in report
        assert "ChipWorks / earthquake (High) [P1]:" in report
        assert "Rationale:" in report
        assert "Activate secondary chip sourcing" in report
        assert len(state["llm_mitigations"]) == 3
