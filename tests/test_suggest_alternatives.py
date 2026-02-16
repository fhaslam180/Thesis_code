"""Tests for LLM-powered alternative supplier suggestions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from bor_risk.graph import run_graph
from bor_risk.models import (
    AlternativeResponse,
    CompanyProfile,
    MitigationResponse,
    TierResponse,
)
from bor_risk.tools import suggest_alternatives_llm

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm_responses"


def _load_alternative_response(filename: str) -> AlternativeResponse:
    return AlternativeResponse.model_validate_json(
        (FIXTURES_DIR / filename).read_text()
    )


def _load_tier_response(filename: str) -> TierResponse:
    return TierResponse.model_validate_json((FIXTURES_DIR / filename).read_text())


def _load_mitigation_response(filename: str) -> MitigationResponse:
    return MitigationResponse.model_validate_json(
        (FIXTURES_DIR / filename).read_text()
    )


def _load_profile(filename: str) -> CompanyProfile:
    return CompanyProfile.model_validate_json((FIXTURES_DIR / filename).read_text())


def _make_mock_chain(*responses: object) -> MagicMock:
    """Mock ChatOpenAI.with_structured_output().invoke() call sequence."""
    mock_llm_cls = MagicMock()
    mock_chain = mock_llm_cls.return_value.with_structured_output.return_value
    mock_chain.invoke.side_effect = list(responses)
    return mock_llm_cls


class TestAlternativeModels:
    """Pydantic model validation."""

    def test_alternative_response_round_trips(self) -> None:
        response = _load_alternative_response("acme_alternatives.json")
        assert response.company == "ACME"
        assert len(response.alternatives) == 2
        assert response.alternatives[0].alternative_for == "ChipWorks"
        assert len(response.alternatives[0].candidates) == 2
        assert 0 <= response.alternatives[0].candidates[0].confidence <= 1


class TestSuggestAlternativesLLM:
    """Unit tests for suggest_alternatives_llm with mocked LLM."""

    def test_parses_structured_output(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        alt_response = _load_alternative_response("acme_alternatives.json")
        mock_cls = _make_mock_chain(alt_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            alternatives = suggest_alternatives_llm(
                company="ACME",
                suppliers=state["suppliers"],
                hazard_scores=state["hazard_scores"],
                summary=state["company_risk_summary"],
            )

        assert len(alternatives) == 2
        assert alternatives[0]["alternative_for"] == "ChipWorks"
        assert alternatives[0]["hazard_type"] == "earthquake"
        assert len(alternatives[0]["candidates"]) == 2

    def test_prompt_includes_risk_context(self) -> None:
        state = run_graph("ACME", tier_depth=1, use_llm=False)
        alt_response = _load_alternative_response("acme_alternatives.json")
        mock_cls = _make_mock_chain(alt_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            suggest_alternatives_llm(
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


class TestSuggestAlternativesNode:
    """Graph node behavior."""

    def test_no_llm_produces_empty_alternatives(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        assert "suggested_alternatives" in state
        assert len(state["suggested_alternatives"]) == 0

    def test_workflow_trace_includes_suggest_alternatives(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        assert "suggest_alternatives" in state["workflow_trace"]

    def test_node_handles_llm_failure(self) -> None:
        """Exception in suggest_alternatives_llm returns empty list gracefully."""
        with patch(
            "bor_risk.graph.suggest_alternatives_llm",
            side_effect=RuntimeError("API down"),
        ):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        assert state["suggested_alternatives"] == []


class TestAlternativesInReport:
    """Report rendering."""

    def test_no_llm_report_shows_no_alternatives(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "--- Suggested Alternative Suppliers ---" in report
        assert "No alternatives suggested." in report

    def test_report_with_alternatives(self) -> None:
        profile = _load_profile("acme_profile.json")
        tier1 = _load_tier_response("acme_tier1.json")
        mitigation_response = _load_mitigation_response("acme_mitigations.json")
        alt_response = _load_alternative_response("acme_alternatives.json")
        mock_cls = _make_mock_chain(profile, tier1, mitigation_response, alt_response)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        report = state["report_text"]
        assert "--- Suggested Alternative Suppliers ---" in report
        assert "Replace ChipWorks" in report
        assert "MalaysiaChip Corp" in report
