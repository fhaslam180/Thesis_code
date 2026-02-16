"""Tests for LLM-based supplier discovery (mocked, no real API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from bor_risk.models import CompanyProfile, Supplier, TierResponse
from bor_risk.tools import _deduplicate_suppliers, discover_suppliers_llm, resolve_company_profile
from bor_risk.graph import run_graph

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm_responses"


def _load_response(filename: str) -> TierResponse:
    return TierResponse.model_validate_json(
        (FIXTURES_DIR / filename).read_text()
    )


def _load_profile(filename: str) -> CompanyProfile:
    return CompanyProfile.model_validate_json(
        (FIXTURES_DIR / filename).read_text()
    )


def _make_mock_chain(*responses: TierResponse) -> MagicMock:
    """Build a mock ChatOpenAI whose .with_structured_output().invoke()
    returns the given TierResponse objects in order."""
    mock_llm_cls = MagicMock()
    mock_chain = mock_llm_cls.return_value.with_structured_output.return_value
    mock_chain.invoke.side_effect = list(responses)
    return mock_llm_cls


class TestDiscoverSuppliersLLM:
    """Unit tests for discover_suppliers_llm with mocked LLM."""

    def test_returns_suppliers_edges_evidence(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            suppliers, edges, evidence = discover_suppliers_llm("ACME", tier_depth=1)

        assert len(suppliers) == 2
        assert len(edges) == 2
        assert len(evidence) == 2

    def test_every_edge_has_evidence(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            _, edges, _ = discover_suppliers_llm("ACME", tier_depth=1)

        for edge in edges:
            assert len(edge["evidence_ids"]) >= 1

    def test_evidence_cites_llm_source(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            _, _, evidence = discover_suppliers_llm("ACME", tier_depth=1)

        for ev in evidence:
            assert "LLM" in ev["source"]

    def test_multi_tier_discovers_deeper_suppliers(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        tier2_steel = _load_response("acme_tier2_steelcorp.json")
        tier2_chip = _load_response("acme_tier2_chipworks.json")
        mock_cls = _make_mock_chain(tier1, tier2_steel, tier2_chip)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            suppliers, edges, evidence = discover_suppliers_llm("ACME", tier_depth=2)

        tiers = {s.tier for s in suppliers}
        assert 1 in tiers
        assert 2 in tiers
        names = {s.name for s in suppliers}
        assert "SteelCorp" in names
        assert "IronOre Global" in names

    def test_supplier_fields_populated(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            suppliers, _, _ = discover_suppliers_llm("ACME", tier_depth=1)

        for s in suppliers:
            assert -90 <= s.lat <= 90
            assert -180 <= s.lon <= 180
            assert 0 <= s.confidence <= 1
            assert len(s.evidence_ids) >= 1


class TestCompanyProfile:
    """Tests for company profile resolution."""

    def test_profile_resolves_structured_output(self) -> None:
        profile = _load_profile("acme_profile.json")
        mock_cls = _make_mock_chain(profile)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            result = resolve_company_profile("ACME")

        assert result["canonical_name"] == "ACME Corporation"
        assert result["industry"] == "Industrial Manufacturing"
        assert len(result["products"]) == 3
        assert result["headquarters"] == "Chicago, USA"

    def test_tier_prompt_includes_industry_context(self) -> None:
        """When company_profile is provided, the prompt includes industry and products."""
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        company_profile = {
            "industry": "Industrial Manufacturing",
            "products": ["Industrial equipment", "Precision tools"],
        }

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            discover_suppliers_llm("ACME", tier_depth=1, company_profile=company_profile)

        invoke_args = (
            mock_cls.return_value.with_structured_output.return_value.invoke.call_args[0][0]
        )
        prompt_text = invoke_args[0].content
        assert "Industrial Manufacturing" in prompt_text
        assert "Industrial equipment" in prompt_text


class TestSupplierNewFields:
    """Tests for enriched supplier fields from LLM discovery."""

    def test_suppliers_have_industry_fields(self) -> None:
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            suppliers, _, _ = discover_suppliers_llm("ACME", tier_depth=1)

        for s in suppliers:
            assert s.industry != ""
            assert s.product_category != ""
            assert s.location_description != ""
            assert s.relationship_type != ""

    def test_fixture_suppliers_have_new_fields(self) -> None:
        """Mock fixtures include industry/product/location/relationship fields."""
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        for s in state["suppliers"]:
            assert s.get("industry", "") != ""
            assert s.get("product_category", "") != ""
            assert s.get("location_description", "") != ""
            assert s.get("relationship_type", "") != ""


class TestDeduplication:
    """Tests for supplier deduplication logic."""

    def test_merges_same_name_suppliers(self) -> None:
        suppliers = [
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=1, confidence=0.8, evidence_ids=["E1"]),
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=2, confidence=0.9, evidence_ids=["E2"]),
        ]
        result = _deduplicate_suppliers(suppliers)
        assert len(result) == 1
        assert result[0].name == "FooCorp"

    def test_keeps_highest_confidence(self) -> None:
        suppliers = [
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=1, confidence=0.6, evidence_ids=["E1"]),
            Supplier(name="FooCorp", lat=3.0, lon=4.0, tier=2, confidence=0.9, evidence_ids=["E2"]),
        ]
        result = _deduplicate_suppliers(suppliers)
        assert result[0].confidence == 0.9
        assert result[0].lat == 3.0  # Uses the higher-confidence entry's location

    def test_merges_evidence_ids(self) -> None:
        suppliers = [
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=1, confidence=0.8, evidence_ids=["E1", "E2"]),
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=2, confidence=0.7, evidence_ids=["E2", "E3"]),
        ]
        result = _deduplicate_suppliers(suppliers)
        assert set(result[0].evidence_ids) == {"E1", "E2", "E3"}

    def test_case_insensitive_matching(self) -> None:
        suppliers = [
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=1, confidence=0.8, evidence_ids=["E1"]),
            Supplier(name="foocorp", lat=1.0, lon=2.0, tier=2, confidence=0.7, evidence_ids=["E2"]),
        ]
        result = _deduplicate_suppliers(suppliers)
        assert len(result) == 1

    def test_no_duplicates_passes_through(self) -> None:
        suppliers = [
            Supplier(name="FooCorp", lat=1.0, lon=2.0, tier=1, confidence=0.8, evidence_ids=["E1"]),
            Supplier(name="BarInc", lat=3.0, lon=4.0, tier=1, confidence=0.9, evidence_ids=["E2"]),
        ]
        result = _deduplicate_suppliers(suppliers)
        assert len(result) == 2


class TestRunGraphWithLLM:
    """Integration test: full graph with mocked LLM."""

    def test_produces_report(self) -> None:
        profile = _load_profile("acme_profile.json")
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(profile, tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        assert "Supply-Chain Risk Report" in state["report_text"]
        assert len(state["suppliers"]) > 0
        assert len(state["hazard_scores"]) > 0
        assert len(state["edges"]) > 0
        assert len(state["evidence"]) > 0

    def test_graph_stores_company_profile(self) -> None:
        profile = _load_profile("acme_profile.json")
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(profile, tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        assert "company_profile" in state
        assert state["company_profile"]["canonical_name"] == "ACME Corporation"
        assert state["company_profile"]["industry"] == "Industrial Manufacturing"

    def test_report_shows_company_profile(self) -> None:
        profile = _load_profile("acme_profile.json")
        tier1 = _load_response("acme_tier1.json")
        mock_cls = _make_mock_chain(profile, tier1)

        with patch("bor_risk.tools.ChatOpenAI", mock_cls):
            state = run_graph("ACME", tier_depth=1, use_llm=True)

        report = state["report_text"]
        assert "--- Company Profile ---" in report
        assert "ACME Corporation" in report
        assert "Industrial Manufacturing" in report

    def test_report_supplier_ranking_shows_industry(self) -> None:
        state = run_graph("ACME", tier_depth=2, use_llm=False)
        report = state["report_text"]
        assert "industry=Steel Manufacturing" in report
        assert "industry=Semiconductor Manufacturing" in report
        assert "location=Tokyo, Japan" in report
