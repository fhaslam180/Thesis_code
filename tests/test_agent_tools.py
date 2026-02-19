"""Tests for agent tools and supplier verification."""

from unittest.mock import MagicMock, patch

from bor_risk.agent_tools import _RELATIONSHIP_CUES, build_agent_tools
from bor_risk.budget import BudgetTracker


def _make_accumulator():
    """Create a fresh shared accumulator for testing."""
    return {
        "company": "TestCo",
        "tier_depth": 2,
        "suppliers": [],
        "edges": [],
        "evidence": [],
        "hazard_scores": [],
        "company_profile": {},
        "company_risk_summary": {},
        "llm_mitigations": [],
        "suggested_alternatives": [],
        "_web_evidence_counter": 1,
    }


class TestBuildAgentTools:
    def test_web_tools_included_by_default(self):
        acc = _make_accumulator()
        budget = BudgetTracker()
        tools = build_agent_tools(acc, budget, enable_web=True)
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "verify_supplier" in names

    def test_web_tools_excluded_when_disabled(self):
        acc = _make_accumulator()
        budget = BudgetTracker()
        tools = build_agent_tools(acc, budget, enable_web=False)
        names = [t.name for t in tools]
        assert "web_search" not in names
        assert "verify_supplier" not in names

    def test_core_tools_always_present(self):
        acc = _make_accumulator()
        budget = BudgetTracker()
        tools = build_agent_tools(acc, budget, enable_web=False)
        names = [t.name for t in tools]
        assert "profile_company" in names
        assert "discover_suppliers" in names
        assert "score_hazard" in names
        assert "aggregate_risk" in names
        assert "generate_mitigations" in names
        assert "suggest_alternatives" in names


class TestScoreHazardTool:
    def test_scores_supplier_and_records_budget(self):
        acc = _make_accumulator()
        budget = BudgetTracker()
        tools = build_agent_tools(acc, budget, enable_web=False)
        score_tool = next(t for t in tools if t.name == "score_hazard")

        result = score_tool.invoke({
            "supplier_name": "TestSupplier",
            "lat": 35.6,
            "lon": 139.7,
            "hazard_type": "earthquake",
        })

        assert "TestSupplier" in result
        assert "earthquake" in result
        assert len(acc["hazard_scores"]) == 1
        assert budget.hazard_scores == 1
        # Hazard scores don't count against LLM/web budget.
        assert budget.llm_calls == 0
        assert budget.web_queries == 0


class TestAggregateRiskTool:
    def test_aggregates_from_accumulator(self):
        acc = _make_accumulator()
        acc["suppliers"] = [
            {"name": "A", "lat": 0, "lon": 0, "tier": 1, "confidence": 0.8},
        ]
        acc["hazard_scores"] = [
            {"supplier_name": "A", "hazard_type": "earthquake",
             "score": 0.5, "score_100": 50, "level": "Medium",
             "dataset_metadata": {}},
        ]
        budget = BudgetTracker()
        tools = build_agent_tools(acc, budget, enable_web=False)
        agg_tool = next(t for t in tools if t.name == "aggregate_risk")

        result = agg_tool.invoke({})
        assert "Company risk" in result
        assert acc["company_risk_summary"] != {}


class TestBudgetEnforcement:
    def test_llm_tool_respects_budget(self):
        acc = _make_accumulator()
        budget = BudgetTracker(max_llm_calls=0)
        tools = build_agent_tools(acc, budget, enable_web=False)
        profile_tool = next(t for t in tools if t.name == "profile_company")

        result = profile_tool.invoke({"company": "TestCo"})
        assert "exhausted" in result.lower()

    def test_web_tool_respects_budget(self):
        acc = _make_accumulator()
        budget = BudgetTracker(max_web_queries=0)
        tools = build_agent_tools(acc, budget, enable_web=True)
        web_tool = next(t for t in tools if t.name == "web_search")

        result = web_tool.invoke({"query": "test"})
        assert "exhausted" in result.lower()

    def test_verify_tool_respects_budget(self):
        acc = _make_accumulator()
        budget = BudgetTracker(max_web_queries=0)
        tools = build_agent_tools(acc, budget, enable_web=True)
        verify_tool = next(t for t in tools if t.name == "verify_supplier")

        result = verify_tool.invoke({
            "supplier_name": "Test",
            "parent_company": "Parent",
        })
        assert "exhausted" in result.lower()


class TestVerifySupplierTool:
    def test_verified_upgrades_evidence_source(self):
        acc = _make_accumulator()
        acc["suppliers"] = [
            {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
             "confidence": 0.7, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker(max_web_queries=5)

        mock_results = [
            {"title": "Apple suppliers",
             "url": "https://example.com/article",
             "content": "Apple sources chips from TSMC, a key supplier in the semiconductor supply chain.",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]

        # Patch BEFORE building tools so the closure captures the mock.
        with patch("bor_risk.agent_tools.search_web_snapshot", return_value=mock_results):
            tools = build_agent_tools(
                acc, budget, enable_web=True, snapshot_mode=True
            )
            verify_tool = next(t for t in tools if t.name == "verify_supplier")
            result = verify_tool.invoke({
                "supplier_name": "TSMC",
                "parent_company": "Apple",
            })

        assert "VERIFIED" in result
        assert acc["suppliers"][0]["evidence_source"] == "web_verified"
        assert abs(acc["suppliers"][0]["confidence"] - 0.8) < 1e-9  # boosted by 0.1

    def test_unverified_stays_llm_only(self):
        acc = _make_accumulator()
        acc["suppliers"] = [
            {"name": "FakeCo", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.5, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker(max_web_queries=5)

        # Patch BEFORE building tools so the closure captures the mock.
        with patch("bor_risk.agent_tools.search_web_snapshot", return_value=[]):
            tools = build_agent_tools(
                acc, budget, enable_web=True, snapshot_mode=True
            )
            verify_tool = next(t for t in tools if t.name == "verify_supplier")
            result = verify_tool.invoke({
                "supplier_name": "FakeCo",
                "parent_company": "Apple",
            })

        assert "UNVERIFIED" in result
        assert acc["suppliers"][0]["evidence_source"] == "llm_only"


class TestRelationshipCues:
    def test_relationship_cues_exist(self):
        assert len(_RELATIONSHIP_CUES) > 5
        assert "supplier" in _RELATIONSHIP_CUES
        assert "vendor" in _RELATIONSHIP_CUES
