"""Tests for pipeline web verification (Condition B) and evidence-source weighting."""

from unittest.mock import patch

from bor_risk.budget import BudgetTracker
from bor_risk.tools import compute_risk_summary, verify_suppliers_batch


class TestVerifySuppliersBatch:
    def _make_suppliers(self):
        return [
            {"name": "HighConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.9, "evidence_source": "llm_only"},
            {"name": "MedConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.7, "evidence_source": "llm_only"},
            {"name": "LowConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.3, "evidence_source": "llm_only"},
        ]

    def test_verifies_in_descending_confidence(self):
        """Pipeline verifies highest-confidence suppliers first."""
        suppliers = self._make_suppliers()
        budget = BudgetTracker(max_web_queries=1)  # Only 1 query allowed
        call_order = []

        def mock_search(query, **kwargs):
            call_order.append(query)
            return []

        with patch("bor_risk.search.search_web_snapshot", side_effect=mock_search):
            updated, evidence = verify_suppliers_batch(
                suppliers, "TestCo", budget, snapshot_mode=True
            )

        # Should have tried to verify HighConf first (highest confidence).
        assert len(call_order) == 1
        assert "HighConf" in call_order[0]

    def test_stops_when_budget_exhausted(self):
        """Pipeline stops verification when web budget runs out."""
        suppliers = self._make_suppliers()
        budget = BudgetTracker(max_web_queries=2)

        with patch("bor_risk.search.search_web_snapshot", return_value=[]):
            updated, evidence = verify_suppliers_batch(
                suppliers, "TestCo", budget, snapshot_mode=True
            )

        assert budget.web_queries == 2
        assert budget.web_budget_remaining == 0

    def test_upgrades_on_co_mention(self):
        """Verification upgrades evidence_source when co-mention found."""
        suppliers = [
            {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
             "confidence": 0.8, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker(max_web_queries=5)

        mock_results = [
            {"title": "Apple suppliers",
             "url": "https://example.com/article",
             "content": "Apple relies on TSMC as a key supplier for its A-series chips.",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]

        with patch("bor_risk.search.search_web_snapshot", return_value=mock_results):
            updated, evidence = verify_suppliers_batch(
                suppliers, "Apple", budget, snapshot_mode=True
            )

        tsmc = next(s for s in updated if s["name"] == "TSMC")
        assert tsmc["evidence_source"] == "web_verified"
        assert tsmc["verification_url"] == "https://example.com/article"
        assert tsmc["confidence"] == 0.9  # boosted by 0.1
        assert len(evidence) == 1

    def test_no_upgrade_without_co_mention(self):
        """No upgrade if only supplier name appears (no company co-mention)."""
        suppliers = [
            {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
             "confidence": 0.8, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker(max_web_queries=5)

        # Content mentions TSMC but not Apple.
        mock_results = [
            {"title": "TSMC news",
             "url": "https://example.com/article",
             "content": "TSMC is the world's largest semiconductor supplier.",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]

        with patch("bor_risk.search.search_web_snapshot", return_value=mock_results):
            updated, evidence = verify_suppliers_batch(
                suppliers, "Apple", budget, snapshot_mode=True
            )

        tsmc = next(s for s in updated if s["name"] == "TSMC")
        assert tsmc["evidence_source"] == "llm_only"


class TestEvidenceSourceWeighting:
    def test_llm_only_gets_halved_confidence(self):
        """Unverified LLM suppliers get halved confidence in aggregation."""
        hazard_scores = [
            {"supplier_name": "Verified", "hazard_type": "earthquake",
             "score": 0.5, "score_100": 50, "level": "Medium",
             "dataset_metadata": {}},
            {"supplier_name": "Unverified", "hazard_type": "earthquake",
             "score": 0.5, "score_100": 50, "level": "Medium",
             "dataset_metadata": {}},
        ]
        suppliers = [
            {"name": "Verified", "tier": 1, "confidence": 0.8,
             "evidence_source": "web_verified"},
            {"name": "Unverified", "tier": 1, "confidence": 0.8,
             "evidence_source": "llm_only"},
        ]
        weights = {"earthquake": 1.0}
        thresholds = {"earthquake": 1.0}

        summary = compute_risk_summary(
            hazard_scores, suppliers, weights, thresholds
        )

        risks = {r["supplier_name"]: r for r in summary["supplier_risks"]}
        # Same base score but Unverified should have lower risk_score
        # due to halved effective confidence.
        assert risks["Verified"]["risk_score"] > risks["Unverified"]["risk_score"]

    def test_fixture_sources_not_penalised(self):
        """Suppliers from test fixtures use full confidence."""
        hazard_scores = [
            {"supplier_name": "Fixture", "hazard_type": "earthquake",
             "score": 0.5, "score_100": 50, "level": "Medium",
             "dataset_metadata": {}},
        ]
        suppliers = [
            {"name": "Fixture", "tier": 1, "confidence": 0.8,
             "evidence_source": "fixture"},
        ]
        weights = {"earthquake": 1.0}
        thresholds = {"earthquake": 1.0}

        summary = compute_risk_summary(
            hazard_scores, suppliers, weights, thresholds
        )

        risk = summary["supplier_risks"][0]
        # With evidence_source="fixture", confidence_factor = 0.5 + 0.5*0.8 = 0.9
        expected = round(0.5 * 0.9, 4)
        assert risk["risk_score"] == expected
