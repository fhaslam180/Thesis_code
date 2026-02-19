"""Tests for evaluation framework."""

from bor_risk.evaluate import (
    _normalize_name,
    compute_ground_truth_metrics,
    compute_hard_metrics,
    format_eval_summary,
    load_ground_truth,
    GROUND_TRUTH_COMPANIES,
)


class TestGroundTruth:
    def test_load_apple(self):
        gt = load_ground_truth("Apple")
        assert gt is not None
        assert "tier_1_suppliers" in gt
        assert len(gt["tier_1_suppliers"]) > 5

    def test_load_toyota(self):
        gt = load_ground_truth("Toyota")
        assert gt is not None
        assert len(gt["tier_1_suppliers"]) > 5

    def test_load_nonexistent(self):
        gt = load_ground_truth("NonexistentCorp")
        assert gt is None

    def test_ground_truth_companies_have_data(self):
        for company in GROUND_TRUTH_COMPANIES:
            gt = load_ground_truth(company)
            assert gt is not None, f"Missing ground truth for {company}"


class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("TSMC") == "tsmc"

    def test_strip_punctuation(self):
        assert _normalize_name("S.K. Hynix, Inc.") == "sk hynix inc"

    def test_strip_whitespace(self):
        assert _normalize_name("  Foxconn  ") == "foxconn"


class TestGroundTruthMetrics:
    def test_perfect_recall(self):
        """When all discovered suppliers are in ground truth."""
        state = {
            "suppliers": [
                {"name": "TSMC", "confidence": 0.9, "evidence_source": "llm_only"},
                {"name": "Foxconn", "confidence": 0.85, "evidence_source": "web_verified"},
            ]
        }
        metrics = compute_ground_truth_metrics("Apple", state)
        assert metrics["has_ground_truth"] is True
        assert metrics["true_positives"] == 2
        assert metrics["precision"] == 1.0  # both are real suppliers

    def test_no_ground_truth(self):
        state = {"suppliers": []}
        metrics = compute_ground_truth_metrics("NoCompany", state)
        assert metrics == {}

    def test_precision_and_recall(self):
        """Mixed results: some correct, some incorrect."""
        state = {
            "suppliers": [
                {"name": "TSMC", "confidence": 0.9, "evidence_source": "llm_only"},
                {"name": "FakeSupplier", "confidence": 0.6, "evidence_source": "llm_only"},
            ]
        }
        metrics = compute_ground_truth_metrics("Apple", state)
        assert metrics["true_positives"] == 1
        assert metrics["precision"] == 0.5
        assert metrics["recall"] > 0  # found at least 1 of many

    def test_calibration_gap(self):
        """Correct suppliers should have higher confidence than incorrect."""
        state = {
            "suppliers": [
                {"name": "TSMC", "confidence": 0.95, "evidence_source": "llm_only"},
                {"name": "Foxconn", "confidence": 0.9, "evidence_source": "llm_only"},
                {"name": "FakeCo", "confidence": 0.3, "evidence_source": "llm_only"},
            ]
        }
        metrics = compute_ground_truth_metrics("Apple", state)
        assert metrics["calibration_gap"] > 0  # correct > incorrect


class TestHardMetrics:
    def test_basic_metrics(self):
        state = {
            "suppliers": [
                {"name": "A", "evidence_source": "web_verified"},
                {"name": "B", "evidence_source": "llm_only"},
                {"name": "C", "evidence_source": "fixture"},
            ],
            "edges": [{"parent": "X", "child": "A"}],
            "hazard_scores": [
                {"supplier_name": "A", "hazard_type": "earthquake"},
                {"supplier_name": "B", "hazard_type": "flood"},
            ],
            "evidence": [
                {"source": "web:https://example.com"},
                {"source": "LLM:gpt-4o"},
            ],
            "budget_summary": {
                "llm_calls": 3,
                "web_queries": 5,
                "hazard_scores": 12,
                "wall_clock_seconds": 45.0,
            },
        }
        metrics = compute_hard_metrics(state)
        assert metrics["supplier_count"] == 3
        assert metrics["edge_count"] == 1
        assert metrics["edge_evidence_rate"] > 0  # 1/3 web_verified
        assert metrics["hazard_scores_count"] == 2
        assert metrics["llm_calls"] == 3
        assert metrics["web_queries"] == 5
        assert metrics["unique_web_sources"] == 1
        assert metrics["unverified_fraction"] > 0  # 1/3 llm_only

    def test_empty_state(self):
        state = {
            "suppliers": [],
            "edges": [],
            "hazard_scores": [],
            "evidence": [],
            "budget_summary": {},
        }
        metrics = compute_hard_metrics(state)
        assert metrics["supplier_count"] == 0
        assert metrics["hazard_coverage"] == 0.0


class TestFormatEvalSummary:
    def test_produces_text(self):
        results = [
            {
                "company": "TestCo",
                "conditions": {
                    "pipeline": {
                        "supplier_count": 5,
                        "edge_evidence_rate": 0.0,
                        "hazard_coverage": 1.0,
                        "llm_calls": 3,
                        "web_queries": 0,
                    },
                },
            }
        ]
        text = format_eval_summary(results)
        assert "EVALUATION SUMMARY" in text
        assert "pipeline" in text
        assert "Companies evaluated: 1" in text
