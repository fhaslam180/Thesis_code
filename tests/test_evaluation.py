import pytest

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

    def test_supplier_aliases_count_as_true_positives(self):
        """Legal suffixes, descriptors, and parentheticals should not hide matches."""
        state = {
            "suppliers": [
                {"name": "Samsung Electronics", "confidence": 0.9, "evidence_source": "web_verified"},
                {"name": "Broadcom Inc.", "confidence": 0.9, "evidence_source": "web_verified"},
                {"name": "TSMC (Taiwan Semiconductor Manufacturing Company)", "confidence": 0.9, "evidence_source": "web_verified"},
                {"name": "FakeSupplier", "confidence": 0.5, "evidence_source": "llm_only"},
            ]
        }
        metrics = compute_ground_truth_metrics("Apple", state)
        assert metrics["true_positives"] == 3
        assert metrics["precision"] == 0.75


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
                {"source": "LLM:gpt-4.1-mini"},
            ],
            "claims": [],
            "evidence_packets": [],
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
        assert metrics["unique_web_domains"] == 1
        assert metrics["unverified_fraction"] > 0  # 1/3 llm_only

    def test_claim_metrics(self):
        """Claim-level metrics computed from claims list."""
        state = {
            "suppliers": [],
            "edges": [],
            "hazard_scores": [],
            "evidence": [],
            "evidence_packets": [],
            "claims": [
                {"verdict": "SUPPORTED", "status": "VERIFIED",
                 "evidence_refs": ["e1"], "supporting_spans": [{"quote": "hello"}]},
                {"verdict": "WEAK", "status": "VERIFIED",
                 "evidence_refs": ["e2"], "supporting_spans": []},
                {"verdict": "DISPUTED", "status": "VERIFIED",
                 "evidence_refs": [], "supporting_spans": []},
                {"verdict": "UNKNOWN", "status": "PROPOSED",
                 "evidence_refs": [], "supporting_spans": []},
            ],
            "budget_summary": {},
        }
        metrics = compute_hard_metrics(state)
        assert metrics["total_claims"] == 4
        assert metrics["claim_support_rate"] == 0.5   # (1+1)/4
        assert metrics["dispute_rate"] == 0.25
        assert metrics["unknown_rate"] == 0.25
        assert metrics["verified_claim_count"] == 3
        # quote_valid_rate: 1 supported claim with non-empty quote / 1 total supported = 1.0
        assert metrics["quote_valid_rate"] == 1.0

    def test_empty_state(self):
        state = {
            "suppliers": [],
            "edges": [],
            "hazard_scores": [],
            "evidence": [],
            "claims": [],
            "evidence_packets": [],
            "budget_summary": {},
        }
        metrics = compute_hard_metrics(state)
        assert metrics["supplier_count"] == 0
        assert metrics["hazard_coverage"] == 0.0
        assert metrics["total_claims"] == 0


class TestFormatEvalSummary:
    def test_produces_text(self):
        results = [
            {
                "company": "TestCo",
                "conditions": {
                    "llm_only": {
                        "supplier_count": 5,
                        "edge_evidence_rate": 0.0,
                        "hazard_coverage": 1.0,
                        "llm_calls": 3,
                        "web_queries": 0,
                        "total_claims": 5,
                        "claim_support_rate": 0.0,
                    },
                },
            }
        ]
        text = format_eval_summary(results)
        assert "EVALUATION SUMMARY" in text
        assert "llm_only" in text
        assert "Companies evaluated: 1" in text

    def test_shows_all_four_conditions_in_header(self):
        results = []
        text = format_eval_summary(results)
        for cond in ["llm_only", "web_retrieve", "web_verify", "strict"]:
            assert cond in text


class TestPrintQualityReport:
    """Tests for print_quality_report()."""

    from unittest.mock import patch

    def _minimal_state(self, n_claims: int = 2, verified: int = 1) -> dict:
        claims = []
        for i in range(n_claims):
            claims.append({
                "claim_id": f"c{i}",
                "object_entity_id": f"Supplier{i}",
                "status": "VERIFIED" if i < verified else "PROPOSED",
                "verdict": "SUPPORTED" if i < verified else "UNKNOWN",
                "evidence_refs": [],
                "location_candidates": [
                    {"lat": 37.3, "lon": -121.9, "confidence": 0.5, "source": "llm", "place_name": ""}
                ],
                "supporting_spans": [],
            })
        suppliers = [{"name": f"Supplier{i}", "lat": 0.0, "lon": 0.0, "tier": 1,
                      "confidence": 0.8, "evidence_source": "llm_only"} for i in range(n_claims)]
        return {
            "company": "TestCo",
            "suppliers": suppliers,
            "claims": claims,
            "evidence_packets": [],
            "hazard_scores": [],
            "company_risk_summary": {},
            "workflow_trace": [],
        }

    def _mock_rs(self, informative: list[str] | None = None):
        return {
            "metadata": {
                "built_at": "2026-01-01T00:00:00+00:00",
                "n_suppliers": 20,
                "source": "build_reference_set.py",
                "corpus": "test",
                "informative_hazards": informative or ["wildfire", "heat_stress", "drought"],
                "non_informative_hazards": ["earthquake", "flood", "cyclone"],
            },
            "hazards": {h: [0.0] * 11 for h in
                        ["earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought"]},
            "supplier_exposure_thresholds": {"medium": 0.2, "high": 0.6},
            "company_exposure_thresholds": {"medium": 0.2, "high": 0.6},
        }

    def test_returns_all_expected_keys(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report
        import io
        state = self._minimal_state()
        with patch("bor_risk.evaluate.load_reference_set", return_value=self._mock_rs()):
            result = print_quality_report("Apple", state, file=io.StringIO())
        for key in ("verified_rate_pct", "location_source_mix", "hazard_informativeness"):
            assert key in result, f"Missing key: {key}"

    def test_duplicate_rate_math(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report
        import io
        state = self._minimal_state(n_claims=8)
        with patch("bor_risk.evaluate.load_reference_set", return_value=self._mock_rs()):
            result = print_quality_report("Apple", state, pre_dedup_supplier_count=10, file=io.StringIO())
        # (10 - 8) / 10 = 0.2
        assert result["duplicate_rate"] == pytest.approx(0.2)

    def test_location_source_mix_uses_best_candidate(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report
        import io
        state = self._minimal_state()
        # Override location_candidates with one evidence and one llm
        state["claims"][0]["location_candidates"] = [
            {"lat": 37.3, "lon": -121.9, "confidence": 0.5, "source": "llm", "place_name": ""},
            {"lat": 37.3, "lon": -121.9, "confidence": 0.85, "source": "evidence", "place_name": "Seoul"},
        ]
        with patch("bor_risk.evaluate.load_reference_set", return_value=self._mock_rs()):
            result = print_quality_report("Apple", state, file=io.StringIO())
        mix = result["location_source_mix"]
        assert mix["evidence"] >= 1

    def test_hazard_informativeness_reads_reference_set(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report
        import io
        state = self._minimal_state()
        rs = self._mock_rs(informative=["wildfire", "heat_stress", "drought"])
        with patch("bor_risk.evaluate.load_reference_set", return_value=rs):
            result = print_quality_report("Apple", state, file=io.StringIO())
        assert result["hazard_informativeness"] == pytest.approx(3 / 6)

    def test_print_output_has_pass_fail(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report
        import io
        state = self._minimal_state()
        buf = io.StringIO()
        with patch("bor_risk.evaluate.load_reference_set", return_value=self._mock_rs()):
            print_quality_report("Apple", state, file=buf)
        out = buf.getvalue()
        assert "PASS" in out or "FAIL" in out
