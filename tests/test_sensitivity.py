"""Tests for sensitivity analysis (new SMHEI API)."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from bor_risk.sensitivity import (
    build_default_scenarios,
    format_sensitivity_report,
    run_sensitivity,
)
from bor_risk.tools import N_HAZARDS, compute_risk_summary

_HAZARD_NAMES = ("earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought")


def _make_hazard_scores(supplier_name: str, score: float = 0.5) -> list[dict]:
    return [
        {
            "supplier_name": supplier_name,
            "hazard_type": h,
            "score": score,
            "score_100": round(score * 100),
            "level": "Medium",
            "dataset_metadata": {},
        }
        for h in _HAZARD_NAMES
    ]


def _make_state(n_suppliers: int = 2) -> dict:
    suppliers = [
        {"name": f"Supplier{i}", "tier": 1, "confidence": 0.8, "evidence_source": "fixture"}
        for i in range(n_suppliers)
    ]
    hazard_scores = []
    for i, s in enumerate(suppliers):
        hazard_scores.extend(_make_hazard_scores(s["name"], score=round(0.3 + i * 0.1, 2)))
    return {"hazard_scores": hazard_scores, "suppliers": suppliers}


# ---------------------------------------------------------------------------
# compute_risk_summary
# ---------------------------------------------------------------------------


class TestComputeRiskSummary:
    def test_equal_weights_arithmetic(self):
        state = _make_state(3)
        summary = compute_risk_summary(state["hazard_scores"], state["suppliers"])
        assert "company_score" in summary
        assert "supplier_risks" in summary
        for sr in summary["supplier_risks"]:
            assert "exposure_index" in sr
            assert "exposure_band" in sr
            assert "dominant_hazard" in sr

    def test_matches_manual_arithmetic(self):
        state = _make_state(1)
        summary = compute_risk_summary(state["hazard_scores"], state["suppliers"])
        # All 6 hazards have score=0.3 → E_s = 0.3
        assert summary["supplier_risks"][0]["exposure_index"] == pytest.approx(0.3, abs=1e-6)

    def test_missing_hazard_raises(self):
        supplier = {"name": "S1", "tier": 1, "confidence": 0.8, "evidence_source": "fixture"}
        # Only 5 hazards — should raise with weights=None (fixed 6)
        scores = _make_hazard_scores("S1")[:5]
        with pytest.raises(ValueError, match="Expected 6 hazard scores"):
            compute_risk_summary(scores, [supplier])

    def test_geometric_mean_with_all_nonzero(self):
        state = _make_state(1)
        summary = compute_risk_summary(
            state["hazard_scores"], state["suppliers"], aggregation="geometric"
        )
        # All scores = 0.3; geometric mean = 0.3
        assert summary["supplier_risks"][0]["exposure_index"] == pytest.approx(0.3, abs=1e-4)

    def test_geometric_mean_with_one_zero(self):
        supplier = {"name": "S1", "tier": 1, "confidence": 0.8, "evidence_source": "fixture"}
        scores = _make_hazard_scores("S1", score=0.5)
        scores[0]["score"] = 0.0  # one hazard is zero
        summary = compute_risk_summary(scores, [supplier], aggregation="geometric")
        assert summary["supplier_risks"][0]["exposure_index"] == pytest.approx(0.0, abs=1e-10)

    def test_negative_weight_raises(self):
        state = _make_state(1)
        bad_w = {h: 1.0 / 6 for h in _HAZARD_NAMES}
        bad_w["earthquake"] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            compute_risk_summary(state["hazard_scores"], state["suppliers"], weights=bad_w)

    def test_zero_sum_weights_raises(self):
        state = _make_state(1)
        zero_w = {h: 0.0 for h in _HAZARD_NAMES}
        with pytest.raises(ValueError, match="sum to zero"):
            compute_risk_summary(state["hazard_scores"], state["suppliers"], weights=zero_w)

    def test_unknown_weight_key_raises(self):
        state = _make_state(1)
        bad_w = {h: 1.0 / 6 for h in _HAZARD_NAMES}
        bad_w["nonexistent_hazard"] = 0.5
        with pytest.raises(ValueError, match="unknown hazard names"):
            compute_risk_summary(state["hazard_scores"], state["suppliers"], weights=bad_w)

    def test_drop_one_explicit_weights_valid(self):
        """Drop-one with explicit weights= does not raise (weights provided bypass N_HAZARDS check)."""
        state = _make_state(1)
        drop_w = {h: (0.0 if h == "earthquake" else 1.0 / 5) for h in _HAZARD_NAMES}
        # Should not raise — weights= provided, no N_HAZARDS check
        summary = compute_risk_summary(state["hazard_scores"], state["suppliers"], weights=drop_w)
        assert "company_score" in summary

    def test_non_informative_hazards_are_excluded_from_primary_smhei(self):
        supplier = {"name": "S1", "tier": 1, "confidence": 0.8, "evidence_source": "fixture"}
        scores = [
            {"supplier_name": "S1", "hazard_type": "earthquake", "score": 0.5, "score_100": 50, "level": "Medium", "dataset_metadata": {}},
            {"supplier_name": "S1", "hazard_type": "flood", "score": 0.5, "score_100": 50, "level": "Medium", "dataset_metadata": {}},
            {"supplier_name": "S1", "hazard_type": "cyclone", "score": 0.5, "score_100": 50, "level": "Medium", "dataset_metadata": {}},
            {"supplier_name": "S1", "hazard_type": "wildfire", "score": 0.0, "score_100": 0, "level": "Low", "dataset_metadata": {}},
            {"supplier_name": "S1", "hazard_type": "heat_stress", "score": 0.0, "score_100": 0, "level": "Low", "dataset_metadata": {}},
            {"supplier_name": "S1", "hazard_type": "drought", "score": 0.0, "score_100": 0, "level": "Low", "dataset_metadata": {}},
        ]
        reference_set = {
            "metadata": {
                "non_informative_hazards": ["earthquake", "flood", "cyclone"],
            },
            "supplier_exposure_thresholds": {"medium": 0.33, "high": 0.67},
            "company_exposure_thresholds": {"medium": 0.35, "high": 0.65},
        }
        with patch("bor_risk.tools.load_reference_set", return_value=reference_set):
            summary = compute_risk_summary(scores, [supplier])
        assert summary["supplier_risks"][0]["exposure_index"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_default_scenarios_count(self):
        scenarios = build_default_scenarios()
        # 1 baseline + 1 geometric + 1 double-eq + 6 drop-one = 9
        assert len(scenarios) == 9

    def test_all_scenarios_have_required_keys(self):
        for s in build_default_scenarios():
            assert "label" in s
            assert "weights" in s
            assert "aggregation" in s
            assert "sensitivity_scenario" in s

    def test_baseline_is_primary(self):
        scenarios = build_default_scenarios()
        baseline = scenarios[0]
        assert baseline["label"] == "Baseline (primary SMHEI)"
        assert baseline["sensitivity_scenario"] is False
        assert baseline["weights"] is None
        assert baseline["aggregation"] == "arithmetic"

    def test_geometric_is_sensitivity(self):
        scenarios = build_default_scenarios()
        geometric = next(s for s in scenarios if "Geometric" in s["label"])
        assert geometric["sensitivity_scenario"] is True
        assert geometric["aggregation"] == "geometric"

    def test_drop_one_scenarios_count(self):
        scenarios = build_default_scenarios()
        drops = [s for s in scenarios if s["label"].startswith("Drop ")]
        assert len(drops) == 6

    def test_drop_one_zeroes_correct_hazard(self):
        scenarios = build_default_scenarios()
        for s in scenarios:
            if not s["label"].startswith("Drop "):
                continue
            dropped = s["label"].replace("Drop ", "")
            assert s["weights"][dropped] == 0.0
            others = [w for h, w in s["weights"].items() if h != dropped]
            assert all(abs(w - others[0]) < 1e-9 for w in others)

    def test_double_earthquake_weight(self):
        scenarios = build_default_scenarios()
        dbl = next(s for s in scenarios if "Double" in s["label"])
        assert dbl["weights"]["earthquake"] == pytest.approx(2.0 / 7.0, abs=1e-9)
        assert dbl["weights"]["flood"] == pytest.approx(1.0 / 7.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Sensitivity results
# ---------------------------------------------------------------------------


class TestSensitivityResults:
    def test_returns_at_least_9_results(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        # 9 scenarios + Kendall tau = 10
        assert len(results) >= 9

    def test_baseline_is_first_and_primary(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        baseline = results[0]
        assert "Baseline" in baseline["label"]
        assert baseline.get("sensitivity_scenario") is False

    def test_kendall_tau_entry_present(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        kendall = next((r for r in results if "kendall" in r["label"].lower()), None)
        assert kendall is not None
        assert "kendall_tau" in kendall or "error" in kendall

    def test_all_results_have_label(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        for r in results:
            assert "label" in r

    def test_non_baseline_results_have_company_score(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        for r in results:
            if "kendall_tau" in r or "error" in r:
                continue
            assert "company_score" in r
            assert "company_band" in r


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestSensitivityReport:
    def test_report_contains_header(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        assert "Sensitivity Analysis" in report

    def test_report_contains_all_scenario_labels(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        for r in results:
            assert r["label"] in report

    def test_report_contains_key_findings(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        assert "Key findings:" in report

    def test_report_labels_primary_smhei(self):
        state = _make_state(2)
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        assert "primary SMHEI" in report
