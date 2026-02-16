"""Tests for sensitivity analysis: weight/threshold perturbation scenarios."""

from __future__ import annotations

from unittest.mock import patch

from bor_risk.graph import run_graph
from bor_risk.sensitivity import (
    build_default_scenarios,
    format_sensitivity_report,
    run_sensitivity,
)
from bor_risk.tools import compute_risk_summary
from bor_risk.utils import load_hazards


def _baseline_state() -> dict:
    """Run the graph once and return completed state for re-aggregation."""
    return run_graph("ACME", tier_depth=2, use_llm=False)


# ---------------------------------------------------------------------------
# compute_risk_summary parity
# ---------------------------------------------------------------------------


class TestComputeRiskSummary:
    """Verify the extracted pure function matches the graph node output."""

    def test_matches_aggregate_node(self) -> None:
        state = _baseline_state()
        hazard_defs = load_hazards()
        weights = {h["name"]: float(h.get("weight", 1.0)) for h in hazard_defs}
        thresholds = {h["name"]: float(h.get("threshold", 1.0)) for h in hazard_defs}

        recomputed = compute_risk_summary(
            state["hazard_scores"],
            state["suppliers"],
            weights,
            thresholds,
        )
        original = state["company_risk_summary"]

        assert recomputed["company_score"] == original["company_score"]
        assert recomputed["risk_band"] == original["risk_band"]
        assert recomputed["critical_alert_count"] == original["critical_alert_count"]
        assert (
            [s["supplier_name"] for s in recomputed["supplier_risks"]]
            == [s["supplier_name"] for s in original["supplier_risks"]]
        )


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


class TestScenarios:
    """Verify default scenarios are well-formed."""

    def test_default_scenarios_count(self) -> None:
        scenarios = build_default_scenarios()
        assert len(scenarios) == 11

    def test_all_scenarios_have_required_keys(self) -> None:
        for s in build_default_scenarios():
            assert "label" in s
            assert "weights" in s
            assert "thresholds" in s

    def test_baseline_matches_config(self) -> None:
        scenarios = build_default_scenarios()
        baseline = scenarios[0]
        assert baseline["label"] == "Baseline"
        hazard_defs = load_hazards()
        for h in hazard_defs:
            assert baseline["weights"][h["name"]] == float(h.get("weight", 1.0))

    def test_ablation_zeroes_one_hazard(self) -> None:
        scenarios = build_default_scenarios()
        drop_scenarios = [s for s in scenarios if s["label"].startswith("Drop ")]
        assert len(drop_scenarios) == 6
        for s in drop_scenarios:
            hazard_name = s["label"].replace("Drop ", "")
            assert s["weights"][hazard_name] == 0.0


# ---------------------------------------------------------------------------
# Sensitivity results
# ---------------------------------------------------------------------------


class TestSensitivityResults:
    """Verify sensitivity analysis produces meaningful variation."""

    def test_equal_weights_changes_score(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        baseline = results[0]
        equal = next(r for r in results if r["label"] == "Equal weights")
        # Scores should differ since weights are different
        assert baseline["company_score"] != equal["company_score"]

    def test_ablation_drops_hazard_contribution(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        baseline = results[0]
        # Dropping any hazard should change the company score
        drop_results = [r for r in results if r["label"].startswith("Drop ")]
        for r in drop_results:
            assert r["company_score"] != baseline["company_score"], (
                f"{r['label']} should produce a different score"
            )

    def test_strict_thresholds_increase_alerts(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        baseline = results[0]
        strict = next(r for r in results if r["label"] == "Strict thresholds")
        assert strict["alert_count"] >= baseline["alert_count"]

    def test_relaxed_thresholds_decrease_alerts(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        baseline = results[0]
        relaxed = next(r for r in results if r["label"] == "Relaxed thresholds")
        assert relaxed["alert_count"] <= baseline["alert_count"]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestSensitivityReport:
    """Verify the sensitivity report is well-formatted."""

    def test_report_contains_header(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        assert "Sensitivity Analysis" in report

    def test_report_contains_all_scenario_labels(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        for r in results:
            assert r["label"] in report

    def test_report_contains_key_findings(self) -> None:
        state = _baseline_state()
        results = run_sensitivity(state)
        report = format_sensitivity_report(results)
        assert "Key findings:" in report


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLISensitivity:
    """Verify --sensitivity flag produces extra output."""

    def test_sensitivity_flag_writes_json(self, tmp_output_dir) -> None:
        from bor_risk.cli import main

        out_path = tmp_output_dir / "acme.txt"
        with patch("bor_risk.cli.run_graph", side_effect=run_graph):
            main([
                "--company", "ACME",
                "--tier-depth", "2",
                "--out", str(out_path),
                "--no-llm",
                "--sensitivity",
            ])

        sensitivity_file = tmp_output_dir / "acme_sensitivity.json"
        assert sensitivity_file.exists()

        import json
        data = json.loads(sensitivity_file.read_text())
        assert len(data) == 11
        assert data[0]["label"] == "Baseline"
