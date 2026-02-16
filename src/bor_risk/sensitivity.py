"""Sensitivity analysis for hazard weight and threshold configurations.

Runs the same set of hazard scores through ``compute_risk_summary`` with
multiple weight/threshold configurations to show how sensitive the risk
outputs are to parameter choices.  Useful for thesis methodology validation.
"""

from __future__ import annotations

from bor_risk.tools import compute_risk_summary
from bor_risk.utils import load_hazards


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def _baseline_config() -> tuple[dict[str, float], dict[str, float]]:
    """Return the baseline weights and thresholds from hazards.yaml."""
    hazard_defs = load_hazards()
    weights = {h["name"]: float(h.get("weight", 1.0)) for h in hazard_defs}
    thresholds = {h["name"]: float(h.get("threshold", 1.0)) for h in hazard_defs}
    return weights, thresholds


def build_default_scenarios() -> list[dict]:
    """Build the default set of 11 sensitivity scenarios.

    Returns a list of dicts, each with:
      ``label``, ``weights``, ``thresholds``
    """
    base_weights, base_thresholds = _baseline_config()
    hazard_names = list(base_weights.keys())
    n = len(hazard_names)

    scenarios: list[dict] = []

    # 1. Baseline
    scenarios.append({
        "label": "Baseline",
        "weights": dict(base_weights),
        "thresholds": dict(base_thresholds),
    })

    # 2. Equal weights
    equal_w = 1.0 / n
    scenarios.append({
        "label": "Equal weights",
        "weights": {h: round(equal_w, 4) for h in hazard_names},
        "thresholds": dict(base_thresholds),
    })

    # 3-8. Ablation: drop each hazard (weight=0), redistribute proportionally
    for drop_name in hazard_names:
        remaining_total = sum(
            w for h, w in base_weights.items() if h != drop_name
        )
        ablation_weights = {}
        for h, w in base_weights.items():
            if h == drop_name:
                ablation_weights[h] = 0.0
            elif remaining_total > 0:
                ablation_weights[h] = round(w / remaining_total, 4)
            else:
                ablation_weights[h] = w
        scenarios.append({
            "label": f"Drop {drop_name}",
            "weights": ablation_weights,
            "thresholds": dict(base_thresholds),
        })

    # 9. Strict thresholds (lower by 0.1)
    scenarios.append({
        "label": "Strict thresholds",
        "weights": dict(base_weights),
        "thresholds": {
            h: round(max(0.1, t - 0.1), 2) for h, t in base_thresholds.items()
        },
    })

    # 10. Relaxed thresholds (raise by 0.1)
    scenarios.append({
        "label": "Relaxed thresholds",
        "weights": dict(base_weights),
        "thresholds": {
            h: round(min(1.0, t + 0.1), 2) for h, t in base_thresholds.items()
        },
    })

    # 11. Double earthquake weight, scale others down
    eq_weight = base_weights.get("earthquake", 0.25)
    doubled = eq_weight * 2.0
    others_total = sum(w for h, w in base_weights.items() if h != "earthquake")
    scale = (1.0 - doubled) / others_total if others_total > 0 else 1.0
    double_eq_weights = {}
    for h, w in base_weights.items():
        if h == "earthquake":
            double_eq_weights[h] = round(doubled, 4)
        else:
            double_eq_weights[h] = round(w * scale, 4)
    scenarios.append({
        "label": "Double earthquake",
        "weights": double_eq_weights,
        "thresholds": dict(base_thresholds),
    })

    return scenarios


# ---------------------------------------------------------------------------
# Run sensitivity
# ---------------------------------------------------------------------------


def run_sensitivity(
    state: dict,
    scenarios: list[dict] | None = None,
) -> list[dict]:
    """Re-aggregate hazard scores under multiple weight/threshold configurations.

    Parameters
    ----------
    state : dict
        Completed graph state containing ``hazard_scores`` and ``suppliers``.
    scenarios : list[dict] | None
        Custom scenario list.  Each entry must have ``label``, ``weights``,
        and ``thresholds``.  If *None*, :func:`build_default_scenarios` is used.

    Returns
    -------
    list[dict]
        One result dict per scenario with keys: ``label``, ``company_score``,
        ``risk_band``, ``critical_alert_count``, ``alert_count``,
        ``supplier_ranking``, ``top_risk_score``, ``rank_changes``.
    """
    if scenarios is None:
        scenarios = build_default_scenarios()

    hazard_scores = state.get("hazard_scores", [])
    suppliers = state.get("suppliers", [])

    results: list[dict] = []
    baseline_ranking: list[str] | None = None

    for scenario in scenarios:
        summary = compute_risk_summary(
            hazard_scores,
            suppliers,
            scenario["weights"],
            scenario["thresholds"],
        )

        ranking = [s["supplier_name"] for s in summary["supplier_risks"]]
        top_score = summary["supplier_risks"][0]["risk_score"] if summary["supplier_risks"] else 0.0

        # Count rank changes vs. baseline
        if baseline_ranking is None:
            baseline_ranking = ranking
            rank_changes = 0
        else:
            rank_changes = sum(
                1 for i, name in enumerate(ranking)
                if i < len(baseline_ranking) and baseline_ranking[i] != name
            )

        results.append({
            "label": scenario["label"],
            "company_score": summary["company_score"],
            "risk_band": summary["risk_band"],
            "critical_alert_count": summary["critical_alert_count"],
            "alert_count": len(summary["hazard_alerts"]),
            "supplier_ranking": ranking,
            "top_risk_score": top_score,
            "rank_changes": rank_changes,
        })

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_sensitivity_report(results: list[dict]) -> str:
    """Format sensitivity analysis results as a plain-text comparison table."""
    lines: list[str] = []
    lines.append("--- Sensitivity Analysis ---\n")

    # Table header
    lines.append(
        f"  {'Scenario':<22}| {'Score':>6} | {'Band':<6} "
        f"| {'Alerts':>6} | {'Rank Chg':>8} | Top Supplier"
    )
    lines.append(
        f"  {'-' * 22}|{'-' * 8}|{'-' * 8}"
        f"|{'-' * 8}|{'-' * 10}|{'-' * 20}"
    )

    baseline = results[0] if results else None

    for r in results:
        top = r["supplier_ranking"][0] if r["supplier_ranking"] else "N/A"
        top_str = f"{top} ({r['top_risk_score']:.2f})"
        rank_str = "-" if r is baseline else str(r["rank_changes"])

        lines.append(
            f"  {r['label']:<22}| {r['company_score']:>6.4f} | {r['risk_band']:<6} "
            f"| {r['alert_count']:>6} | {rank_str:>8} | {top_str}"
        )

    # Key findings
    if baseline and len(results) > 1:
        lines.append("")
        lines.append("  Key findings:")

        # Check ranking stability
        any_rank_change = any(r["rank_changes"] > 0 for r in results[1:])
        if not any_rank_change:
            lines.append(
                "    - Supplier ranking is stable across all weight variations"
            )
        else:
            changed = [r["label"] for r in results[1:] if r["rank_changes"] > 0]
            lines.append(
                f"    - Supplier ranking changes under: {', '.join(changed)}"
            )

        # Check band changes
        band_changes = [
            r for r in results[1:]
            if r["risk_band"] != baseline["risk_band"]
        ]
        if band_changes:
            for r in band_changes:
                lines.append(
                    f"    - '{r['label']}' shifts risk band from "
                    f"{baseline['risk_band']} to {r['risk_band']}"
                )
        else:
            lines.append(
                f"    - Risk band ({baseline['risk_band']}) is consistent "
                f"across all scenarios"
            )

        # Threshold sensitivity
        strict = next((r for r in results if r["label"] == "Strict thresholds"), None)
        relaxed = next((r for r in results if r["label"] == "Relaxed thresholds"), None)
        if strict and relaxed:
            delta_strict = strict["alert_count"] - baseline["alert_count"]
            delta_relaxed = relaxed["alert_count"] - baseline["alert_count"]
            lines.append(
                f"    - Threshold sensitivity: strict {delta_strict:+d} alerts, "
                f"relaxed {delta_relaxed:+d} alerts vs. baseline ({baseline['alert_count']})"
            )

    lines.append("")
    return "\n".join(lines)
