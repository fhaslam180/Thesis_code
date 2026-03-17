"""Sensitivity analysis for the Supplier Multi-Hazard Exposure Index (SMHEI).

Runs the same set of hazard scores through ``compute_risk_summary`` with
multiple aggregation configurations to assess robustness of the primary
equal-weight arithmetic mean.

IMPORTANT: All non-baseline scenarios are SENSITIVITY-ONLY and must NOT
be presented in the primary SMHEI results table. They belong in a separate
"Sensitivity Analysis" section of the report and thesis.

Scenarios
---------
1. Baseline          — equal weights, arithmetic mean (primary SMHEI)
2. Geometric mean    — equal weights, geometric mean (sensitivity only)
3. Double earthquake — earthquake weight 2/7, others 1/7, arithmetic
4–9. Drop-one × 6   — each hazard dropped (weight=0), others 1/5, arithmetic
10. Ranking comparison — Kendall's τ between scenario 1 and scenario 2 rankings
"""

from __future__ import annotations

import math

from bor_risk.tools import N_HAZARDS, compute_risk_summary

_ALL_HAZARDS = ("earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought")


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def build_default_scenarios() -> list[dict]:
    """Build the default set of sensitivity scenarios.

    Returns a list of dicts, each with:
      ``label``, ``weights`` (or None), ``aggregation``, ``sensitivity_scenario``
    """
    scenarios: list[dict] = []

    # 1. Baseline (primary SMHEI — equal-weight arithmetic mean)
    scenarios.append({
        "label": "Baseline (primary SMHEI)",
        "weights": None,
        "aggregation": "arithmetic",
        "sensitivity_scenario": False,
    })

    # 2. Geometric mean (sensitivity only — zeros penalised harshly)
    scenarios.append({
        "label": "Geometric mean",
        "weights": None,
        "aggregation": "geometric",
        "sensitivity_scenario": True,
    })

    # 3. Double earthquake (arithmetic; seismic-stress test)
    double_eq_w = {
        "earthquake": 2.0 / 7.0,
        "flood": 1.0 / 7.0,
        "wildfire": 1.0 / 7.0,
        "cyclone": 1.0 / 7.0,
        "heat_stress": 1.0 / 7.0,
        "drought": 1.0 / 7.0,
    }
    scenarios.append({
        "label": "Double earthquake",
        "weights": double_eq_w,
        "aggregation": "arithmetic",
        "sensitivity_scenario": True,
    })

    # 4–9. Drop-one × 6 (redistribute equally among remaining 5)
    for drop in _ALL_HAZARDS:
        w = {h: (0.0 if h == drop else 1.0 / (N_HAZARDS - 1)) for h in _ALL_HAZARDS}
        scenarios.append({
            "label": f"Drop {drop}",
            "weights": w,
            "aggregation": "arithmetic",
            "sensitivity_scenario": True,
        })

    return scenarios


# ---------------------------------------------------------------------------
# Run sensitivity
# ---------------------------------------------------------------------------


def run_sensitivity(
    state: dict,
    scenarios: list[dict] | None = None,
) -> list[dict]:
    """Re-aggregate hazard scores under multiple weight/aggregation configurations.

    Parameters
    ----------
    state : dict
        Completed graph state with ``hazard_scores`` and ``suppliers``.
    scenarios : list[dict] | None
        Custom scenario list.  If None, :func:`build_default_scenarios` is used.

    Returns
    -------
    list[dict]
        One result dict per scenario, including:
        ``label``, ``sensitivity_scenario``, ``company_score``,
        ``company_band``, ``supplier_ranking``, ``top_exposure_index``,
        ``rank_changes`` (vs. baseline).

    Also appends scenario 10: Kendall's τ between baseline and geometric mean rankings.
    """
    if scenarios is None:
        scenarios = build_default_scenarios()

    hazard_scores = state.get("hazard_scores", [])
    suppliers = state.get("suppliers", [])

    results: list[dict] = []
    baseline_ranking: list[str] | None = None

    for scenario in scenarios:
        try:
            summary = compute_risk_summary(
                hazard_scores,
                suppliers,
                weights=scenario.get("weights"),
                aggregation=scenario.get("aggregation", "arithmetic"),
            )
        except ValueError as e:
            results.append({
                "label": scenario["label"],
                "sensitivity_scenario": scenario.get("sensitivity_scenario", True),
                "error": str(e),
            })
            continue

        ranking = [s["supplier_name"] for s in summary["supplier_risks"]]
        top_score = (
            summary["supplier_risks"][0]["exposure_index"]
            if summary["supplier_risks"] else 0.0
        )

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
            "sensitivity_scenario": scenario.get("sensitivity_scenario", True),
            "company_score": summary["company_score"],
            "company_band": summary["company_band"],
            "supplier_ranking": ranking,
            "top_exposure_index": top_score,
            "rank_changes": rank_changes,
        })

    # Scenario 10: Kendall's τ between baseline and geometric-mean rankings
    kendall_result = _compute_kendall_tau(results)
    if kendall_result is not None:
        results.append(kendall_result)

    return results


def _compute_kendall_tau(results: list[dict]) -> dict | None:
    """Compute Kendall's τ between baseline and geometric-mean supplier rankings."""
    try:
        from scipy.stats import kendalltau  # type: ignore
    except ImportError:
        return {
            "label": "Kendall tau (baseline vs geometric)",
            "sensitivity_scenario": True,
            "error": "scipy not installed",
        }

    baseline = next(
        (r for r in results if r.get("label") == "Baseline (primary SMHEI)"
         and "supplier_ranking" in r),
        None,
    )
    geometric = next(
        (r for r in results if r.get("label") == "Geometric mean"
         and "supplier_ranking" in r),
        None,
    )
    if baseline is None or geometric is None:
        return None

    base_names = baseline["supplier_ranking"]
    geom_names = geometric["supplier_ranking"]
    all_names = list(dict.fromkeys(base_names + geom_names))  # preserve order, deduplicate

    rank_base = [base_names.index(n) if n in base_names else len(base_names) for n in all_names]
    rank_geom = [geom_names.index(n) if n in geom_names else len(geom_names) for n in all_names]

    try:
        tau, p_value = kendalltau(rank_base, rank_geom)
    except Exception as e:
        return {
            "label": "Kendall tau (baseline vs geometric)",
            "sensitivity_scenario": True,
            "error": str(e),
        }

    return {
        "label": "Kendall tau (baseline vs geometric)",
        "sensitivity_scenario": True,
        "kendall_tau": round(float(tau), 4),
        "p_value": round(float(p_value), 4),
        "interpretation": (
            "tau=1.0: identical rankings; tau=0.0: no correlation; "
            "tau=-1.0: reversed. p_value tests H0: rankings are independent."
        ),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_sensitivity_report(results: list[dict]) -> str:
    """Format sensitivity analysis results as plain text.

    Baseline (sensitivity_scenario=False) is shown first and labelled as
    the primary SMHEI. Non-baseline scenarios are shown in a separate
    'Sensitivity Analysis' section.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("--- Sensitivity Analysis ---")
    lines.append("")
    lines.append(
        "  NOTE: The baseline (equal-weight arithmetic mean) is the primary SMHEI."
    )
    lines.append(
        "  All other scenarios are sensitivity checks and are NOT the primary index."
    )
    lines.append("")

    # Table header
    lines.append(
        f"  {'Scenario':<34}| {'E_s':>6} | {'Band':<6} | {'Rank Chg':>8} | Top Supplier"
    )
    lines.append(
        f"  {'-' * 34}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 20}"
    )

    baseline = next((r for r in results if not r.get("sensitivity_scenario", True)), None)

    for r in results:
        if "error" in r:
            lines.append(f"  {r['label']:<34}  ERROR: {r['error']}")
            continue
        if "kendall_tau" in r:
            lines.append(
                f"\n  {r['label']}: tau={r['kendall_tau']:.4f}, p={r['p_value']:.4f}"
            )
            continue

        top = r.get("supplier_ranking", ["N/A"])[0] if r.get("supplier_ranking") else "N/A"
        top_str = f"{top} ({r.get('top_exposure_index', 0.0):.3f})"
        rank_str = "-" if not r.get("sensitivity_scenario", True) else str(r.get("rank_changes", "?"))
        tag = "" if not r.get("sensitivity_scenario", True) else " [sensitivity]"
        label = r["label"] + tag

        lines.append(
            f"  {label:<34}| {r.get('company_score', 0):>6.4f} | {r.get('company_band', '?'):<6} "
            f"| {rank_str:>8} | {top_str}"
        )

    # Key findings
    if baseline and len(results) > 1:
        lines.append("")
        lines.append("  Key findings:")

        any_rank_change = any(
            r.get("rank_changes", 0) > 0
            for r in results
            if r.get("sensitivity_scenario", True) and "rank_changes" in r
        )
        if not any_rank_change:
            lines.append("    - Supplier ranking is stable across all weight variations")
        else:
            changed = [
                r["label"] for r in results
                if r.get("sensitivity_scenario", True) and r.get("rank_changes", 0) > 0
            ]
            lines.append(f"    - Supplier ranking changes under: {', '.join(changed)}")

        band_changes = [
            r for r in results
            if r.get("sensitivity_scenario", True)
            and r.get("company_band") and r["company_band"] != baseline.get("company_band")
        ]
        if band_changes:
            for r in band_changes:
                lines.append(
                    f"    - '{r['label']}' shifts company band from "
                    f"{baseline.get('company_band')} to {r['company_band']}"
                )
        else:
            lines.append(
                f"    - Company band ({baseline.get('company_band')}) is consistent "
                f"across all scenarios"
            )

    lines.append("")
    return "\n".join(lines)
