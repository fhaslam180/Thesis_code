"""Evaluation framework for comparing pipeline vs agent conditions.

Three-tier evaluation:
  1. Ground-truth metrics (PRIMARY): precision, recall, calibration
  2. Hard computed metrics: edge evidence rate, hazard coverage, budget usage
  3. LLM-as-judge (SUPPLEMENTARY): narrative quality only

Experimental conditions:
  A: pipeline        — fixed tool selection, no web
  B: pipeline-web    — fixed tool selection, web verification
  C: agent           — dynamic tool selection, no web
  D: agent-web       — dynamic tool selection, web verification
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from bor_risk.budget import BudgetTracker
from bor_risk.graph import run_graph
from bor_risk.report import format_report
from bor_risk.utils import load_prompts

_GROUND_TRUTH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ground_truth"

# Default evaluation companies.
EVAL_COMPANIES = [
    "Apple", "Samsung", "TSMC", "Intel", "Nvidia",
    "Toyota", "BMW", "Tesla", "Ford", "Volkswagen",
    "Nike", "Unilever", "Procter & Gamble", "Nestle", "IKEA",
    "Pfizer", "Novartis", "Johnson & Johnson",
    "Siemens", "General Electric", "Caterpillar",
    "Walmart", "Amazon", "Zara",
    "Shell", "ExxonMobil",
    "Boeing", "Airbus",
    "Coca-Cola", "McDonalds",
]

# Companies with curated ground truth files.
GROUND_TRUTH_COMPANIES = ["Apple", "Toyota", "Nike"]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(company: str) -> dict | None:
    """Load ground truth supplier list for a company, or None if unavailable."""
    name = company.lower().replace(" ", "_").replace("'", "")
    path = _GROUND_TRUTH_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_name(name: str) -> str:
    """Normalise supplier name for fuzzy matching."""
    return name.strip().lower().replace(".", "").replace(",", "")


# ---------------------------------------------------------------------------
# Tier 1: Ground-truth metrics
# ---------------------------------------------------------------------------

def compute_ground_truth_metrics(company: str, state: dict) -> dict:
    """Compute precision, recall, and calibration against ground truth.

    Returns empty dict if no ground truth is available for this company.
    """
    gt = load_ground_truth(company)
    if gt is None:
        return {}

    gt_names = set()
    for name in gt.get("tier_1_suppliers", []):
        gt_names.add(_normalize_name(name))
    for name in gt.get("tier_2_suppliers", []):
        gt_names.add(_normalize_name(name))

    discovered = state.get("suppliers", [])
    discovered_names = {_normalize_name(s["name"]) for s in discovered}

    # True positives: discovered AND in ground truth.
    tp = discovered_names & gt_names
    precision = len(tp) / len(discovered_names) if discovered_names else 0.0
    recall = len(tp) / len(gt_names) if gt_names else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Confidence calibration: are high-confidence suppliers more accurate?
    correct_confidences = []
    incorrect_confidences = []
    for s in discovered:
        norm = _normalize_name(s["name"])
        conf = float(s.get("confidence", 0.5))
        if norm in gt_names:
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)

    avg_correct_conf = (
        sum(correct_confidences) / len(correct_confidences)
        if correct_confidences
        else 0.0
    )
    avg_incorrect_conf = (
        sum(incorrect_confidences) / len(incorrect_confidences)
        if incorrect_confidences
        else 0.0
    )
    calibration_gap = avg_correct_conf - avg_incorrect_conf

    # Verification accuracy: how many web_verified suppliers are correct?
    web_verified = [
        s for s in discovered if s.get("evidence_source") == "web_verified"
    ]
    wv_correct = sum(
        1 for s in web_verified if _normalize_name(s["name"]) in gt_names
    )
    verification_accuracy = (
        wv_correct / len(web_verified) if web_verified else 0.0
    )

    return {
        "has_ground_truth": True,
        "gt_supplier_count": len(gt_names),
        "discovered_count": len(discovered_names),
        "true_positives": len(tp),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_correct_confidence": round(avg_correct_conf, 4),
        "avg_incorrect_confidence": round(avg_incorrect_conf, 4),
        "calibration_gap": round(calibration_gap, 4),
        "web_verified_count": len(web_verified),
        "verification_accuracy": round(verification_accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Tier 2: Hard computed metrics
# ---------------------------------------------------------------------------

def compute_hard_metrics(state: dict) -> dict:
    """Compute deterministic metrics from the run state."""
    suppliers = state.get("suppliers", [])
    hazard_scores = state.get("hazard_scores", [])
    evidence = state.get("evidence", [])
    budget = state.get("budget_summary", {})

    total_suppliers = len(suppliers)
    total_edges = len(state.get("edges", []))

    # Edge evidence rate: fraction of suppliers with web verification.
    web_verified = sum(
        1 for s in suppliers if s.get("evidence_source") == "web_verified"
    )
    edge_evidence_rate = (
        web_verified / total_suppliers if total_suppliers > 0 else 0.0
    )

    # Hazard coverage: scored pairs vs possible pairs (suppliers * 6).
    possible_pairs = total_suppliers * 6
    scored_pairs = len(hazard_scores)
    hazard_coverage = (
        scored_pairs / possible_pairs if possible_pairs > 0 else 0.0
    )

    # Unique web sources.
    web_urls = {
        e.get("source", "").replace("web:", "")
        for e in evidence
        if e.get("source", "").startswith("web:")
    }

    # Unverified fraction.
    llm_only = sum(
        1 for s in suppliers if s.get("evidence_source", "fixture") == "llm_only"
    )
    unverified_fraction = (
        llm_only / total_suppliers if total_suppliers > 0 else 0.0
    )

    return {
        "supplier_count": total_suppliers,
        "edge_count": total_edges,
        "edge_evidence_rate": round(edge_evidence_rate, 4),
        "hazard_scores_count": scored_pairs,
        "hazard_coverage": round(hazard_coverage, 4),
        "llm_calls": budget.get("llm_calls", 0),
        "web_queries": budget.get("web_queries", 0),
        "hazard_score_calls": budget.get("hazard_scores", 0),
        "wall_clock_seconds": budget.get("wall_clock_seconds", 0),
        "unique_web_sources": len(web_urls),
        "unverified_fraction": round(unverified_fraction, 4),
        "evidence_count": len(evidence),
    }


# ---------------------------------------------------------------------------
# Tier 3: LLM-as-judge (narrative quality only)
# ---------------------------------------------------------------------------

def judge_report_quality(company: str, state: dict) -> dict:
    """Rate report narrative quality using GPT-4o as judge.

    Returns ratings for completeness, actionability, risk_communication
    (each 1-5). Falls back to empty dict on failure.
    """
    report_text = state.get("report_text", "")
    if not report_text:
        return {"judge_scores": {}}

    try:
        prompts = load_prompts()
        template = prompts.get("judge_prompt", "")
        if not template:
            return {"judge_scores": {}}

        prompt_text = template.format(report_text=report_text[:3000])

        llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        content = response.content.strip()

        # Extract JSON from response.
        if "{" in content:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            scores = json.loads(json_str)
            return {
                "judge_scores": {
                    "completeness": int(scores.get("completeness", 0)),
                    "actionability": int(scores.get("actionability", 0)),
                    "risk_communication": int(scores.get("risk_communication", 0)),
                }
            }
    except Exception:
        pass

    return {"judge_scores": {}}


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(
    company: str,
    mode: str,
    tier_depth: int = 2,
    budget: BudgetTracker | None = None,
    snapshot: bool = True,
) -> dict:
    """Run one experimental condition and return the final state.

    Parameters
    ----------
    mode : str
        One of: "pipeline", "pipeline-web", "agent", "agent-web".
    """
    budget = budget or BudgetTracker()

    if mode in ("agent", "agent-web"):
        from bor_risk.agent import run_agent

        enable_web = mode == "agent-web"
        state = run_agent(
            company=company,
            tier_depth=tier_depth,
            enable_web=enable_web,
            budget=budget,
            snapshot_mode=snapshot,
        )
        state["report_text"] = format_report(state)
    else:
        enable_web = mode == "pipeline-web"
        state = run_graph(
            company=company,
            tier_depth=tier_depth,
            use_llm=True,
            enable_web=enable_web,
            max_web_queries=budget.max_web_queries,
            snapshot_mode=snapshot,
            budget=budget,
        )
        # Ensure budget summary is in state (format_report_node may have set it).
        if not state.get("budget_summary"):
            state["budget_summary"] = budget.summary()

    return state


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_company(
    company: str,
    tier_depth: int = 2,
    budget_llm: int = 20,
    budget_web: int = 30,
    snapshot: bool = True,
    modes: list[str] | None = None,
    skip_judge: bool = False,
) -> dict:
    """Run all conditions on one company, compute all metrics.

    Parameters
    ----------
    modes : list[str] | None
        Which conditions to evaluate. Defaults to all four.
    skip_judge : bool
        Skip LLM-as-judge scoring (saves LLM calls during development).
    """
    if modes is None:
        modes = ["pipeline", "pipeline-web", "agent", "agent-web"]

    results: dict = {}

    for mode in modes:
        budget = BudgetTracker(
            max_llm_calls=budget_llm, max_web_queries=budget_web
        )
        try:
            state = run_condition(
                company, mode, tier_depth, budget, snapshot
            )
            metrics = {
                **compute_ground_truth_metrics(company, state),
                **compute_hard_metrics(state),
                "budget": budget.summary(),
            }
            if not skip_judge:
                metrics.update(judge_report_quality(company, state))
            results[mode] = metrics
        except Exception as e:
            results[mode] = {"error": str(e)}

    return {"company": company, "conditions": results}


def evaluate_batch(
    companies: list[str],
    tier_depth: int = 2,
    budget_llm: int = 20,
    budget_web: int = 30,
    snapshot: bool = True,
    modes: list[str] | None = None,
    skip_judge: bool = False,
) -> list[dict]:
    """Evaluate multiple companies across all conditions."""
    results = []
    for company in companies:
        result = evaluate_company(
            company,
            tier_depth=tier_depth,
            budget_llm=budget_llm,
            budget_web=budget_web,
            snapshot=snapshot,
            modes=modes,
            skip_judge=skip_judge,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _safe_stdev(values: list[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for n < 2."""
    if len(values) < 2:
        return 0.0
    from statistics import stdev
    return stdev(values)


def _paired_ttest(a: list[float], b: list[float]) -> dict:
    """Compute paired t-test between two matched samples.

    Returns dict with mean_diff, t_stat, p_value, cohens_d.
    Falls back to manual computation if scipy is unavailable.
    """
    n = min(len(a), len(b))
    if n < 2:
        return {}

    diffs = [a[i] - b[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    sd_diff = _safe_stdev(diffs)

    if sd_diff == 0:
        return {"mean_diff": round(mean_diff, 4), "t_stat": float("inf"),
                "p_value": 0.0, "cohens_d": float("inf"), "n": n}

    t_stat = mean_diff / (sd_diff / (n ** 0.5))

    # Try scipy for proper p-value, fall back to approximation.
    try:
        from scipy.stats import t as t_dist
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
    except ImportError:
        # Rough two-tailed p-value approximation using normal distribution.
        import math
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    # Cohen's d: mean_diff / pooled SD of the two samples.
    sd_a = _safe_stdev(a[:n])
    sd_b = _safe_stdev(b[:n])
    pooled_sd = ((sd_a ** 2 + sd_b ** 2) / 2) ** 0.5
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "mean_diff": round(mean_diff, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "cohens_d": round(cohens_d, 4),
        "n": n,
    }


def _compute_stat_tests(
    condition_metrics: dict[str, list[dict]],
) -> list[str]:
    """Compute paired statistical tests between conditions."""
    lines = ["", "--- Statistical Tests ---"]

    comparisons = [
        ("pipeline", "agent-web", "Pipeline vs Agent+Web"),
        ("pipeline", "pipeline-web", "Pipeline vs Pipeline+Web"),
        ("agent", "agent-web", "Agent vs Agent+Web"),
    ]
    test_metrics = [
        ("precision", "Precision"),
        ("edge_evidence_rate", "Edge Evidence Rate"),
        ("f1", "F1 Score"),
        ("calibration_gap", "Calibration Gap"),
    ]

    any_tests = False
    for mode_a, mode_b, label in comparisons:
        vals_a_all = condition_metrics.get(mode_a, [])
        vals_b_all = condition_metrics.get(mode_b, [])
        if not vals_a_all or not vals_b_all:
            continue

        for metric_key, metric_label in test_metrics:
            a = [m.get(metric_key) for m in vals_a_all if metric_key in m]
            b = [m.get(metric_key) for m in vals_b_all if metric_key in m]
            a = [x for x in a if x is not None]
            b = [x for x in b if x is not None]

            if len(a) < 2 or len(b) < 2:
                continue

            result = _paired_ttest(a, b)
            if not result:
                continue

            any_tests = True
            sig = "*" if result["p_value"] < 0.05 else ""
            lines.append(
                f"  {label} ({metric_label}): "
                f"diff={result['mean_diff']:+.4f}, "
                f"t={result['t_stat']:.2f}, "
                f"p={result['p_value']:.3f}{sig}, "
                f"d={result['cohens_d']:.2f} "
                f"(n={result['n']})"
            )

    if not any_tests:
        lines.append("  Insufficient data for statistical tests (need >= 2 companies).")

    return lines


def format_eval_summary(results: list[dict]) -> str:
    """Format a comparison table from evaluation results."""
    lines = ["=" * 80, "EVALUATION SUMMARY", "=" * 80, ""]

    # Aggregate by condition.
    condition_metrics: dict[str, list[dict]] = {}
    for r in results:
        for mode, metrics in r.get("conditions", {}).items():
            if "error" not in metrics:
                condition_metrics.setdefault(mode, []).append(metrics)

    # Header.
    modes = ["pipeline", "pipeline-web", "agent", "agent-web"]
    header = f"{'Metric':<30}"
    for mode in modes:
        header += f"  {mode:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    # Metric rows with mean +/- std.
    metric_keys = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("edge_evidence_rate", "Edge Evidence Rate"),
        ("hazard_coverage", "Hazard Coverage"),
        ("unverified_fraction", "Unverified Fraction"),
        ("supplier_count", "Supplier Count"),
        ("llm_calls", "LLM Calls"),
        ("web_queries", "Web Queries"),
        ("wall_clock_seconds", "Wall Clock (s)"),
    ]

    for key, label in metric_keys:
        row = f"{label:<30}"
        for mode in modes:
            vals = condition_metrics.get(mode, [])
            nums = [m.get(key, 0) for m in vals if key in m]
            if nums:
                avg = sum(nums) / len(nums)
                sd = _safe_stdev(nums)
                if isinstance(nums[0], float):
                    if sd > 0:
                        row += f"  {avg:>7.4f}+/-{sd:.2f}"
                    else:
                        row += f"  {avg:>14.4f}"
                else:
                    if sd > 0:
                        row += f"  {avg:>7.1f}+/-{sd:.1f}"
                    else:
                        row += f"  {avg:>14.1f}"
            else:
                row += f"  {'N/A':>14}"
        lines.append(row)

    lines.append("")
    lines.append(f"Companies evaluated: {len(results)}")
    gt_count = sum(
        1 for r in results
        for m in r.get("conditions", {}).values()
        if m.get("has_ground_truth")
    )
    lines.append(f"With ground truth: {gt_count}")

    # Statistical significance tests.
    lines += _compute_stat_tests(condition_metrics)

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def eval_main(argv: list[str] | None = None) -> None:
    """CLI for running evaluation across conditions."""
    import argparse

    from dotenv import load_dotenv

    from bor_risk.utils import ensure_dir

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate pipeline vs agent across experimental conditions.",
    )
    parser.add_argument(
        "--companies",
        default=None,
        help="Comma-separated company names (default: full benchmark set).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full 30-company benchmark.",
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Only evaluate companies with ground truth data.",
    )
    parser.add_argument(
        "--out",
        default="outputs/eval",
        help="Output directory (default: outputs/eval).",
    )
    parser.add_argument(
        "--tier-depth",
        type=int,
        default=2,
        help="Supplier tiers to explore (default: 2).",
    )
    parser.add_argument(
        "--budget-llm",
        type=int,
        default=20,
        help="Max LLM calls per condition (default: 20).",
    )
    parser.add_argument(
        "--budget-web",
        type=int,
        default=30,
        help="Max web queries per condition (default: 30).",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use cached web results only (reproducible).",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated modes to evaluate (default: all four).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (saves LLM calls).",
    )
    args = parser.parse_args(argv)

    # Determine company list.
    if args.companies:
        companies = [c.strip() for c in args.companies.split(",")]
    elif args.ground_truth_only:
        companies = GROUND_TRUTH_COMPANIES
    elif args.full:
        companies = EVAL_COMPANIES
    else:
        companies = GROUND_TRUTH_COMPANIES

    modes = (
        [m.strip() for m in args.modes.split(",")]
        if args.modes
        else None
    )

    out_dir = ensure_dir(args.out)

    print(f"Evaluating {len(companies)} companies across "
          f"{len(modes) if modes else 4} conditions...")
    print(f"Budget: {args.budget_llm} LLM / {args.budget_web} web per condition")
    print()

    results = evaluate_batch(
        companies,
        tier_depth=args.tier_depth,
        budget_llm=args.budget_llm,
        budget_web=args.budget_web,
        snapshot=args.snapshot,
        modes=modes,
        skip_judge=args.skip_judge,
    )

    # Write outputs.
    results_path = out_dir / "eval_results.json"
    results_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Wrote {results_path}")

    summary_text = format_eval_summary(results)
    summary_path = out_dir / "eval_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Wrote {summary_path}")

    # Print summary to console.
    print()
    print(summary_text)


if __name__ == "__main__":
    eval_main()
