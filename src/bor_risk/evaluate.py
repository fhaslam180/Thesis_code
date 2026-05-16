"""Evaluation framework for the Verified Claim Graph (VCG) pipeline.

Three-tier evaluation:
  1. Ground-truth metrics (PRIMARY): precision, recall, calibration
  2. Hard computed metrics: edge evidence rate, hazard coverage, claim metrics
  3. LLM-as-judge (SUPPLEMENTARY): narrative quality only

Experimental conditions (ablations of the VCG method):
  llm_only     — LLM-proposed claims, no web retrieval, no entailment verification
  web_retrieve — Evidence fetched; claims unverified (retrieval ablation)
  web_verify   — Full pipeline: retrieval + LLM entailment verification
  strict       — web_verify + only SUPPORTED/WEAK claims included in report
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from bor_risk.budget import BudgetTracker
from bor_risk.graph import run_vcg_graph
from bor_risk.names import canonical_name_key, names_match
from bor_risk.utils import load_prompts, load_reference_set

_GROUND_TRUTH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ground_truth"
_DEFAULT_JUDGE_MODEL = os.getenv(
    "BOR_RISK_JUDGE_MODEL",
    os.getenv("BOR_RISK_LLM_MODEL", "gpt-4.1-mini"),
)

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

# Condition definitions: name → (enable_web, no_verify, strict_mode)
CONDITIONS: dict[str, tuple[bool, bool, bool]] = {
    "llm_only":     (False, True,  False),
    "web_retrieve": (True,  True,  False),
    "web_verify":   (True,  False, False),
    "strict":       (True,  False, True),
}


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


def _unique_names(names: list[str]) -> list[str]:
    """Deduplicate names by canonical supplier key while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        key = canonical_name_key(name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def _matches_any_ground_truth(name: str, gt_names: list[str]) -> bool:
    return any(names_match(name, gt_name) for gt_name in gt_names)


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

    gt_names = _unique_names(
        list(gt.get("tier_1_suppliers", [])) + list(gt.get("tier_2_suppliers", []))
    )

    discovered = state.get("suppliers", [])
    discovered_names = _unique_names([s["name"] for s in discovered])

    tp_discovered = [
        name for name in discovered_names
        if _matches_any_ground_truth(name, gt_names)
    ]
    matched_gt = [
        gt_name for gt_name in gt_names
        if any(names_match(name, gt_name) for name in discovered_names)
    ]
    precision = len(tp_discovered) / len(discovered_names) if discovered_names else 0.0
    recall = len(matched_gt) / len(gt_names) if gt_names else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    for s in discovered:
        conf = float(s.get("confidence", 0.5))
        if _matches_any_ground_truth(s["name"], gt_names):
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)

    avg_correct_conf = (
        sum(correct_confidences) / len(correct_confidences)
        if correct_confidences else 0.0
    )
    avg_incorrect_conf = (
        sum(incorrect_confidences) / len(incorrect_confidences)
        if incorrect_confidences else 0.0
    )
    calibration_gap = avg_correct_conf - avg_incorrect_conf

    web_verified = [
        s for s in discovered if s.get("evidence_source") == "web_verified"
    ]
    wv_correct = sum(
        1 for s in web_verified if _matches_any_ground_truth(s["name"], gt_names)
    )
    verification_accuracy = (
        wv_correct / len(web_verified) if web_verified else 0.0
    )

    # text_supported_precision: of SUPPORTED claims, fraction matching ground truth
    supported_names = {
        c["object_entity_id"]
        for c in state.get("claims", [])
        if c.get("verdict") == "SUPPORTED"
    }
    text_supported = [
        s for s in discovered
        if any(names_match(s["name"], sup) for sup in supported_names)
    ]
    ts_correct = sum(
        1 for s in text_supported if _matches_any_ground_truth(s["name"], gt_names)
    )
    text_supported_precision = ts_correct / len(text_supported) if text_supported else 0.0

    return {
        "has_ground_truth": True,
        "gt_supplier_count": len(gt_names),
        "discovered_count": len(discovered_names),
        "true_positives": len(tp_discovered),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_correct_confidence": round(avg_correct_conf, 4),
        "avg_incorrect_confidence": round(avg_incorrect_conf, 4),
        "calibration_gap": round(calibration_gap, 4),
        "web_verified_count": len(web_verified),
        "verification_accuracy": round(verification_accuracy, 4),
        "text_supported_count": len(text_supported),
        "text_supported_precision": round(text_supported_precision, 4),
    }


# ---------------------------------------------------------------------------
# Tier 2: Hard computed metrics
# ---------------------------------------------------------------------------

def compute_hard_metrics(state: dict) -> dict:
    """Compute deterministic metrics from the run state."""
    suppliers = state.get("suppliers", [])
    # Prefer strict-filtered scores when present (key set by aggregate_risk_node
    # in strict mode). Use key presence — not truthiness — so an intentional
    # empty strict result is not overridden by the full accumulated list.
    hazard_scores = (
        state["report_hazard_scores"]
        if "report_hazard_scores" in state
        else state.get("hazard_scores", [])
    )
    evidence = state.get("evidence", [])
    budget = state.get("budget_summary", {})
    claims = state.get("claims", [])
    evidence_packets = state.get("evidence_packets", [])

    total_suppliers = len(suppliers)
    total_claims = len(claims)
    total_edges = len(state.get("edges", []))

    web_verified = sum(
        1 for s in suppliers if s.get("evidence_source") == "web_verified"
    )
    edge_evidence_rate = (
        web_verified / total_suppliers if total_suppliers > 0 else 0.0
    )

    possible_pairs = total_suppliers * 6
    scored_pairs = len(hazard_scores)
    hazard_coverage = (
        scored_pairs / possible_pairs if possible_pairs > 0 else 0.0
    )

    def _to_domain(raw: str) -> str:
        try:
            return urlparse(raw).netloc or raw
        except Exception:
            return raw

    # Collect unique web domains from evidence_packets (VCG pipeline) and
    # from legacy evidence items (backward compat); normalize all to domain-only.
    web_domains: set[str] = {
        p.get("domain", "") for p in evidence_packets if p.get("domain")
    }
    web_domains |= {
        _to_domain(e.get("source", "").replace("web:", ""))
        for e in evidence
        if e.get("source", "").startswith("web:")
    }
    web_domains.discard("")

    llm_only = sum(
        1 for s in suppliers if s.get("evidence_source", "fixture") == "llm_only"
    )
    unverified_fraction = (
        llm_only / total_suppliers if total_suppliers > 0 else 0.0
    )

    # Claim-level metrics
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    weak = sum(1 for c in claims if c.get("verdict") == "WEAK")
    disputed = sum(1 for c in claims if c.get("verdict") == "DISPUTED")
    unknown = sum(1 for c in claims if c.get("verdict") == "UNKNOWN")
    verified_count = sum(1 for c in claims if c.get("status") == "VERIFIED")

    claim_support_rate = (supported + weak) / total_claims if total_claims > 0 else 0.0
    dispute_rate = disputed / total_claims if total_claims > 0 else 0.0
    unknown_rate = unknown / total_claims if total_claims > 0 else 0.0

    # Quote validity: supported claims with non-empty supporting_spans
    supported_claims = [c for c in claims if c.get("verdict") == "SUPPORTED"]
    quotes_valid = sum(
        1 for c in supported_claims
        if any(s.get("quote") for s in c.get("supporting_spans", []))
    )
    quote_valid_rate = (
        quotes_valid / len(supported_claims) if supported_claims else 0.0
    )

    # Mean unique domains per supported claim
    domain_counts: list[float] = []
    for c in supported_claims:
        eids = set(c.get("evidence_refs", []))
        domains = {
            p.get("domain", "")
            for p in evidence_packets
            if p.get("evidence_id") in eids and p.get("domain")
        }
        domain_counts.append(float(len(domains)))
    mean_evidence_per_claim = (
        sum(len(c.get("evidence_refs", [])) for c in claims) / total_claims
        if total_claims > 0 else 0.0
    )
    unique_domain_per_supported = (
        sum(domain_counts) / len(domain_counts) if domain_counts else 0.0
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
        "unique_web_domains": len(web_domains),
        "unverified_fraction": round(unverified_fraction, 4),
        "evidence_count": len(evidence),
        # Claim metrics
        "total_claims": total_claims,
        "claim_support_rate": round(claim_support_rate, 4),
        "verified_claim_count": verified_count,
        "dispute_rate": round(dispute_rate, 4),
        "unknown_rate": round(unknown_rate, 4),
        "quote_valid_rate": round(quote_valid_rate, 4),
        "mean_evidence_per_claim": round(mean_evidence_per_claim, 4),
        "unique_domain_per_supported_claim": round(unique_domain_per_supported, 4),
    }


# ---------------------------------------------------------------------------
# Stage-separated evaluation metrics
# ---------------------------------------------------------------------------

def compute_stage_metrics(state: dict) -> dict:
    """Return metrics broken down by pipeline stage. Uses claims, not filtered suppliers."""
    claims = state.get("claims", [])
    suppliers = state.get("suppliers", [])   # strict-filtered; used only for report count
    packets = state.get("evidence_packets", [])
    n = len(claims)

    # Discovery stage (counts from claims, never strict-filtered suppliers)
    pre = state.get("pre_dedup_supplier_count") or n
    rag_seeded = sum(1 for c in claims if c.get("discovery_url"))
    llm_only_count = sum(1 for c in claims if not c.get("discovery_url"))
    discovery = {
        "candidate_count": n,
        "pre_dedup_count": pre,
        "dedup_rate": round((pre - n) / pre, 4) if pre > 0 else 0.0,
        "rag_seeded_count": rag_seeded,
        "llm_only_count": llm_only_count,
        "final_report_count": len(suppliers),
    }

    # Retrieval stage
    retrieved = sum(1 for c in claims if c.get("status") in ("RETRIEVED", "VERIFIED"))
    retrieval = {
        "retrieved_count": retrieved,
        "retrieval_rate": round(retrieved / n, 4) if n > 0 else 0.0,
        "mean_evidence_refs": round(
            sum(len(c.get("evidence_refs", [])) for c in claims) / n, 4
        ) if n > 0 else 0.0,
        "unique_domains": len({p.get("domain") for p in packets if p.get("domain")}),
    }

    # Verification stage
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    weak = sum(1 for c in claims if c.get("verdict") == "WEAK")
    disputed = sum(1 for c in claims if c.get("verdict") == "DISPUTED")
    unknown = sum(1 for c in claims if c.get("verdict") == "UNKNOWN")
    verified = sum(1 for c in claims if c.get("status") == "VERIFIED")
    direction_clear = sum(
        1 for c in claims
        if "direction=supplier_to_company" in c.get("verdict_explanation", "")
        or "direction=company_to_supplier" in c.get("verdict_explanation", "")
    )
    supported_claims = [c for c in claims if c.get("verdict") == "SUPPORTED"]
    quotes_valid = sum(
        1 for c in supported_claims
        if any(s.get("quote") for s in c.get("supporting_spans", []))
    )
    verification = {
        "verified_count": verified,
        "supported": supported,
        "weak": weak,
        "disputed": disputed,
        "unknown": unknown,
        "supported_rate": round(supported / n, 4) if n > 0 else 0.0,
        "unknown_rate": round(unknown / n, 4) if n > 0 else 0.0,
        "quote_valid_rate": (
            round(quotes_valid / len(supported_claims), 4) if supported_claims else 0.0
        ),
        "direction_clear_rate": (
            round(direction_clear / verified, 4) if verified > 0 else 0.0
        ),
    }

    # Source quality stage (from evidence_packets)
    qualities = [p.get("source_quality", 0.5) for p in packets]
    type_counts: dict[str, int] = {}
    for p in packets:
        stype = p.get("source_type", "unknown")
        type_counts[stype] = type_counts.get(stype, 0) + 1
    source = {
        "mean_source_quality": round(sum(qualities) / len(qualities), 4) if qualities else 0.0,
        "high_quality_count": sum(1 for q in qualities if q >= 0.7),
        "low_quality_count": sum(1 for q in qualities if q < 0.4),
        "total_packets": len(packets),
        "source_type_counts": type_counts,
    }

    return {
        "discovery": discovery,
        "retrieval": retrieval,
        "verification": verification,
        "source_quality": source,
    }


# ---------------------------------------------------------------------------
# Tier 3: LLM-as-judge (narrative quality only)
# ---------------------------------------------------------------------------

def judge_report_quality(company: str, state: dict) -> dict:
    """Rate report narrative quality using GPT-4o as judge."""
    report_text = state.get("report_text", "")
    if not report_text:
        return {"judge_scores": {}}

    try:
        prompts = load_prompts()
        template = prompts.get("judge_prompt", "")
        if not template:
            return {"judge_scores": {}}

        prompt_text = template.format(report_text=report_text[:3000])
        llm = ChatOpenAI(
            model=_DEFAULT_JUDGE_MODEL,
            temperature=0,
            timeout=60,
            max_retries=5,
        )
        response = llm.invoke([HumanMessage(content=prompt_text)])
        content = response.content.strip()

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
    condition: str,
    tier_depth: int = 2,
    snapshot: bool = True,
    no_llm: bool = False,
) -> dict:
    """Run one ablation condition and return the final state.

    Parameters
    ----------
    condition : str
        One of: "llm_only", "web_retrieve", "web_verify", "strict".
    no_llm : bool
        When True, use fixture supplier data instead of LLM discovery.
        Requires fixture data to exist for *company* in ``data/fixtures/``.
        Useful for hard-metric benchmarking without LLM API calls.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Valid: {list(CONDITIONS.keys())}"
        )

    enable_web, no_verify, strict_mode = CONDITIONS[condition]
    budget = BudgetTracker()

    state = run_vcg_graph(
        company=company,
        tier_depth=tier_depth,
        use_llm=not no_llm,
        enable_web=enable_web,
        no_verify=no_verify,
        strict_mode=strict_mode,
        snapshot_mode=snapshot,
        budget=budget,
    )

    if not state.get("budget_summary"):
        state["budget_summary"] = budget.summary()

    return state


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_company(
    company: str,
    tier_depth: int = 2,
    snapshot: bool = True,
    conditions: list[str] | None = None,
    skip_judge: bool = False,
    no_llm: bool = False,
) -> dict:
    """Run all ablation conditions on one company, compute all metrics."""
    if conditions is None:
        conditions = list(CONDITIONS.keys())

    results: dict = {}

    for condition in conditions:
        try:
            state = run_condition(
                company, condition, tier_depth, snapshot, no_llm=no_llm
            )
            metrics = {
                **compute_ground_truth_metrics(company, state),
                **compute_hard_metrics(state),
                "stages": compute_stage_metrics(state),
            }
            if not skip_judge:
                metrics.update(judge_report_quality(company, state))
            results[condition] = metrics
        except Exception as e:
            results[condition] = {"error": str(e)}

    return {"company": company, "conditions": results}


def evaluate_batch(
    companies: list[str],
    tier_depth: int = 2,
    snapshot: bool = True,
    conditions: list[str] | None = None,
    skip_judge: bool = False,
    no_llm: bool = False,
) -> list[dict]:
    """Evaluate multiple companies across all ablation conditions."""
    results = []
    for company in companies:
        result = evaluate_company(
            company,
            tier_depth=tier_depth,
            snapshot=snapshot,
            conditions=conditions,
            skip_judge=skip_judge,
            no_llm=no_llm,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    from statistics import stdev
    return stdev(values)


def _paired_ttest(a: list[float], b: list[float]) -> dict:
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

    try:
        from scipy.stats import t as t_dist
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
    except ImportError:
        import math
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

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


def _compute_stat_tests(condition_metrics: dict[str, list[dict]]) -> list[str]:
    """Paired statistical tests comparing VCG ablation conditions."""
    lines = ["", "--- Statistical Tests (VCG Ablations) ---"]

    # Ablation comparisons: what does adding each component contribute?
    comparisons = [
        ("llm_only", "web_verify",   "LLM-only vs Web+Verify (full pipeline)"),
        ("web_retrieve", "web_verify", "Web-retrieve vs Web+Verify (entailment adds?)"),
        ("web_verify", "strict",      "Web+Verify vs Strict (filtering disputed?)"),
    ]
    test_metrics = [
        ("precision", "Precision"),
        ("claim_support_rate", "Claim Support Rate"),
        ("edge_evidence_rate", "Edge Evidence Rate"),
        ("f1", "F1 Score"),
    ]

    any_tests = False
    for cond_a, cond_b, label in comparisons:
        vals_a = condition_metrics.get(cond_a, [])
        vals_b = condition_metrics.get(cond_b, [])
        if not vals_a or not vals_b:
            continue

        for metric_key, metric_label in test_metrics:
            a = [m.get(metric_key) for m in vals_a if metric_key in m]
            b = [m.get(metric_key) for m in vals_b if metric_key in m]
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
    """Format an ablation comparison table from evaluation results."""
    lines = ["=" * 80, "VCG ABLATION EVALUATION SUMMARY", "=" * 80, ""]

    condition_metrics: dict[str, list[dict]] = {}
    for r in results:
        for cond, metrics in r.get("conditions", {}).items():
            if "error" not in metrics:
                condition_metrics.setdefault(cond, []).append(metrics)

    all_conditions = list(CONDITIONS.keys())
    header = f"{'Metric':<35}"
    for cond in all_conditions:
        header += f"  {cond:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    metric_keys = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("edge_evidence_rate", "Edge Evidence Rate"),
        ("hazard_coverage", "Hazard Coverage"),
        ("unverified_fraction", "Unverified Fraction"),
        ("total_claims", "Total Claims"),
        ("claim_support_rate", "Claim Support Rate"),
        ("verified_claim_count", "Verified Claims"),
        ("dispute_rate", "Dispute Rate"),
        ("unknown_rate", "Unknown Rate"),
        ("quote_valid_rate", "Quote Valid Rate"),
        ("mean_evidence_per_claim", "Mean Evidence/Claim"),
        ("unique_domain_per_supported_claim", "Unique Domains/SUPPORTED"),
        ("unique_web_domains", "Unique Web Domains"),
        ("supplier_count", "Supplier Count"),
        ("llm_calls", "LLM Calls"),
        ("web_queries", "Web Queries"),
        ("wall_clock_seconds", "Wall Clock (s)"),
    ]

    for key, label in metric_keys:
        row = f"{label:<35}"
        for cond in all_conditions:
            vals = condition_metrics.get(cond, [])
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

    lines += _compute_stat_tests(condition_metrics)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-run quality report
# ---------------------------------------------------------------------------

import sys as _sys

_QUALITY_TARGETS: dict[str, tuple[str, float]] = {
    "precision":              (">",  0.60),
    "recall":                 (">",  0.60),
    "unknown_rate":           ("<",  0.25),
    "duplicate_rate":         ("<",  0.05),
    "hazard_informativeness": (">=", 0.50),
}


def print_quality_report(
    company: str,
    state: dict,
    pre_dedup_supplier_count: int | None = None,
    *,
    file=None,
) -> dict:
    """Print a formatted quality report and return the full metrics dict.

    Parameters
    ----------
    company:
        Company name (used to load ground truth).
    state:
        GraphState dict from a completed pipeline run.
    pre_dedup_supplier_count:
        Raw supplier count before _deduplicate_suppliers(); taken from
        state["pre_dedup_supplier_count"] if not supplied explicitly.
    file:
        Output stream (default: sys.stdout).
    """
    if file is None:
        file = _sys.stdout

    hard = compute_hard_metrics(state)
    gt = compute_ground_truth_metrics(company, state)
    merged = {**hard, **gt}

    # -- pre_dedup_count: caller arg > state field > None ----------------
    pre_count = pre_dedup_supplier_count
    if pre_count is None:
        pre_count = state.get("pre_dedup_supplier_count")

    # Use len(claims) as post-dedup count: claims are created one-per-supplier
    # at discovery and are never filtered by strict mode, so this is the correct
    # denominator even when strict mode has reduced state["suppliers"] to a small
    # verified subset.
    post_count = len(state.get("claims", []))
    if pre_count is not None and pre_count > 0:
        merged["duplicate_rate"] = round((pre_count - post_count) / pre_count, 4)
    else:
        merged["duplicate_rate"] = None

    # -- verified_rate_pct -----------------------------------------------
    total_claims = hard.get("total_claims", 0)
    verified_count = hard.get("verified_claim_count", 0)
    merged["verified_rate_pct"] = (
        round(verified_count / total_claims * 100, 1) if total_claims > 0 else 0.0
    )

    # -- location_source_mix ---------------------------------------------
    mix: dict[str, int] = {"llm": 0, "evidence": 0, "fixture": 0}
    for c in state.get("claims", []):
        candidates = c.get("location_candidates", [])
        if not candidates:
            mix["llm"] += 1
            continue
        best = max(candidates, key=lambda lc: lc["confidence"] if isinstance(lc, dict) else lc.confidence)
        src = best["source"] if isinstance(best, dict) else best.source
        mix[src] = mix.get(src, 0) + 1
    merged["location_source_mix"] = mix

    # -- hazard_informativeness (from reference_set metadata) ------------
    try:
        rs = load_reference_set()
        informative = rs.get("metadata", {}).get("informative_hazards", [])
        merged["hazard_informativeness"] = round(len(informative) / 6, 4)
    except Exception:
        merged["hazard_informativeness"] = None

    # -- Print table -----------------------------------------------------
    W = 80
    print("=" * W, file=file)
    print(f"VCG QUALITY REPORT — {company}", file=file)
    print("=" * W, file=file)
    print(f"{'Metric':<35} {'Value':>10}  {'Target':>10}  {'Status':>6}", file=file)
    print("-" * W, file=file)

    def _fmt(v) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _status(key: str, v) -> str:
        if v is None or key not in _QUALITY_TARGETS:
            return "—"
        op, threshold = _QUALITY_TARGETS[key]
        if op == ">":
            ok = v > threshold
        elif op == ">=":
            ok = v >= threshold
        else:
            ok = v < threshold
        return "PASS" if ok else "FAIL"

    rows = [
        ("Precision", "precision", merged.get("precision")),
        ("Recall", "recall", merged.get("recall")),
        ("F1 Score", "f1", merged.get("f1")),
        ("Unknown Rate", "unknown_rate", merged.get("unknown_rate")),
        ("Duplicate Rate", "duplicate_rate", merged.get("duplicate_rate")),
        ("Unverified Fraction", "unverified_fraction", merged.get("unverified_fraction")),
        ("Verified Claims (%)", "verified_rate_pct", merged.get("verified_rate_pct")),
        ("Claim Support Rate", "claim_support_rate", merged.get("claim_support_rate")),
        ("Hazard Informativeness", "hazard_informativeness", merged.get("hazard_informativeness")),
        ("Unique Web Domains", "unique_web_domains", merged.get("unique_web_domains")),
        ("Supplier Count", "supplier_count", merged.get("supplier_count")),
    ]

    for label, key, val in rows:
        target_str = ""
        if key in _QUALITY_TARGETS:
            op, thr = _QUALITY_TARGETS[key]
            target_str = f"{op} {thr}"
        status = _status(key, val)
        if key == "location_source_mix":
            continue
        print(
            f"{label:<35} {_fmt(val):>10}  {target_str:>10}  {status:>6}",
            file=file,
        )

    loc = merged["location_source_mix"]
    print(
        f"{'Location Source Mix':<35} llm={loc['llm']} evidence={loc['evidence']} fixture={loc['fixture']}",
        file=file,
    )
    print("=" * W, file=file)

    # Stage Breakdown
    stages = compute_stage_metrics(state)
    d = stages["discovery"]
    r = stages["retrieval"]
    v = stages["verification"]
    sq = stages["source_quality"]
    type_str = " ".join(f"{k}={v2}" for k, v2 in sorted(sq["source_type_counts"].items()))
    print("", file=file)
    print("--- Stage Breakdown ---", file=file)
    print(
        f"Discovery:    {d['candidate_count']} candidates "
        f"(pre-dedup={d['pre_dedup_count']}, dedup_rate={d['dedup_rate']:.2f}) "
        f"| rag_seeded={d['rag_seeded_count']} llm_only={d['llm_only_count']} "
        f"| report={d['final_report_count']}",
        file=file,
    )
    print(
        f"Retrieval:    {r['retrieved_count']} retrieved ({r['retrieval_rate']:.1%}) "
        f"| mean_refs={r['mean_evidence_refs']:.1f} | unique_domains={r['unique_domains']}",
        file=file,
    )
    print(
        f"Verification: {v['verified_count']} verified "
        f"| SUPPORTED={v['supported']} WEAK={v['weak']} "
        f"DISPUTED={v['disputed']} UNKNOWN={v['unknown']} "
        f"| quotes_valid={v['quote_valid_rate']:.1%} dir_clear={v['direction_clear_rate']:.1%}",
        file=file,
    )
    print(
        f"Src quality:  mean={sq['mean_source_quality']:.2f} "
        f"| high={sq['high_quality_count']}(>=0.7) low={sq['low_quality_count']}(<0.4) "
        f"| types: {type_str or 'none'}",
        file=file,
    )

    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def eval_main(argv: list[str] | None = None) -> None:
    """CLI for running VCG ablation evaluation."""
    import argparse

    from dotenv import load_dotenv
    from bor_risk.utils import ensure_dir

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate VCG pipeline across ablation conditions.",
    )
    parser.add_argument(
        "--companies",
        default=None,
        help="Comma-separated company names (default: ground-truth set).",
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
        "--snapshot",
        action="store_true",
        help="Use cached web results only (reproducible).",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated conditions to evaluate (default: all four).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Use fixture supplier data instead of LLM discovery. "
            "Requires fixtures to exist for each company. "
            "Useful for hard-metric benchmarking without LLM API calls."
        ),
    )
    args = parser.parse_args(argv)

    if args.companies:
        companies = [c.strip() for c in args.companies.split(",")]
    elif args.ground_truth_only:
        companies = GROUND_TRUTH_COMPANIES
    elif args.full:
        companies = EVAL_COMPANIES
    else:
        companies = GROUND_TRUTH_COMPANIES

    selected_conditions = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions
        else None
    )

    out_dir = ensure_dir(args.out)

    print(
        f"Evaluating {len(companies)} companies across "
        f"{len(selected_conditions) if selected_conditions else 4} conditions..."
    )
    print()

    results = evaluate_batch(
        companies,
        tier_depth=args.tier_depth,
        snapshot=args.snapshot,
        conditions=selected_conditions,
        skip_judge=args.skip_judge,
        no_llm=args.no_llm,
    )

    results_path = out_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {results_path}")

    summary_text = format_eval_summary(results)
    summary_path = out_dir / "eval_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Wrote {summary_path}")

    print()
    print(summary_text)


if __name__ == "__main__":
    eval_main()
