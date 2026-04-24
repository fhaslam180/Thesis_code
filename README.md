# bor-risk-agent

A supply-chain risk analysis system built with [LangGraph](https://github.com/langchain-ai/langgraph). Given a company name, it discovers suppliers across multiple tiers (via GPT-4o or deterministic fixtures), scores six geospatial hazards from real public data sources, aggregates risk with configurable weights, and produces an evidence-backed report with IEEE-style references.

The system implements a **Verified Claim Graph (VCG)** — a fixed 11-node pipeline that models supplier relationships as claims progressing through a `PROPOSED → RETRIEVED → VERIFIED` lifecycle. A built-in evaluation framework compares four ablation conditions, computing ground-truth precision/recall, hard metrics (edge evidence rate, hazard coverage, claim support rate), and LLM-as-judge narrative quality.

## Experimental Conditions

The core thesis question: *Under a fixed LLM-call and web-query budget, does adding web evidence retrieval and LLM entailment verification to a fixed Verified Claim Graph pipeline improve claim support rate, evidence coverage, and precision compared to an LLM-only baseline?*

| Condition | Flags | Web Search | Verification | Description |
|-----------|-------|------------|--------------|-------------|
| `llm_only` | `--no-web` | No | No | Baseline: LLM-proposed claims only, no web evidence |
| `web_retrieve` | `--no-verify` | Yes | No | Adds web evidence retrieval; claims unverified |
| `web_verify` | *(default)* | Yes | Yes | Full pipeline: retrieval + LLM entailment verification |
| `strict` | `--strict` | Yes | Yes | web_verify + filter out DISPUTED/UNKNOWN claims |

All conditions operate under the same budget caps (LLM calls and web queries tracked separately; hazard scoring is deterministic and uncapped).

## Architecture

### VCG Pipeline (all conditions)

```
resolve_entity → discover_claims
  → [retrieve_evidence]       (skipped when --no-web)
  → [extract_mentions]        (skipped when --no-web)
  → [link_claims]             (skipped when --no-web)
  → [verify_claims]           (skipped when --no-verify)
  → resolve_locations
  → [score_earthquake | score_flood | score_heat_stress |
     score_drought | score_wildfire | score_cyclone]   (parallel fan-out)
  → aggregate_risk → generate_report → export_artifacts → END
```

| Node | Role |
|------|------|
| `resolve_entity` | Resolves company profile via GPT-4o (or fixture) |
| `discover_claims` | LLM proposes supplier relationships as PROPOSED claims with deterministic `claim_id = sha256(subject\|object\|type\|tier)[:12]` |
| `retrieve_evidence` | Web-searches for each claim; stores full page content in content-addressed `EvidenceStore`; writes `evidence_refs` back onto claims (PROPOSED → RETRIEVED) |
| `extract_mentions` | LLM extracts supplier mentions from each fetched document |
| `link_claims` | Fuzzy-matches mentions to claims via rapidfuzz token_set_ratio ≥ 85 |
| `verify_claims` | Two-stage verification: heuristic co-mention + negation check (Stage 1), LLM entailment (Stage 2), verbatim substring guard (Stage 3); iterates all evidence refs and keeps best verdict by SUPPORTED > DISPUTED > WEAK > UNKNOWN |
| `resolve_locations` | Enriches supplier lat/lon from evidence text |
| `score_{hazard}` (x6) | Scores all suppliers for one hazard type (parallel fan-out) |
| `aggregate_risk` | Weighted hazard average with confidence, tier, and evidence-source adjustments |
| `generate_report` | GPT-4o generates mitigations and alternatives; formats plain-text report |
| `export_artifacts` | Writes report, graph JSON, evidence JSONL, optional PROV-O |

### Claim Lifecycle

LLM-discovered suppliers are modelled as **claims** (verifiable hypotheses):

- **PROPOSED**: LLM has proposed a supplier relationship
- **RETRIEVED**: Web evidence found and linked (`evidence_refs` populated)
- **VERIFIED**: LLM entailment run; verdict assigned

Verdicts: `SUPPORTED` / `WEAK` / `DISPUTED` / `UNKNOWN`

Verified suppliers (`evidence_source: "web_verified"`) carry full confidence weight in aggregation. Unverified suppliers (`evidence_source: "llm_only"`) have their effective confidence halved (×0.5) to reflect epistemic uncertainty.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a `.env` file with API keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
BOR_RISK_LLM_MODEL=gpt-4.1-mini
# Optional: use a different model only for the evaluation judge
# BOR_RISK_JUDGE_MODEL=gpt-4.1-mini
```

The OpenAI key is required for LLM-backed supplier discovery, verification, and report-generation steps. `BOR_RISK_LLM_MODEL` defaults to `gpt-4.1-mini`; set it to `gpt-4o-mini` for very cheap smoke tests or `gpt-4o` for final comparison runs. The Tavily key is required for web search in the `web_retrieve`, `web_verify`, and `strict` conditions.

### Data preprocessing (one-time)

Wildfire and cyclone scoring use preprocessed local grid files. To regenerate from raw sources:

```bash
python3 scripts/download_data.py
```

The preprocessed grids are included in the repository, so this step is only needed to update source data.

## Usage

### LLM-only baseline (no web, no verification)

```bash
python3 -m bor_risk.cli --company "Apple" --no-web --out outputs/apple.txt
```

### Web retrieval only (no entailment verification)

```bash
python3 -m bor_risk.cli --company "Apple" --no-verify --budget-web 30 --out outputs/apple.txt
```

### Full pipeline — web retrieval + verification (default)

```bash
python3 -m bor_risk.cli --company "Apple" --out outputs/apple.txt
```

### Strict mode (filter DISPUTED/UNKNOWN claims)

```bash
python3 -m bor_risk.cli --company "Apple" --strict --out outputs/apple.txt
```

### Deterministic run (no API key needed)

```bash
python3 -m bor_risk.cli --company "ACME" --out outputs/acme.txt --no-llm
```

### Graph visualisation

```bash
python3 -m bor_risk.cli --company "Apple" --out outputs/apple.txt --visualize
```

### CLI flags reference

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--company` | Yes | — | Target company name |
| `--out` | Yes | — | Output path (directory derived from parent) |
| `--no-web` | No | off | Disable web retrieval (llm_only condition) |
| `--no-verify` | No | off | Skip LLM entailment verification (web_retrieve condition) |
| `--strict` | No | off | Exclude DISPUTED/UNKNOWN claims from report (strict condition) |
| `--tier-depth` | No | 2 | Number of supplier tiers to expand |
| `--no-llm` | No | off | Use deterministic JSON fixtures instead of GPT-4o |
| `--suppliers-path` | No | — | Path to custom supplier JSON file |
| `--budget-llm` | No | 20 | Max LLM calls per run |
| `--budget-web` | No | 30 | Max web queries per run |
| `--snapshot` | No | off | Use cached web results + persistent URL index (reproducible) |
| `--export-prov` | No | off | Export PROV-O JSON-LD provenance file |
| `--visualize` | No | off | Print Mermaid graph diagram and exit |

## Evaluation Framework

The evaluation framework runs all four conditions on a set of companies and computes three tiers of metrics:

### Tier 1: Ground-Truth Metrics (Primary)

Computed for companies with curated supplier lists (Apple, Toyota, Nike):

| Metric | Formula |
|--------|---------|
| Supplier precision | `discovered ∩ ground_truth / discovered` |
| Supplier recall | `discovered ∩ ground_truth / ground_truth` |
| Confidence calibration | Mean confidence gap between correct and incorrect suppliers |
| Verification accuracy | `web_verified ∩ ground_truth / web_verified` |

### Tier 2: Hard Computed Metrics

| Metric | Description |
|--------|-------------|
| Edge evidence rate | Fraction of suppliers with web verification |
| Hazard coverage | Scored pairs / (suppliers × 6) |
| Claim support rate | (SUPPORTED + WEAK) / total claims |
| Unknown rate | UNKNOWN / total claims |
| Quote valid rate | Supported claims with verbatim quote / supported claims |
| Mean evidence/claim | Average `evidence_refs` count per claim |
| Unique web domains | Distinct source domains in evidence_packets |
| LLM calls used | Count from BudgetTracker |
| Web queries used | Count from BudgetTracker |
| Wall clock time | End-to-end execution time |
| Unverified fraction | `llm_only / total` suppliers |

### Tier 3: LLM-as-Judge (Supplementary)

GPT-4o rates narrative quality (1-5) on completeness, actionability, and risk communication. Used for report quality only — factual accuracy is measured by Tier 1.

### Running evaluations

```bash
# Ground-truth case study (Apple, Toyota, Nike — precision/recall metrics)
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Nike" --skip-judge --out outputs/eval/

# Reproducible snapshot rerun (cached web results + persistent URL index)
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Nike" --snapshot --skip-judge --out outputs/eval/

# Broader hard-metric benchmark (fixture suppliers, no LLM discovery)
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Nike" --no-llm --snapshot --out outputs/eval_broad/

# Specific conditions only
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Nike" --conditions "llm_only,web_verify" --skip-judge --out outputs/eval/

# Custom budget
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Nike" --budget-llm 15 --budget-web 25 --out outputs/eval/
```

Evaluation outputs:
- `eval_results.json` — raw per-company, per-condition metrics (all 3 tiers)
- `eval_summary.txt` — comparison table with means across conditions

## Hazard Scoring Methodology

All hazard scores are deterministic and computed from real data — never from an LLM. Each raw score is normalised to 0.0–1.0, then mapped to 0–100 with risk levels: **Low** (0–33), **Medium** (34–66), **High** (67–100).

| Hazard | Data Source | Formula | Parameters |
|--------|------------|---------|------------|
| Earthquake | USGS FDSNWS event count API | `min(1, log10(1 + count) / 3.5)` | M4.0+ quakes within 200 km, 2015–2025 |
| Flood | Open-Meteo GloFAS river discharge | `min(1, annual_flood_days / 90)` | Annual days exceeding 2x mean discharge, 2015–2025 |
| Wildfire | NASA FIRMS VIIRS (preprocessed grid) | `min(1, log10(1 + fire_count) / 4)` | 0.5-degree grid cells, annual fire detections |
| Cyclone | NOAA IBTrACS (preprocessed grid) | `min(1, storm_count / 50)` | 1-degree grid cells, storms >= 34 kt, 1980–2024 |
| Heat stress | Open-Meteo ERA5 reanalysis | `min(1, annual_extreme_days / 130)` | Days with apparent temp max > 35 C, 2015–2025 |
| Drought | Open-Meteo ERA5 precipitation | `min(1, dry_month_fraction * 1.5)` | Months below 50% of mean monthly precipitation |

## Risk Aggregation

The `aggregate_risk` node computes supplier and company risk scores:

**1. Weighted hazard average per supplier:**

```
base_score = sum(score_i * weight_i) / sum(weight_i)
```

Default weights from `configs/hazards.yaml`: earthquake 0.25, flood 0.20, wildfire 0.15, cyclone 0.15, heat_stress 0.10, drought 0.15.

**2. Evidence-source, confidence, and tier adjustment:**

```
effective_confidence = confidence * 0.5  (if evidence_source == "llm_only")
effective_confidence = confidence         (if "web_verified" or "fixture")

confidence_factor = 0.5 + (0.5 * effective_confidence)
tier_factor = max(0.8, 1.0 - 0.05 * (tier - 1))
risk_score = base_score * confidence_factor * tier_factor
```

**3. Company score and risk band:**

```
company_score = mean(supplier_risk_scores)
```

| Company Score | Risk Band |
|---------------|-----------|
| >= 0.40 | High |
| >= 0.25 | Medium |
| < 0.25 | Low |

**Threshold alerts** fire when any individual hazard score exceeds its configured threshold. Alerts with exceedance >= 0.2 are classified as critical.

## Output Files

Each run produces three files in the output directory:

| File | Format | Contents |
|------|--------|----------|
| `{company}_report.txt` | Plain text | Executive summary, company profile, supplier risk ranking, threshold alerts, risk register matrix, hazard summary, mitigations, alternatives, evidence appendix, IEEE references |
| `{company}_graph.json` | JSON | Suppliers, edges, risk summary, workflow decision/actions/trace, budget summary |
| `{company}_evidence.jsonl` | JSONL | One evidence item per line (source, description, retrieval timestamp) |

With `--sensitivity`, an additional `{company}_sensitivity.json` is produced.

## Configuration

### `configs/hazards.yaml`

Defines hazard types with weights (aggregation) and thresholds (alert generation):

```yaml
hazards:
  - name: earthquake
    weight: 0.25
    threshold: 0.5
  - name: flood
    weight: 0.20
    threshold: 0.4
  # ...
```

### `configs/prompts.yaml`

LLM prompt templates for company profile resolution, supplier discovery, mitigation generation, alternative suggestions, agent system prompt, and evaluation judge prompt.

## Testing

```bash
python3 -m pytest -v
```

| Test File | Coverage |
|-----------|----------|
| `test_hazard_scoring.py` | Score normalisation, all 6 hazard APIs, edge cases, hash stub fallback |
| `test_enhanced_report.py` | Report sections, risk matrix, IEEE references, evidence appendix |
| `test_langgraph_features.py` | Graph topology, parallel fan-out/fan-in, streaming |
| `test_llm_discovery.py` | Company profile resolution, contextual prompts, deduplication |
| `test_sensitivity.py` | Pure function parity, scenario generation, weight/threshold perturbation |
| `test_tier_expansion.py` | End-to-end graph run, CLI output files, supplier loading |
| `test_evaluation.py` | Ground truth loading, precision/recall, hard metrics, summary formatting |
| `test_claim_lifecycle.py` | Claim PROPOSED→RETRIEVED→VERIFIED lifecycle, link + verify tools |
| `test_evidence_store.py` | Content-addressed store, dedup, snapshot read, URL index |
| `test_provenance.py` | PROV-O JSON-LD export |
| `test_suggest_alternatives.py` | Alternative supplier suggestions, LLM mock, report rendering |
| `test_budget.py` | BudgetTracker state, budget exhaustion, summary |
| `test_verification.py` | Co-mention matching, alias variants, evidence-source weighting |
| `test_search.py` | Snapshot mode, cache hits/misses, max_results |

### Mock strategy

An `autouse` fixture in `conftest.py` patches `urllib.request.urlopen` with URL-aware routing. LLM tests mock `ChatOpenAI` at the class level. Agent tool tests patch `search_web_snapshot` before building tool closures to ensure the mock is captured. Verification tests patch `bor_risk.search.search_web_snapshot` directly.

## Project Structure

```
bor-risk-agent/
  configs/
    hazards.yaml              # Hazard weights and thresholds
    prompts.yaml               # LLM prompt templates (including agent system prompt)
  data/
    wildfire_grid.json         # Preprocessed 0.5-degree fire grid (NASA FIRMS)
    cyclone_grid.json          # Preprocessed 1-degree cyclone grid (NOAA IBTrACS)
    ground_truth/              # Curated supplier lists for evaluation
      apple.json
      toyota.json
      nike.json
    evidence_cache/            # Content-addressed full-page snapshots + URL index
      packet_index.json        # Persistent URL → packet metadata (for snapshot reruns)
    search_cache/              # Cached Tavily search snippets (not tracked in git)
    raw/                       # Source data files (not tracked in git)
  scripts/
    download_data.py           # Downloads IBTrACS + finds FIRMS CSV
    preprocess_cyclone.py      # Converts IBTrACS CSV to 1-degree grid JSON
    preprocess_wildfire.py     # Converts FIRMS VIIRS CSV to 0.5-degree grid JSON
  src/bor_risk/
    cli.py                     # CLI entry point (--no-web, --no-verify, --strict, etc.)
    graph.py                   # 11-node VCG LangGraph pipeline
    budget.py                  # BudgetTracker (LLM calls, web queries, hazard scores)
    search.py                  # Tavily web search with disk cache + snapshot mode
    evaluate.py                # 3-tier evaluation framework + CLI
    tools.py                   # Hazard scoring, LLM discovery, risk aggregation, batch verification
    models.py                  # Pydantic models and GraphState TypedDict
    report.py                  # Plain-text report formatting
    sensitivity.py             # Weight/threshold sensitivity analysis
    utils.py                   # Config loaders (cached YAML)
  tests/
    conftest.py                # URL-aware mock fixtures
    fixtures/
      mock_suppliers.json      # Deterministic supplier data (ACME, GlobalMfg)
    test_*.py                  # 185 tests across 14 files
  outputs/                     # Generated reports (not tracked in git)
```
