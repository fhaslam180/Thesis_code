# bor-risk-agent

A supply-chain risk analysis system built with [LangGraph](https://github.com/langchain-ai/langgraph). Given a company name, it discovers suppliers across multiple tiers (via GPT-4o or deterministic fixtures), scores six geospatial hazards from real public data sources, aggregates risk with configurable weights, and produces an evidence-backed report with IEEE-style references.

The system supports two execution modes — a **fixed pipeline** and an **autonomous ReAct agent** — enabling controlled comparison of dynamic vs fixed tool selection under identical budget constraints. A built-in evaluation framework computes ground-truth precision/recall, hard metrics (edge evidence rate, hazard coverage, cost), and LLM-as-judge narrative quality across four ablation conditions.

## Experimental Conditions

The core thesis question: *Under a fixed LLM-call and web-query budget, does a dynamic tool-using agent produce a more evidence-grounded and accurate supplier-risk assessment than a fixed pipeline?*

| Condition | Flag | Tool Selection | Web Search | Description |
|-----------|------|---------------|------------|-------------|
| A: Pipeline | `--mode pipeline` | Fixed | No | Baseline: fixed-order discovery and scoring |
| B: Pipeline + Web | `--mode pipeline-web` | Fixed | Yes | Adds deterministic web verification (descending confidence order) |
| C: Agent | `--mode agent` | Dynamic | No | ReAct agent chooses tools dynamically, no web access |
| D: Agent + Web | `--mode agent-web` | Dynamic | Yes | Full system: agent chooses tools + web verification |

All conditions operate under the same budget caps (LLM calls and web queries tracked separately; hazard scoring is deterministic and uncapped).

## Architecture

### Pipeline Mode (Conditions A/B)

```
START
  -> discover_suppliers
       -> [verify_suppliers]          (Condition B only)
            |-> score_earthquake --\
            |-> score_flood --------\
            |-> score_wildfire -------> aggregate_risk
            |-> score_cyclone ------/     -> decide_workflow (+ HITL interrupt)
            |-> score_heat_stress -/          |-> high_risk_response --\
            |-> score_drought ----/           |-> monitoring_response ---> generate_mitigations
                                                                              -> suggest_alternatives
                                                                                 -> format_report
                                                                                    -> END
```

| Node | Role |
|------|------|
| `discover_suppliers` | Two-step contextual discovery: resolves company profile via GPT-4o, then discovers suppliers with industry context. Falls back to JSON fixtures with `--no-llm` |
| `verify_suppliers` | (Condition B only) Verifies suppliers against web evidence in descending confidence order until web budget is exhausted |
| `score_{hazard}` (x6) | Scores all suppliers for one hazard type (runs in parallel) |
| `aggregate_risk` | Weighted average of hazard scores with confidence, tier, and evidence-source adjustments |
| `decide_workflow` | Routes to escalation or monitoring; pauses for human approval when interactive |
| `high_risk_response` | Builds P1/P2 escalation actions for high-risk cases |
| `monitoring_response` | Builds P3 monitoring actions for lower-risk cases |
| `generate_mitigations` | GPT-4o generates location-specific mitigation strategies (skipped with `--no-llm`) |
| `suggest_alternatives` | GPT-4o suggests lower-risk alternative suppliers (skipped with `--no-llm`) |
| `format_report` | Produces the final plain-text report |

### Agent Mode (Conditions C/D)

The autonomous agent uses `create_react_agent` from LangGraph with 8 tools as closures over a shared accumulator and budget tracker:

| Tool | Budget Cost | Purpose |
|------|------------|---------|
| `profile_company` | 1 LLM call | Understand company industry and products |
| `discover_suppliers` | 1 LLM call | Generate candidate suppliers (hypotheses) |
| `web_search` | 1 web query | Search for supplier data from news/filings (Condition D only) |
| `verify_supplier` | 1 web query | Verify a supply relationship via co-mention matching (Condition D only) |
| `score_hazard` | Free | Score one supplier for one hazard (deterministic) |
| `aggregate_risk` | Free | Compute company risk summary |
| `generate_mitigations` | 1 LLM call | Create mitigation strategies |
| `suggest_alternatives` | 1 LLM call | Suggest lower-risk alternatives |

The agent decides which suppliers to verify first, which hazards to prioritise per geography, and when to stop — spending budget where uncertainty reduction has the most impact.

### Supplier Verification

LLM-discovered suppliers are treated as **hypotheses**. Verification searches for **relationship evidence** — a web source that co-mentions both the company and supplier with relationship cues (supplier, vendor, manufactures, supplies, partner, contract, etc.).

Verified suppliers (`evidence_source: "web_verified"`) carry full confidence weight in aggregation. Unverified suppliers (`evidence_source: "llm_only"`) have their effective confidence halved (x0.5) to reflect epistemic uncertainty.

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
```

The OpenAI key is required for GPT-4o supplier discovery and mitigations. The Tavily key is required for web search in Conditions B and D.

### Data preprocessing (one-time)

Wildfire and cyclone scoring use preprocessed local grid files. To regenerate from raw sources:

```bash
python3 scripts/download_data.py
```

The preprocessed grids are included in the repository, so this step is only needed to update source data.

## Usage

### Pipeline mode (Condition A — baseline)

```bash
python3 -m bor_risk.cli --company "Apple" --mode pipeline --out outputs/apple.txt
```

### Pipeline + web verification (Condition B)

```bash
python3 -m bor_risk.cli --company "Apple" --mode pipeline-web --budget-web 30 --out outputs/apple.txt
```

### Agent mode (Condition C — no web)

```bash
python3 -m bor_risk.cli --company "Apple" --mode agent --budget-llm 20 --out outputs/apple.txt
```

### Agent + web (Condition D — full system)

```bash
python3 -m bor_risk.cli --company "Apple" --mode agent-web --budget-llm 20 --budget-web 30 --out outputs/apple.txt
```

### Deterministic run (no API key needed)

```bash
python3 -m bor_risk.cli --company "ACME" --out outputs/acme.txt --no-llm
```

### Interactive mode (streaming + human-in-the-loop)

```bash
python3 -m bor_risk.cli --company "Apple" --out outputs/apple.txt --interactive
```

### With sensitivity analysis

```bash
python3 -m bor_risk.cli --company "ACME" --out outputs/acme.txt --no-llm --sensitivity
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
| `--mode` | No | `pipeline` | Experimental condition: `pipeline`, `pipeline-web`, `agent`, `agent-web` |
| `--tier-depth` | No | 2 | Number of supplier tiers to expand |
| `--no-llm` | No | off | Use deterministic JSON fixtures instead of GPT-4o |
| `--suppliers-path` | No | — | Path to custom supplier JSON file |
| `--interactive` | No | off | Streaming output + human-in-the-loop approval |
| `--sensitivity` | No | off | Run weight/threshold sensitivity analysis |
| `--visualize` | No | off | Print Mermaid graph diagram and exit |
| `--budget-llm` | No | 20 | Max LLM calls (agent and pipeline-web modes) |
| `--budget-web` | No | 30 | Max web queries (agent-web and pipeline-web modes) |
| `--snapshot` | No | off | Use cached web results only (reproducible evaluation) |

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
| Hazard coverage | Scored pairs / (suppliers x 6) |
| LLM calls used | Count from BudgetTracker |
| Web queries used | Count from BudgetTracker |
| Wall clock time | End-to-end execution time |
| Unique web sources | Distinct URLs in evidence |
| Unverified fraction | `llm_only / total` suppliers |

### Tier 3: LLM-as-Judge (Supplementary)

GPT-4o rates narrative quality (1-5) on completeness, actionability, and risk communication. Used for report quality only — factual accuracy is measured by Tier 1.

### Running evaluations

```bash
# Evaluate ground-truth companies across all 4 conditions
python3 -m bor_risk.evaluate --ground-truth-only --out outputs/eval/

# Evaluate specific companies
python3 -m bor_risk.evaluate --companies "Apple,Toyota,Boeing" --out outputs/eval/

# Full 30-company benchmark
python3 -m bor_risk.evaluate --full --out outputs/eval/

# Reproducible run (snapshot mode — cached web results only)
python3 -m bor_risk.evaluate --full --snapshot --out outputs/eval/

# Custom budget
python3 -m bor_risk.evaluate --full --budget-llm 15 --budget-web 25 --out outputs/eval/

# Skip LLM judge (saves API calls during development)
python3 -m bor_risk.evaluate --ground-truth-only --skip-judge --out outputs/eval/
```

Evaluation outputs:
- `eval_results.json` — raw per-company, per-condition metrics (all 3 tiers)
- `eval_summary.txt` — comparison table with means across conditions

## Hazard Scoring Methodology

All hazard scores are deterministic and computed from real data — never from an LLM. Each raw score is normalised to 0.0–1.0, then mapped to 0–100 with risk levels: **Low** (0–33), **Medium** (34–66), **High** (67–100).

| Hazard | Data Source | Formula | Parameters |
|--------|------------|---------|------------|
| Earthquake | USGS FDSNWS event count API | `min(1, log10(1 + count) / 3)` | M4.0+ quakes within 200 km, 2015–2025 |
| Flood | Open-Meteo GloFAS river discharge | `min(1, days_above_2x_mean / 365)` | Daily discharge, days exceeding 2x mean |
| Wildfire | NASA FIRMS VIIRS (preprocessed grid) | `min(1, log10(1 + fire_count) / 4)` | 0.5-degree grid cells, annual fire detections |
| Cyclone | NOAA IBTrACS (preprocessed grid) | `min(1, storm_count / 50)` | 1-degree grid cells, storms >= 34 kt, 1980–2024 |
| Heat stress | Open-Meteo ERA5 reanalysis | `min(1, annual_extreme_days / 90)` | Days with apparent temp max > 35 C, 2015–2025 |
| Drought | Open-Meteo ERA5 precipitation | `min(1, dry_month_fraction * 2)` | Months below 50% of mean monthly precipitation |

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

185 tests across 14 test files:

```bash
python3 -m pytest -v
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_hazard_scoring.py` | 40 | Score normalisation, all 6 hazard APIs, edge cases, hash stub fallback |
| `test_enhanced_report.py` | 24 | Report sections, risk matrix, IEEE references, evidence appendix |
| `test_langgraph_features.py` | 21 | Graph topology, parallel fan-out/fan-in, HITL interrupts, streaming |
| `test_llm_discovery.py` | 18 | Company profile resolution, contextual prompts, deduplication |
| `test_sensitivity.py` | 13 | Pure function parity, scenario generation, weight/threshold perturbation |
| `test_agent_tools.py` | 11 | Tool building, budget enforcement, verify/unverify, relationship cues |
| `test_tier_expansion.py` | 10 | End-to-end graph run, CLI output files, supplier loading |
| `test_evaluation.py` | 10 | Ground truth loading, precision/recall, hard metrics, summary formatting |
| `test_suggest_alternatives.py` | 8 | Alternative supplier suggestions, LLM mock, report rendering |
| `test_budget.py` | 7 | BudgetTracker state, budget exhaustion, summary |
| `test_verification.py` | 6 | Pipeline batch verification, co-mention matching, evidence-source weighting |
| `test_mitigation_generation.py` | 6 | Mitigation prompts, priority levels, report integration |
| `test_workflow_decision.py` | 4 | Routing logic, risk band assignment |
| `test_search.py` | 3 | Snapshot mode, cache hits/misses, max_results |

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
    search_cache/              # Cached web search results (not tracked in git)
    raw/                       # Source data files (not tracked in git)
  scripts/
    download_data.py           # Downloads IBTrACS + finds FIRMS CSV
    preprocess_cyclone.py      # Converts IBTrACS CSV to 1-degree grid JSON
    preprocess_wildfire.py     # Converts FIRMS VIIRS CSV to 0.5-degree grid JSON
  src/bor_risk/
    cli.py                     # CLI entry point (--mode, --budget-llm, --budget-web, etc.)
    graph.py                   # LangGraph pipeline (15 nodes, fan-out/fan-in, conditional verify)
    agent.py                   # ReAct agent via create_react_agent
    agent_tools.py             # 8 LangChain @tool closures for the agent
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
