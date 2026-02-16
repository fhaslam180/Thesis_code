# bor-risk-agent

A supply-chain risk analysis agent built with [LangGraph](https://github.com/langchain-ai/langgraph). Given a company name, the agent discovers suppliers across multiple tiers (via GPT-4o or deterministic fixtures), scores six geospatial hazards from real public data sources, aggregates risk with configurable weights, and produces an evidence-backed report with IEEE-style references.

The agent architecture demonstrates several LangGraph capabilities: parallel fan-out/fan-in for concurrent hazard scoring, human-in-the-loop interrupts for high-risk escalation decisions, checkpointed state for interrupt/resume workflows, and streaming node-level progress output.

## Architecture

The graph has 14 nodes arranged in a parallel fan-out/fan-in topology:

```
START
  -> discover_suppliers
       |-> score_earthquake --\
       |-> score_flood --------\
       |-> score_wildfire -------> aggregate_risk
       |-> score_cyclone ------/     -> decide_workflow (+ human-in-the-loop interrupt)
       |-> score_heat_stress -/          |-> high_risk_response --\
       |-> score_drought ----/           |-> monitoring_response ---> generate_mitigations
                                                                         -> suggest_alternatives
                                                                            -> format_report
                                                                               -> END
```

| Node | Role |
|------|------|
| `discover_suppliers` | Two-step contextual discovery: resolves company profile via GPT-4o, then discovers suppliers with industry context. Deduplicates across tiers. Falls back to JSON fixtures with `--no-llm` |
| `score_{hazard}` (x6) | Scores all suppliers for one hazard type (runs in parallel) |
| `aggregate_risk` | Weighted average of hazard scores with confidence and tier adjustments |
| `decide_workflow` | Routes to escalation or monitoring; pauses for human approval when interactive |
| `high_risk_response` | Builds P1/P2 escalation actions for high-risk cases |
| `monitoring_response` | Builds P3 monitoring actions for lower-risk cases |
| `generate_mitigations` | GPT-4o generates location-specific mitigation strategies (skipped with `--no-llm`) |
| `suggest_alternatives` | GPT-4o suggests lower-risk alternative suppliers for high-risk nodes (skipped with `--no-llm`) |
| `format_report` | Produces the final plain-text report with all sections |

The `hazard_scores` and `workflow_trace` fields use `Annotated[list, operator.add]` reducers so the six parallel scorer nodes can each return partial results that LangGraph merges automatically at the fan-in point.

## Supplier Discovery

The `discover_suppliers` node uses a two-step contextual discovery pipeline when running with GPT-4o:

**Step 1: Company profile resolution** — A single GPT-4o call resolves the input company name to a structured profile containing: canonical name, industry sector, main products, headquarters location, and description. This profile is stored in the graph state and rendered in the report.

**Step 2: Contextual tier expansion** — For each (parent, tier) pair, GPT-4o discovers suppliers with the company's industry and products injected into the prompt. This context enables more accurate supplier identification. Each discovered supplier includes:

| Field | Description |
|-------|-------------|
| `name` | Company name |
| `lat` / `lon` | Primary facility coordinates |
| `confidence` | Supplier relationship confidence (0–1) |
| `rationale` | Why this is a plausible supplier |
| `industry` | Supplier's industry sector |
| `product_category` | What they supply (e.g. "Silicon wafers") |
| `location_description` | City, country of facility |
| `relationship_type` | One of: `raw_material`, `component`, `service`, `logistics` |

**Deduplication** — After multi-tier expansion, suppliers appearing from multiple parents are merged: the entry with the highest confidence is kept, and evidence IDs from all occurrences are combined. Matching is case-insensitive.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For GPT-powered supplier discovery and mitigation generation, create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

### Data preprocessing (one-time)

Wildfire and cyclone scoring use preprocessed local grid files. To regenerate from raw sources:

```bash
python3 scripts/download_data.py
```

This downloads IBTrACS cyclone data from NOAA and processes FIRMS VIIRS fire data into the `data/` grid files. The preprocessed grids are included in the repository, so this step is only needed if you want to update the source data.

## Usage

### Basic run (with LLM)

```bash
python3 -m bor_risk.cli --company "ACME" --tier-depth 2 --out outputs/acme.txt
```

### Deterministic run (no API key needed)

```bash
python3 -m bor_risk.cli --company "ACME" --tier-depth 2 --out outputs/acme.txt --no-llm
```

### Interactive mode (streaming + human-in-the-loop)

```bash
python3 -m bor_risk.cli --company "ACME" --tier-depth 2 --out outputs/acme.txt --interactive
```

Displays real-time progress as each node completes. When a high-risk decision is reached, prompts for human approval before escalating.

### With sensitivity analysis

```bash
python3 -m bor_risk.cli --company "ACME" --tier-depth 2 --out outputs/acme.txt --no-llm --sensitivity
```

Appends a sensitivity analysis table to the report, showing how 11 weight/threshold variations affect the risk scores.

### Graph visualisation

```bash
python3 -m bor_risk.cli --company "ACME" --out outputs/acme.txt --visualize
```

Prints a Mermaid diagram of the graph topology and exits.

### CLI flags reference

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--company` | Yes | — | Target company name |
| `--out` | Yes | — | Output path (directory derived from parent) |
| `--tier-depth` | No | 2 | Number of supplier tiers to expand |
| `--no-llm` | No | off | Use deterministic JSON fixtures instead of GPT-4o |
| `--suppliers-path` | No | — | Path to custom supplier JSON file |
| `--interactive` | No | off | Streaming output + human-in-the-loop approval |
| `--sensitivity` | No | off | Run weight/threshold sensitivity analysis |
| `--visualize` | No | off | Print Mermaid graph diagram and exit |

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

## Data Sources

| Dataset | Provider | Access | Period | Resolution |
|---------|----------|--------|--------|------------|
| USGS Earthquake Catalog | USGS | REST API (no key) | 2015–2025 | Point query, 200 km radius |
| GloFAS River Discharge | Copernicus via Open-Meteo | REST API (no key) | 2015–2025 | ~5 km grid |
| ERA5 Temperature | ECMWF via Open-Meteo | REST API (no key) | 2015–2025 | ~25 km grid |
| ERA5 Precipitation | ECMWF via Open-Meteo | REST API (no key) | 2015–2025 | ~25 km grid |
| VIIRS Active Fire | NASA FIRMS | Preprocessed local JSON | Annual aggregate | 0.5-degree grid |
| IBTrACS Cyclones | NOAA NCEI | Preprocessed local JSON | 1980–2024 | 1-degree grid |

All REST API functions include SSL certificate fallback for environments where CA verification fails.

## Risk Aggregation

The `aggregate_risk` node computes supplier and company risk scores in three steps:

**1. Weighted hazard average per supplier:**

```
base_score = sum(score_i * weight_i) / sum(weight_i)
```

Default weights from `configs/hazards.yaml`: earthquake 0.25, flood 0.20, wildfire 0.15, cyclone 0.15, heat_stress 0.10, drought 0.15.

**2. Confidence and tier adjustment:**

```
confidence_factor = 0.5 + (0.5 * confidence)     # range: 0.5 - 1.0
tier_factor = max(0.8, 1.0 - 0.05 * (tier - 1))  # tier 1: 1.0, tier 2: 0.95, tier 3+: 0.8
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
| `{company}_report.txt` | Plain text | Executive summary, company profile, enriched supplier risk ranking, threshold alerts, risk register matrix, hazard summary, mitigations, suggested alternatives, evidence appendix, IEEE references |
| `{company}_graph.json` | JSON | Suppliers, edges, company risk summary, workflow decision and actions, workflow trace |
| `{company}_evidence.jsonl` | JSONL | One evidence item per line (source, description, retrieval timestamp) |

With `--sensitivity`, an additional `{company}_sensitivity.json` is produced containing per-scenario results.

## Configuration

### `configs/hazards.yaml`

Defines the six hazard types with weights (used in aggregation) and thresholds (used for alert generation):

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

Weights are normalised during aggregation, so they represent relative importance rather than absolute values.

### `configs/prompts.yaml`

Contains LLM prompt templates for company profile resolution, supplier discovery, mitigation strategy generation, and alternative supplier suggestions. Each template uses Python string formatting placeholders.

## Testing

144 tests across 9 test files:

```bash
python3 -m pytest -v
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_hazard_scoring.py` | 40 | Score normalisation, all 6 hazard APIs, edge cases, hash stub fallback |
| `test_enhanced_report.py` | 24 | Report sections, risk matrix, IEEE references, evidence appendix |
| `test_langgraph_features.py` | 21 | Graph topology, parallel fan-out/fan-in, HITL interrupts, streaming, backward compatibility |
| `test_llm_discovery.py` | 18 | Company profile resolution, contextual prompts, enriched supplier fields, deduplication, graph integration |
| `test_sensitivity.py` | 13 | Pure function parity, scenario generation, weight/threshold perturbation, CLI flag |
| `test_tier_expansion.py` | 10 | End-to-end graph run, CLI output files, supplier loading |
| `test_suggest_alternatives.py` | 8 | Alternative supplier suggestions, LLM mock, report rendering |
| `test_mitigation_generation.py` | 6 | Mitigation prompts, priority levels, report integration |
| `test_workflow_decision.py` | 4 | Routing logic, risk band assignment |

### Mock strategy

An `autouse` fixture in `conftest.py` patches `urllib.request.urlopen` with URL-aware routing:

- `earthquake.usgs.gov` requests return location-aware seismicity counts (high for Pacific Ring of Fire latitudes, low for stable regions)
- `flood-api.open-meteo.com` returns canned GloFAS river discharge data
- `archive-api.open-meteo.com` returns canned ERA5 temperature or precipitation data depending on the query parameter
- Wildfire and cyclone tests use the actual preprocessed grid files in `data/`

LLM tests mock `ChatOpenAI` at the class level with pre-built structured output responses.

## Project Structure

```
bor-risk-agent/
  configs/
    hazards.yaml          # Hazard weights and thresholds
    prompts.yaml           # LLM prompt templates
  data/
    wildfire_grid.json     # Preprocessed 0.5-degree fire grid (NASA FIRMS)
    cyclone_grid.json      # Preprocessed 1-degree cyclone grid (NOAA IBTrACS)
    raw/                   # Source data files (not tracked in git)
  scripts/
    download_data.py       # Downloads IBTrACS + finds FIRMS CSV
    preprocess_cyclone.py  # Converts IBTrACS CSV to 1-degree grid JSON
    preprocess_wildfire.py # Converts FIRMS VIIRS CSV to 0.5-degree grid JSON
  src/bor_risk/
    cli.py                 # CLI entry point with all flags
    graph.py               # LangGraph workflow (14 nodes, fan-out/fan-in)
    tools.py               # Hazard scoring, LLM discovery, risk aggregation
    models.py              # Pydantic models and GraphState TypedDict
    report.py              # Plain-text report formatting
    sensitivity.py         # Weight/threshold sensitivity analysis
    utils.py               # Config loaders (cached YAML)
  tests/
    conftest.py            # URL-aware mock fixtures
    fixtures/
      mock_suppliers.json  # Deterministic supplier data (ACME, GlobalMfg)
    test_*.py              # 144 tests across 9 files
  outputs/                 # Generated reports (not tracked in git)
```
