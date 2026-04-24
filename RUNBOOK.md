# Reproducibility Runbook — Apple SMHEI Result

This document records how the committed Apple result was produced and how to reproduce it.

---

## 1. Environment

```bash
python3 --version     # 3.11.x
pip install -e .      # install bor_risk package from source
cp .env.example .env  # add OPENAI_API_KEY and TAVILY_API_KEY
```

---

## 2. Frozen Artifacts

Three files define the normalization basis. They must be committed to the repository and must not be regenerated mid-study. Commit them immediately after any build with:

```bash
git add data/reference_locations.json data/reference_set.json data/api_hazard_cache.json
git commit -m "chore: freeze reference corpus and hazard cache"
```

| File | Role | Notes |
|------|------|-------|
| `data/reference_locations.json` | Corpus of 20 Apple supplier locations | Collected 2026-04-04 via `llm_only` pipeline |
| `data/reference_set.json` | Percentile-rank reference arrays + exposure thresholds | Built 2026-04-12 from the above |
| `data/api_hazard_cache.json` | Open-Meteo API results for resolved grid cells | See §3 for coverage |

**Do not delete or overwrite these files.** The exposure indices in any published result are relative to this corpus.

---

## 3. API Cache Coverage and Reproducibility

The committed `api_hazard_cache.json` currently contains **heat_stress only** (87 grid cells). Wildfire and drought values were not pre-warmed before the reference-set build (the prewarm script had a JSON-parsing bug, now fixed). Wildfire and drought scores for the Apple run were computed from live Open-Meteo API calls.

**Consequence**: rerunning the Apple pipeline as-is will make fresh wildfire and drought API calls. Results may differ slightly if ERA5 data or Open-Meteo API behavior changes.

**To fully freeze wildfire and drought** (recommended before thesis submission):

```bash
python3 scripts/prewarm_hazard_cache.py \
    --locations data/reference_locations.json
# Then commit the updated cache:
git add data/api_hazard_cache.json
git commit -m "chore: add wildfire and drought to hazard cache"
```

After prewarming, all three API-based hazards will be served from cache on every subsequent run.

---

## 4. Reference Corpus Scope

The corpus was collected for **Apple only** (1 company, 20 supplier locations).

Implications:
- **Supplier-level thresholds** (`supplier_exposure_thresholds`) are based on 20 suppliers and are valid for ranking Apple's suppliers against each other.
- **Company-level thresholds** (`company_exposure_thresholds`) are based on 1 company mean. With a single data point both tertile boundaries collapse to the same value (medium == high == 0.235088). This is mathematically expected with a single-company corpus; the company band should be treated as indicative.
- **Non-informative hazards**: earthquake, flood, and cyclone reference arrays are all-zero (SQLite grid DBs were not populated). These receive a neutral score of 0.5 and are excluded from the composite E_s. Only wildfire, heat_stress, and drought are informative.

---

## 5. Rebuilding from Scratch (if corpus changes)

> Only do this if the study corpus genuinely changes. Regenerating these files mid-study invalidates comparability with prior results.

### Phase 1 — Prewarm API cache

```bash
python3 scripts/prewarm_hazard_cache.py \
    --locations data/reference_locations.json
```

Pre-fetches Open-Meteo values for all supplier locations. API failures during prewarm are safe (they will be retried during the build). The prewarm script now parses the company-grouped JSON format correctly.

### Phase 2 — Build reference set

```bash
python3 scripts/build_reference_set.py \
    --locations data/reference_locations.json \
    --out data/reference_set.json
```

The reference-set builder uses `_strict=True` when calling the API fetchers: transient network failures now raise `RuntimeError` and skip the location cleanly, rather than silently accepting 0.0 as a valid zero-exposure value. Company-level E_s slices are computed by actual success count (not attempted count).

Commit all three frozen files immediately after a successful build.

---

## 6. Apple Run

The committed `outputs/apple_report.txt` was produced by:

```bash
python3 -m bor_risk.cli \
    --company "Apple" \
    --no-web \
    --out outputs/apple_report.txt
```

`--no-web` uses GPT-4o for supplier discovery only (no web retrieval or claim verification). Hazard scoring is independent of this flag.

---

## 7. Reproducibility Scope

| Component | Reproducible? | Condition |
|-----------|--------------|-----------|
| Earthquake, flood, cyclone scores | Yes | Always 0.0 (SQLite DBs absent; non-informative) |
| Heat stress scores | Yes (given cache) | 87 grid cells in committed cache |
| Wildfire scores | No (live API) | Not yet cached; prewarm needed |
| Drought scores | No (live API) | Not yet cached; prewarm needed |
| Company E_s, E_s bands | Yes (given above) | Pure arithmetic on hazard scores |
| Company band | Indicative | Single-company corpus → degenerate thresholds |
| Supplier list | Approximately | GPT-4o at temperature=0; not bit-reproducible |
| Claim verification | Approximately | LLM-based; same caveat |

**What is being defended**: the SMHEI hazard computation — the percentile-ranking methodology, the 3-hazard composition (wildfire, heat_stress, drought), and the supplier exposure band assignments. The supplier discovery step is model-dependent and is acknowledged as a limitation.

---

## 8. Verification

```bash
python3 -m pytest -q
```

All tests must pass. Key groups:
- `tests/test_hazard_scoring.py` — includes successful-path tests for heat, wildfire, drought
- `tests/test_reference_set_build.py` — includes regression for company-slicing bug
- `tests/test_normalization.py`, `tests/test_grid_lookup.py` — percentile-rank and grid-key correctness
- `tests/test_sensitivity.py` — SMHEI sensitivity scenarios

---

## 9. Known Limitations (for thesis defence)

1. **Company-level band is indicative**: single-company corpus → degenerate thresholds (medium == high). The more meaningful metric is the per-supplier E_s distribution.
2. **Three hazards non-informative**: earthquake, flood, cyclone all score 0.5 (neutral) because the SQLite grid DBs were not loaded. The composite is effectively a 3-hazard index.
3. **Wildfire/drought not cached**: live API calls on rerun; results may vary if Open-Meteo updates ERA5 data. Prewarm before submission.
4. **Supplier discovery is LLM-dependent**: the 104 Apple suppliers are GPT-4o outputs and vary across runs. The hazard scores assigned to each named supplier are deterministic; the set of names is not.
