# Agent Rules (must follow)

## Goal
Build a Python project that:
- takes a company name
- expands suppliers from Tier 1 to Tier X (with uncertainty + confidence)
- computes deterministic geospatial hazard indicators for supplier locations
- outputs a single business-ready text report with IEEE-style references
- uses LangGraph for orchestration and LangSmith tracing (if env vars present)

## Non-negotiables
- Keep the repo SIMPLE (do not add lots of folders).
- Prefer minimal dependencies.
- Do not introduce complex frameworks.
- Do not invent data sources or claims in the report without evidence items.
- Geospatial hazard scores must come from deterministic functions (no LLM “guessing”).

## Workflow rules
- Work in small steps; each step must run locally.
- After edits, run tests (pytest) and fix failures.
- Do not refactor unrelated code.
- If something is unclear, choose the simplest implementation that still meets the goal.

## Output expectations
- CLI works: `python -m bor_risk.cli --company "X" --tier-depth 3 --out outputs/x.txt`
- Produces:
  - outputs/<company>_report.txt
  - outputs/<company>_graph.json
  - outputs/<company>_evidence.jsonl

## Evidence rule
Every supplier relationship edge must include >=1 evidence_id.
Every hazard indicator must include dataset metadata (name/version/retrieved_at/method).
Report claims must reference evidence_ids.
