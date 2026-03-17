"""Shared utilities."""

from __future__ import annotations

import bisect
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_REFERENCE_SET_PATH = _DATA_DIR / "reference_set.json"

# ---------------------------------------------------------------------------
# Grid-key quantizer (half-away-from-zero)
# ---------------------------------------------------------------------------


def grid_key(v: float, decimals: int) -> int:
    """Half-away-from-zero integer grid key.

    Python's built-in round() uses banker's rounding (round-half-to-even), which
    means round(12.5) == 12 and round(-12.5) == -12. That creates inconsistencies
    at cell boundaries. This function always rounds half away from zero:
      grid_key(12.5, 0) == 13
      grid_key(-12.5, 0) == -13

    Must be imported and called identically in every preprocessing script and in
    every runtime lookup, or preprocessing and query may disagree on boundary cells.
    """
    factor = 10 ** decimals
    return int(math.floor(abs(v) * factor + 0.5) * (1 if v >= 0 else -1))


# ---------------------------------------------------------------------------
# Percentile-rank normalisation (explicit-boundary mid-rank empirical CDF)
# ---------------------------------------------------------------------------


def percentile_rank(raw: float, sorted_ref: list[float]) -> float:
    """Explicit-boundary mid-rank empirical CDF.

    Lowest reference value → 0.0; highest → 1.0; ties share average rank.
    Constant (all-equal) reference → 0.5 (non-informative; callers should warn).
    """
    n = len(sorted_ref)
    if n <= 1:
        return 0.5
    if sorted_ref[0] == sorted_ref[-1]:
        # Constant reference: hazard cannot discriminate suppliers.
        # Return 0.5 (neutral) rather than 0.0 (would systematically
        # drag all suppliers toward low exposure for this hazard).
        # Callers should log a warning: this hazard is non-informative.
        return 0.5
    if raw <= sorted_ref[0]:
        return 0.0
    if raw >= sorted_ref[-1]:
        return 1.0
    lo = bisect.bisect_left(sorted_ref, raw)
    hi = bisect.bisect_right(sorted_ref, raw)
    mid_rank = (lo + hi - 1) / 2.0   # 0-indexed average rank
    return mid_rank / (n - 1)


# ---------------------------------------------------------------------------
# Reference-set loading and validation
# ---------------------------------------------------------------------------

EXPECTED_HAZARDS = {"earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought"}


def validate_reference_set(rs: dict) -> None:
    """Validate reference_set.json structure. Called by load_reference_set().

    Raises ValueError on any structural problem so a stale or malformed file
    fails immediately at startup rather than silently producing wrong scores.
    """
    missing = EXPECTED_HAZARDS - set(rs.get("hazards", {}))
    if missing:
        raise ValueError(f"reference_set.json missing hazard arrays: {sorted(missing)}")
    for name, arr in rs["hazards"].items():
        if arr != sorted(arr):
            raise ValueError(f"reference_set.json hazard '{name}' array is not sorted ascending")
    for key in ("supplier_exposure_thresholds", "company_exposure_thresholds"):
        if key not in rs:
            raise ValueError(f"reference_set.json missing '{key}'")
        for sub in ("medium", "high"):
            if sub not in rs[key]:
                raise ValueError(f"reference_set.json['{key}'] missing '{sub}' threshold")
    if "metadata" not in rs or "corpus" not in rs.get("metadata", {}):
        raise ValueError("reference_set.json missing metadata.corpus field")


@lru_cache(maxsize=1)
def load_reference_set() -> dict:
    """Load and validate data/reference_set.json (cached)."""
    rs = json.loads(_REFERENCE_SET_PATH.read_text())
    validate_reference_set(rs)
    return rs


# ---------------------------------------------------------------------------
# Config loaders (cached)
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_company_name(name: str) -> str:
    """Lowercase, strip, replace non-alphanumerics with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, str]:
    """Load prompt templates from configs/prompts.yaml (cached)."""
    path = _CONFIGS_DIR / "prompts.yaml"
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def load_hazards() -> list[dict]:
    """Load hazard definitions from configs/hazards.yaml (cached)."""
    path = _CONFIGS_DIR / "hazards.yaml"
    return yaml.safe_load(path.read_text())["hazards"]
