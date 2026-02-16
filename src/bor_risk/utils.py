"""Shared utilities."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


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


