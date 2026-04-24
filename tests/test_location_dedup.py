"""Tests for location deduplication and cache key normalization."""

import pytest
import bor_risk.tools as _tools
from bor_risk.utils import grid_key


# ---------------------------------------------------------------------------
# Cache isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_api_cache(tmp_path):
    """Redirect API cache to a temp file and reset in-memory state between tests."""
    orig_path = _tools._API_CACHE_PATH
    orig_cache = _tools._api_cache
    _tools._API_CACHE_PATH = tmp_path / "test_cache.json"
    _tools._api_cache = None  # force re-init from the (empty) temp file
    yield
    _tools._API_CACHE_PATH = orig_path
    _tools._api_cache = orig_cache


# ---------------------------------------------------------------------------
# Test 1: Cache key normalization uses grid_key, not round()
# ---------------------------------------------------------------------------


def test_same_01_cell_is_cache_hit():
    """Two coords in the same 0.1° cell share a cache entry."""
    # grid_key(22.45, 1) == grid_key(22.54, 1) == 225
    assert grid_key(22.45, 1) == grid_key(22.54, 1) == 225
    _tools._api_cache_store("heat_stress", 22.45, 114.0, "apparent_temp_32_v1", 0.42)
    assert _tools._api_cache_lookup("heat_stress", 22.54, 114.0, "apparent_temp_32_v1") == 0.42


def test_different_01_cell_is_cache_miss():
    """Two coords on opposite sides of a 0.1° boundary do not share a cache entry."""
    # grid_key(22.45, 1) == 225; grid_key(22.55, 1) == 226 (half-away-from-zero)
    assert grid_key(22.45, 1) == 225
    assert grid_key(22.55, 1) == 226
    _tools._api_cache_store("heat_stress", 22.45, 114.0, "apparent_temp_32_v1", 0.42)
    assert _tools._api_cache_lookup("heat_stress", 22.55, 114.0, "apparent_temp_32_v1") is None


# ---------------------------------------------------------------------------
# Test 2: Hazard-specific dedup in _make_hazard_scorer
# ---------------------------------------------------------------------------


def _sup(name: str, lat: float, lon: float) -> dict:
    return {
        "name": name, "lat": lat, "lon": lon, "tier": 1,
        "confidence": 0.8, "evidence_source": "llm_only",
        "industry": "", "product_category": "", "location_description": "",
        "relationship_type": "", "verification_url": "", "verification_snippet": "",
        "evidence_ids": [],
    }


def test_earthquake_deduplicates_at_01_degree():
    """Two suppliers in the same 0.1° cell trigger only one compute_hazard call."""
    from unittest.mock import patch, MagicMock
    from bor_risk.graph import _make_hazard_scorer

    # grid_key(24.85, 1) == grid_key(24.87, 1) == 249
    assert grid_key(24.85, 1) == grid_key(24.87, 1) == 249

    scorer = _make_hazard_scorer("earthquake")
    state = {"suppliers": [_sup("A", 24.85, 121.0), _sup("B", 24.87, 121.02)]}

    fake = MagicMock()
    fake.model_dump.return_value = {
        "score": 0.5, "supplier_name": "A", "hazard_type": "earthquake",
        "level": "Low", "score_100": 50, "dataset_metadata": {},
    }

    with patch("bor_risk.graph.compute_hazard", return_value=fake) as mock_compute:
        result = scorer(state)

    assert mock_compute.call_count == 1        # 1 unique 0.1° cell
    assert len(result["hazard_scores"]) == 2   # but 2 score entries (one per supplier)


def test_flood_deduplicates_at_001_degree():
    """Two suppliers in different 0.01° cells each trigger a compute_hazard call."""
    from unittest.mock import patch, MagicMock
    from bor_risk.graph import _make_hazard_scorer

    # grid_key(24.851, 2) == 2485; grid_key(24.857, 2) == 2486 (different cells)
    assert grid_key(24.851, 2) == 2485
    assert grid_key(24.857, 2) == 2486

    scorer = _make_hazard_scorer("flood")
    state = {"suppliers": [_sup("A", 24.851, 121.0), _sup("B", 24.857, 121.0)]}

    fake = MagicMock()
    fake.model_dump.return_value = {
        "score": 0.3, "supplier_name": "A", "hazard_type": "flood",
        "level": "Low", "score_100": 30, "dataset_metadata": {},
    }

    with patch("bor_risk.graph.compute_hazard", return_value=fake) as mock_compute:
        result = scorer(state)

    assert mock_compute.call_count == 2        # 2 distinct 0.01° cells
    assert len(result["hazard_scores"]) == 2


# ---------------------------------------------------------------------------
# Test 3: Prewarm deduplicate_locations helper
# ---------------------------------------------------------------------------


def test_prewarm_deduplicates_by_01_degree():
    """deduplicate_locations returns one representative per 0.1° cell."""
    from scripts.prewarm_hazard_cache import deduplicate_locations

    locations = [
        {"name": "A", "lat": 22.45, "lon": 114.0},   # grid_key → 225, 1140
        {"name": "B", "lat": 22.54, "lon": 114.0},   # grid_key → 225, 1140 — same cell as A
        {"name": "C", "lat": 22.55, "lon": 114.0},   # grid_key → 226, 1140 — different cell
    ]
    unique = deduplicate_locations(locations)
    assert len(unique) == 2
