"""Tests for hazard scoring across all hazard types (new SMHEI scorers)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from bor_risk.models import Supplier
from bor_risk.tools import _normalize_score, compute_hazard


def _make_supplier(
    name: str = "TestCorp",
    lat: float = 0.0,
    lon: float = 0.0,
) -> Supplier:
    return Supplier(
        name=name, lat=lat, lon=lon,
        tier=1, confidence=0.8, evidence_ids=["E_TEST"],
    )


# -------------------------------------------------------------------
# Score normalisation tests
# -------------------------------------------------------------------


class TestNormalizeScore:
    def test_zero(self):
        assert _normalize_score(0.0) == (0, "Low")

    def test_low_boundary(self):
        assert _normalize_score(0.33) == (33, "Low")

    def test_medium_start(self):
        assert _normalize_score(0.34) == (34, "Medium")

    def test_medium_boundary(self):
        assert _normalize_score(0.66) == (66, "Medium")

    def test_high_start(self):
        assert _normalize_score(0.67) == (67, "High")

    def test_max(self):
        assert _normalize_score(1.0) == (100, "High")

    def test_mid_value(self):
        s100, level = _normalize_score(0.5)
        assert s100 == 50
        assert level == "Medium"


# -------------------------------------------------------------------
# Earthquake scoring (GEM 2023 PGA, SQLite)
# -------------------------------------------------------------------


class TestEarthquakeScoring:
    """Earthquake scorer returns 0.0 when DB is absent (logged warning)."""

    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_raw_value_field_present(self):
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert hasattr(score, "raw_value")
        assert isinstance(score.raw_value, float)

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        meta = score.dataset_metadata
        assert meta["dataset"] == "GEM_2023"
        assert meta["return_period_yr"] == 475
        assert "source" in meta

    def test_no_db_gives_zero_raw(self):
        # DB absent → pga_g = 0.0 → percentile_rank(0.0, ref) = 0.0
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.raw_value == 0.0
        assert score.score == 0.0


# -------------------------------------------------------------------
# Flood scoring (WRI Aqueduct, SQLite, 0.01° grid)
# -------------------------------------------------------------------


class TestFloodScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        meta = score.dataset_metadata
        assert meta["dataset"] == "Aqueduct_Floods"
        assert meta["return_period_yr"] == 100
        assert meta["grid_resolution_deg"] == 0.01
        assert "depth_m" in meta

    def test_no_db_gives_zero_raw(self):
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.raw_value == 0.0


# -------------------------------------------------------------------
# Heat stress scoring (Open-Meteo ERA5, 32°C threshold)
# -------------------------------------------------------------------


class TestHeatStressScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        meta = score.dataset_metadata
        assert meta["dataset"] == "Open-Meteo ERA5"
        assert meta["heat_metric"] == "apparent_temperature_max >= 32C"
        assert "threshold_note" in meta
        assert "metric_source" in meta

    def test_no_extreme_days_gives_zero_fraction(self, mock_urlopen):
        cold = json.dumps({
            "daily": {"apparent_temperature_max": [10.0, 12.0, 8.0, 15.0]},
        }).encode()
        from conftest import _make_mock_response
        mock_urlopen.side_effect = lambda url, *a, **kw: _make_mock_response(cold)
        score = compute_hazard(_make_supplier(lat=99.0, lon=99.0), hazard_type="heat_stress")
        assert score.raw_value == 0.0

    def test_api_failure_returns_zero(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            score = compute_hazard(_make_supplier(lat=88.0, lon=88.0), hazard_type="heat_stress")
        assert score.raw_value == 0.0
        assert score.score == 0.0

    def test_successful_path_gives_correct_fraction(self):
        """Mock provides 5/20 days >= 32 C → heat fraction = 0.25."""
        score = compute_hazard(_make_supplier(lat=1.0, lon=1.0), hazard_type="heat_stress")
        assert score.raw_value == pytest.approx(0.25)
        assert score.score > 0.0


# -------------------------------------------------------------------
# Drought scoring (SPEI-12, ERA5)
# -------------------------------------------------------------------


class TestDroughtScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="drought")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="drought")
        meta = score.dataset_metadata
        assert meta["dataset"] == "Open-Meteo ERA5"
        assert meta["indicator"] == "SPEI-12"
        assert "Vicente-Serrano" in meta["source"]

    def test_api_failure_returns_zero(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            score = compute_hazard(_make_supplier(lat=77.0, lon=77.0), hazard_type="drought")
        assert score.raw_value == 0.0
        assert score.score == 0.0

    def test_successful_path_runs_spei_to_completion(self):
        """Mock provides 200 months with drought years → SPEI-12 fraction > 0."""
        score = compute_hazard(_make_supplier(lat=3.0, lon=3.0), hazard_type="drought")
        assert score.raw_value > 0.0
        assert 0.0 <= score.score <= 1.0


# -------------------------------------------------------------------
# Wildfire scoring (Approx. FWI Q95, ERA5 daily stats)
# -------------------------------------------------------------------


class TestWildfireScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="wildfire")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="wildfire")
        meta = score.dataset_metadata
        assert meta["dataset"] == "Open-Meteo ERA5"
        assert meta["fwi_computation"] == "daily_stats_approximation"
        assert "Approx. FWI Q95" in meta["indicator"]
        assert "Van Wagner" in meta["method"]

    def test_api_failure_returns_zero(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            score = compute_hazard(_make_supplier(lat=66.0, lon=66.0), hazard_type="wildfire")
        assert score.raw_value == 0.0
        assert score.score == 0.0

    def test_successful_path_gives_nonzero_fwi(self):
        """Mock provides 100 days of fire conditions → FWI Q95 > 0."""
        score = compute_hazard(_make_supplier(lat=2.0, lon=2.0), hazard_type="wildfire")
        assert score.raw_value > 0.0
        assert 0.0 <= score.score <= 1.0


# -------------------------------------------------------------------
# Cyclone scoring (STORM dataset, SQLite)
# -------------------------------------------------------------------


class TestCycloneScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="cyclone")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata(self):
        score = compute_hazard(_make_supplier(), hazard_type="cyclone")
        meta = score.dataset_metadata
        assert meta["dataset"] == "STORM_2020"
        assert meta["return_period_yr"] == 100
        assert "Bloemendaal" in meta["source"]
        assert "v100_kmh" in meta

    def test_no_db_gives_zero_raw(self):
        score = compute_hazard(_make_supplier(), hazard_type="cyclone")
        assert score.raw_value == 0.0


# -------------------------------------------------------------------
# Hash-stub fallback tests (unknown hazard types)
# -------------------------------------------------------------------


class TestHashStubScoring:
    def test_returns_valid_hazard_score(self):
        score = compute_hazard(_make_supplier(), hazard_type="unknown_hazard")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0

    def test_dataset_metadata_populated(self):
        score = compute_hazard(_make_supplier(), hazard_type="unknown_hazard")
        assert score.dataset_metadata["dataset"] == "stub_unknown_hazard_v1"
        assert score.dataset_metadata["method"] == "sha256_hash_stub"

    def test_deterministic(self):
        s = _make_supplier("DeterministicCo")
        a = compute_hazard(s, "unknown_hazard")
        b = compute_hazard(s, "unknown_hazard")
        assert a.score == b.score

    def test_different_hazard_types_differ(self):
        s = _make_supplier()
        a = compute_hazard(s, "unknown_a")
        b = compute_hazard(s, "unknown_b")
        assert a.score != b.score
