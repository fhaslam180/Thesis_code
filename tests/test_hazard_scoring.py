"""Tests for hazard scoring across all hazard types."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import pytest

from bor_risk.models import Supplier
from bor_risk.tools import _normalize_score, compute_hazard

from conftest import _make_mock_response


def _make_supplier(
    name: str = "TestCorp",
    lat: float = 0.0,
    lon: float = 0.0,
) -> Supplier:
    return Supplier(
        name=name,
        lat=lat,
        lon=lon,
        tier=1,
        confidence=0.8,
        evidence_ids=["E_TEST"],
    )


def _mock_urlopen_count(count: int):
    """Return a side_effect that always returns *count* for any URL."""
    return lambda url, *a, **kw: _make_mock_response(str(count).encode())


# -------------------------------------------------------------------
# Score normalisation tests
# -------------------------------------------------------------------


class TestNormalizeScore:
    """Tests for the _normalize_score helper."""

    def test_zero(self) -> None:
        assert _normalize_score(0.0) == (0, "Low")

    def test_low_boundary(self) -> None:
        assert _normalize_score(0.33) == (33, "Low")

    def test_medium_start(self) -> None:
        assert _normalize_score(0.34) == (34, "Medium")

    def test_medium_boundary(self) -> None:
        assert _normalize_score(0.66) == (66, "Medium")

    def test_high_start(self) -> None:
        assert _normalize_score(0.67) == (67, "High")

    def test_max(self) -> None:
        assert _normalize_score(1.0) == (100, "High")

    def test_mid_value(self) -> None:
        score_100, level = _normalize_score(0.5)
        assert score_100 == 50
        assert level == "Medium"


# -------------------------------------------------------------------
# USGS earthquake scoring tests
# -------------------------------------------------------------------


class TestUSGSEarthquakeScoring:
    """Tests for earthquake scoring via the USGS API (mocked)."""

    def test_earthquake_uses_usgs_metadata(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.dataset_metadata["dataset"] == "usgs_earthquake_catalog"
        assert score.dataset_metadata["method"] == "usgs_fdsnws_count"

    def test_has_score_100_and_level(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_zero_count_gives_zero_score(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = _mock_urlopen_count(0)
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.score == 0.0
        assert score.score_100 == 0
        assert score.level == "Low"

    def test_high_count_gives_max_score(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = _mock_urlopen_count(1072)
        score = compute_hazard(
            _make_supplier("TokyoCo", lat=35.67, lon=139.65),
            hazard_type="earthquake",
        )
        assert score.score == 1.0
        assert score.score_100 == 100
        assert score.level == "High"

    def test_moderate_count_gives_mid_score(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = _mock_urlopen_count(46)
        score = compute_hazard(
            _make_supplier("SFCo", lat=37.77, lon=-122.42),
            hazard_type="earthquake",
        )
        expected = round(min(1.0, math.log10(1 + 46) / 3.0), 4)
        assert score.score == expected

    def test_passes_correct_lat_lon_to_api(self, mock_urlopen) -> None:
        supplier = _make_supplier("GeoTest", lat=51.51, lon=-0.13)
        compute_hazard(supplier, hazard_type="earthquake")
        url = mock_urlopen.call_args[0][0]
        assert "latitude=51.51" in url
        assert "longitude=-0.13" in url

    def test_api_failure_returns_zero_score(self) -> None:
        """If USGS API fails, score should default to 0."""
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.score == 0.0
        assert score.score_100 == 0

    def test_earthquake_count_in_metadata(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = _mock_urlopen_count(46)
        score = compute_hazard(_make_supplier(), hazard_type="earthquake")
        assert score.dataset_metadata["earthquake_count"] == 46


# -------------------------------------------------------------------
# Flood scoring tests (Open-Meteo GloFAS)
# -------------------------------------------------------------------


class TestFloodScoring:
    """Tests for flood scoring via the Open-Meteo GloFAS API (mocked)."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_dataset_metadata(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.dataset_metadata["dataset"] == "glofas_river_discharge"
        assert score.dataset_metadata["method"] == "open_meteo_flood_api"
        assert "mean_discharge_m3s" in score.dataset_metadata
        assert "days_above_2x_mean" in score.dataset_metadata

    def test_api_failure_returns_zero(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.score == 0.0
        assert score.score_100 == 0

    def test_no_discharge_gives_zero(self, mock_urlopen) -> None:
        empty = json.dumps({"daily": {"river_discharge": []}}).encode()
        mock_urlopen.side_effect = lambda url, *a, **kw: _make_mock_response(empty)
        score = compute_hazard(_make_supplier(), hazard_type="flood")
        assert score.score == 0.0


# -------------------------------------------------------------------
# Heat stress scoring tests (Open-Meteo ERA5)
# -------------------------------------------------------------------


class TestHeatStressScoring:
    """Tests for heat stress scoring via Open-Meteo ERA5 (mocked)."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_dataset_metadata(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        assert score.dataset_metadata["dataset"] == "era5_reanalysis"
        assert score.dataset_metadata["method"] == "open_meteo_historical_api"
        assert score.dataset_metadata["extreme_heat_threshold_c"] == 35

    def test_no_extreme_days_gives_zero(self, mock_urlopen) -> None:
        cold = json.dumps({
            "daily": {"apparent_temperature_max": [10.0, 12.0, 8.0, 15.0]},
        }).encode()
        mock_urlopen.side_effect = lambda url, *a, **kw: _make_mock_response(cold)
        score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        assert score.score == 0.0

    def test_api_failure_returns_zero(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            score = compute_hazard(_make_supplier(), hazard_type="heat_stress")
        assert score.score == 0.0


# -------------------------------------------------------------------
# Drought scoring tests (Open-Meteo ERA5 precipitation)
# -------------------------------------------------------------------


class TestDroughtScoring:
    """Tests for drought scoring via Open-Meteo ERA5 precipitation (mocked)."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="drought")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_dataset_metadata(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="drought")
        assert score.dataset_metadata["dataset"] == "era5_precipitation"
        assert score.dataset_metadata["method"] == "dry_month_fraction"
        assert "mean_monthly_precip_mm" in score.dataset_metadata
        assert "dry_months" in score.dataset_metadata

    def test_api_failure_returns_zero(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            score = compute_hazard(_make_supplier(), hazard_type="drought")
        assert score.score == 0.0


# -------------------------------------------------------------------
# Wildfire scoring tests (local FIRMS grid)
# -------------------------------------------------------------------


class TestWildfireScoring:
    """Tests for wildfire scoring via preprocessed NASA FIRMS grid."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(
            _make_supplier("SFCo", lat=37.7749, lon=-122.4194),
            hazard_type="wildfire",
        )
        assert score.supplier_name == "SFCo"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100
        assert score.level in ("Low", "Medium", "High")

    def test_dataset_metadata(self) -> None:
        score = compute_hazard(
            _make_supplier("SFCo", lat=37.7749, lon=-122.4194),
            hazard_type="wildfire",
        )
        assert score.dataset_metadata["dataset"] == "viirs_active_fire_annual"
        assert score.dataset_metadata["method"] == "grid_cell_fire_count"
        assert "fire_count" in score.dataset_metadata

    def test_no_fire_area_gives_zero(self) -> None:
        score = compute_hazard(
            _make_supplier("OceanCo", lat=0.0, lon=0.0),
            hazard_type="wildfire",
        )
        assert score.score == 0.0
        assert score.score_100 == 0

    def test_high_fire_area_gives_positive_score(self) -> None:
        score = compute_hazard(
            _make_supplier("LACo", lat=34.0, lon=-118.0),
            hazard_type="wildfire",
        )
        assert score.score > 0.0
        assert score.score_100 > 0

    def test_formula_log_scaled(self) -> None:
        score = compute_hazard(
            _make_supplier("LACo", lat=34.0, lon=-118.0),
            hazard_type="wildfire",
        )
        count = score.dataset_metadata["fire_count"]
        expected = round(min(1.0, math.log10(1 + count) / 4.0), 4)
        assert score.score == expected


# -------------------------------------------------------------------
# Cyclone scoring tests (local IBTrACS grid)
# -------------------------------------------------------------------


class TestCycloneScoring:
    """Tests for cyclone scoring via preprocessed NOAA IBTrACS grid."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(
            _make_supplier("ManilaCo", lat=14.0, lon=121.0),
            hazard_type="cyclone",
        )
        assert score.supplier_name == "ManilaCo"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100

    def test_dataset_metadata(self) -> None:
        score = compute_hazard(
            _make_supplier("ManilaCo", lat=14.0, lon=121.0),
            hazard_type="cyclone",
        )
        assert score.dataset_metadata["dataset"] == "ibtracs"
        assert score.dataset_metadata["method"] == "grid_cell_storm_count"
        assert "storm_count" in score.dataset_metadata
        assert "max_wind_kt" in score.dataset_metadata

    def test_no_cyclone_area_gives_zero(self) -> None:
        score = compute_hazard(
            _make_supplier("LondonCo", lat=52.0, lon=0.0),
            hazard_type="cyclone",
        )
        assert score.score == 0.0
        assert score.score_100 == 0

    def test_high_cyclone_area_gives_high_score(self) -> None:
        # Eastern Pacific (18,-112) has 73 storms in real IBTrACS data
        score = compute_hazard(
            _make_supplier("PacificCo", lat=18.0, lon=-112.0),
            hazard_type="cyclone",
        )
        assert score.score > 0.5
        assert score.level == "High"

    def test_formula_linear_capped(self) -> None:
        score = compute_hazard(
            _make_supplier("ManilaCo", lat=14.0, lon=121.0),
            hazard_type="cyclone",
        )
        count = score.dataset_metadata["storm_count"]
        expected = round(min(1.0, count / 50.0), 4)
        assert score.score == expected


# -------------------------------------------------------------------
# Hash-stub fallback tests (unknown hazard types)
# -------------------------------------------------------------------


class TestHashStubScoring:
    """Tests for unknown hazard types that fall back to SHA-256 stub."""

    def test_returns_hazard_score(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="unknown_hazard")
        assert score.supplier_name == "TestCorp"
        assert 0.0 <= score.score <= 1.0
        assert 0 <= score.score_100 <= 100

    def test_dataset_metadata_populated(self) -> None:
        score = compute_hazard(_make_supplier(), hazard_type="unknown_hazard")
        assert score.dataset_metadata["dataset"] == "stub_unknown_hazard_v1"
        assert score.dataset_metadata["method"] == "sha256_hash_stub"

    def test_deterministic(self) -> None:
        s = _make_supplier("DeterministicCo")
        a = compute_hazard(s, "unknown_hazard")
        b = compute_hazard(s, "unknown_hazard")
        assert a.score == b.score

    def test_different_hazard_types_differ(self) -> None:
        s = _make_supplier()
        a = compute_hazard(s, "unknown_a")
        b = compute_hazard(s, "unknown_b")
        assert a.score != b.score
