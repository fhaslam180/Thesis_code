"""Shared pytest fixtures."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Mock reference set (autouse) — avoids needing data/reference_set.json
# ---------------------------------------------------------------------------

_MOCK_REFERENCE_SET = {
    "metadata": {
        "built_at": "2024-01-01T00:00:00+00:00",
        "n_suppliers": 50,
        "source": "build_reference_set.py",
        "corpus": "study-specific: supplier locations for benchmark companies",
        "non_informative_hazards": [],
        "informative_hazards": ["wildfire", "heat_stress", "drought"],
    },
    "hazards": {
        "earthquake": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "flood":      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "wildfire":   [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        "cyclone":    [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
        "heat_stress":[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "drought":    [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
    },
    "supplier_exposure_thresholds": {"medium": 0.33, "high": 0.67},
    "company_exposure_thresholds": {"medium": 0.35, "high": 0.65},
}


@pytest.fixture(autouse=True)
def mock_reference_set():
    """Patch load_reference_set to return a deterministic mock.

    Avoids needing data/reference_set.json to exist during tests.
    Patches both bor_risk.utils and bor_risk.tools (which does a from-import).
    Tests that specifically test load_reference_set must override this fixture.
    """
    from bor_risk import utils as utils_module
    from bor_risk import tools as tools_module
    from bor_risk import evaluate as evaluate_module
    utils_module.load_reference_set.cache_clear()
    with patch.object(utils_module, "load_reference_set", return_value=_MOCK_REFERENCE_SET):
        with patch.object(tools_module, "load_reference_set", return_value=_MOCK_REFERENCE_SET):
            with patch.object(evaluate_module, "load_reference_set", return_value=_MOCK_REFERENCE_SET):
                yield
    utils_module.load_reference_set.cache_clear()


@pytest.fixture()
def mock_suppliers() -> list[dict]:
    """Load the mock-suppliers fixture."""
    return json.loads((FIXTURES_DIR / "mock_suppliers.json").read_text())


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Canned API responses for each hazard data source
# ---------------------------------------------------------------------------

# High-seismicity regions (e.g. Tokyo, San Francisco) get 1000 quakes;
# low-seismicity regions (e.g. London, Sydney) get 5.
_USGS_HIGH = b"1000"
_USGS_LOW = b"5"

# Latitude bands with high seismicity (Pacific Ring of Fire, etc.)
_HIGH_SEISMICITY_BANDS = [(25, 45), (-10, 10)]  # (min_lat, max_lat)

_FLOOD_RESPONSE = json.dumps({
    "daily": {
        "river_discharge": [
            10.5, 11.2, 12.8, 10.1, 9.8,
            25.3, 30.1, 28.4, 15.2, 11.0,
            10.2, 10.8, 9.5, 10.3, 10.0,
            11.1, 10.7, 10.4, 10.9, 10.6,
        ],
    },
}).encode()

# 20 days, 5 of them >= 32 C → heat fraction = 5/20 = 0.25 (deterministic)
_HEAT_RESPONSE = json.dumps({
    "daily": {
        "apparent_temperature_max": [
            12.5, 13.1, 11.8, 14.2, 10.9,
            33.0, 35.5, 34.2, 36.1, 32.8,   # 5 hot days
            29.2, 27.8, 24.5, 20.3, 18.0,
            15.1, 12.7, 10.4, 14.9, 16.6,
        ],
    },
}).encode()

# 100 days of fire-prone conditions → FWI series converges to non-zero Q95
_WILDFIRE_RESPONSE = json.dumps({
    "daily": {
        "temperature_2m_max": [35.0] * 100,
        "relative_humidity_2m_min": [20.0] * 100,
        "wind_speed_10m_max": [30.0] * 100,
        "precipitation_sum": [0.0] * 100,
    },
}).encode()


def _make_monthly_dates(n: int) -> list[str]:
    """Generate n unique YYYY-MM-15 dates starting from 1991-01."""
    dates: list[str] = []
    year, month = 1991, 1
    for _ in range(n):
        dates.append(f"{year}-{month:02d}-15")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


_DROUGHT_N = 200
_DROUGHT_DATES = _make_monthly_dates(_DROUGHT_N)
# Seasonal precip with inter-annual drought years (every 5th year: halved)
_DROUGHT_PRECIP = [
    max(5.0, (60.0 + 30.0 * math.sin(2 * math.pi * (i % 12) / 12))
        * (0.5 if (i // 12) % 5 == 3 else 1.0))
    for i in range(_DROUGHT_N)
]
_DROUGHT_ET0 = [
    35.0 + 15.0 * math.sin(2 * math.pi * (i % 12) / 12 + math.pi / 4)
    for i in range(_DROUGHT_N)
]
# 200 unique months (> 180 minimum), with drought years → SPEI-12 fraction > 0
_DROUGHT_RESPONSE = json.dumps({
    "daily": {
        "time": _DROUGHT_DATES,
        "precipitation_sum": _DROUGHT_PRECIP,
        "et0_fao_evapotranspiration": _DROUGHT_ET0,
    },
}).encode()


def _make_mock_response(data: bytes) -> MagicMock:
    """Create a mock urllib response returning *data*."""
    mock = MagicMock()
    mock.read.return_value = data
    mock.__enter__ = lambda self: self
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _usgs_response_for_url(url_str: str) -> bytes:
    """Return a seismicity-appropriate USGS count based on the latitude in the URL."""
    m = re.search(r"latitude=([-\d.]+)", url_str)
    if m:
        lat = abs(float(m.group(1)))
        for lo, hi in _HIGH_SEISMICITY_BANDS:
            if lo <= lat <= hi:
                return _USGS_HIGH
    return _USGS_LOW


def _url_router(url, *args, **kwargs) -> MagicMock:
    """Route mock urllib responses based on the URL being called.

    Specificity-first: drought and wildfire both use archive-api.open-meteo.com,
    so they are distinguished by their unique query parameters before the
    generic precipitation_sum check would match both.
    """
    url_str = url if isinstance(url, str) else str(url)
    if "earthquake.usgs.gov" in url_str:
        return _make_mock_response(_usgs_response_for_url(url_str))
    if "flood-api.open-meteo.com" in url_str:
        return _make_mock_response(_FLOOD_RESPONSE)
    if "archive-api.open-meteo.com" in url_str:
        if "apparent_temperature_max" in url_str:       # heat_stress
            return _make_mock_response(_HEAT_RESPONSE)
        if "et0_fao_evapotranspiration" in url_str:     # drought (unique parameter)
            return _make_mock_response(_DROUGHT_RESPONSE)
        if "temperature_2m_max" in url_str:             # wildfire (unique parameter)
            return _make_mock_response(_WILDFIRE_RESPONSE)
    # Default fallback — valid empty JSON so callers don't crash on decode
    return _make_mock_response(b'{"daily": {}}')


@pytest.fixture(autouse=True)
def mock_urlopen(tmp_path: Path):
    """Globally mock urllib.request.urlopen with URL-aware routing.

    Also isolates the API hazard cache per test:
    - Sets _api_cache to an empty dict (blocks reads from committed cache file)
    - Redirects _API_CACHE_PATH to tmp_path (blocks writes to committed cache file)
    Both are restored to their original values after each test.

    Routes responses based on URL domain and unique query parameters:
    - earthquake.usgs.gov → location-aware count
    - flood-api.open-meteo.com → canned GloFAS discharge
    - archive-api.open-meteo.com + apparent_temperature_max → heat (5/20 days >= 32 C)
    - archive-api.open-meteo.com + et0_fao_evapotranspiration → drought (200 months)
    - archive-api.open-meteo.com + temperature_2m_max → wildfire (100 fire-prone days)
    """
    import bor_risk.tools as _tools

    saved_cache = _tools._api_cache
    saved_path = _tools._API_CACHE_PATH

    _tools._api_cache = {}                                         # blocks disk reads
    _tools._API_CACHE_PATH = tmp_path / "api_hazard_cache.json"   # blocks disk writes

    with patch("urllib.request.urlopen", side_effect=_url_router) as m:
        yield m

    _tools._api_cache = saved_cache   # restore original value, not just None
    _tools._API_CACHE_PATH = saved_path
