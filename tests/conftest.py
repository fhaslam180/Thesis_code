"""Shared pytest fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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

_HEAT_RESPONSE = json.dumps({
    "daily": {
        "apparent_temperature_max": [
            12.5, 13.1, 11.8, 14.2, 10.9,
            22.3, 25.1, 28.4, 30.2, 31.0,
            29.2, 27.8, 24.5, 20.3, 18.0,
            15.1, 12.7, 10.4, 14.9, 16.6,
        ],
    },
}).encode()

_PRECIP_RESPONSE = json.dumps({
    "daily": {
        "time": [
            "2015-01-01", "2015-01-15", "2015-02-01", "2015-02-15",
            "2015-03-01", "2015-03-15", "2015-04-01", "2015-04-15",
            "2015-05-01", "2015-05-15", "2015-06-01", "2015-06-15",
            "2015-07-01", "2015-07-15", "2015-08-01", "2015-08-15",
            "2015-09-01", "2015-09-15", "2015-10-01", "2015-10-15",
            "2015-11-01", "2015-11-15", "2015-12-01", "2015-12-15",
        ],
        "precipitation_sum": [
            12.5, 10.8, 14.2, 11.5,
            9.8, 8.4, 10.1, 12.3,
            8.6, 7.5, 9.2, 11.0,
            7.8, 8.1, 9.5, 10.4,
            11.2, 9.8, 12.1, 10.5,
            13.4, 11.8, 14.0, 12.2,
        ],
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
    """Route mock urllib responses based on the URL being called."""
    url_str = url if isinstance(url, str) else str(url)
    if "earthquake.usgs.gov" in url_str:
        return _make_mock_response(_usgs_response_for_url(url_str))
    if "flood-api.open-meteo.com" in url_str:
        return _make_mock_response(_FLOOD_RESPONSE)
    if "archive-api.open-meteo.com" in url_str:
        if "apparent_temperature_max" in url_str:
            return _make_mock_response(_HEAT_RESPONSE)
        if "precipitation_sum" in url_str:
            return _make_mock_response(_PRECIP_RESPONSE)
    # Default fallback
    return _make_mock_response(b"0")


@pytest.fixture(autouse=True)
def mock_urlopen():
    """Globally mock urllib.request.urlopen with URL-aware routing.

    Routes responses based on the URL domain and parameters:
    - earthquake.usgs.gov -> location-aware count (1000 for seismic zones, 5 for stable regions)
    - flood-api.open-meteo.com -> canned GloFAS discharge data
    - archive-api.open-meteo.com (temperature) -> canned ERA5 heat data
    - archive-api.open-meteo.com (precipitation) -> canned ERA5 precip data

    Tests can override by accepting this fixture and changing side_effect.
    """
    with patch("urllib.request.urlopen", side_effect=_url_router) as m:
        yield m
