"""Tests for reference-set threshold construction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_reference_set.py"
_SPEC = importlib.util.spec_from_file_location("build_reference_set_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_compute_supplier_exposure_indices = _MODULE._compute_supplier_exposure_indices


_HAZARDS = ["earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought"]


def test_failed_location_does_not_shift_company_b_thresholds(tmp_path):
    """Regression: a skipped Company A location must not shift Company B's E_s slice.

    With the old bug (slice by len(company_locs) including failures), Company A's
    slice would consume both supplier_Es values [0.0, 1.0], leaving Company B with
    an empty slice. company_Es ends up as [0.5] so medium == high == 0.5.

    With the fix (slice by len(per_company_raws), actual successes), Company A gets
    [0.0] and Company B gets [1.0], giving distinct thresholds medium=0.0, high=1.0.
    """
    locations_data = {
        "metadata": {"collected_at": "2024-01-01T00:00:00+00:00", "corpus": "test"},
        "companies": {
            "CompanyA": [
                {"name": "A-fail", "lat": 1.0, "lon": 1.0},   # will raise
                {"name": "A-ok",   "lat": 2.0, "lon": 2.0},   # low hazard (0.1)
            ],
            "CompanyB": [
                {"name": "B-only", "lat": 3.0, "lon": 3.0},   # high hazard (0.9)
            ],
        },
    }
    loc_path = tmp_path / "locs.json"
    loc_path.write_text(json.dumps(locations_data))
    out_path = tmp_path / "ref.json"

    def _fake_compute(lat, lon):
        if lat == 1.0:
            raise ValueError("Simulated fetch failure")
        value = 0.1 if lat == 2.0 else 0.9
        return {h: value for h in _HAZARDS}

    with patch.object(_MODULE, "_compute_raw_hazards", side_effect=_fake_compute):
        _MODULE._build_reference_set(loc_path, out_path)

    result = json.loads(out_path.read_text())
    thresholds = result["company_exposure_thresholds"]

    # Bug would produce medium == high == 0.5 (one company mean, wrong slice)
    assert thresholds["medium"] != thresholds["high"], (
        "Bug present: Company B's E_s was absorbed by Company A's slice; "
        "only one company mean was computed (medium == high)."
    )
    assert thresholds["medium"] == pytest.approx(0.0)
    assert thresholds["high"] == pytest.approx(1.0)


def test_reference_threshold_builder_excludes_non_informative_hazards():
    supplier_raws = [
        {
            "earthquake": 0.0,
            "flood": 0.0,
            "wildfire": 2.0,
            "cyclone": 0.0,
            "heat_stress": 0.1,
            "drought": 0.05,
        },
        {
            "earthquake": 0.0,
            "flood": 0.0,
            "wildfire": 8.0,
            "cyclone": 0.0,
            "heat_stress": 0.4,
            "drought": 0.20,
        },
    ]
    hazard_raws = {
        "earthquake": [0.0, 0.0],
        "flood": [0.0, 0.0],
        "wildfire": [2.0, 8.0],
        "cyclone": [0.0, 0.0],
        "heat_stress": [0.1, 0.4],
        "drought": [0.05, 0.20],
    }

    exposures, informative = _compute_supplier_exposure_indices(
        supplier_raws,
        hazard_raws,
        non_informative=["earthquake", "flood", "cyclone"],
    )

    assert informative == ["wildfire", "heat_stress", "drought"]
    assert exposures == [0.0, 1.0]
