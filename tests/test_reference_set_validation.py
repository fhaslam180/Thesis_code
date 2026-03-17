"""Tests for validate_reference_set() and load_reference_set() in bor_risk.utils."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bor_risk.utils import EXPECTED_HAZARDS, load_reference_set as _real_load_reference_set, validate_reference_set


def _valid_reference_set() -> dict:
    """Minimal valid reference set for testing."""
    return {
        "metadata": {
            "built_at": "2024-01-01T00:00:00+00:00",
            "n_suppliers": 50,
            "source": "build_reference_set.py",
            "corpus": "study-specific corpus",
            "non_informative_hazards": [],
        },
        "hazards": {
            "earthquake": [0.0, 0.1, 0.2, 0.3, 0.4],
            "flood": [0.0, 0.5, 1.0, 1.5, 2.0],
            "wildfire": [0.0, 5.0, 10.0, 15.0, 20.0],
            "cyclone": [0.0, 50.0, 100.0, 150.0, 200.0],
            "heat_stress": [0.0, 0.1, 0.2, 0.3, 0.4],
            "drought": [0.0, 0.05, 0.10, 0.15, 0.20],
        },
        "supplier_exposure_thresholds": {"medium": 0.33, "high": 0.67},
        "company_exposure_thresholds": {"medium": 0.35, "high": 0.65},
    }


class TestValidateReferenceSetPasses:
    def test_valid_reference_set(self):
        rs = _valid_reference_set()
        validate_reference_set(rs)  # must not raise

    def test_expected_hazards_constant(self):
        assert EXPECTED_HAZARDS == {
            "earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought"
        }


class TestValidateReferenceSetMissingHazard:
    def test_missing_one_hazard_raises(self):
        rs = _valid_reference_set()
        del rs["hazards"]["earthquake"]
        with pytest.raises(ValueError, match="missing hazard arrays"):
            validate_reference_set(rs)

    def test_empty_hazards_dict_raises(self):
        rs = _valid_reference_set()
        rs["hazards"] = {}
        with pytest.raises(ValueError, match="missing hazard arrays"):
            validate_reference_set(rs)

    def test_missing_hazards_key_raises(self):
        rs = _valid_reference_set()
        del rs["hazards"]
        with pytest.raises(ValueError, match="missing hazard arrays"):
            validate_reference_set(rs)


class TestValidateReferenceSetUnsortedArray:
    def test_unsorted_array_raises(self):
        rs = _valid_reference_set()
        rs["hazards"]["earthquake"] = [0.4, 0.1, 0.3]  # not sorted
        with pytest.raises(ValueError, match="not sorted ascending"):
            validate_reference_set(rs)

    def test_reversed_array_raises(self):
        rs = _valid_reference_set()
        rs["hazards"]["flood"] = [5.0, 3.0, 1.0]
        with pytest.raises(ValueError, match="not sorted ascending"):
            validate_reference_set(rs)

    def test_constant_array_passes(self):
        # Constant (all-equal) is technically sorted — validator allows it
        rs = _valid_reference_set()
        rs["hazards"]["drought"] = [0.1, 0.1, 0.1]
        validate_reference_set(rs)  # must not raise


class TestValidateReferenceSetThresholds:
    def test_missing_supplier_thresholds_raises(self):
        rs = _valid_reference_set()
        del rs["supplier_exposure_thresholds"]
        with pytest.raises(ValueError, match="missing 'supplier_exposure_thresholds'"):
            validate_reference_set(rs)

    def test_missing_company_thresholds_raises(self):
        rs = _valid_reference_set()
        del rs["company_exposure_thresholds"]
        with pytest.raises(ValueError, match="missing 'company_exposure_thresholds'"):
            validate_reference_set(rs)

    def test_missing_medium_in_supplier_thresholds_raises(self):
        rs = _valid_reference_set()
        del rs["supplier_exposure_thresholds"]["medium"]
        with pytest.raises(ValueError, match="missing 'medium'"):
            validate_reference_set(rs)

    def test_missing_high_in_company_thresholds_raises(self):
        rs = _valid_reference_set()
        del rs["company_exposure_thresholds"]["high"]
        with pytest.raises(ValueError, match="missing 'high'"):
            validate_reference_set(rs)


class TestValidateReferenceSetMetadata:
    def test_missing_metadata_raises(self):
        rs = _valid_reference_set()
        del rs["metadata"]
        with pytest.raises(ValueError, match="missing metadata.corpus"):
            validate_reference_set(rs)

    def test_missing_corpus_field_raises(self):
        rs = _valid_reference_set()
        del rs["metadata"]["corpus"]
        with pytest.raises(ValueError, match="missing metadata.corpus"):
            validate_reference_set(rs)


class TestLoadReferenceSetValidatesOnLoad:
    def test_malformed_file_raises_at_load(self, tmp_path: Path):
        """load_reference_set() must call validate_reference_set() and raise on bad input.

        Note: the autouse mock_reference_set fixture patches utils_module.load_reference_set,
        so we call the real function directly via the module-level import captured before patching.
        """
        bad_rs = _valid_reference_set()
        del bad_rs["hazards"]["drought"]  # missing hazard
        rs_path = tmp_path / "reference_set.json"
        rs_path.write_text(json.dumps(bad_rs))

        from bor_risk import utils as utils_module
        with patch.object(utils_module, "_REFERENCE_SET_PATH", rs_path):
            _real_load_reference_set.cache_clear()
            with pytest.raises(ValueError, match="missing hazard arrays"):
                _real_load_reference_set()
            _real_load_reference_set.cache_clear()

    def test_valid_file_loads_without_error(self, tmp_path: Path):
        rs = _valid_reference_set()
        rs_path = tmp_path / "reference_set.json"
        rs_path.write_text(json.dumps(rs))

        from bor_risk import utils as utils_module
        with patch.object(utils_module, "_REFERENCE_SET_PATH", rs_path):
            _real_load_reference_set.cache_clear()
            loaded = _real_load_reference_set()
            assert "hazards" in loaded
            _real_load_reference_set.cache_clear()
