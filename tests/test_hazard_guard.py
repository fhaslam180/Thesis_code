"""Tests for filter_located_suppliers_node — the (0,0) coordinate guard."""
from __future__ import annotations

from bor_risk.graph import filter_located_suppliers_node


def _state(suppliers: list[dict]) -> dict:
    return {"suppliers": suppliers}


class TestFilterLocatedSuppliersNode:

    def test_zero_zero_excluded(self):
        """A (0,0) supplier must be absent from suppliers and present in location_excluded_suppliers."""
        result = filter_located_suppliers_node(_state([
            {"name": "Ghost Corp", "lat": 0.0, "lon": 0.0},
            {"name": "TSMC", "lat": 24.8, "lon": 121.0},
        ]))
        names = [s["name"] for s in result["suppliers"]]
        excluded_names = [s["name"] for s in result["location_excluded_suppliers"]]
        assert "Ghost Corp" not in names
        assert "Ghost Corp" in excluded_names

    def test_valid_location_kept(self):
        """A supplier with real coordinates stays in suppliers and is not excluded."""
        result = filter_located_suppliers_node(_state([
            {"name": "TSMC", "lat": 24.8, "lon": 121.0},
        ]))
        assert result["suppliers"][0]["name"] == "TSMC"
        assert result["location_excluded_suppliers"] == []

    def test_all_unresolved_all_excluded(self):
        """When every supplier has (0,0), suppliers becomes [] and all are excluded."""
        result = filter_located_suppliers_node(_state([
            {"name": "A", "lat": 0.0, "lon": 0.0},
            {"name": "B", "lat": 0.0, "lon": 0.0},
        ]))
        assert result["suppliers"] == []
        assert len(result["location_excluded_suppliers"]) == 2

    def test_count_invariant(self):
        """len(suppliers) + len(location_excluded_suppliers) == original count."""
        original = [
            {"name": "A", "lat": 0.0, "lon": 0.0},
            {"name": "B", "lat": 35.0, "lon": 139.0},
            {"name": "C", "lat": 0.0, "lon": 0.0},
            {"name": "D", "lat": 1.3, "lon": 103.8},
        ]
        result = filter_located_suppliers_node(_state(original))
        assert len(result["suppliers"]) + len(result["location_excluded_suppliers"]) == len(original)

    def test_no_op_when_all_located(self):
        """location_excluded_suppliers is [] when no (0,0) supplier is present."""
        result = filter_located_suppliers_node(_state([
            {"name": "Samsung", "lat": 37.5, "lon": 127.0},
            {"name": "TSMC", "lat": 24.8, "lon": 121.0},
        ]))
        assert result["location_excluded_suppliers"] == []
        assert len(result["suppliers"]) == 2
