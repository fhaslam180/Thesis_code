"""Smoke tests for the three SQLite hazard databases.

All tests are skipped when the database files don't exist — run
the preprocessors first to generate them from raw data:

    python3 scripts/preprocess_gem_seismic.py
    python3 scripts/preprocess_aqueduct_flood.py
    python3 scripts/preprocess_storm_cyclone.py --src data/storm/
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_GEM_DB = _DATA_DIR / "gem_seismic.db"
_FLOOD_DB = _DATA_DIR / "aqueduct_flood.db"
_CYCLONE_DB = _DATA_DIR / "storm_cyclone.db"


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM grid").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.skipif(not _GEM_DB.exists(), reason="gem_seismic.db not built yet")
def test_gem_db_row_count():
    assert _row_count(_GEM_DB) > 0


@pytest.mark.skipif(not _FLOOD_DB.exists(), reason="aqueduct_flood.db not built yet")
def test_aqueduct_db_row_count():
    assert _row_count(_FLOOD_DB) > 0


@pytest.mark.skipif(not _CYCLONE_DB.exists(), reason="storm_cyclone.db not built yet")
def test_cyclone_db_row_count():
    assert _row_count(_CYCLONE_DB) > 0


@pytest.mark.skipif(not _GEM_DB.exists(), reason="gem_seismic.db not built yet")
def test_gem_known_coordinate_taiwan():
    """Taiwan (TSMC region) should return a non-negative PGA value."""
    from bor_risk.tools import _db_lookup
    from bor_risk.utils import grid_key
    value = _db_lookup(str(_GEM_DB), grid_key(24.8, 1), grid_key(121.0, 1))
    assert isinstance(value, float)
    assert value >= 0.0


@pytest.mark.skipif(not _FLOOD_DB.exists(), reason="aqueduct_flood.db not built yet")
def test_aqueduct_absent_cell_returns_zero():
    """A cell not covered by the dataset should return 0.0 (not raise)."""
    from bor_risk.tools import _db_lookup
    from bor_risk.utils import grid_key
    value = _db_lookup(str(_FLOOD_DB), grid_key(0.0, 2), grid_key(0.0, 2))
    assert value == 0.0
