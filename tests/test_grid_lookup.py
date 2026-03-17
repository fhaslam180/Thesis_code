"""Tests for SQLite grid helpers and grid_key quantization.

Covers:
- grid_key boundary: half-away-from-zero at .5 boundaries
- _db_lookup: exact cell hit, missing cell (→ 0.0)
- Thread safety: concurrent reads on the same fixture DB
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from bor_risk.utils import grid_key
from bor_risk.tools import _db_lookup, _open_grid_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grid (
    lat_key INTEGER NOT NULL,
    lon_key INTEGER NOT NULL,
    value   REAL    NOT NULL,
    PRIMARY KEY (lat_key, lon_key)
);
"""


@pytest.fixture()
def tmp_grid_db(tmp_path: Path) -> str:
    """Create a small in-memory-like SQLite grid for tests."""
    db_path = str(tmp_path / "test_grid.db")
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    # Insert a few known cells: (lat=12.1, lon=45.6) at 0.1° resolution
    #   grid_key(12.1, 1) = 121, grid_key(45.6, 1) = 456
    rows = [
        (grid_key(12.1, 1), grid_key(45.6, 1), 0.25),   # seismic/cyclone hit
        (grid_key(-34.9, 1), grid_key(138.6, 1), 0.80),  # high-value cell
    ]
    conn.executemany("INSERT OR REPLACE INTO grid VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    # Clear the lru_cache so a fresh connection is created for this path
    _open_grid_db.cache_clear()
    return db_path


# ---------------------------------------------------------------------------
# grid_key tests
# ---------------------------------------------------------------------------

class TestGridKey:
    def test_positive_exact(self):
        assert grid_key(12.1, 1) == 121

    def test_positive_half_rounds_away_from_zero(self):
        # 12.05 * 10 = 120.5 → floor(120.5 + 0.5) = floor(121.0) = 121
        assert grid_key(12.05, 1) == 121
        # 12.15 * 10 = 121.5 → floor(121.5 + 0.5) = floor(122.0) = 122
        assert grid_key(12.15, 1) == 122

    def test_negative_half_rounds_away_from_zero(self):
        # -12.05: abs * 10 = 120.5 → 121, negate → -121
        assert grid_key(-12.05, 1) == -121
        assert grid_key(-12.15, 1) == -122

    def test_zero_decimals(self):
        # grid_key(12.5, 0) → 13 (not 12 as banker's rounding would give)
        assert grid_key(12.5, 0) == 13
        assert grid_key(-12.5, 0) == -13

    def test_two_decimal_places(self):
        # 0.01° resolution for flood
        assert grid_key(51.505, 2) == 5151  # 51.505 * 100 = 5150.5 → 5151
        assert grid_key(-34.905, 2) == -3491  # 34.905 * 100 = 3490.5 → 3491

    def test_zero_value(self):
        assert grid_key(0.0, 1) == 0


# ---------------------------------------------------------------------------
# SQLite lookup tests
# ---------------------------------------------------------------------------

class TestDbLookup:
    def test_exact_hit_returns_correct_value(self, tmp_grid_db: str):
        lk = grid_key(12.1, 1)
        lok = grid_key(45.6, 1)
        val = _db_lookup(tmp_grid_db, lk, lok)
        assert val == pytest.approx(0.25)

    def test_missing_cell_returns_zero(self, tmp_grid_db: str):
        val = _db_lookup(tmp_grid_db, 9999, 9999)
        assert val == 0.0

    def test_negative_coords_hit(self, tmp_grid_db: str):
        lk = grid_key(-34.9, 1)
        lok = grid_key(138.6, 1)
        val = _db_lookup(tmp_grid_db, lk, lok)
        assert val == pytest.approx(0.80)

    def test_nonexistent_db_raises_operational_error(self, tmp_path: Path):
        _open_grid_db.cache_clear()
        # SQLite creates an empty file when opened, but the 'grid' table won't exist.
        # _db_lookup raises OperationalError. Scorers guard with .exists() before calling.
        import sqlite3 as _sqlite3
        db_path = str(tmp_path / "nonexistent.db")
        with pytest.raises(_sqlite3.OperationalError, match="no such table"):
            _db_lookup(db_path, 100, 200)
        _open_grid_db.cache_clear()


class TestDbLookupConcurrency:
    def test_concurrent_reads_no_errors(self, tmp_grid_db: str):
        """8 threads reading the same DB concurrently must not raise."""
        errors: list[Exception] = []
        results: list[float] = []
        lock = threading.Lock()

        def _read():
            try:
                v = _db_lookup(tmp_grid_db, grid_key(12.1, 1), grid_key(45.6, 1))
                with lock:
                    results.append(v)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert all(r == pytest.approx(0.25) for r in results), f"Inconsistent results: {results}"
