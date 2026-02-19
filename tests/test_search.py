"""Tests for web search + caching."""

import json
from pathlib import Path

from bor_risk.search import search_web_snapshot, _CACHE_DIR


class TestSearchWebSnapshot:
    def test_returns_empty_on_cache_miss(self, tmp_path, monkeypatch):
        """Snapshot mode returns [] when there's no cached result."""
        monkeypatch.setattr("bor_risk.search._CACHE_DIR", tmp_path)
        result = search_web_snapshot("nonexistent query")
        assert result == []

    def test_returns_cached_data(self, tmp_path, monkeypatch):
        """Snapshot mode returns cached data when available."""
        monkeypatch.setattr("bor_risk.search._CACHE_DIR", tmp_path)

        import hashlib
        query = "test query"
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_path = tmp_path / f"{cache_key}.json"

        cached = [
            {"title": "Test", "url": "https://example.com", "content": "data",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]
        cache_path.write_text(json.dumps(cached))

        result = search_web_snapshot(query)
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    def test_max_results_limits_output(self, tmp_path, monkeypatch):
        """Snapshot mode respects max_results."""
        monkeypatch.setattr("bor_risk.search._CACHE_DIR", tmp_path)

        import hashlib
        query = "many results"
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_path = tmp_path / f"{cache_key}.json"

        cached = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}",
             "content": f"content {i}", "retrieved_at": "2025-01-01T00:00:00Z"}
            for i in range(10)
        ]
        cache_path.write_text(json.dumps(cached))

        result = search_web_snapshot(query, max_results=3)
        assert len(result) == 3
