"""Web search with disk caching for reproducible evaluation.

Uses Tavily for web search. Every query+response is cached to disk
so evaluation runs can be pinned to a saved corpus via --snapshot mode.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "search_cache"


def search_web(
    query: str,
    max_results: int = 5,
    use_cache: bool = True,
) -> list[dict]:
    """Search the web via Tavily. Results are cached to disk.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int
        Maximum number of results to return.
    use_cache : bool
        When *True* (default), return cached results if available and
        write new results to cache. When running in snapshot mode,
        only cached results are returned (no live queries).

    Returns
    -------
    list[dict]
        Each dict has keys: ``title``, ``url``, ``content``, ``retrieved_at``.
    """
    cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
    cache_path = _CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # Import lazily so tests/offline runs don't require tavily installed.
    from tavily import TavilyClient  # type: ignore[import-untyped]

    client = TavilyClient()  # reads TAVILY_API_KEY from env
    raw = client.search(query, max_results=max_results)
    results = [
        {
            "title": r["title"],
            "url": r["url"],
            "content": r["content"],
            "retrieved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        for r in raw["results"]
    ]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def search_web_snapshot(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Cache-only search (no live queries). Returns [] on cache miss.

    Used in snapshot/evaluation mode for reproducibility.
    """
    cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
    cache_path = _CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        return results[:max_results]
    return []
