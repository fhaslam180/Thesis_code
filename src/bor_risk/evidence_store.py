"""Content-addressed store for fetched web page snapshots.

Distinct from search.py, which caches Tavily query snippets (200-300 chars)
keyed by query-hash. EvidenceStore caches full page text keyed by SHA-256
content-hash, deduplicates the same URL fetched via different queries, and
exports an audit index as JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from bor_risk.models import EvidencePacket

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_OFFICIAL_TLDS = {".gov", ".edu", ".org"}
_OFFICIAL_DOMAIN_KEYWORDS = {
    "reuters", "bloomberg", "wsj", "ft.com", "sec.gov",
    "ir.", "investor.", "corporate.", "annual-report",
}


def _extract_domain(url: str) -> str:
    """Extract netloc from URL, returning empty string on failure."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def _is_official_domain(domain: str) -> bool:
    """Heuristic: is this an official or high-quality source domain?"""
    dl = domain.lower()
    if any(dl.endswith(tld) for tld in _OFFICIAL_TLDS):
        return True
    if any(kw in dl for kw in _OFFICIAL_DOMAIN_KEYWORDS):
        return True
    return False


class EvidenceStore:
    """Content-addressed store for full web page content.

    Cache path: ``data/evidence_cache/{hash[:2]}/{hash}.txt``
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or (_DATA_DIR / "evidence_cache")

    def store(
        self,
        url: str,
        final_url: str,
        content: str,
        content_hash: str,
        mime_type: str,
        retrieved_at: str,
        http_status: int = 200,
        title: str = "",
    ) -> EvidencePacket:
        """Store *content* and return an :class:`EvidencePacket`.

        Writes a snapshot file only when content is non-empty and the file
        does not already exist (deduplication by hash).
        """
        evidence_id = content_hash[:16]
        domain = _extract_domain(final_url or url)

        snapshot_path: str | None = None
        if content:
            bucket = self._cache_dir / content_hash[:2]
            bucket.mkdir(parents=True, exist_ok=True)
            snap_file = bucket / f"{content_hash}.txt"
            if not snap_file.exists():
                snap_file.write_text(content, encoding="utf-8")
            snapshot_path = str(snap_file)

        quality_signals: dict = {
            "is_official_domain": _is_official_domain(domain),
            "is_pdf": mime_type == "application/pdf",
            "word_count": len(content.split()) if content else 0,
        }

        return EvidencePacket(
            evidence_id=evidence_id,
            url=url,
            final_url=final_url,
            domain=domain,
            title=title,
            retrieved_at=retrieved_at,
            http_status=http_status,
            content_hash=content_hash,
            mime_type=mime_type,
            snapshot_path=snapshot_path,
            quality_signals=quality_signals,
        )

    def read_snapshot(self, packet: EvidencePacket) -> str | None:
        """Return the stored plain-text snapshot for *packet*, or ``None``."""
        if packet.snapshot_path:
            path = Path(packet.snapshot_path)
            if path.exists():
                return path.read_text(encoding="utf-8")
        # Fallback: look up by hash
        if packet.content_hash:
            snap_file = (
                self._cache_dir
                / packet.content_hash[:2]
                / f"{packet.content_hash}.txt"
            )
            if snap_file.exists():
                return snap_file.read_text(encoding="utf-8")
        return None

    def export_index(self, packets: list[dict], out_path: Path) -> None:
        """Write one JSON line per packet to *out_path* (JSONL format)."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for p in packets:
                fh.write(json.dumps(p) + "\n")

    def exists(self, content_hash: str) -> bool:
        """Return ``True`` if a snapshot for *content_hash* is already stored."""
        snap_file = (
            self._cache_dir / content_hash[:2] / f"{content_hash}.txt"
        )
        return snap_file.exists()
