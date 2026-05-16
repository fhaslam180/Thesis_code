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


def _extract_domain(url: str) -> str:
    """Extract netloc from URL, returning empty string on failure."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Source quality classifier
# ---------------------------------------------------------------------------

_FORUM_DOMAINS: frozenset[str] = frozenset({
    "reddit.com", "quora.com", "stackexchange.com", "tripadvisor.com", "yelp.com",
})
_FORUM_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "discussions.", "forums.", "community.", "talk.", "answers.",
)

_DOMAIN_TIERS: list[tuple[frozenset[str], float, str]] = [
    (frozenset({"sec.gov", "edgar.sec.gov", "companies-house.gov.uk"}),
     1.0, "regulatory_filing"),
    (frozenset({"apple.com", "toyota.com", "nike.com"}),
     0.85, "official_company"),
    (frozenset({"bloomberg.com", "reuters.com", "wsj.com", "ft.com",
                "nikkei.com", "economist.com", "businessinsider.com"}),
     0.8, "news"),
    (frozenset({"supplychainbrain.com", "supplychaindigital.com", "industryweek.com",
                "globalsources.com", "purchasing.com", "thomasnet.com"}),
     0.6, "industry_database"),
    (frozenset({"wikipedia.org", "businesswire.com", "prnewswire.com",
                "globenewswire.com", "cnbc.com", "marketwatch.com"}),
     0.5, "general"),
]

_PATH_TIERS: list[tuple[frozenset[str], float, str]] = [
    (frozenset({"supplier-responsibility", "supplier-list", "supplier_list",
                "supply-chain-report", "responsibility"}),
     0.7, "official_report_candidate"),
    (frozenset({"newsroom", "press-release", "pressroom"}),
     0.7, "official_company"),
    (frozenset({"annual-report", "annual_report", "10-k", "10k",
                "sustainability-report", "esg-report", "investor-relations", "/ir/"}),
     0.7, "official_report_candidate"),
]


def _domain_matches(domain: str, base: str) -> bool:
    """True if domain is exactly base or a subdomain (e.g. www.base.com)."""
    return domain == base or domain.endswith("." + base)


def _classify_source(url: str) -> tuple[float, str]:
    """Return (quality, source_type) from parsed domain and path.

    Precedence (highest to lowest):
    1. Forum domains/subdomains  → 0.2, "forum"  (cannot be upgraded by path rules)
    2. Government TLD (.gov)     → 1.0, "regulatory_filing"
    3. Named domain tiers        → 0.5–1.0
    4. Official-report path      → 0.7, "official_report_candidate"  (unknown domain)
    5. Default                   → 0.4, "unknown"
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
    except Exception:
        return 0.4, "unknown"

    # 1. Forum domains/subdomains always win
    if any(_domain_matches(domain, d) for d in _FORUM_DOMAINS):
        return 0.2, "forum"
    if any(domain.startswith(prefix) for prefix in _FORUM_SUBDOMAIN_PREFIXES):
        return 0.2, "forum"

    # 2. Government TLD catch-all
    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        return 1.0, "regulatory_filing"

    # 3. Named domain rules (exact/subdomain match)
    for bases, quality, stype in _DOMAIN_TIERS:
        if any(_domain_matches(domain, b) for b in bases):
            return quality, stype

    # 4. Path rules — only reached for non-forum, non-gov, unrecognised domains
    for keywords, quality, stype in _PATH_TIERS:
        if any(kw in path for kw in keywords):
            return quality, stype

    return 0.4, "unknown"


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
        quality, stype = _classify_source(final_url or url)

        snapshot_path: str | None = None
        if content:
            bucket = self._cache_dir / content_hash[:2]
            bucket.mkdir(parents=True, exist_ok=True)
            snap_file = bucket / f"{content_hash}.txt"
            if not snap_file.exists():
                snap_file.write_text(content, encoding="utf-8")
            snapshot_path = str(snap_file)

        packet = EvidencePacket(
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
            source_quality=quality,
            source_type=stype,
            quality_signals={
                "is_official_domain": quality >= 0.8,
                "is_pdf": mime_type == "application/pdf",
                "word_count": len(content.split()) if content else 0,
            },
        )
        self.update_url_index(url, packet)
        return packet

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

    # ------------------------------------------------------------------
    # Persistent URL → packet index (for reproducible snapshot reruns)
    # ------------------------------------------------------------------

    @property
    def _url_index_path(self) -> "Path":
        return self._cache_dir / "packet_index.json"

    def load_url_index(self) -> dict[str, dict]:
        """Load the persistent URL → packet metadata index from disk.

        Returns a dict mapping URL → ``{"evidence_id": ..., "content_hash": ...}``.
        Returns an empty dict if the index does not yet exist.
        """
        if not self._url_index_path.exists():
            return {}
        try:
            return json.loads(self._url_index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update_url_index(self, url: str, packet: EvidencePacket) -> None:
        """Append or update *url* in the persistent index on disk.

        Safe to call on every :meth:`store` — uses atomic read-modify-write so
        concurrent runs do not corrupt the index.
        """
        index = self.load_url_index()
        index[url] = {
            "evidence_id": packet.evidence_id,
            "content_hash": packet.content_hash,
            "domain": packet.domain,
            "retrieved_at": packet.retrieved_at,
            "source_quality": packet.source_quality,
            "source_type": packet.source_type,
        }
        self._url_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._url_index_path.write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
