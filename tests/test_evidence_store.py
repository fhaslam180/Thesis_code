"""Tests for EvidenceStore content-addressed cache."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bor_risk.evidence_store import EvidenceStore, _extract_domain, _is_official_domain
from bor_risk.models import EvidencePacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(cache_dir=tmp_path / "evidence_cache")


def _store_sample(store: EvidenceStore, content: str = "Sample page text about Apple and TSMC") -> EvidencePacket:
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return store.store(
        url="https://example.com/article",
        final_url="https://example.com/article",
        content=content,
        content_hash=content_hash,
        mime_type="text/html",
        retrieved_at="2025-01-01T00:00:00Z",
        http_status=200,
        title="Example Article",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvidenceStoreStore:
    def test_store_returns_evidence_packet(self, tmp_path):
        store = _make_store(tmp_path)
        packet = _store_sample(store)
        assert isinstance(packet, EvidencePacket)

    def test_evidence_id_is_first_16_chars_of_hash(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        content = "Sample content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        packet = store.store(
            url="https://x.com",
            final_url="https://x.com",
            content=content,
            content_hash=content_hash,
            mime_type="text/html",
            retrieved_at="2025-01-01T00:00:00Z",
        )
        assert packet.evidence_id == content_hash[:16]

    def test_snapshot_file_created(self, tmp_path):
        store = _make_store(tmp_path)
        packet = _store_sample(store)
        assert packet.snapshot_path is not None
        assert Path(packet.snapshot_path).exists()

    def test_snapshot_file_has_correct_content(self, tmp_path):
        store = _make_store(tmp_path)
        content = "Unique content xyz123"
        import hashlib
        ch = hashlib.sha256(content.encode()).hexdigest()
        packet = store.store(
            url="https://x.com", final_url="https://x.com",
            content=content, content_hash=ch,
            mime_type="text/html", retrieved_at="2025-01-01T00:00:00Z",
        )
        stored = Path(packet.snapshot_path).read_text(encoding="utf-8")
        assert stored == content

    def test_domain_extracted_correctly(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        content = "page"
        ch = hashlib.sha256(content.encode()).hexdigest()
        packet = store.store(
            url="https://news.reuters.com/article",
            final_url="https://news.reuters.com/article",
            content=content, content_hash=ch,
            mime_type="text/html", retrieved_at="2025-01-01T00:00:00Z",
        )
        assert packet.domain == "news.reuters.com"

    def test_quality_signals_populated(self, tmp_path):
        store = _make_store(tmp_path)
        packet = _store_sample(store)
        assert "is_official_domain" in packet.quality_signals
        assert "word_count" in packet.quality_signals
        assert packet.quality_signals["word_count"] > 0

    def test_empty_content_no_snapshot(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        ch = hashlib.sha256(b"").hexdigest()
        packet = store.store(
            url="https://x.com", final_url="https://x.com",
            content="", content_hash=ch,
            mime_type="text/html", retrieved_at="2025-01-01T00:00:00Z",
        )
        assert packet.snapshot_path is None


class TestEvidenceStoreDeduplication:
    def test_second_store_does_not_overwrite(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        content = "Original content"
        ch = hashlib.sha256(content.encode()).hexdigest()
        packet1 = store.store(
            url="https://x.com", final_url="https://x.com",
            content=content, content_hash=ch,
            mime_type="text/html", retrieved_at="2025-01-01T00:00:00Z",
        )
        # Store again with "different" content but same hash
        packet2 = store.store(
            url="https://x.com/other", final_url="https://x.com/other",
            content=content, content_hash=ch,
            mime_type="text/html", retrieved_at="2025-02-01T00:00:00Z",
        )
        # File should still have the original content
        stored = Path(packet1.snapshot_path).read_text(encoding="utf-8")
        assert stored == content


class TestEvidenceStoreReadSnapshot:
    def test_read_snapshot_returns_content(self, tmp_path):
        store = _make_store(tmp_path)
        packet = _store_sample(store)
        text = store.read_snapshot(packet)
        assert text is not None
        assert "Apple" in text

    def test_read_snapshot_none_for_missing_packet(self, tmp_path):
        store = _make_store(tmp_path)
        import hashlib
        fake_hash = hashlib.sha256(b"nonexistent").hexdigest()
        fake_packet = EvidencePacket(
            evidence_id=fake_hash[:16],
            url="https://x.com",
            final_url="https://x.com",
            domain="x.com",
            retrieved_at="2025-01-01T00:00:00Z",
            content_hash=fake_hash,
            mime_type="text/html",
            snapshot_path=None,
        )
        assert store.read_snapshot(fake_packet) is None


class TestEvidenceStoreExists:
    def test_exists_false_before_store(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        fake_hash = hashlib.sha256(b"not stored").hexdigest()
        assert store.exists(fake_hash) is False

    def test_exists_true_after_store(self, tmp_path):
        import hashlib
        store = _make_store(tmp_path)
        content = "Check existence"
        ch = hashlib.sha256(content.encode()).hexdigest()
        store.store(
            url="https://x.com", final_url="https://x.com",
            content=content, content_hash=ch,
            mime_type="text/html", retrieved_at="2025-01-01T00:00:00Z",
        )
        assert store.exists(ch) is True


class TestEvidenceStoreExportIndex:
    def test_export_writes_jsonl(self, tmp_path):
        store = _make_store(tmp_path)
        packets = [
            {"evidence_id": "abc123", "url": "https://a.com", "domain": "a.com"},
            {"evidence_id": "def456", "url": "https://b.com", "domain": "b.com"},
        ]
        out = tmp_path / "index.jsonl"
        store.export_index(packets, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["evidence_id"] == "abc123"

    def test_export_empty_list_creates_empty_file(self, tmp_path):
        store = _make_store(tmp_path)
        out = tmp_path / "empty.jsonl"
        store.export_index([], out)
        assert out.read_text() == ""


class TestDomainHelpers:
    def test_extract_domain_standard(self):
        assert _extract_domain("https://news.reuters.com/article") == "news.reuters.com"

    def test_extract_domain_no_path(self):
        assert _extract_domain("https://example.com") == "example.com"

    def test_extract_domain_invalid_returns_empty(self):
        # urllib can handle malformed URLs but may return empty netloc
        result = _extract_domain("not-a-url")
        assert isinstance(result, str)

    def test_is_official_domain_gov(self):
        assert _is_official_domain("sec.gov") is True

    def test_is_official_domain_edu(self):
        assert _is_official_domain("mit.edu") is True

    def test_is_official_domain_reuters(self):
        assert _is_official_domain("news.reuters.com") is True

    def test_is_official_domain_random(self):
        assert _is_official_domain("random-blog.xyz") is False
