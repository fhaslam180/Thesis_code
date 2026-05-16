"""Tests for _classify_source, _domain_matches, and EvidenceStore source quality."""
from __future__ import annotations

import pytest

from bor_risk.evidence_store import _classify_source, _domain_matches


# ---------------------------------------------------------------------------
# _domain_matches
# ---------------------------------------------------------------------------

class TestDomainMatches:

    def test_exact_match(self):
        assert _domain_matches("reddit.com", "reddit.com") is True

    def test_subdomain_match(self):
        assert _domain_matches("www.reddit.com", "reddit.com") is True

    def test_deep_subdomain(self):
        assert _domain_matches("news.bloomberg.com", "bloomberg.com") is True

    def test_no_match_substring(self):
        # "notreddit.com" must NOT match "reddit.com"
        assert _domain_matches("notreddit.com", "reddit.com") is False

    def test_no_match_different(self):
        assert _domain_matches("example.com", "reddit.com") is False


# ---------------------------------------------------------------------------
# _classify_source — forum rules (always win)
# ---------------------------------------------------------------------------

class TestClassifySourceForum:

    def test_reddit_plain(self):
        q, t = _classify_source("https://reddit.com/r/apple")
        assert q == 0.2
        assert t == "forum"

    def test_reddit_with_supplier_list_path(self):
        # Forum domain beats official-report path — the key correctness check
        q, t = _classify_source("https://reddit.com/supplier-list")
        assert q == 0.2
        assert t == "forum"

    def test_www_reddit(self):
        q, t = _classify_source("https://www.reddit.com/r/apple")
        assert q == 0.2
        assert t == "forum"

    def test_notreddit_not_forum(self):
        # Substring "reddit" in domain name must NOT trigger forum rule
        q, _ = _classify_source("https://notreddit.com/r/apple")
        assert q != 0.2

    def test_discussions_subdomain(self):
        # discussions.apple.com → forum via subdomain prefix
        q, t = _classify_source("https://discussions.apple.com/thread/12345")
        assert q == 0.2
        assert t == "forum"

    def test_forums_subdomain(self):
        q, t = _classify_source("https://forums.example.com/topic/1")
        assert q == 0.2
        assert t == "forum"

    def test_community_subdomain(self):
        q, t = _classify_source("https://community.example.com/post")
        assert q == 0.2
        assert t == "forum"

    def test_quora(self):
        q, t = _classify_source("https://quora.com/What-suppliers-does-Apple-use")
        assert q == 0.2
        assert t == "forum"


# ---------------------------------------------------------------------------
# _classify_source — government / regulatory
# ---------------------------------------------------------------------------

class TestClassifySourceGov:

    def test_sec_gov(self):
        q, t = _classify_source("https://sec.gov/filing/0001234")
        assert q == 1.0
        assert t == "regulatory_filing"

    def test_gov_tld_catchall(self):
        q, t = _classify_source("https://commerce.gov/report")
        assert q == 1.0
        assert t == "regulatory_filing"

    def test_gov_uk(self):
        q, t = _classify_source("https://companieshouse.gov.uk/company/12345")
        assert q == 1.0
        assert t == "regulatory_filing"


# ---------------------------------------------------------------------------
# _classify_source — named domain tiers
# ---------------------------------------------------------------------------

class TestClassifySourceDomains:

    def test_apple_official_domain(self):
        q, t = _classify_source("https://www.apple.com/newsroom/2026/03/apple-suppliers/")
        assert q == 0.85
        assert t == "official_company"

    def test_bloomberg(self):
        q, t = _classify_source("https://bloomberg.com/news/apple-supply-chain")
        assert q == 0.8
        assert t == "news"

    def test_bloomberg_subdomain(self):
        q, t = _classify_source("https://www.bloomberg.com/news/apple")
        assert q == 0.8
        assert t == "news"

    def test_supplychainbrain(self):
        q, t = _classify_source("https://supplychainbrain.com/article/apple")
        assert q == 0.6
        assert t == "industry_database"

    def test_unknown_domain(self):
        q, t = _classify_source("https://unknown-site.io/page")
        assert q == 0.4
        assert t == "unknown"


# ---------------------------------------------------------------------------
# _classify_source — path rules (official_report_candidate, unknown domain)
# ---------------------------------------------------------------------------

class TestClassifySourcePath:

    def test_supplier_responsibility_path(self):
        q, t = _classify_source("https://apple.com/supplier-responsibility/pdf")
        assert q == 0.85
        assert t == "official_company"

    def test_annual_report_path(self):
        q, t = _classify_source("https://apple.com/annual-report-2023")
        assert q == 0.85
        assert t == "official_company"

    def test_ir_path(self):
        q, t = _classify_source("https://somecompany.com/investor-relations/2023")
        assert q == 0.7
        assert t == "official_report_candidate"

    def test_10k_path(self):
        q, t = _classify_source("https://somecompany.com/10-k-filing.pdf")
        assert q == 0.7
        assert t == "official_report_candidate"


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

class TestModelDefaults:

    def test_evidence_packet_defaults(self):
        from bor_risk.models import EvidencePacket
        p = EvidencePacket(
            evidence_id="abc123",
            url="https://x.com",
            final_url="https://x.com",
            domain="x.com",
            retrieved_at="2025-01-01T00:00:00Z",
            content_hash="deadbeef" * 8,
            mime_type="text/html",
        )
        assert p.source_quality == 0.5
        assert p.source_type == "unknown"

    def test_claim_best_evidence_quality_default(self):
        from bor_risk.models import Claim, LocationCandidate
        c = Claim(
            claim_id="abc123456789",
            display_id="C001",
            subject_entity_id="Apple",
            object_entity_id="TSMC",
            normalized_name="tsmc",
            claim_type="SUPPLIES_TO",
            confidence=0.9,
            location_candidates=[
                LocationCandidate(lat=24.8, lon=121.0, confidence=0.5, source="llm")
            ],
        )
        assert c.best_evidence_quality == 0.5


# ---------------------------------------------------------------------------
# EvidenceStore.store() populates source_quality / source_type
# ---------------------------------------------------------------------------

class TestEvidenceStoreQuality:

    def test_store_populates_source_quality(self, tmp_path):
        from bor_risk.evidence_store import EvidenceStore
        store = EvidenceStore(cache_dir=tmp_path)
        packet = store.store(
            url="https://bloomberg.com/news/apple",
            final_url="https://bloomberg.com/news/apple",
            content="Apple sources chips from TSMC",
            content_hash="a" * 64,
            mime_type="text/html",
            retrieved_at="2025-01-01T00:00:00Z",
        )
        assert packet.source_quality == 0.8
        assert packet.source_type == "news"

    def test_update_url_index_persists_quality(self, tmp_path):
        from bor_risk.evidence_store import EvidenceStore
        store = EvidenceStore(cache_dir=tmp_path)
        packet = store.store(
            url="https://bloomberg.com/news/apple",
            final_url="https://bloomberg.com/news/apple",
            content="Apple sources chips from TSMC",
            content_hash="b" * 64,
            mime_type="text/html",
            retrieved_at="2025-01-01T00:00:00Z",
        )
        index = store.load_url_index()
        entry = index["https://bloomberg.com/news/apple"]
        assert entry["source_quality"] == 0.8
        assert entry["source_type"] == "news"


# ---------------------------------------------------------------------------
# Tier-2 gate: _classify_source applied in discover_suppliers_rag
# ---------------------------------------------------------------------------

class TestTier2Gate:

    def _make_supplier(self, name, confidence=0.9, source_url="https://x.com", snippet="confirmed"):
        from bor_risk.models import Supplier
        return Supplier(
            name=name, lat=24.8, lon=121.0, tier=1,
            confidence=confidence,
            evidence_ids=[f"RAG-{name[:4]}"],
            evidence_source="llm_only",
            source_url=source_url,
            verification_snippet=snippet,
        )

    def test_reddit_source_excluded_from_tier2(self):
        from bor_risk.evidence_store import _classify_source
        sup = self._make_supplier("TSMC", source_url="https://reddit.com/r/apple")
        q, _ = _classify_source(sup.source_url)
        assert q < 0.5, "reddit should be quality 0.2 — below tier-2 gate"

    def test_bloomberg_source_admitted_to_tier2(self):
        from bor_risk.evidence_store import _classify_source
        sup = self._make_supplier("TSMC", source_url="https://bloomberg.com/news/apple-tsmc")
        q, _ = _classify_source(sup.source_url)
        assert q >= 0.5, "bloomberg should be quality 0.8 — above tier-2 gate"
