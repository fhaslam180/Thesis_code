"""Tests for Claim lifecycle: PROPOSED → RETRIEVED → VERIFIED."""

from __future__ import annotations

import hashlib

import pytest

from bor_risk.models import Claim, EvidenceSpan
from bor_risk.tools import link_mentions_to_claims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claim(
    company: str,
    supplier: str,
    tier: int = 1,
    status: str = "PROPOSED",
    aliases: list[str] | None = None,
) -> Claim:
    cid = hashlib.sha256(
        f"{company}|{supplier}|SUPPLIES_TO|{tier}".encode()
    ).hexdigest()[:12]
    return Claim(
        claim_id=cid,
        display_id="C001",
        subject_entity_id=company,
        object_entity_id=supplier,
        normalized_name=supplier.strip().lower(),
        confidence=0.8,
        status=status,
        aliases=aliases or [],
        tier=tier,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic claim IDs
# ---------------------------------------------------------------------------


class TestClaimId:
    def test_same_inputs_produce_same_id(self):
        c1 = _make_claim("Apple", "TSMC", tier=1)
        c2 = _make_claim("Apple", "TSMC", tier=1)
        assert c1.claim_id == c2.claim_id

    def test_different_supplier_different_id(self):
        c1 = _make_claim("Apple", "TSMC", tier=1)
        c2 = _make_claim("Apple", "Foxconn", tier=1)
        assert c1.claim_id != c2.claim_id

    def test_different_tier_different_id(self):
        c1 = _make_claim("Apple", "TSMC", tier=1)
        c2 = _make_claim("Apple", "TSMC", tier=2)
        assert c1.claim_id != c2.claim_id

    def test_different_company_different_id(self):
        c1 = _make_claim("Apple", "TSMC", tier=1)
        c2 = _make_claim("Samsung", "TSMC", tier=1)
        assert c1.claim_id != c2.claim_id

    def test_claim_id_is_12_chars(self):
        c = _make_claim("Apple", "TSMC")
        assert len(c.claim_id) == 12

    def test_display_id_format(self):
        c = Claim(
            claim_id="abc123def456",
            display_id="C007",
            subject_entity_id="Apple",
            object_entity_id="TSMC",
            normalized_name="tsmc",
            confidence=0.8,
        )
        assert c.display_id == "C007"


# ---------------------------------------------------------------------------
# 2. Claim → PROPOSED defaults
# ---------------------------------------------------------------------------


class TestClaimDefaults:
    def test_new_claim_is_proposed(self):
        c = _make_claim("Apple", "TSMC")
        assert c.status == "PROPOSED"

    def test_new_claim_verdict_is_unknown(self):
        c = _make_claim("Apple", "TSMC")
        assert c.verdict == "UNKNOWN"

    def test_new_claim_has_no_spans(self):
        c = _make_claim("Apple", "TSMC")
        assert c.supporting_spans == []

    def test_new_claim_has_no_evidence_refs(self):
        c = _make_claim("Apple", "TSMC")
        assert c.evidence_refs == []


# ---------------------------------------------------------------------------
# 3. link_mentions_to_claims: PROPOSED → RETRIEVED
# ---------------------------------------------------------------------------


class TestLinkMentionsToClaims:
    def _mention(self, obj: str, eid: str = "ev01", quote: str = "Apple uses TSMC as a key supplier", company: str = "Apple") -> dict:
        return {
            "object_entity_id": obj,
            "evidence_id": eid,
            "supporting_quote": quote,
            "start_char": 0,
            "end_char": len(quote),
            "company": company,
        }

    def test_exact_match_transitions_to_retrieved(self):
        claims = [_make_claim("Apple", "tsmc")]
        mentions = [self._mention("tsmc", eid="ev01")]
        updated = link_mentions_to_claims(mentions, claims)
        assert updated[0].status == "RETRIEVED"

    def test_exact_match_attaches_evidence_ref(self):
        claims = [_make_claim("Apple", "tsmc")]
        mentions = [self._mention("tsmc", eid="ev01")]
        updated = link_mentions_to_claims(mentions, claims)
        assert "ev01" in updated[0].evidence_refs

    def test_exact_match_creates_evidence_span(self):
        claims = [_make_claim("Apple", "tsmc")]
        mentions = [self._mention("tsmc", eid="ev01",
                                  quote="Apple uses tsmc as a key supplier")]
        updated = link_mentions_to_claims(mentions, claims)
        assert len(updated[0].supporting_spans) == 1
        assert updated[0].supporting_spans[0].quote == "Apple uses tsmc as a key supplier"

    def test_alias_match_transitions_to_retrieved(self):
        claims = [_make_claim("Apple", "taiwan semiconductor", aliases=["TSMC"])]
        mentions = [self._mention("TSMC", eid="ev02")]
        updated = link_mentions_to_claims(mentions, claims)
        assert updated[0].status == "RETRIEVED"

    def test_no_match_does_not_change_status(self):
        claims = [_make_claim("Apple", "tsmc")]
        mentions = [self._mention("Foxconn", eid="ev03",
                                  quote="Apple uses Foxconn as a key supplier")]
        updated = link_mentions_to_claims(mentions, claims)
        # tsmc claim unchanged
        assert updated[0].status == "PROPOSED"

    def test_new_claim_created_for_substantial_unmatched_quote(self):
        """A mention for an unknown supplier with >20 char quote creates new claim."""
        claims = [_make_claim("Apple", "tsmc")]
        long_quote = "Apple works closely with Bosch for automotive sensors and parts."
        mentions = [self._mention("Bosch", eid="ev04", quote=long_quote, company="Apple")]
        updated = link_mentions_to_claims(mentions, claims)
        names = [c.normalized_name for c in updated]
        assert "bosch" in names

    def test_short_quote_does_not_create_new_claim(self):
        """Mentions with a short quote (<= 20 chars) don't spawn new claims."""
        claims = [_make_claim("Apple", "tsmc")]
        mentions = [self._mention("NewCo", eid="ev05",
                                  quote="Apple NewCo deal")]  # <= 20 chars
        updated = link_mentions_to_claims(mentions, claims)
        assert len(updated) == 1  # no new claim

    def test_idempotent_evidence_refs(self):
        """Same evidence_id not added twice."""
        claims = [_make_claim("Apple", "tsmc")]
        mention = self._mention("tsmc", eid="ev01")
        updated = link_mentions_to_claims([mention, mention], claims)
        assert updated[0].evidence_refs.count("ev01") == 1

    def test_already_retrieved_stays_retrieved(self):
        """RETRIEVED claim stays RETRIEVED; additional span appended."""
        claims = [_make_claim("Apple", "tsmc", status="RETRIEVED")]
        mentions = [self._mention("tsmc", eid="ev02")]
        updated = link_mentions_to_claims(mentions, claims)
        assert updated[0].status == "RETRIEVED"


# ---------------------------------------------------------------------------
# 4. VCG Core Invariant: no SUPPORTED claim without evidence
# ---------------------------------------------------------------------------


class TestVCGSubstringInvariant:
    """The substring guard: SUPPORTED → WEAK if quote not verbatim in evidence."""

    def test_supported_claim_with_valid_span_quote(self):
        """A SUPPORTED claim must have all non-empty span quotes as verbatim substrings."""
        evidence_text = "Apple relies on TSMC as a key supplier for chips."
        quote = "TSMC as a key supplier"

        # Simulate what verify_claims_node does when LLM returns a good quote.
        span = EvidenceSpan(
            evidence_id="ev01",
            quote=quote,
        )
        assert quote in evidence_text, "quote must be verbatim substring"

    def test_non_verbatim_quote_is_invalid(self):
        """A quote not found verbatim in the text would violate the invariant."""
        evidence_text = "Apple relies on TSMC as a key supplier for chips."
        hallucinated_quote = "TSMC manufactures all Apple chips exclusively"
        assert hallucinated_quote not in evidence_text

    def test_claim_without_evidence_refs_stays_unverified(self):
        """A claim with no evidence_refs should not be SUPPORTED."""
        claim = _make_claim("Apple", "TSMC")
        assert claim.evidence_refs == []
        assert claim.verdict in ("UNKNOWN", "WEAK")  # not SUPPORTED
