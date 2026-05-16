"""Tests for verification rubric gates (direction, quality, corroboration, strict mode)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bor_risk.models import (
    Claim,
    EvidencePacket,
    LocationCandidate,
    VerificationVerdict,
)


def _make_claim(**kwargs):
    defaults = dict(
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
        evidence_refs=["E1"],
    )
    defaults.update(kwargs)
    return Claim(**defaults)


def _make_packet(evidence_id="E1", source_quality=0.8, domain="bloomberg.com", **kwargs):
    return {
        "evidence_id": evidence_id,
        "url": "https://bloomberg.com/x",
        "final_url": "https://bloomberg.com/x",
        "domain": domain,
        "retrieved_at": "2025-01-01T00:00:00Z",
        "content_hash": ("a" * 64),
        "mime_type": "text/html",
        "source_quality": source_quality,
        "source_type": "news",
        **kwargs,
    }


def _run_verify_node(claims_dicts, packets_list, state_extras=None):
    """Run verify_claims_node with patched evidence store."""
    from bor_risk.graph import verify_claims_node

    packets_by_id = {p["evidence_id"]: p for p in packets_list}
    contents = {p["evidence_id"]: "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports" for p in packets_list}

    state = {
        "company": "Apple",
        "claims": claims_dicts,
        "evidence_packets": packets_list,
        "no_verify": False,
        **(state_extras or {}),
    }

    with patch("bor_risk.graph._STORE") as mock_store:
        mock_store.read_snapshot.side_effect = lambda p: contents.get(p.evidence_id, "")
        with patch("bor_risk.graph.verify_claim_against_evidence") as mock_verify:
            yield mock_verify, mock_store, state


# ---------------------------------------------------------------------------
# Direction gate
# ---------------------------------------------------------------------------

class TestDirectionGate:

    def test_supported_unclear_direction_downgraded_to_weak(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        packet = _make_packet(source_quality=0.8)

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="unclear",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        out_claim = result["claims"][0]
        assert out_claim["verdict"] == "WEAK"
        assert "direction unclear" in out_claim["verdict_explanation"]

    def test_supported_clear_direction_stays_supported(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        packet = _make_packet(source_quality=0.8)

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        assert result["claims"][0]["verdict"] == "SUPPORTED"


# ---------------------------------------------------------------------------
# Source quality gate
# ---------------------------------------------------------------------------

class TestSourceQualityGate:

    def test_supported_low_quality_capped_to_weak(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        packet = _make_packet(source_quality=0.2, domain="reddit.com")

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        out = result["claims"][0]
        assert out["verdict"] == "WEAK"
        assert "low source quality" in out["verdict_explanation"]

    def test_supported_sufficient_quality_stays_supported(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        packet = _make_packet(source_quality=0.5, domain="wikipedia.org")

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        assert result["claims"][0]["verdict"] == "SUPPORTED"

    def test_best_evidence_quality_stored_on_claim(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        packet = _make_packet(source_quality=0.8)

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        assert result["claims"][0]["best_evidence_quality"] == 0.8

    def test_old_packet_without_source_quality_defaults_to_0_5(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim()
        # Simulate a legacy packet missing source_quality
        packet = {
            "evidence_id": "E1",
            "url": "https://x.com",
            "final_url": "https://x.com",
            "domain": "x.com",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "content_hash": "a" * 64,
            "mime_type": "text/html",
            # no source_quality key
        }

        verdict = VerificationVerdict(
            verdict="SUPPORTED", rationale="x", supporting_quote="TSMC supplies",
            direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [packet],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=verdict):
                result = verify_claims_node(state)

        # quality defaults to 0.5 — no spurious downgrade, claim stays SUPPORTED
        out = result["claims"][0]
        assert out["verdict"] == "SUPPORTED"
        assert out["best_evidence_quality"] == 0.5


# ---------------------------------------------------------------------------
# Corroboration
# ---------------------------------------------------------------------------

class TestCorroboration:

    def _run_two_packet(self, q1, q2, quote1="TSMC supplies chips", direction="supplier_to_company"):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim(evidence_refs=["E1", "E2"])
        p1 = _make_packet("E1", source_quality=q1, domain="bloomberg.com")
        p2 = _make_packet("E2", source_quality=q2, domain="reuters.com",
                          content_hash="b" * 64)

        weak = VerificationVerdict(
            verdict="WEAK", rationale="co-mention",
            supporting_quote=quote1, direction=direction,
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [p1, p2],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=weak):
                result = verify_claims_node(state)

        return result["claims"][0]

    def test_weak_plus_two_domains_upgrades_to_supported(self):
        out = self._run_two_packet(0.8, 0.8)
        assert out["verdict"] == "SUPPORTED"
        assert "corroborated" in out["verdict_explanation"]

    def test_corroboration_blocked_empty_quote(self):
        out = self._run_two_packet(0.8, 0.8, quote1="")
        assert out["verdict"] != "SUPPORTED"

    def test_corroboration_blocked_unclear_direction(self):
        out = self._run_two_packet(0.8, 0.8, direction="unclear")
        assert out["verdict"] != "SUPPORTED"

    def test_corroboration_blocked_low_quality_domains(self):
        out = self._run_two_packet(0.2, 0.2)
        assert out["verdict"] != "SUPPORTED"

    def test_corroboration_blocked_single_domain(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim(evidence_refs=["E1", "E2"])
        # Both packets from same domain — won't accumulate 2 independent domains
        p1 = _make_packet("E1", source_quality=0.8, domain="bloomberg.com")
        p2 = _make_packet("E2", source_quality=0.8, domain="bloomberg.com",
                          content_hash="b" * 64)

        weak = VerificationVerdict(
            verdict="WEAK", rationale="co-mention",
            supporting_quote="TSMC supplies", direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [p1, p2],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=weak):
                result = verify_claims_node(state)

        assert result["claims"][0]["verdict"] != "SUPPORTED"


# ---------------------------------------------------------------------------
# Tie-breaking: higher quality wins for equal-rank verdicts
# ---------------------------------------------------------------------------

class TestTieBreaking:

    def test_higher_quality_packet_wins_for_equal_verdicts(self):
        from bor_risk.graph import verify_claims_node

        claim = _make_claim(evidence_refs=["E1", "E2"])
        p_low = _make_packet("E1", source_quality=0.3, domain="unknown.io")
        p_high = _make_packet("E2", source_quality=0.8, domain="bloomberg.com",
                              content_hash="b" * 64)

        # Both return WEAK — high quality one should win, so best_evidence_quality = 0.8
        weak = VerificationVerdict(
            verdict="WEAK", rationale="co-mention",
            supporting_quote="TSMC supplies", direction="supplier_to_company",
        )

        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [p_low, p_high],
            "no_verify": False,
        }

        with patch("bor_risk.graph._STORE") as mock_store:
            mock_store.read_snapshot.return_value = "TSMC manufactures and supplies advanced semiconductor chips to Apple for use in iPhone and Mac products, according to industry reports"
            with patch("bor_risk.graph.verify_claim_against_evidence", return_value=weak):
                result = verify_claims_node(state)

        # With only 1 supporting domain (both packets give WEAK but different domains,
        # and low-quality packet (0.3) doesn't reach threshold for domain accumulation),
        # bloomberg domain is accumulated, unknown.io is not (quality 0.3 < 0.5).
        # The best_evidence_quality should be the higher-quality packet's quality.
        assert result["claims"][0]["best_evidence_quality"] == 0.8


# ---------------------------------------------------------------------------
# Strict-mode gate
# ---------------------------------------------------------------------------

class TestStrictModeGate:

    def _make_verified_claim(self, verdict, best_evidence_quality):
        return {
            "claim_id": "abc123456789",
            "display_id": "C001",
            "subject_entity_id": "Apple",
            "object_entity_id": "TSMC",
            "normalized_name": "tsmc",
            "aliases": [],
            "claim_type": "SUPPLIES_TO",
            "status": "VERIFIED",
            "verdict": verdict,
            "confidence": 0.9,
            "evidence_refs": [],
            "supporting_spans": [],
            "location_candidates": [],
            "tier": 1,
            "rationale": "",
            "verdict_explanation": "",
            "best_evidence_quality": best_evidence_quality,
            "discovery_url": "",
        }

    def _run_aggregate(self, claim_dict):
        from bor_risk.graph import aggregate_risk_node
        supplier = {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
                    "confidence": 0.9, "evidence_source": "web_verified",
                    "evidence_ids": [], "industry": "", "product_category": "",
                    "location_description": "", "relationship_type": "",
                    "verification_url": "", "verification_snippet": "", "source_url": ""}
        state = {
            "claims": [claim_dict],
            "suppliers": [supplier],
            "hazard_scores": [],
            "strict_mode": True,
        }
        with patch("bor_risk.graph.compute_risk_summary", return_value={}):
            return aggregate_risk_node(state)

    def test_supported_low_quality_excluded(self):
        claim = self._make_verified_claim("SUPPORTED", 0.3)
        result = self._run_aggregate(claim)
        assert result.get("suppliers") == [] or "suppliers" not in result or (
            result.get("suppliers", []) == [] and result.get("company_risk_summary") == {}
        )

    def test_weak_low_quality_excluded(self):
        claim = self._make_verified_claim("WEAK", 0.3)
        result = self._run_aggregate(claim)
        assert result.get("suppliers", []) == []

    def test_weak_sufficient_quality_included(self):
        claim = self._make_verified_claim("WEAK", 0.6)
        result = self._run_aggregate(claim)
        assert len(result.get("suppliers", [])) == 1

    def test_supported_sufficient_quality_included(self):
        claim = self._make_verified_claim("SUPPORTED", 0.5)
        result = self._run_aggregate(claim)
        assert len(result.get("suppliers", [])) == 1
