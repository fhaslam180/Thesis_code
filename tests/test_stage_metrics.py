"""Tests for compute_stage_metrics and print_quality_report stage breakdown."""
from __future__ import annotations

import io

from bor_risk.evaluate import compute_stage_metrics


def _claim(status="PROPOSED", verdict="UNKNOWN", discovery_url="", evidence_refs=None,
           supporting_spans=None, verdict_explanation=""):
    return {
        "claim_id": "abc123456789",
        "display_id": "C001",
        "subject_entity_id": "Apple",
        "object_entity_id": "TSMC",
        "normalized_name": "tsmc",
        "aliases": [],
        "claim_type": "SUPPLIES_TO",
        "status": status,
        "verdict": verdict,
        "confidence": 0.9,
        "evidence_refs": evidence_refs or [],
        "supporting_spans": supporting_spans or [],
        "location_candidates": [],
        "tier": 1,
        "rationale": "",
        "verdict_explanation": verdict_explanation,
        "best_evidence_quality": 0.5,
        "discovery_url": discovery_url,
    }


def _packet(evidence_id="E1", domain="bloomberg.com", source_quality=0.8, source_type="news"):
    return {
        "evidence_id": evidence_id,
        "url": "https://bloomberg.com/x",
        "final_url": "https://bloomberg.com/x",
        "domain": domain,
        "retrieved_at": "2025-01-01T00:00:00Z",
        "content_hash": evidence_id * 16,
        "mime_type": "text/html",
        "source_quality": source_quality,
        "source_type": source_type,
    }


class TestComputeStageMetrics:

    def test_returns_all_four_stage_keys(self):
        state = {"claims": [], "suppliers": [], "evidence_packets": []}
        result = compute_stage_metrics(state)
        assert set(result.keys()) == {"discovery", "retrieval", "verification", "source_quality"}

    def test_rag_seeded_count_from_claims(self):
        claims = [
            _claim(discovery_url="https://example.com/supplier"),
            _claim(discovery_url=""),
            _claim(discovery_url=""),
        ]
        state = {"claims": claims, "suppliers": [], "evidence_packets": []}
        result = compute_stage_metrics(state)
        assert result["discovery"]["rag_seeded_count"] == 1
        assert result["discovery"]["llm_only_count"] == 2

    def test_final_report_count_from_suppliers(self):
        # strict mode: claims=3, suppliers=1 (after filtering)
        claims = [_claim(), _claim(), _claim()]
        suppliers = [{"name": "TSMC"}]
        state = {"claims": claims, "suppliers": suppliers, "evidence_packets": []}
        result = compute_stage_metrics(state)
        assert result["discovery"]["final_report_count"] == 1
        assert result["discovery"]["candidate_count"] == 3

    def test_retrieval_rate(self):
        claims = [
            _claim(status="RETRIEVED"),
            _claim(status="VERIFIED"),
            _claim(status="PROPOSED"),
        ]
        state = {"claims": claims, "suppliers": [], "evidence_packets": []}
        result = compute_stage_metrics(state)
        assert result["retrieval"]["retrieved_count"] == 2
        assert result["retrieval"]["retrieval_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_verification_verdict_counts(self):
        claims = [
            _claim(status="VERIFIED", verdict="SUPPORTED"),
            _claim(status="VERIFIED", verdict="WEAK"),
            _claim(status="VERIFIED", verdict="DISPUTED"),
            _claim(status="VERIFIED", verdict="UNKNOWN"),
        ]
        state = {"claims": claims, "suppliers": [], "evidence_packets": []}
        result = compute_stage_metrics(state)
        v = result["verification"]
        assert v["supported"] == 1
        assert v["weak"] == 1
        assert v["disputed"] == 1
        assert v["unknown"] == 1
        assert v["verified_count"] == 4

    def test_source_type_counts_from_packets(self):
        packets = [
            _packet("E1", source_type="news"),
            _packet("E2", source_type="news", source_quality=0.8),
            _packet("E3", source_type="official_report_candidate", source_quality=0.7),
        ]
        state = {"claims": [], "suppliers": [], "evidence_packets": packets}
        result = compute_stage_metrics(state)
        sc = result["source_quality"]
        assert sc["source_type_counts"]["news"] == 2
        assert sc["source_type_counts"]["official_report_candidate"] == 1
        assert sc["high_quality_count"] == 3  # all >= 0.7


class TestPrintQualityReportStageBreakdown:

    def test_stage_breakdown_section_present(self):
        from unittest.mock import patch
        from bor_risk.evaluate import print_quality_report

        state = {
            "claims": [],
            "suppliers": [],
            "evidence_packets": [],
            "hazard_scores": [],
            "pre_dedup_supplier_count": 0,
        }
        buf = io.StringIO()
        with patch("bor_risk.evaluate.load_reference_set", return_value={"metadata": {}}):
            print_quality_report("Apple", state, file=buf)

        output = buf.getvalue()
        assert "Stage Breakdown" in output
        assert "Discovery:" in output
        assert "Retrieval:" in output
        assert "Verification:" in output
        assert "Src quality:" in output


# ---------------------------------------------------------------------------
# filter_verified_suppliers_node
# ---------------------------------------------------------------------------

def _supplier(name: str) -> dict:
    return {"name": name, "lat": 24.8, "lon": 121.0, "tier": 1}


def _verified_claim(name: str, verdict: str, quality: float = 0.8) -> dict:
    return {
        "object_entity_id": name,
        "status": "VERIFIED",
        "verdict": verdict,
        "best_evidence_quality": quality,
    }


class TestFilterVerifiedSuppliersNode:

    def test_no_op_when_not_strict(self):
        from bor_risk.graph import filter_verified_suppliers_node
        suppliers = [_supplier("TSMC"), _supplier("Foxconn")]
        state = {"strict_mode": False, "suppliers": suppliers, "claims": []}
        result = filter_verified_suppliers_node(state)
        assert result["suppliers"] == suppliers

    def test_removes_unknown_in_strict(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "UNKNOWN", 0.8)],
        }
        result = filter_verified_suppliers_node(state)
        assert result["suppliers"] == []

    def test_keeps_supported_quality_04(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "SUPPORTED", 0.4)],
        }
        result = filter_verified_suppliers_node(state)
        assert len(result["suppliers"]) == 1

    def test_drops_supported_below_quality(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "SUPPORTED", 0.3)],
        }
        result = filter_verified_suppliers_node(state)
        assert result["suppliers"] == []

    def test_keeps_weak_quality_05(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "WEAK", 0.5)],
        }
        result = filter_verified_suppliers_node(state)
        assert len(result["suppliers"]) == 1

    def test_drops_weak_below_quality(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "WEAK", 0.4)],
        }
        result = filter_verified_suppliers_node(state)
        assert result["suppliers"] == []

    def test_claims_unchanged(self):
        from bor_risk.graph import filter_verified_suppliers_node
        state = {
            "strict_mode": True,
            "suppliers": [_supplier("TSMC")],
            "claims": [_verified_claim("TSMC", "SUPPORTED", 0.8)],
        }
        result = filter_verified_suppliers_node(state)
        assert "claims" not in result


import pytest
