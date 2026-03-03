"""Tests for PROV-O JSON-LD provenance export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bor_risk.provenance import ProvenanceExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exporter(
    company: str = "Apple",
    claims: list[dict] | None = None,
    evidence_packets: list[dict] | None = None,
    condition: str = "web_verify",
) -> ProvenanceExporter:
    return ProvenanceExporter(
        company=company,
        claims=claims or [],
        evidence_packets=evidence_packets or [],
        condition=condition,
        run_id="test-run-01",
    )


def _sample_claim() -> dict:
    return {
        "claim_id": "abc123def456",
        "display_id": "C001",
        "subject_entity_id": "Apple",
        "object_entity_id": "TSMC",
        "verdict": "SUPPORTED",
        "status": "VERIFIED",
        "tier": 1,
        "evidence_refs": ["ev0001"],
        "supporting_spans": [{"evidence_id": "ev0001", "quote": "Apple uses TSMC"}],
        "verdict_explanation": "Co-mention verified [direction=supplier_to_company]",
    }


def _sample_packet() -> dict:
    return {
        "evidence_id": "ev0001",
        "url": "https://reuters.com/article",
        "final_url": "https://reuters.com/article",
        "domain": "reuters.com",
        "title": "Apple TSMC partnership",
        "content_hash": "abcd1234" * 8,
        "http_status": 200,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProvenanceExporterBasic:
    def test_export_returns_dict(self):
        exp = _make_exporter()
        doc = exp.export()
        assert isinstance(doc, dict)

    def test_has_jsonld_context(self):
        doc = _make_exporter().export()
        assert "@context" in doc
        ctx = doc["@context"]
        assert "prov" in ctx
        assert "schema" in ctx
        assert "bor" in ctx

    def test_prov_context_value(self):
        doc = _make_exporter().export()
        assert "http://www.w3.org/ns/prov#" in doc["@context"]["prov"]

    def test_has_graph_key(self):
        doc = _make_exporter().export()
        assert "@graph" in doc
        assert isinstance(doc["@graph"], list)

    def test_graph_has_run_activity(self):
        doc = _make_exporter().export()
        activities = [
            node for node in doc["@graph"]
            if node.get("@type") == "prov:Activity"
        ]
        assert len(activities) == 1
        run = activities[0]
        assert run["bor:company"] == "Apple"
        assert run["bor:condition"] == "web_verify"

    def test_run_activity_has_id(self):
        doc = _make_exporter(company="Nike", condition="strict").export()
        activities = [n for n in doc["@graph"] if n.get("@type") == "prov:Activity"]
        assert activities[0]["@id"].startswith("bor:run/")


class TestProvenanceEvidenceEntities:
    def test_evidence_packets_become_entities(self):
        packets = [_sample_packet()]
        doc = _make_exporter(evidence_packets=packets).export()
        entities = [n for n in doc["@graph"] if n.get("@type") == "prov:Entity"
                    and n["@id"].startswith("bor:evidence/")]
        assert len(entities) == 1

    def test_evidence_entity_has_url(self):
        packets = [_sample_packet()]
        doc = _make_exporter(evidence_packets=packets).export()
        ev = next(
            n for n in doc["@graph"]
            if n.get("@id") == "bor:evidence/ev0001"
        )
        assert ev["schema:url"] == "https://reuters.com/article"

    def test_evidence_entity_has_domain(self):
        packets = [_sample_packet()]
        doc = _make_exporter(evidence_packets=packets).export()
        ev = next(n for n in doc["@graph"] if n.get("@id") == "bor:evidence/ev0001")
        assert ev["bor:domain"] == "reuters.com"

    def test_evidence_entity_attributed_to_run(self):
        packets = [_sample_packet()]
        doc = _make_exporter(evidence_packets=packets).export()
        ev = next(n for n in doc["@graph"] if n.get("@id") == "bor:evidence/ev0001")
        assert ev["prov:wasAttributedTo"]["@id"].startswith("bor:run/")


class TestProvenanceClaimEntities:
    def test_claims_become_entities(self):
        claims = [_sample_claim()]
        doc = _make_exporter(claims=claims).export()
        claim_entities = [
            n for n in doc["@graph"]
            if n.get("@type") == "prov:Entity"
            and n["@id"].startswith("bor:claim/")
        ]
        assert len(claim_entities) == 1

    def test_claim_entity_has_verdict(self):
        claims = [_sample_claim()]
        doc = _make_exporter(claims=claims).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == "bor:claim/abc123def456")
        assert ce["bor:verdict"] == "SUPPORTED"

    def test_claim_entity_has_subject_and_object(self):
        claims = [_sample_claim()]
        doc = _make_exporter(claims=claims).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == "bor:claim/abc123def456")
        assert ce["bor:subject"] == "Apple"
        assert ce["bor:object"] == "TSMC"

    def test_claim_was_generated_by_run(self):
        claims = [_sample_claim()]
        doc = _make_exporter(claims=claims).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == "bor:claim/abc123def456")
        assert ce["prov:wasGeneratedBy"]["@id"].startswith("bor:run/")

    def test_claim_derived_from_evidence(self):
        claims = [_sample_claim()]
        packets = [_sample_packet()]
        doc = _make_exporter(claims=claims, evidence_packets=packets).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == "bor:claim/abc123def456")
        assert "prov:wasDerivedFrom" in ce
        derived_ids = [d["@id"] for d in ce["prov:wasDerivedFrom"]]
        assert "bor:evidence/ev0001" in derived_ids

    def test_claim_without_evidence_refs_has_no_derived_from(self):
        claim = dict(_sample_claim())
        claim["evidence_refs"] = []
        doc = _make_exporter(claims=[claim]).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == f"bor:claim/{claim['claim_id']}")
        assert "prov:wasDerivedFrom" not in ce

    def test_claim_with_supporting_span_quote(self):
        claims = [_sample_claim()]
        doc = _make_exporter(claims=claims).export()
        ce = next(n for n in doc["@graph"] if n.get("@id") == "bor:claim/abc123def456")
        assert ce.get("bor:supportingQuote") == "Apple uses TSMC"


class TestProvenanceFileExport:
    def test_export_writes_valid_json_file(self, tmp_path):
        out = tmp_path / "provenance.jsonld"
        exp = _make_exporter(claims=[_sample_claim()], evidence_packets=[_sample_packet()])
        exp.export(out_path=out)
        assert out.exists()
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "@context" in doc
        assert "@graph" in doc

    def test_export_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "prov.jsonld"
        _make_exporter().export(out_path=out)
        assert out.exists()

    def test_multiple_claims_all_in_graph(self, tmp_path):
        claim1 = dict(_sample_claim())
        claim2 = dict(_sample_claim())
        claim2["claim_id"] = "xyz999aaa111"
        claim2["object_entity_id"] = "Foxconn"
        claim2["evidence_refs"] = []
        claim2["supporting_spans"] = []

        exp = _make_exporter(claims=[claim1, claim2])
        doc = exp.export()
        claim_nodes = [
            n for n in doc["@graph"]
            if n["@id"].startswith("bor:claim/")
        ]
        assert len(claim_nodes) == 2
