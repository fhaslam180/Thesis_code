"""PROV-O JSON-LD provenance export.

Produces a simplified JSON-LD document using core PROV-O classes:
  prov:Entity, prov:Activity, prov:Agent
  prov:wasGeneratedBy, prov:wasDerivedFrom, prov:wasAssociatedWith,
  prov:wasAttributedTo, prov:used
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ProvenanceExporter:
    """Export PROV-O JSON-LD linking Claims ↔ EvidencePackets ↔ pipeline run."""

    def __init__(
        self,
        company: str,
        claims: list[dict],
        evidence_packets: list[dict],
        condition: str,
        run_id: str | None = None,
    ) -> None:
        self.company = company
        self.claims = claims
        self.evidence_packets = evidence_packets
        self.condition = condition
        self.run_id = run_id or str(uuid.uuid4())[:8]

    def export(self, out_path: Path | None = None) -> dict:
        """Build the provenance document and optionally write it to *out_path*."""
        doc = self._build_document()
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return doc

    def _build_document(self) -> dict:
        run_iri = f"bor:run/{self.run_id}"

        graph: list[dict] = [
            # Pipeline run activity
            {
                "@id": run_iri,
                "@type": "prov:Activity",
                "bor:company": self.company,
                "bor:condition": self.condition,
                "bor:timestamp": datetime.now(timezone.utc).isoformat(),
                "prov:wasAssociatedWith": {"@id": "bor:agent/vcg-pipeline"},
            },
        ]

        # Evidence entities
        for ep in self.evidence_packets:
            ev_id = f"bor:evidence/{ep.get('evidence_id', 'unknown')}"
            graph.append({
                "@id": ev_id,
                "@type": "prov:Entity",
                "schema:url": ep.get("final_url") or ep.get("url", ""),
                "schema:name": ep.get("title", ""),
                "bor:contentHash": ep.get("content_hash", ""),
                "bor:httpStatus": ep.get("http_status", 0),
                "bor:domain": ep.get("domain", ""),
                "prov:wasAttributedTo": {"@id": run_iri},
            })

        # Claim entities
        for claim in self.claims:
            claim_id_val = claim.get("claim_id", "unknown")
            claim_iri = f"bor:claim/{claim_id_val}"

            derived_from = [
                {"@id": f"bor:evidence/{eid}"}
                for eid in claim.get("evidence_refs", [])
            ]

            entry: dict = {
                "@id": claim_iri,
                "@type": "prov:Entity",
                "bor:subject": claim.get("subject_entity_id", ""),
                "bor:object": claim.get("object_entity_id", ""),
                "bor:verdict": claim.get("verdict", "UNKNOWN"),
                "bor:status": claim.get("status", "PROPOSED"),
                "bor:tier": claim.get("tier", 1),
                "prov:wasGeneratedBy": {"@id": run_iri},
            }

            if claim.get("verdict_explanation"):
                entry["bor:verdictExplanation"] = claim["verdict_explanation"]

            # Include first supporting span quote if present
            spans = claim.get("supporting_spans", [])
            if spans and spans[0].get("quote"):
                entry["bor:supportingQuote"] = spans[0]["quote"]

            if derived_from:
                entry["prov:wasDerivedFrom"] = derived_from

            graph.append(entry)

        return {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "schema": "https://schema.org/",
                "bor": "https://example.org/bor#",
            },
            "@graph": graph,
        }
