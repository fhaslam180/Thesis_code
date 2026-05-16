"""Tests for RAG-based supplier discovery and its evidence bridge."""
from __future__ import annotations

from unittest.mock import patch


def _make_supplier(
    name: str,
    confidence: float = 0.9,
    source_url: str = "https://x.com",
    verification_snippet: str = "confirmed supplier",
):
    from bor_risk.models import Supplier
    return Supplier(
        name=name, lat=24.8, lon=121.0, tier=1, confidence=confidence,
        evidence_ids=[f"RAG-{name[:4]}"], evidence_source="llm_only",
        source_url=source_url, verification_snippet=verification_snippet,
    )


_SNIPPETS = [{"url": "https://x.com", "content": "TSMC manufactures chips for Apple."}]


class TestRAGDiscovery:

    def test_tier1_suppliers_in_result(self):
        from bor_risk.tools import discover_suppliers_rag
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]):
            with patch("bor_risk.search.search_web", return_value=_SNIPPETS):
                with patch("bor_risk.tools.extract_suppliers_from_snippets",
                           return_value=[_make_supplier("TSMC")]):
                    suppliers, *_ = discover_suppliers_rag("Apple", tier_depth=1)
        assert any(s.name == "TSMC" for s in suppliers)

    def test_evidence_source_stays_llm_only(self):
        from bor_risk.tools import discover_suppliers_rag
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]):
            with patch("bor_risk.search.search_web", return_value=_SNIPPETS):
                with patch("bor_risk.tools.extract_suppliers_from_snippets",
                           return_value=[_make_supplier("TSMC")]):
                    suppliers, *_ = discover_suppliers_rag("Apple", tier_depth=1)
        assert all(s.evidence_source == "llm_only" for s in suppliers)

    def test_source_url_populated(self):
        from bor_risk.tools import discover_suppliers_rag
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]):
            with patch("bor_risk.search.search_web", return_value=_SNIPPETS):
                with patch("bor_risk.tools.extract_suppliers_from_snippets",
                           return_value=[_make_supplier("TSMC", source_url="https://x.com")]):
                    suppliers, *_ = discover_suppliers_rag("Apple", tier_depth=1)
        assert all(s.source_url for s in suppliers)

    def test_tier2_gate_low_confidence_skipped(self):
        from bor_risk.tools import discover_suppliers_rag
        low = _make_supplier("Weak", confidence=0.5)
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]) as mock_gq:
            with patch("bor_risk.search.search_web", return_value=[]):
                with patch("bor_risk.tools.extract_suppliers_from_snippets", return_value=[low]):
                    discover_suppliers_rag("Apple", tier_depth=2)
        assert mock_gq.call_count == 1  # no tier-2 query call

    def test_tier2_gate_no_snippet_skipped(self):
        from bor_risk.tools import discover_suppliers_rag
        no_quote = _make_supplier("NQ", confidence=0.9, verification_snippet="")
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]) as mock_gq:
            with patch("bor_risk.search.search_web", return_value=[]):
                with patch("bor_risk.tools.extract_suppliers_from_snippets",
                           return_value=[no_quote]):
                    discover_suppliers_rag("Apple", tier_depth=2)
        assert mock_gq.call_count == 1

    def test_tier2_gate_high_confidence_expands(self):
        from bor_risk.tools import discover_suppliers_rag
        # Use a quality URL (>= 0.5) so the source quality gate passes
        high = _make_supplier("TSMC", confidence=0.9, source_url="https://bloomberg.com/apple-tsmc")
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]) as mock_gq:
            with patch("bor_risk.search.search_web", return_value=[]):
                with patch("bor_risk.tools.extract_suppliers_from_snippets",
                           side_effect=[[high], []]):
                    discover_suppliers_rag("Apple", tier_depth=2)
        assert mock_gq.call_count == 2

    def test_snapshot_mode_uses_cache(self):
        from bor_risk.tools import discover_suppliers_rag
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]):
            with patch("bor_risk.search.search_web_snapshot", return_value=[]) as mock_snap:
                with patch("bor_risk.tools.extract_suppliers_from_snippets", return_value=[]):
                    discover_suppliers_rag("Apple", tier_depth=1, snapshot_mode=True)
        mock_snap.assert_called()

    def test_pre_dedup_count_in_meta(self):
        from bor_risk.tools import discover_suppliers_rag
        with patch("bor_risk.tools.generate_discovery_queries", return_value=["q"]):
            with patch("bor_risk.search.search_web", return_value=[]):
                with patch("bor_risk.tools.extract_suppliers_from_snippets", return_value=[]):
                    _, _, _, meta = discover_suppliers_rag("Apple", tier_depth=1)
        assert "pre_dedup_supplier_count" in meta

    def test_source_id_validated_against_snippet(self):
        """LLM returning candidate_id not in the index must be dropped."""
        from bor_risk.tools import extract_suppliers_from_snippets
        from bor_risk.models import NERCandidateResult, NERExtractionResponse
        bad = NERExtractionResponse(
            focal_company="Apple", tier=1,
            verified_suppliers=[NERCandidateResult(
                candidate_id="C99",  # not in candidate_index
                evidence_quote="TSMC manufactures chips for Apple.",
                confidence=0.9,
            )],
        )
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = bad
                result = extract_suppliers_from_snippets("Apple", "Apple", _SNIPPETS, tier=1)
        assert result == []  # C99 not in candidate_index → dropped

    def test_evidence_quote_validated_verbatim(self):
        """A quote that is NOT in the snippet content must result in confidence=0."""
        from bor_risk.tools import extract_suppliers_from_snippets
        from bor_risk.models import NERCandidateResult, NERExtractionResponse
        fabricated = NERExtractionResponse(
            focal_company="Apple", tier=1,
            verified_suppliers=[NERCandidateResult(
                candidate_id="C1",  # valid — C1 maps to "TSMC" via mocked extract_orgs
                evidence_quote="this phrase does not appear in the snippet",
                confidence=0.9,
            )],
        )
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = fabricated
                result = extract_suppliers_from_snippets("Apple", "Apple", _SNIPPETS, tier=1)
        assert len(result) == 1
        assert result[0].confidence == 0.0
        assert result[0].verification_snippet == ""

    def test_discover_claims_node_routes_rag_when_web_enabled(self):
        from bor_risk.graph import discover_claims_node
        state = {"company": "Apple", "tier_depth": 1, "use_llm": True,
                 "enable_web": True, "snapshot_mode": False}
        with patch("bor_risk.graph.discover_suppliers_rag",
                   return_value=([], [], [], {"pre_dedup_supplier_count": 0})) as mock_rag:
            with patch("bor_risk.graph.discover_suppliers_llm") as mock_llm:
                discover_claims_node(state)
        mock_rag.assert_called_once()
        mock_llm.assert_not_called()

    def test_discover_claims_node_routes_llm_when_web_disabled(self):
        from bor_risk.graph import discover_claims_node
        state = {"company": "Apple", "tier_depth": 1, "use_llm": True,
                 "enable_web": False, "snapshot_mode": False}
        with patch("bor_risk.graph.discover_suppliers_llm",
                   return_value=([], [], [], {"pre_dedup_supplier_count": 0})) as mock_llm:
            with patch("bor_risk.graph.discover_suppliers_rag") as mock_rag:
                discover_claims_node(state)
        mock_llm.assert_called_once()
        mock_rag.assert_not_called()

    def test_rag_claim_has_discovery_url(self):
        """Claims from RAG-discovered suppliers must carry discovery_url."""
        from bor_risk.graph import discover_claims_node
        from bor_risk.models import Supplier
        sup = Supplier(name="TSMC", lat=24.8, lon=121.0, tier=1, confidence=0.9,
                       evidence_ids=["RAG-tsmc"], evidence_source="llm_only",
                       source_url="https://example.com/suppliers")
        state = {"company": "Apple", "tier_depth": 1, "use_llm": True,
                 "enable_web": True, "snapshot_mode": False}
        with patch("bor_risk.graph.discover_suppliers_rag",
                   return_value=([sup], [], [], {"pre_dedup_supplier_count": 1})):
            result = discover_claims_node(state)
        assert result["claims"][0]["discovery_url"] == "https://example.com/suppliers"

    def test_rag_claim_status_is_proposed(self):
        """RAG-discovered claims must be PROPOSED until retrieve_evidence fetches the page."""
        from bor_risk.graph import discover_claims_node
        from bor_risk.models import Supplier
        sup = Supplier(name="TSMC", lat=24.8, lon=121.0, tier=1, confidence=0.9,
                       evidence_ids=["RAG-tsmc"], evidence_source="llm_only",
                       source_url="https://example.com")
        state = {"company": "Apple", "tier_depth": 1, "use_llm": True,
                 "enable_web": True, "snapshot_mode": False}
        with patch("bor_risk.graph.discover_suppliers_rag",
                   return_value=([sup], [], [], {"pre_dedup_supplier_count": 1})):
            result = discover_claims_node(state)
        assert result["claims"][0]["status"] == "PROPOSED"

    def test_retrieve_evidence_fetches_discovery_url(self):
        """retrieve_evidence_node must prepend discovery_url to fetched URLs."""
        from bor_risk.graph import retrieve_evidence_node
        from bor_risk.models import Claim, LocationCandidate
        import os
        claim = Claim(
            claim_id="abc123", display_id="C001",
            subject_entity_id="Apple", object_entity_id="TSMC",
            normalized_name="tsmc", aliases=[], claim_type="SUPPLIES_TO",
            status="PROPOSED", verdict="UNKNOWN", confidence=0.9, tier=1,
            rationale="", location_candidates=[
                LocationCandidate(lat=24.8, lon=121.0, confidence=0.5, source="llm")
            ],
            discovery_url="https://apple-suppliers.example.com/tsmc",
        )
        state = {
            "company": "Apple",
            "claims": [claim.model_dump()],
            "evidence_packets": [],
            "snapshot_mode": False,
            "enable_web": True,
        }

        fetched_urls: list[str] = []

        def mock_fetch(url):
            fetched_urls.append(url)
            return "", "", "", url, 200

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            with patch("bor_risk.search.search_web", return_value=[]):
                with patch("bor_risk.graph.fetch_url_content", side_effect=mock_fetch):
                    retrieve_evidence_node(state)

        assert "https://apple-suppliers.example.com/tsmc" in fetched_urls
