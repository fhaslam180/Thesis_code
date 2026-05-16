"""Tests for NER-first supplier extraction (extract_suppliers_from_snippets replacement)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bor_risk.tools import (
    MAX_CANDIDATES,
    _clean_supplier_candidate_name,
    _has_company_suffix,
    _is_location_or_header_name,
    _is_supplier_list_source,
    _is_valid_supplier_candidate,
    _quote_mentions_candidate,
    _score_candidate_proximity,
    extract_orgs_from_text,
    extract_suppliers_from_snippets,
)


# ---------------------------------------------------------------------------
# extract_orgs_from_text
# ---------------------------------------------------------------------------

class TestExtractOrgsFromText:

    def test_spacy_orgs_returned_when_available(self):
        """When spaCy is available, ORG entities are included in results."""
        mock_ent_tsmc = MagicMock()
        mock_ent_tsmc.label_ = "ORG"
        mock_ent_tsmc.text = "TSMC"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent_tsmc]

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("bor_risk.tools._get_spacy_nlp", return_value=mock_nlp):
            result = extract_orgs_from_text("TSMC manufactures chips for Apple.")

        assert "TSMC" in result

    def test_regex_catches_acronyms_when_spacy_absent(self):
        """Acronym regex finds ALL-CAPS names even when spaCy raises ImportError."""
        with patch("bor_risk.tools._get_spacy_nlp", side_effect=ImportError):
            result = extract_orgs_from_text("TSMC and CATL supply components.")
        assert "TSMC" in result
        assert "CATL" in result

    def test_suffix_regex_catches_proper_nouns_when_spacy_absent(self):
        """Suffix regex finds capitalised multi-word names when spaCy unavailable."""
        with patch("bor_risk.tools._get_spacy_nlp", side_effect=OSError):
            result = extract_orgs_from_text("Samsung Display Co. supplies OLED panels.")
        assert any("Samsung" in r for r in result)

    def test_both_spacy_and_regex_results_merged(self):
        """spaCy ORGs and regex ORGs both appear in the merged list."""
        mock_ent = MagicMock()
        mock_ent.label_ = "ORG"
        mock_ent.text = "Apple"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("bor_risk.tools._get_spacy_nlp", return_value=mock_nlp):
            result = extract_orgs_from_text("Apple and TSMC announced a deal.")

        assert "Apple" in result   # from spaCy
        assert "TSMC" in result    # from acronym regex


# ---------------------------------------------------------------------------
# _has_company_suffix
# ---------------------------------------------------------------------------

class TestHasCompanySuffix:

    def test_inc_suffix(self):
        assert _has_company_suffix("Foxconn Inc") is True

    def test_ltd_suffix(self):
        assert _has_company_suffix("Murata Ltd") is True

    def test_kk_suffix(self):
        assert _has_company_suffix("Toyota K.K.") is True

    def test_no_suffix(self):
        assert _has_company_suffix("Supply Chain") is False


# ---------------------------------------------------------------------------
# candidate cleanup / rejection
# ---------------------------------------------------------------------------

class TestCandidateCleanup:

    def test_trims_trailing_location_after_legal_suffix(self):
        assert (
            _clean_supplier_candidate_name("Pegatron Corporation Jiangsu")
            == "Pegatron Corporation"
        )
        assert (
            _clean_supplier_candidate_name(
                "Pioneer Material Precision Tech Company Limited Chongqing"
            )
            == "Pioneer Material Precision Tech Company Limited"
        )

    def test_strips_leading_location_tokens(self):
        assert (
            _clean_supplier_candidate_name(
                "Taiwan Taiwan Platinum Optics Technology Incorporated"
            )
            == "Platinum Optics Technology Incorporated"
        )

    def test_rejects_location_only_candidates(self):
        assert _is_valid_supplier_candidate("China", list_context=True) is False
        assert _is_valid_supplier_candidate("Jiangsu China", list_context=True) is False
        assert _is_valid_supplier_candidate("Bangkok Thailand Supplier List", list_context=True) is False

    def test_rejects_table_header_candidates(self):
        assert (
            _is_valid_supplier_candidate(
                "SUPPLIER NAME PRIMARY LOCATIONS WHERE", list_context=True
            )
            is False
        )
        assert _is_valid_supplier_candidate("CLEAN ENERGY PROGRAM SUPPLIER", list_context=True) is False
        assert _is_valid_supplier_candidate("Contract Manufacturers", list_context=False) is False
        assert _is_valid_supplier_candidate("Shenzhen-based", list_context=False) is False
        assert _is_valid_supplier_candidate("FOR", list_context=True) is False

    def test_allows_real_supplier_list_names(self):
        assert _is_valid_supplier_candidate("TSMC", list_context=True) is True
        assert _is_valid_supplier_candidate("Murata Manufacturing", list_context=True) is True
        assert _is_valid_supplier_candidate("Jabil Incorporated", list_context=True) is True


# ---------------------------------------------------------------------------
# _quote_mentions_candidate
# ---------------------------------------------------------------------------

class TestQuoteMentionsCandidate:

    def test_exact_match(self):
        assert _quote_mentions_candidate("TSMC", "TSMC supplies chips to Apple") is True

    def test_alias_stripped_suffix(self):
        # "Samsung" should match full legal name via token overlap
        assert _quote_mentions_candidate(
            "Samsung Electronics Co., Ltd.",
            "Samsung manufactures display panels for Apple.",
        ) is True

    def test_unrelated_org_in_quote(self):
        # Quote mentions a different org entirely
        assert _quote_mentions_candidate("TSMC", "Apple buys from Foxconn.") is False

    def test_empty_quote(self):
        assert _quote_mentions_candidate("TSMC", "") is False


# ---------------------------------------------------------------------------
# supplier-list source handling
# ---------------------------------------------------------------------------

class TestSupplierListSource:

    def test_official_supplier_list_source_detected(self):
        assert _is_supplier_list_source(
            "https://www.apple.com/supplier-responsibility/pdf/Apple-Supplier-List.pdf",
            "Apple Supplier List 2024\nTSMC\nFoxconn",
            "Apple",
        ) is True

    def test_forum_supplier_list_source_rejected(self):
        assert _is_supplier_list_source(
            "https://www.reddit.com/r/apple/comments/supplier-list",
            "Apple Supplier List 2024\nTSMC",
            "Apple",
        ) is False


# ---------------------------------------------------------------------------
# extract_suppliers_from_snippets — integration (LLM mocked)
# ---------------------------------------------------------------------------

_SNIPPET_TSMC = {
    "url": "https://example.com/tsmc",
    "content": "TSMC manufactures chips and supplies Apple with advanced semiconductors.",
}


def _make_ner_response(candidate_id: str, quote: str, confidence: float = 0.9):
    from bor_risk.models import NERCandidateResult, NERExtractionResponse
    result = NERCandidateResult(
        candidate_id=candidate_id,
        evidence_quote=quote,
        confidence=confidence,
        relationship_type="component",
    )
    return NERExtractionResponse(focal_company="Apple", tier=1, verified_suppliers=[result])


class TestExtractSuppliersFromSnippets:

    def _call(self, snippets, ner_response, extra_patches=None):
        """Helper: patch extract_orgs_from_text to return TSMC and run extraction."""
        extra_patches = extra_patches or {}
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = (
                    ner_response
                )
                return extract_suppliers_from_snippets("Apple", "Apple", snippets, tier=1)

    def test_empty_snippets_returns_empty(self):
        with patch("bor_risk.tools._make_llm") as mock_llm:
            result = extract_suppliers_from_snippets("Apple", "Apple", [], tier=1)
        assert result == []
        mock_llm.assert_not_called()

    def test_no_candidates_after_filter_returns_empty(self):
        """If NER only finds the focal company, no LLM call is made."""
        snippets = [{"url": "https://x.com", "content": "Apple reported strong earnings."}]
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["Apple"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                result = extract_suppliers_from_snippets("Apple", "Apple", snippets, tier=1)
        assert result == []
        mock_llm.assert_not_called()

    def test_junk_candidate_filtered(self):
        """'Inc' alone must not become a candidate."""
        snippets = [{"url": "https://x.com", "content": "TSMC Inc supplies Apple."}]
        captured_prompts: list[str] = []

        def fake_invoke(msgs):
            captured_prompts.append(msgs[0].content)
            return _make_ner_response("C1", "TSMC Inc supplies Apple")

        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC Inc", "Inc"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.side_effect = (
                    fake_invoke
                )
                extract_suppliers_from_snippets("Apple", "Apple", snippets, tier=1)

        if captured_prompts:
            assert "\nInc " not in captured_prompts[0]

    def test_unknown_candidate_id_hard_rejected(self):
        """LLM returning C99 (not in index) must yield zero suppliers."""
        response = _make_ner_response("C99", "TSMC supplies Apple.")
        result = self._call([_SNIPPET_TSMC], response)
        assert result == []

    def test_fabricated_quote_not_in_content_yields_zero_confidence(self):
        """Quote not present in snippet → confidence zeroed, verification_snippet empty."""
        response = _make_ner_response("C1", "this phrase is completely fabricated")
        result = self._call([_SNIPPET_TSMC], response)
        assert len(result) == 1
        assert result[0].confidence == 0.0
        assert result[0].verification_snippet == ""

    def test_quote_in_content_but_not_mentioning_candidate_yields_zero_confidence(self):
        """Quote exists in snippet but does not contain candidate name → Gate 3 fires.

        Only TSMC is in the candidate list (Foxconn is not mocked as a candidate),
        so C1=TSMC is guaranteed. The LLM returns a quote that IS verbatim in the
        content but only mentions Foxconn, not TSMC.
        """
        snippet = {
            "url": "https://x.com",
            "content": "TSMC supplies chips to Apple. Foxconn assembles iPhones.",
        }
        # C1 = TSMC (only candidate); quote mentions Foxconn only → Gate 3 rejects
        response = _make_ner_response("C1", "Foxconn assembles iPhones.")
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = (
                    response
                )
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)
        tsmc_result = next((s for s in result if s.name == "TSMC"), None)
        assert tsmc_result is not None
        assert tsmc_result.confidence == 0.0
        assert tsmc_result.verification_snippet == ""

    def test_valid_candidate_and_quote_creates_supplier(self):
        """Valid C1 with verbatim quote containing org name → Supplier with confidence."""
        response = _make_ner_response("C1", "TSMC manufactures chips and supplies Apple")
        result = self._call([_SNIPPET_TSMC], response)
        assert len(result) == 1
        assert result[0].name == "TSMC"
        assert result[0].confidence == pytest.approx(0.9)
        assert result[0].source_url == "https://example.com/tsmc"

    def test_shortened_alias_in_quote_passes_gate3(self):
        """'Samsung' in quote must satisfy Gate 3 for candidate 'Samsung Electronics Co., Ltd.'"""
        snippet = {
            "url": "https://example.com/samsung",
            "content": "Samsung Electronics Co., Ltd. supplies OLED panels. Samsung is a key vendor.",
        }
        from bor_risk.models import NERCandidateResult, NERExtractionResponse
        full_name = "Samsung Electronics Co., Ltd."
        response = NERExtractionResponse(
            focal_company="Apple", tier=1,
            verified_suppliers=[NERCandidateResult(
                candidate_id="C1",
                evidence_quote="Samsung is a key vendor.",
                confidence=0.85,
                relationship_type="component",
            )],
        )
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=[full_name]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = (
                    response
                )
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)
        assert len(result) == 1
        assert result[0].name == full_name
        assert result[0].confidence == pytest.approx(0.85)

    def test_official_supplier_list_yields_suppliers_without_llm(self):
        """Structured supplier lists can seed claims using document-level context."""
        snippet = {
            "url": "https://www.apple.com/supplier-responsibility/pdf/Apple-Supplier-List.pdf",
            "content": (
                "Apple Supplier List 2024\n"
                "The following companies represent manufacturing locations.\n"
                "TSMC\n"
                "Hon Hai Precision Industry Co., Ltd.\n"
            ),
        }
        with patch("bor_risk.tools.extract_orgs_from_text",
                   return_value=["Apple", "TSMC", "Hon Hai Precision Industry Co., Ltd."]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        mock_llm.assert_not_called()
        assert {s.name for s in result} == {"TSMC", "Hon Hai Precision Industry Co., Ltd."}
        assert all(s.confidence == pytest.approx(0.8) for s in result)
        assert all(s.source_url == snippet["url"] for s in result)
        assert any("Apple Supplier List 2024" in s.verification_snippet for s in result)

    def test_supplier_list_filters_headers_locations_and_cleans_names(self):
        snippet = {
            "url": "https://www.apple.com/supplier-responsibility/pdf/Apple-Supplier-List.pdf",
            "content": (
                "Apple Supplier List 2024\n"
                "SUPPLIER NAME PRIMARY LOCATIONS WHERE MANUFACTURING FOR APPLE OCCURS\n"
                "J.Pond Industry (Dongguan) Company Limited Guangdong, Jiangsu China mainland "
                "Jabil Incorporated Guangdong, Jiangsu, Sichuan China mainland "
                "Pegatron Corporation Jiangsu, Shanghai China mainland\n"
            ),
        }
        with patch("bor_risk.tools.extract_orgs_from_text",
                   return_value=[
                       "China",
                       "Jiangsu",
                       "CLEAN ENERGY PROGRAM SUPPLIER",
                       "SUPPLIER NAME PRIMARY LOCATIONS WHERE",
                       "FOR",
                       "J.Pond Industry (Dongguan) Company Limited Guangdong",
                       "Jabil Incorporated Guangdong",
                       "Pegatron Corporation Jiangsu",
                   ]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        mock_llm.assert_not_called()
        assert {s.name for s in result} == {
            "J.Pond Industry (Dongguan) Company Limited",
            "Jabil Incorporated",
            "Pegatron Corporation",
        }

    def test_supplier_list_context_keeps_zero_proximity_name(self):
        """List-mode keeps listed proper nouns even when trigger words are distant."""
        filler = "x " * 120
        snippet = {
            "url": "https://www.apple.com/supplier-responsibility/pdf/Apple-Supplier-List.pdf",
            "content": f"Apple Supplier List 2024\n{filler}\nMurata Manufacturing\n",
        }
        with patch("bor_risk.tools.extract_orgs_from_text",
                   return_value=["Murata Manufacturing"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        mock_llm.assert_not_called()
        assert [s.name for s in result] == ["Murata Manufacturing"]

    def test_forum_supplier_list_still_uses_llm_classifier(self):
        """List-looking forum content is not accepted as structured source evidence."""
        snippet = {
            "url": "https://www.reddit.com/r/apple/comments/supplier-list",
            "content": "Apple Supplier List 2024\nTSMC supplies Apple with chips.",
        }
        response = _make_ner_response("C1", "TSMC supplies Apple with chips.")
        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.return_value = (
                    response
                )
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        mock_llm.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "TSMC"
        assert result[0].confidence == pytest.approx(0.9)

    def test_location_candidate_not_sent_to_llm_prompt(self):
        snippet = {
            "url": "https://x.com",
            "content": "Jiangsu is a province. TSMC supplies chips to Apple.",
        }
        captured_prompts: list[str] = []

        def fake_invoke(msgs):
            captured_prompts.append(msgs[0].content)
            return _make_ner_response("C1", "TSMC supplies chips to Apple")

        with patch("bor_risk.tools.extract_orgs_from_text", return_value=["Jiangsu", "TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.side_effect = (
                    fake_invoke
                )
                result = extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        assert [s.name for s in result] == ["TSMC"]
        assert captured_prompts
        candidate_lines = [
            line for line in captured_prompts[0].splitlines()
            if line.startswith("C")
        ]
        assert not any("Jiangsu" in line for line in candidate_lines)

    def test_zero_score_no_acronym_no_suffix_filtered(self):
        """'Annual Report' has no trigger proximity, is not an acronym, has no suffix → dropped.

        Content is long enough that the ±100-char window around 'Annual Report'
        does not reach 'TSMC supplies chips' at the end.
        """
        filler = "x " * 80  # ~160 chars of non-trigger text
        snippet = {
            "url": "https://x.com",
            "content": f"Annual Report: Corporate Overview. {filler} TSMC supplies chips to Apple.",
        }
        captured_prompts: list[str] = []

        def fake_invoke(msgs):
            captured_prompts.append(msgs[0].content)
            return _make_ner_response("C1", "TSMC supplies chips to Apple")

        with patch("bor_risk.tools.extract_orgs_from_text",
                   return_value=["Annual Report", "TSMC"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.side_effect = (
                    fake_invoke
                )
                extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        if captured_prompts:
            # Candidate ID lines only (C\d+: ...) — Annual Report should not be listed
            import re as _re
            cand_lines = [l for l in captured_prompts[0].splitlines()
                          if _re.match(r"C\d+:", l)]
            assert not any("Annual Report" in l for l in cand_lines)

    def test_duplicate_canonical_key_same_snippet_not_double_counted(self):
        """'TSMC' and 'TSMC Inc' normalize to same canonical key in same snippet → one ID."""
        snippet = {
            "url": "https://x.com",
            "content": "TSMC supplies Apple. TSMC Inc is a supplier.",
        }
        captured_candidates: list[str] = []

        def fake_invoke(msgs):
            captured_candidates.append(msgs[0].content)
            return _make_ner_response("C1", "TSMC supplies Apple")

        with patch("bor_risk.tools.extract_orgs_from_text",
                   return_value=["TSMC", "TSMC Inc"]):
            with patch("bor_risk.tools._make_llm") as mock_llm:
                mock_llm.return_value.with_structured_output.return_value.invoke.side_effect = (
                    fake_invoke
                )
                extract_suppliers_from_snippets("Apple", "Apple", [snippet], tier=1)

        if captured_candidates:
            import re as _re
            # Count only candidate ID lines (C1: ..., C2: ...) that mention TSMC
            cand_lines = [l for l in captured_candidates[0].splitlines()
                          if _re.match(r"C\d+:", l) and "tsmc" in l.lower()]
            assert len(cand_lines) <= 1


# ---------------------------------------------------------------------------
# _BAD_LAST_TOKENS — _is_location_or_header_name last-token checks
# ---------------------------------------------------------------------------

class TestBadLastTokens:

    def test_program_suffix_rejected(self):
        assert _is_location_or_header_name("American Manufacturing Program") is True

    def test_initiative_suffix_rejected(self):
        assert _is_location_or_header_name("Clean Energy Initiative") is True

    def test_council_suffix_rejected(self):
        assert _is_location_or_header_name("Global Supply Council") is True

    def test_samsung_manufacturing_kept(self):
        assert _is_location_or_header_name("Samsung Manufacturing") is False

    def test_tsmc_kept(self):
        assert _is_location_or_header_name("TSMC") is False
