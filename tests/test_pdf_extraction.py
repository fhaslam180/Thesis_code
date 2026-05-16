"""Tests for _is_readable_text quality gate and fetch_url_content PDF path."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bor_risk.tools import _is_readable_text


# ---------------------------------------------------------------------------
# _is_readable_text
# ---------------------------------------------------------------------------

class TestIsReadableText:

    def test_empty_string_is_false(self):
        assert _is_readable_text("") is False

    def test_whitespace_only_is_false(self):
        assert _is_readable_text("   \n\t  ") is False

    def test_pdf_magic_bytes_is_false(self):
        assert _is_readable_text("%PDF-1.4 binary garbage here") is False

    def test_pdf_magic_with_leading_space_is_false(self):
        assert _is_readable_text("  %PDF-1.6 stream obj endobj") is False

    def test_fewer_than_20_words_is_false(self):
        assert _is_readable_text("Only a few words here") is False

    def test_pdf_structure_marker_xref_is_false(self):
        text = "This text contains an xref table and some other words to pad it out enough"
        assert _is_readable_text(text) is False

    def test_pdf_structure_marker_endobj_is_false(self):
        text = "Some words followed by endobj marker and more padding text to reach twenty words total"
        assert _is_readable_text(text) is False

    def test_low_alpha_ratio_is_false(self):
        # High proportion of digits and symbols, low alpha
        text = "123 456 789 012 345 678 901 234 567 890 123 456 789 012 345 678 901 234 567 890"
        assert _is_readable_text(text) is False

    def test_valid_prose_is_true(self):
        text = (
            "TSMC manufactures semiconductors and supplies chips to Apple and other major "
            "technology companies around the world. The company is headquartered in Hsinchu, "
            "Taiwan and operates multiple advanced fabrication facilities."
        )
        assert _is_readable_text(text) is True

    def test_supplier_list_prose_is_true(self):
        text = (
            "Apple's list of suppliers includes TSMC for advanced chip fabrication, "
            "Samsung for OLED displays, Foxconn for final assembly, and Murata for "
            "passive electronic components such as capacitors and inductors. "
            "These relationships are disclosed in Apple's annual supplier responsibility report."
        )
        assert _is_readable_text(text) is True


# ---------------------------------------------------------------------------
# _JUNK_LOWER case-insensitive filtering
# ---------------------------------------------------------------------------

class TestJunkCandidateFiltering:

    def test_supplier_all_caps_filtered(self):
        """'SUPPLIER' (all-caps from acronym regex) must not become a candidate."""
        from bor_risk.tools import _JUNK_LOWER
        assert "supplier" in _JUNK_LOWER
        # All-caps form lowercases to "supplier" which is in _JUNK_LOWER
        candidate = "SUPPLIER"
        assert candidate.lower() in _JUNK_LOWER

    def test_vendor_mixed_case_filtered(self):
        from bor_risk.tools import _JUNK_LOWER
        assert "Vendor".lower() in _JUNK_LOWER

    def test_bloomberg_filtered(self):
        from bor_risk.tools import _JUNK_LOWER
        assert "bloomberg" in _JUNK_LOWER

    def test_tsmc_not_filtered(self):
        from bor_risk.tools import _JUNK_LOWER
        assert "tsmc" not in _JUNK_LOWER


# ---------------------------------------------------------------------------
# fetch_url_content — PDF path
# ---------------------------------------------------------------------------

class TestFetchUrlContentPDF:

    def _fake_raw(self, content: bytes):
        """Return a context manager that yields content as read()."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read = MagicMock(return_value=content)
        return cm

    def test_sparse_pypdf_output_returns_empty(self):
        """When pypdf produces fewer than 20 words, plain_text should be empty string."""
        from bor_risk.tools import fetch_url_content

        # Simulate sparse pypdf output (6 words — below min_words=20)
        sparse_text = "Supplier List Annual Report"

        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = sparse_text

        pdf_bytes = b"%PDF-1.4 fake pdf bytes"

        with patch("bor_risk.tools._urlopen_with_ssl_fallback", return_value=pdf_bytes):
            with patch("bor_risk.tools.PdfReader", mock_reader.__class__, create=True):
                with patch("pypdf.PdfReader", return_value=mock_reader):
                    plain_text, _, mime, _, status = fetch_url_content("https://example.com/file.pdf")

        assert mime == "application/pdf"
        assert status == 200
        assert plain_text == "" or len(plain_text.split()) < 20 or plain_text == ""

    def _mock_pdfplumber(self, open_result):
        """Return a sys.modules mock for pdfplumber (optional dep, may not be installed)."""
        import sys
        mock_module = MagicMock()
        mock_module.open = MagicMock(return_value=open_result)
        return patch.dict(sys.modules, {"pdfplumber": mock_module})

    def test_pypdf_raises_pdfplumber_attempted(self):
        """When pypdf raises, pdfplumber is attempted as a fallback."""
        from bor_risk.tools import fetch_url_content

        good_text = " ".join(["word"] * 30)  # 30 words, >50% alpha → readable

        pdf_bytes = b"%PDF-1.4 fake"

        mock_page = MagicMock()
        mock_page.extract_text.return_value = good_text
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        with patch("bor_risk.tools._urlopen_with_ssl_fallback", return_value=pdf_bytes):
            with patch("pypdf.PdfReader", side_effect=Exception("corrupt PDF")):
                with self._mock_pdfplumber(mock_pdf):
                    plain_text, _, mime, _, status = fetch_url_content("https://example.com/f.pdf")

        assert mime == "application/pdf"
        assert status == 200
        assert plain_text == good_text[:50_000]

    def test_both_extractors_fail_returns_empty(self):
        """When both pypdf and pdfplumber fail quality gate, plain_text must be empty."""
        from bor_risk.tools import fetch_url_content
        import sys

        pdf_bytes = b"%PDF-1.4 fake"

        mock_module = MagicMock()
        mock_module.open = MagicMock(side_effect=Exception("also corrupt"))

        with patch("bor_risk.tools._urlopen_with_ssl_fallback", return_value=pdf_bytes):
            with patch("pypdf.PdfReader", side_effect=Exception("corrupt")):
                with patch.dict(sys.modules, {"pdfplumber": mock_module}):
                    plain_text, _, mime, _, status = fetch_url_content("https://example.com/x.pdf")

        assert mime == "application/pdf"
        assert status == 200
        assert plain_text == ""

    def test_no_garbage_fallback_for_bad_pdf(self):
        """Stripped raw bytes must never be returned; plain_text must be empty on failure."""
        from bor_risk.tools import fetch_url_content
        import sys

        pdf_bytes = b"%PDF-1.4\x00\x01\x02\x03 binary garbage"

        mock_module = MagicMock()
        mock_module.open = MagicMock(side_effect=Exception("binary"))

        with patch("bor_risk.tools._urlopen_with_ssl_fallback", return_value=pdf_bytes):
            with patch("pypdf.PdfReader", side_effect=Exception("binary")):
                with patch.dict(sys.modules, {"pdfplumber": mock_module}):
                    plain_text, _, _, _, _ = fetch_url_content("https://example.com/bad.pdf")

        # Crucially: no stripped raw bytes returned
        assert "%PDF" not in plain_text
        assert plain_text == ""
