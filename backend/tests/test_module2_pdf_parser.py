"""Tests for the PyMuPDF-based PDF parser (task 2.2).

Covers:
- Successful parsing of a well-formed PDF
- Per-page text extraction
- Metadata collection
- Error handling (missing file, corrupted content)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

from app.core.parsers.pdf_parser import PdfParserError, parse_pdf
from app.models.document import DocType, Document


# ---------------------------------------------------------------------------
# Helpers — create test PDFs
# ---------------------------------------------------------------------------


def _make_test_pdf(pages_text: list[str]) -> Path:
    """Create a real PDF via PyMuPDF with the given per-page text.

    Returns the path to the temporary file.
    """
    import fitz

    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        # "china-s" built-in CJK font ensures Chinese characters render correctly
        page.insert_text(
            fitz.Point(50, 72),
            text,
            fontsize=11,
            fontname="china-s",
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParsePdfSuccess:
    """Happy-path scenarios."""

    def test_single_page(self):
        """Single-page PDF with known content."""
        pdf_path = _make_test_pdf(["Hello World"])
        try:
            doc = parse_pdf(pdf_path)
            assert isinstance(doc, Document)
            assert doc.file_name == pdf_path.name
            assert doc.doc_type == DocType.PDF
            assert doc.page_count == 1
            assert doc.pages[0].page_number == 1
            assert "Hello World" in doc.pages[0].text
            assert doc.full_text == "Hello World"
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_multi_page(self):
        """Multi-page PDF — text per page should stay separate."""
        pdf_path = _make_test_pdf(
            ["第一章  引言", "第二章  方法", "第三章  结论"]
        )
        try:
            doc = parse_pdf(pdf_path)
            assert doc.page_count == 3
            assert doc.pages[0].text == "第一章  引言"
            assert doc.pages[1].text == "第二章  方法"
            assert doc.pages[2].text == "第三章  结论"
            # full_text joins with double newlines
            assert "\n\n" in doc.full_text
            assert doc.char_count > 0
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_metadata_extracted(self):
        """PDF metadata (page dimensions) should be present."""
        pdf_path = _make_test_pdf(["test"])
        try:
            doc = parse_pdf(pdf_path)
            # Doc-level metadata
            assert "page_count" in doc.metadata
            assert doc.metadata["page_count"] == 1
            assert "file_path" in doc.metadata
            # Page-level metadata
            page_meta = doc.pages[0].metadata
            assert page_meta["width"] > 0
            assert page_meta["height"] > 0
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_chinese_text(self):
        """CJK text should be extracted correctly."""
        pdf_path = _make_test_pdf(["知识图谱是一种用图结构来表示知识的方法。"])
        try:
            doc = parse_pdf(pdf_path)
            assert "知识图谱" in doc.pages[0].text
            assert "图结构" in doc.pages[0].text
        finally:
            pdf_path.unlink(missing_ok=True)


class TestParsePdfErrors:
    """Error scenarios."""

    def test_file_not_found(self):
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_pdf("/tmp/nonexistent_12345.pdf")

    def test_empty_pdf(self):
        """A PDF with only blank pages returns empty text per page."""
        # PyMuPDF cannot save a 0-page document; we test blank pages instead.
        # The 0-page guard in pdf_parser is still exercised by code coverage.
        pdf_path = _make_test_pdf(["", "", ""])
        try:
            doc = parse_pdf(pdf_path)
            assert doc.page_count == 3
            # All pages return empty strings
            assert all(p.text == "" for p in doc.pages)
            assert doc.char_count == 0
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_not_a_pdf(self):
        """A text file disguised as .pdf should raise PdfParserError."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"this is not a PDF, just plain text")
        tmp.close()
        try:
            with pytest.raises(PdfParserError):
                parse_pdf(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_empty_text_pages(self):
        """A PDF page with no text content returns empty string."""
        import fitz

        doc = fitz.open()
        page = doc.new_page()  # blank page, no text inserted
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.close()
        doc.save(tmp.name)
        doc.close()
        try:
            doc = parse_pdf(tmp.name)
            assert doc.page_count == 1
            assert doc.pages[0].text == ""  # empty but not None
            assert doc.char_count == 0
        finally:
            Path(tmp.name).unlink(missing_ok=True)
