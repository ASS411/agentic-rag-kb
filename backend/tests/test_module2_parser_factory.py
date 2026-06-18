"""Tests for the parser factory (task 2.5).

Covers:
- Routing to PDF / Markdown / TXT parsers by file extension
- Unsupported file type rejection
- Direct imports of individual parsers
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.parsers import parse_document, parse_markdown, parse_pdf, parse_txt
from app.models.document import DocType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(filename: str, content: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False)
    tmp.write(content)
    tmp.close()
    # Rename to match the desired extension
    path = Path(tmp.name)
    new_path = path.with_name(filename)
    path.rename(new_path)
    return new_path


def _make_test_pdf() -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), "factory test", fontsize=11)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFactoryRouting:
    """parse_document() routes to the correct parser."""

    def test_routes_to_pdf_parser(self):
        path = _make_test_pdf()
        try:
            doc = parse_document(path)
            assert doc.doc_type == DocType.PDF
            assert "factory test" in doc.full_text
        finally:
            path.unlink(missing_ok=True)

    def test_routes_to_md_parser(self):
        path = _make_file("readme.md", b"# Hello\n\nWorld")
        try:
            doc = parse_document(path)
            assert doc.doc_type == DocType.MARKDOWN
            assert "[H1] Hello" in doc.full_text
        finally:
            path.unlink(missing_ok=True)

    def test_routes_to_txt_parser(self):
        path = _make_file("notes.txt", "plain text".encode("utf-8"))
        try:
            doc = parse_document(path)
            assert doc.doc_type == DocType.TXT
            assert "plain text" in doc.full_text
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_extension(self):
        path = _make_file("data.csv", b"a,b,c")
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                parse_document(path)
        finally:
            path.unlink(missing_ok=True)


class TestDirectImports:
    """Individual parsers are importable from the package."""

    def test_parse_pdf_direct(self):
        path = _make_test_pdf()
        try:
            doc = parse_pdf(path)
            assert doc.doc_type == DocType.PDF
        finally:
            path.unlink(missing_ok=True)

    def test_parse_markdown_direct(self):
        path = _make_file("test.md", b"# Title")
        try:
            doc = parse_markdown(path)
            assert doc.doc_type == DocType.MARKDOWN
        finally:
            path.unlink(missing_ok=True)

    def test_parse_txt_direct(self):
        path = _make_file("test.txt", "hello".encode("utf-8"))
        try:
            doc = parse_txt(path)
            assert doc.doc_type == DocType.TXT
        finally:
            path.unlink(missing_ok=True)
