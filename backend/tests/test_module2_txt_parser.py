"""Tests for the TXT parser (task 2.4).

Covers:
- UTF-8 decoding
- GBK (Chinese) encoding detection and decoding
- Empty file handling
- Error scenarios
- Encoding metadata in document
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.parsers.txt_parser import TxtParserError, parse_txt
from app.models.document import DocType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_txt(content: bytes, suffix: str = ".txt") -> Path:
    """Write raw bytes to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseTxtSuccess:
    """Happy-path scenarios."""

    def test_utf8_text(self):
        """Standard UTF-8 text file."""
        content = "Hello, 世界! This is a test.\nSecond line."
        path = _write_txt(content.encode("utf-8"))
        try:
            doc = parse_txt(path)
            assert doc.file_name == path.name
            assert doc.doc_type == DocType.TXT
            assert doc.page_count == 1
            assert doc.pages[0].page_number == 1
            assert "Hello, 世界!" in doc.full_text
            assert "Second line" in doc.full_text
            assert doc.metadata["encoding"] in ("utf-8", "ascii")
        finally:
            path.unlink(missing_ok=True)

    def test_gbk_chinese(self):
        """GBK-encoded Chinese file — encoding should be detected."""
        text = "这是GBK编码的中文文本。\n第二行内容。"
        path = _write_txt(text.encode("gbk"))
        try:
            doc = parse_txt(path)
            assert "GBK编码" in doc.full_text
            assert "第二行" in doc.full_text
            # chardet should detect GBK (or GB2312) at high confidence
            enc = doc.metadata["encoding"]
            assert enc is not None
            # GBK/GB2312/GB18030 are all valid detections
            assert "gb" in enc.lower() or enc.lower() == "gb2312"
        finally:
            path.unlink(missing_ok=True)

    def test_ascii_only(self):
        """Pure ASCII text — encoding should be detected with high confidence."""
        content = b"The quick brown fox jumps over the lazy dog.\nLine two."
        path = _write_txt(content)
        try:
            doc = parse_txt(path)
            assert "quick brown fox" in doc.full_text
            # ASCII is a subset of UTF-8; chardet may report either
            enc = doc.metadata["encoding"].lower()
            assert enc in ("ascii", "utf-8")
            assert doc.metadata["confidence"] > 0.7
        finally:
            path.unlink(missing_ok=True)

    def test_empty_file(self):
        """Empty file returns a document with an empty page."""
        path = _write_txt(b"")
        try:
            doc = parse_txt(path)
            assert doc.page_count == 1
            assert doc.pages[0].text == ""
            assert doc.char_count == 0
        finally:
            path.unlink(missing_ok=True)

    def test_metadata_populated(self):
        """Document metadata includes encoding, confidence, and file path."""
        path = _write_txt("你好".encode("utf-8"))
        try:
            doc = parse_txt(path)
            meta = doc.metadata
            assert "encoding" in meta
            assert "confidence" in meta
            assert meta["size_bytes"] > 0
            assert meta["source_path"] == str(path)
        finally:
            path.unlink(missing_ok=True)


class TestParseTxtErrors:
    """Error scenarios."""

    def test_file_not_found(self):
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_txt("/tmp/nonexistent_abc_123.txt")

    def test_binary_garbage(self):
        """Random binary data — should not crash; decodes via fallback."""
        import os

        garbage = os.urandom(256)
        path = _write_txt(garbage)
        try:
            doc = parse_txt(path)
            # Must not raise — Latin-1 always decodes any byte sequence
            assert doc.page_count == 1
            assert len(doc.full_text) == 256
        finally:
            path.unlink(missing_ok=True)
