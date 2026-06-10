"""Tests for the text chunker (task 3.1).

Covers:
- Default-parameter chunking
- Custom chunk_size / chunk_overlap
- Single vs multi-page document splitting
- Empty / very-short text edge cases
- Chinese text handling
- Chunk metadata (page, index, ID format, character count)
- ``split_text`` convenience method
- Separator cascade behaviour
"""

from __future__ import annotations

import pytest

from app.core.chunker import Chunk, Chunker
from app.models.document import DocType, Document, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    pages: list[str],
    file_name: str = "test.txt",
    doc_type: DocType = DocType.TXT,
    doc_id: str = "",
) -> Document:
    """Build a ``Document`` from a list of page-text strings."""
    return Document(
        file_name=file_name,
        doc_type=doc_type,
        pages=[Page(page_number=i + 1, text=t) for i, t in enumerate(pages)],
        metadata={"doc_id": doc_id} if doc_id else {},
    )


def _make_chunker(**kwargs) -> Chunker:
    """Create a Chunker with fixed parameters (no config dependency)."""
    defaults = {"chunk_size": 200, "chunk_overlap": 40}
    defaults.update(kwargs)
    return Chunker(**defaults)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


class TestChunkerBasic:
    """Happy-path tests with English text."""

    def test_single_page_single_chunk(self):
        """Short text fits in one chunk."""
        doc = _make_doc(["Hello world. This is a test."], doc_id="doc1")
        chunker = _make_chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world. This is a test."
        assert chunks[0].chunk_index == 0
        assert chunks[0].page == 1
        assert chunks[0].doc_id == "doc1"

    def test_single_page_multiple_chunks(self):
        """Long text produces multiple overlapping chunks."""
        # 50 chars per sentence, 20 sentences ≈ 1000 chars
        sentence = "The quick brown fox jumps over the lazy dog.  "  # ~50 chars
        text = sentence * 20
        doc = _make_doc([text], doc_id="doc1")
        chunker = _make_chunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.split(doc)

        assert len(chunks) > 1
        # Every chunk should be <= chunk_size
        for c in chunks:
            assert c.char_count <= 300
            assert c.char_count == len(c.content)
            assert c.doc_id == "doc1"
            assert c.page == 1

    def test_multi_page_document(self):
        """Chunks from different pages have correct page numbers."""
        doc = _make_doc(
            ["Page one text. " * 10, "Page two text. " * 10],
            doc_id="doc2",
        )
        chunker = _make_chunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.split(doc)

        # Every chunk should belong to the page it came from
        page_one_chunks = [c for c in chunks if c.page == 1]
        page_two_chunks = [c for c in chunks if c.page == 2]
        assert len(page_one_chunks) > 0
        assert len(page_two_chunks) > 0
        assert len(page_one_chunks) + len(page_two_chunks) == len(chunks)

        # Chunk indices increase monotonically
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_id_format_with_doc_id(self):
        """Chunk IDs follow {doc_id}_chunk_{index} when doc_id is present."""
        doc = _make_doc(["Hello world. " * 20], doc_id="abc123")
        chunker = _make_chunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.split(doc)

        for i, c in enumerate(chunks):
            assert c.id == f"abc123_chunk_{i}"

    def test_chunk_id_fallback_when_no_doc_id(self):
        """Chunk IDs use a random hex prefix when doc_id is missing."""
        doc = _make_doc(["Hello world. " * 20], doc_id="")  # no doc_id
        chunker = _make_chunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.split(doc)

        for c in chunks:
            # Format: _{8_hex_chars}_chunk_{index}
            assert "_chunk_" in c.id
            parts = c.id.split("_chunk_")
            assert len(parts) == 2
            # The prefix should be 9 chars: "_" + 8 hex chars
            assert len(parts[0]) == 9  # e.g. "_a1b2c3d4"
            assert parts[1].isdigit()


# ---------------------------------------------------------------------------
# Parameters and configuration
# ---------------------------------------------------------------------------


class TestChunkerParameters:
    """Tests around chunk_size, chunk_overlap, and separators."""

    def test_custom_chunk_size(self):
        """chunk_size parameter is respected."""
        sentence = "A" * 100 + ". "  # 102 chars per sentence
        text = sentence * 10  # 1020 chars
        doc = _make_doc([text])
        chunker = _make_chunker(chunk_size=300, chunk_overlap=0)
        chunks = chunker.split(doc)

        for c in chunks:
            assert c.char_count <= 300

    def test_custom_chunk_overlap(self):
        """Overlap between consecutive chunks exists."""
        # Use predictable text so we can detect overlap
        words = [f"word{i:03d}" for i in range(100)]
        text = " ".join(words)  # ~1300 chars
        doc = _make_doc([text])
        chunker = _make_chunker(chunk_size=300, chunk_overlap=80)
        chunks = chunker.split(doc)

        if len(chunks) >= 2:
            # Last few words of chunk 0 should appear near the start of chunk 1
            last_words = chunks[0].content.split()[-5:]
            chunk1_start = chunks[1].content[:200]
            overlap_found = any(w in chunk1_start for w in last_words)
            assert overlap_found, "Expected overlap between consecutive chunks"

    def test_zero_overlap(self):
        """chunk_overlap=0 produces non-overlapping chunks."""
        doc = _make_doc(["hello world. " * 50])
        chunker = _make_chunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.split(doc)

        assert len(chunks) >= 1

        # With overlap=0, chunks should not contain overlapping content.
        # Verify by checking that the end of chunk[i] does not appear at
        # the start of chunk[i+1] (allowing for small separator differences).
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-50:] if len(chunks[i].content) >= 50 else chunks[i].content
            head = chunks[i + 1].content[:50] if len(chunks[i + 1].content) >= 50 else chunks[i + 1].content
            # With zero overlap there should be no shared long substring
            # (allow accidental very short matches like " " or ".")
            assert tail != head


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestChunkerEdgeCases:
    """Empty, very short, and boundary inputs."""

    def test_empty_document(self):
        """Document with no pages returns empty chunk list."""
        doc = _make_doc([])
        chunker = _make_chunker()
        chunks = chunker.split(doc)
        assert chunks == []

    def test_empty_page(self):
        """A page with empty text produces no chunks from that page."""
        doc = _make_doc(["Some text. " * 5, "", "More text. " * 5])
        chunker = _make_chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(doc)

        # The empty page should be skipped
        pages = {c.page for c in chunks}
        assert 2 not in pages  # page 2 was empty

    def test_all_empty_pages(self):
        """All pages empty → empty chunk list."""
        doc = _make_doc(["", "", ""])
        chunker = _make_chunker()
        chunks = chunker.split(doc)
        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        """Text shorter than chunk_size returns exactly one chunk."""
        doc = _make_doc(["Hi."])
        chunker = _make_chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi."
        assert chunks[0].char_count == 3

    def test_exactly_chunk_size(self):
        """Text that is exactly chunk_size chars returns one chunk."""
        text = "X" * 200
        doc = _make_doc([text])
        chunker = _make_chunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.split(doc)

        assert len(chunks) == 1
        assert chunks[0].char_count == 200

    def test_single_character(self):
        """Single-character text produces one chunk."""
        doc = _make_doc(["X"])
        chunker = _make_chunker()
        chunks = chunker.split(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "X"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestChunkerMetadata:
    """Chunk metadata fields are correctly populated."""

    def test_doc_name_and_type(self):
        """Every chunk carries the document name and type."""
        doc = _make_doc(
            ["Hello world. " * 20],
            file_name="report.md",
            doc_type=DocType.MARKDOWN,
            doc_id="doc-md-1",
        )
        chunker = _make_chunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.split(doc)

        for c in chunks:
            assert c.doc_name == "report.md"
            assert c.doc_type == DocType.MARKDOWN
            assert c.metadata["doc_name"] == "report.md"
            assert c.metadata["doc_type"] == "md"

    def test_metadata_keys(self):
        """Required metadata keys are present."""
        doc = _make_doc(["Hello world. " * 20], doc_id="doc1")
        chunker = _make_chunker()
        chunks = chunker.split(doc)

        for c in chunks:
            meta = c.metadata
            assert "doc_name" in meta
            assert "doc_type" in meta
            assert "page" in meta
            assert "chunk_index" in meta
            assert "chunk_size" in meta
            assert "chunk_overlap" in meta

    def test_page_metadata_preserved(self):
        """Page number is correctly tracked per chunk."""
        doc = _make_doc(
            ["Page1 " * 30, "Page2 " * 30, "Page3 " * 30],
            doc_id="doc1",
        )
        chunker = _make_chunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(doc)

        for c in chunks:
            assert c.metadata["page"] == c.page
            assert c.page in (1, 2, 3)


# ---------------------------------------------------------------------------
# Chinese text
# ---------------------------------------------------------------------------


class TestChunkerChinese:
    """Chinese text splitting — separators should respect Chinese punctuation."""

    def test_chinese_text_split_on_period(self):
        """Chinese periods (。) should be used as split boundaries."""
        sentences = [
            "这是第一句话。",
            "这是第二句话，带逗号。",
            "这是第三句话！",
            "这是第四句话？",
        ]
        text = "".join(sentences)
        doc = _make_doc([text])
        # Small chunk_size so the splitter is forced to use punctuation boundaries
        chunker = _make_chunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.split(doc)

        # With chunk_size=20, we should get multiple chunks
        assert len(chunks) >= 2

    def test_chinese_paragraphs(self):
        """Paragraph separators (\\n\\n) should be respected in Chinese text."""
        para1 = "这是第一段内容，包含一些中文文本。\n这是第一段的第二句。"
        para2 = "这是第二段内容，与第一段用空行分隔。"
        text = para1 + "\n\n" + para2
        doc = _make_doc([text])
        chunker = _make_chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(doc)

        assert len(chunks) >= 1
        # All chunks should be non-empty
        for c in chunks:
            assert len(c.content.strip()) > 0

    def test_chinese_document_type(self):
        """DocType.MARKDOWN is correctly stored on chunks."""
        doc = _make_doc(
            ["# 标题\n\n这是正文。"],
            file_name="readme.md",
            doc_type=DocType.MARKDOWN,
            doc_id="md1",
        )
        chunker = _make_chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(doc)

        for c in chunks:
            assert c.doc_type == DocType.MARKDOWN
            assert c.metadata["doc_type"] == "md"


# ---------------------------------------------------------------------------
# split_text convenience method
# ---------------------------------------------------------------------------


class TestSplitText:
    """The ``split_text`` convenience wrapper."""

    def test_returns_strings(self):
        """split_text returns list[str]."""
        chunker = _make_chunker(chunk_size=100, chunk_overlap=20)
        result = chunker.split_text("Hello world. " * 30)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_all_segments_non_empty(self):
        """No empty strings in the result."""
        chunker = _make_chunker(chunk_size=50, chunk_overlap=10)
        result = chunker.split_text("Short text here.")
        for seg in result:
            assert len(seg) > 0


# ---------------------------------------------------------------------------
# DocType integration
# ---------------------------------------------------------------------------


class TestDocTypeOnChunks:
    """DocType enum is propagated to every chunk."""

    @pytest.mark.parametrize(
        "doc_type",
        [DocType.PDF, DocType.MARKDOWN, DocType.TXT],
    )
    def test_doc_type_roundtrip(self, doc_type: DocType):
        """Each DocType value is preserved in chunks."""
        doc = _make_doc(
            ["Sample text. " * 15],
            doc_type=doc_type,
            doc_id="doc1",
        )
        chunker = _make_chunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.split(doc)

        for c in chunks:
            assert c.doc_type == doc_type
