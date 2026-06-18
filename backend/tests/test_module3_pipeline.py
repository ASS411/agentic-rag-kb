"""Tests for the ingestion pipeline (task 3.4).

Covers:
- Full pipeline: parse → chunk → embed → store
- Dependency injection (custom chunker/embedder/chroma)
- Empty document handling
- Error propagation
- run_from_storage integration
- PipelineResult properties
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.chunker import Chunk, Chunker
from app.core.embedder import Embedder
from app.core.pipeline import IngestionPipeline, PipelineResult
from app.db.chroma import ChromaStore
from app.models.document import DocType, Document, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    file_name: str = "test.txt",
    doc_type: DocType = DocType.TXT,
    text: str = "hello world",
) -> Document:
    """Build a minimal parsed Document for testing."""
    page = Page(page_number=1, text=text)
    return Document(
        file_name=file_name,
        doc_type=doc_type,
        pages=[page],
        metadata={"doc_id": "abc123"},
    )


def _make_chunks(n: int = 3, prefix: str = "doc1_chunk_") -> list[Chunk]:
    return [
        Chunk(
            id=f"{prefix}{i}",
            content=f"chunk {i} content",
            doc_id="doc1",
            doc_name="test.txt",
            doc_type=DocType.TXT,
            page=1,
            chunk_index=i,
            char_count=len(f"chunk {i} content"),
        )
        for i in range(n)
    ]


def _make_embeddings(n: int = 3, dim: int = 8) -> list[list[float]]:
    return [[0.1 * (i + j + 1) for j in range(dim)] for i in range(n)]


# ---------------------------------------------------------------------------
# Patch helpers for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_components():
    """Return a triple of mock Chunker, Embedder, ChromaStore."""
    mock_chunker = MagicMock(spec=Chunker)
    mock_embedder = MagicMock(spec=Embedder)
    mock_chroma = MagicMock(spec=ChromaStore)
    return mock_chunker, mock_embedder, mock_chroma


# ---------------------------------------------------------------------------
# Unit: pipeline with mocked components
# ---------------------------------------------------------------------------


class TestPipelineUnit:
    """Test the pipeline using mock components."""

    @pytest.mark.asyncio
    async def test_full_flow_calls_all_components(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        doc = _make_doc()
        chunks = _make_chunks(3)
        embeddings = _make_embeddings(3)

        mock_chunker.split.return_value = chunks
        mock_embedder.embed_batch = AsyncMock(return_value=embeddings)

        with patch("app.core.pipeline.parse_document", return_value=doc) as mock_parse:
            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            result = await pipeline.run("fake/path.txt", doc_id="abc123")

        mock_parse.assert_called_once()
        mock_chunker.split.assert_called_once_with(doc)
        mock_embedder.embed_batch.assert_called_once()
        mock_chroma.add.assert_called_once_with(
            chunks=chunks, embeddings=embeddings
        )
        assert result.chunk_count == 3

    @pytest.mark.asyncio
    async def test_propagates_doc_id_to_chunks(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        doc = _make_doc()
        mock_chunker.split.return_value = _make_chunks(2)
        mock_embedder.embed_batch = AsyncMock(
            return_value=_make_embeddings(2)
        )

        with patch("app.core.pipeline.parse_document", return_value=doc):
            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            await pipeline.run("f.txt", doc_id="doc-xyz")

        # Verify doc_id was set on the document before chunking
        passed_doc: Document = mock_chunker.split.call_args[0][0]
        assert passed_doc.metadata["doc_id"] == "doc-xyz"

    @pytest.mark.asyncio
    async def test_empty_chunks_skips_embed_and_store(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components
        mock_chunker.split.return_value = []

        doc = _make_doc()
        with patch("app.core.pipeline.parse_document", return_value=doc):
            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            result = await pipeline.run("empty.txt", doc_id="x")

        mock_embedder.embed_batch.assert_not_called()
        mock_chroma.add.assert_not_called()
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    async def test_embedding_mismatch_raises(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components
        mock_chunker.split.return_value = _make_chunks(3)
        # Return only 2 embeddings for 3 chunks
        mock_embedder.embed_batch = AsyncMock(
            return_value=_make_embeddings(2)
        )

        doc = _make_doc()
        with patch("app.core.pipeline.parse_document", return_value=doc):
            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            with pytest.raises(RuntimeError, match="count mismatch"):
                await pipeline.run("f.txt", doc_id="x")

    @pytest.mark.asyncio
    async def test_parse_error_propagates(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        with patch(
            "app.core.pipeline.parse_document",
            side_effect=FileNotFoundError("no such file"),
        ):
            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            with pytest.raises(FileNotFoundError):
                await pipeline.run("nope.txt", doc_id="x")


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """PipelineResult data class properties."""

    def test_properties(self):
        doc = _make_doc()
        doc.metadata["doc_id"] = "xyz"
        chunks = _make_chunks(5)

        result = PipelineResult(doc=doc, chunks=chunks)
        assert result.doc_id == "xyz"
        assert result.chunk_count == 5
        assert result.total_chars > 0

    def test_repr(self):
        result = PipelineResult(doc=_make_doc(), chunks=_make_chunks(2))
        r = repr(result)
        assert "test.txt" in r
        assert "chunks=2" in r


# ---------------------------------------------------------------------------
# run_from_storage
# ---------------------------------------------------------------------------


class TestRunFromStorage:
    """Integration with FileStorage upload directory layout."""

    @pytest.mark.asyncio
    async def test_finds_file_in_doc_dir(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        chunks = _make_chunks(1)
        mock_chunker.split.return_value = chunks
        mock_embedder.embed_batch = AsyncMock(
            return_value=_make_embeddings(1)
        )

        with tempfile.TemporaryDirectory() as upload_dir:
            # Create the expected layout: upload_dir/{doc_id}/file.txt
            doc_dir = Path(upload_dir) / "doc123"
            doc_dir.mkdir()
            test_file = doc_dir / "report.pdf"
            test_file.write_text("PDF content", encoding="utf-8")

            with patch(
                "app.core.pipeline.parse_document",
                return_value=_make_doc(file_name="report.pdf", doc_type=DocType.PDF),
            ):
                pipeline = IngestionPipeline(
                    chunker=mock_chunker,
                    embedder=mock_embedder,
                    chroma=mock_chroma,
                )
                result = await pipeline.run_from_storage(
                    "doc123", upload_dir=upload_dir
                )

            assert result is not None
            assert result.chunk_count == 1

    @pytest.mark.asyncio
    async def test_missing_dir_returns_none(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            chroma=mock_chroma,
        )
        result = await pipeline.run_from_storage(
            "nonexistent_doc_id", upload_dir="/tmp/no_such_dir_xyz"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_dir_returns_none(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components

        with tempfile.TemporaryDirectory() as upload_dir:
            empty_doc_dir = Path(upload_dir) / "empty_doc"
            empty_doc_dir.mkdir()

            pipeline = IngestionPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                chroma=mock_chroma,
            )
            result = await pipeline.run_from_storage(
                "empty_doc", upload_dir=upload_dir
            )
            assert result is None


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestPipelineInit:
    """Default constructor behaviour."""

    def test_creates_defaults_when_none_provided(self):
        pipeline = IngestionPipeline()
        assert pipeline._chunker is not None
        assert pipeline._embedder is not None
        assert pipeline._chroma is not None

    def test_accepts_custom_instances(self, mock_components):
        mock_chunker, mock_embedder, mock_chroma = mock_components
        pipeline = IngestionPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            chroma=mock_chroma,
        )
        assert pipeline._chunker is mock_chunker
        assert pipeline._embedder is mock_embedder
        assert pipeline._chroma is mock_chroma
