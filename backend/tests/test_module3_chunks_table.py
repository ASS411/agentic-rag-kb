
"""Tests for chunks metadata persistence (task 3.5)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Column, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase

from app.core.chunker import Chunk, DocType
from app.models.chunk import ChunkModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id="doc1_chunk_0", content="hello world", doc_id="doc1",
    doc_name="test.txt"
):
    return Chunk(
        id=chunk_id, content=content, doc_id=doc_id, doc_name=doc_name,
        doc_type=DocType.TXT, page=1, chunk_index=0, char_count=len(content),
        metadata={"doc_name": doc_name, "page": 1, "chunk_index": 0},
    )


class TestChunkModel:
    """ChunkModel ORM entity."""

    def test_create_instance(self):
        cm = ChunkModel(
            chunk_id="abc_chunk_0",
            doc_id="abc",
            content_hash="a" * 64,
            char_count=800,
        )
        assert cm.chunk_id == "abc_chunk_0"
        assert cm.doc_id == "abc"
        assert cm.content_hash == "a" * 64
        assert cm.char_count == 800

    def test_repr(self):
        cm = ChunkModel(chunk_id="x", doc_id="y", content_hash="z" * 64, char_count=10)
        r = repr(cm)
        assert "ChunkModel" in r
        assert "x" in r
        assert "y" in r

    def test_table_name(self):
        assert ChunkModel.__tablename__ == "chunks"

    def test_fk_references_documents(self):
        fk = ChunkModel.__table__.c.doc_id
        assert fk.foreign_keys
        fk_def = list(fk.foreign_keys)[0]
        assert "documents" in str(fk_def)


class TestContentHash:
    """SHA-256 content hashing."""

    def test_hash_is_64_hex_chars(self):
        h = hashlib.sha256(b"hello").hexdigest()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self):
        h1 = hashlib.sha256(b"hello").hexdigest()
        h2 = hashlib.sha256(b"world").hexdigest()
        assert h1 != h2

    def test_same_content_same_hash(self):
        h1 = hashlib.sha256(b"hello").hexdigest()
        h2 = hashlib.sha256(b"hello").hexdigest()
        assert h1 == h2

    def test_chunk_hash_is_deterministic(self):
        c = _make_chunk(content="test content")
        h1 = hashlib.sha256(c.content.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256("test content".encode("utf-8")).hexdigest()
        assert h1 == h2


class TestPersistChunkMetadata:
    """The _persist_chunk_metadata background helper."""

    @pytest.mark.asyncio
    async def test_persists_rows_to_db(self):
        from app.api.documents import _persist_chunk_metadata
        from app.models.chunk import ChunkModel

        chunks = [_make_chunk("c0"), _make_chunk("c1", content="second")]

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.__aenter__.return_value = mock_session

        with patch(
            "app.api.documents.async_session_factory",
            return_value=mock_factory,
        ):
            await _persist_chunk_metadata("doc-1", chunks)

        mock_session.add_all.assert_called_once()
        mock_session.commit.assert_called_once()
        rows = mock_session.add_all.call_args[0][0]
        assert len(rows) == 2
        assert all(isinstance(r, ChunkModel) for r in rows)
        assert rows[0].chunk_id == "c0"
        assert rows[0].doc_id == "doc-1"
        assert len(rows[0].content_hash) == 64

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_immediately(self):
        from app.api.documents import _persist_chunk_metadata

        mock_factory = MagicMock()

        with patch(
            "app.api.documents.async_session_factory",
            new=mock_factory,
        ):
            await _persist_chunk_metadata("doc-1", [])

        mock_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_error_is_silent(self):
        from app.api.documents import _persist_chunk_metadata

        chunks = [_make_chunk("c0")]
        mock_factory = MagicMock()
        mock_factory.__aenter__.side_effect = RuntimeError("db down")

        with patch(
            "app.api.documents.async_session_factory",
            return_value=mock_factory,
        ):
            # Must not raise
            await _persist_chunk_metadata("doc-1", chunks)

    @pytest.mark.asyncio
    async def test_chunk_id_and_doc_id_concatenated(self):
        from app.api.documents import _persist_chunk_metadata

        chunks = [
            _make_chunk("d1_chunk_0", doc_id="d1"),
            _make_chunk("d1_chunk_1", doc_id="d1"),
            _make_chunk("d2_chunk_0", doc_id="d2"),
        ]

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.__aenter__.return_value = mock_session

        with patch(
            "app.api.documents.async_session_factory",
            return_value=mock_factory,
        ):
            await _persist_chunk_metadata("master-doc", chunks)

        rows = mock_session.add_all.call_args[0][0]
        # doc_id in rows matches the arg passed to _persist, not the chunk.doc_id
        assert all(r.doc_id == "master-doc" for r in rows)


class TestForeignKeyCascade:
    """Verify the cascade delete is properly defined in DDL."""

    def test_ondelete_cascade(self):
        fk = ChunkModel.__table__.c.doc_id
        fk_list = list(fk.foreign_keys)
        assert len(fk_list) == 1
        fk_def = fk_list[0]
        assert "documents" in str(fk_def.column)
        # ondelete="CASCADE" should be reflected
        assert fk_def.ondelete == "CASCADE"
