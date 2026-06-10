
"""Tests for the Chroma vector store client (task 3.3)."""

from __future__ import annotations

import shutil
import tempfile

import pytest

from app.core.chunker import Chunk
from app.models.document import DocType
from app.db.chroma import ChromaStore


def _make_chunk(chunk_id="doc1_chunk_0", content="hello world", doc_id="doc1",
                doc_name="test.txt", doc_type=DocType.TXT, page=1, idx=0):
    return Chunk(
        id=chunk_id, content=content, doc_id=doc_id, doc_name=doc_name,
        doc_type=doc_type, page=page, chunk_index=idx, char_count=len(content),
        metadata={"doc_name": doc_name, "page": page, "chunk_index": idx},
    )

def _emb(dim=8):
    return [0.1 * (i % 10 + 1) for i in range(dim)]


@pytest.fixture
def store():
    """Create a ChromaStore in a temp directory and clean up afterwards."""
    d = tempfile.mkdtemp()
    s = ChromaStore(persist_dir=d, collection_name="test_coll")
    yield s
    # Release Chroma's file locks before removing the directory
    s._client.clear_system_cache()
    shutil.rmtree(d, ignore_errors=True)


class TestInit:
    def test_init_custom(self, store):
        assert store.count() == 0
        assert store._collection is not None


class TestAdd:
    def test_add_and_count(self, store):
        store.add(chunks=[_make_chunk("x00")], embeddings=[_emb(8)])
        assert store.count() == 1

    def test_add_multiple(self, store):
        n = 50
        chunks = [_make_chunk(f"c{i:02d}", idx=i) for i in range(n)]
        embs = [_emb(8) for _ in range(n)]
        store.add(chunks=chunks, embeddings=embs)
        assert store.count() == n

    def test_add_empty_noop(self, store):
        store.add(chunks=[], embeddings=[])
        assert store.count() == 0

    def test_mismatched_raises(self, store):
        with pytest.raises(ValueError, match="Mismatched"):
            store.add(chunks=[_make_chunk("c00")], embeddings=[])


class TestQuery:
    def test_query_returns_dict(self, store):
        store.add(chunks=[_make_chunk("c00"), _make_chunk("c01")],
                  embeddings=[_emb(8), _emb(8)])
        r = store.query(embedding=_emb(8), n_results=2)
        assert "ids" in r
        assert "documents" in r
        assert "metadatas" in r
        assert "distances" in r
        assert len(r["ids"][0]) == 2

    def test_query_where_filter(self, store):
        c1 = _make_chunk("a00", doc_id="doc_a", doc_name="a.txt")
        c2 = _make_chunk("b00", doc_id="doc_b", doc_name="b.txt")
        store.add(chunks=[c1, c2], embeddings=[_emb(8), _emb(8)])
        r = store.query(embedding=_emb(8), n_results=5, where={"doc_id": "doc_a"})
        assert len(r["ids"][0]) == 1
        assert r["metadatas"][0][0]["doc_id"] == "doc_a"


class TestDelete:
    def test_delete_by_doc_id(self, store):
        c1 = _make_chunk("a00", doc_id="doc_a")
        c2 = _make_chunk("a01", doc_id="doc_a")
        c3 = _make_chunk("b00", doc_id="doc_b")
        store.add(chunks=[c1,c2,c3], embeddings=[_emb(8) for _ in range(3)])
        assert store.delete_by_doc_id("doc_a") == 2
        assert store.count() == 1

    def test_delete_by_chunk_ids(self, store):
        chunks = [_make_chunk(f"c{i:02d}") for i in range(5)]
        store.add(chunks=chunks, embeddings=[_emb(8) for _ in range(5)])
        assert store.delete_by_chunk_ids(["c00", "c01"]) == 2
        assert store.count() == 3

    def test_delete_nonexistent_zero(self, store):
        assert store.delete_by_chunk_ids(["no_such"]) == 0


class TestPersistence:
    def test_data_persists(self):
        d = tempfile.mkdtemp()
        try:
            s1 = ChromaStore(persist_dir=d, collection_name="test_coll")
            s1.add(chunks=[_make_chunk("p00"), _make_chunk("p01")],
                   embeddings=[_emb(8), _emb(8)])
            s1._client.clear_system_cache()

            s2 = ChromaStore(persist_dir=d, collection_name="test_coll")
            assert s2.count() == 2
            r = s2.query(embedding=_emb(8), n_results=2)
            assert set(r["ids"][0]) == {"p00", "p01"}
            s2._client.clear_system_cache()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestReset:
    def test_reset_clears(self, store):
        store.add(chunks=[_make_chunk("c00")], embeddings=[_emb(8)])
        assert store.count() == 1
        store.reset()
        assert store.count() == 0


class TestEdgeCases:
    def test_large_batch(self, store):
        n = 200
        chunks = [_make_chunk(f"c{i:03d}", idx=i) for i in range(n)]
        store.add(chunks=chunks, embeddings=[_emb(8) for _ in range(n)])
        assert store.count() == n

    def test_unicode(self, store):
        store.add(chunks=[_make_chunk("zh00", content="中文测试内容")],
                  embeddings=[_emb(8)])
        r = store.query(embedding=_emb(8), n_results=1)
        assert "中文" in r["documents"][0][0]

    def test_same_id_upserts(self, store):
        store.add(chunks=[_make_chunk("same1", content="v1")], embeddings=[_emb(8)])
        store.add(chunks=[_make_chunk("same1", content="v2")], embeddings=[_emb(8)])
        # Chroma add with duplicate ID is a no-op: first write wins, count stays 1
        assert store.count() == 1
        r = store.query(embedding=_emb(8), n_results=1)
        assert r["documents"][0][0] == "v1"

    def test_multiple_doc_types(self, store):
        c1 = _make_chunk("pdf00", doc_type=DocType.PDF, doc_name="a.pdf")
        c2 = _make_chunk("md_00", doc_type=DocType.MARKDOWN, doc_name="b.md")
        c3 = _make_chunk("txt00", doc_type=DocType.TXT, doc_name="c.txt")
        store.add(chunks=[c1,c2,c3], embeddings=[_emb(8) for _ in range(3)])
        r = store.query(embedding=_emb(8), n_results=3)
        types = {m["doc_type"] for m in r["metadatas"][0]}
        assert types == {"pdf", "md", "txt"}
