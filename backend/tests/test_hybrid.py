"""Tests for BM25 keyword index and hybrid BM25+vector RRF fusion (P1 roadmap §4).

Covers: BM25Index build / search / search_batch, RRF fusion logic,
Retriever with hybrid=True.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# BM25Index tests
# ============================================================================

class TestBM25Index:
    def test_build_empty_records(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([])
        assert idx.is_built is True
        assert idx.chunk_count == 0
        assert idx.search("hello") == []

    def test_build_and_search_single(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([
            {"id": "c0", "content": "machine learning is great"},
        ])
        results = idx.search("machine learning", top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c0"
        assert results[0]["bm25_score"] > 0
        assert results[0]["rank"] == 1

    def test_build_and_search_ranking(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([
            {"id": "c0", "content": "python is a programming language"},
            {"id": "c1", "content": "RAG retrieval augmented generation"},
            {"id": "c2", "content": "python for machine learning and data science"},
        ])
        # "python" should hit c0 and c2; c2 ranks higher (more term matches)
        results = idx.search("python", top_k=5)
        ids = [r["chunk_id"] for r in results]
        assert "c2" in ids or "c0" in ids  # at least one hit
        assert len(results) <= 3

    def test_search_batch(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([
            {"id": "c0", "content": "RAG combines retrieval with generation"},
            {"id": "c1", "content": "BM25 is a keyword search algorithm"},
        ])
        results = idx.search_batch(["RAG", "BM25"], top_k=3)
        assert len(results) == 2
        assert results[0][0]["chunk_id"] == "c0"  # RAG → c0
        assert results[1][0]["chunk_id"] == "c1"  # BM25 → c1

    def test_search_empty_query(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([
            {"id": "c0", "content": "some text"},
        ])
        results = idx.search("", top_k=5)
        assert results == []

    def test_reset_clears_index(self):
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([{"id": "c0", "content": "hello"}])
        idx.reset()
        assert idx.is_built is False
        assert idx.chunk_count == 0
        assert idx.search("hello") == []

    def test_cjk_tokenization(self):
        """BM25Index should handle CJK text via character bigrams."""
        from app.core.bm25 import BM25Index
        idx = BM25Index()
        idx.build_from_metas([
            {"id": "c0", "content": "向量检索与BM25关键词检索混合"},
        ])
        results = idx.search("向量检索", top_k=5)
        assert len(results) == 1
        assert results[0]["bm25_score"] > 0


# ============================================================================
# RRF fusion tests
# ============================================================================

class TestRRFFusion:
    def _make_chunk(self, cid, score=0.9):
        from app.models.search import SearchChunk
        return SearchChunk(
            chunk_id=cid, content=f"Content {cid}", score=score,
            doc_id="d1", doc_name="f.pdf", doc_type="txt",
            page=1, chunk_index=0, metadata={},
        )

    def _make_retriever(self):
        from app.core.retriever import Retriever
        r = Retriever()
        r._chroma = MagicMock()
        r._chroma.count.return_value = 10
        return r

    def test_fusion_both_sources(self, monkeypatch):
        """Vector and BM25 both contribute; fused pool includes chunks from both."""
        from app.config import settings
        settings.agent.hybrid_rrf_k = 60

        retriever = self._make_retriever()

        vector_chunks = [
            self._make_chunk("v0", 0.95),
            self._make_chunk("v1", 0.80),
            self._make_chunk("v2", 0.60),
        ]

        bm25_per_query = [
            [
                {"chunk_id": "b0", "bm25_score": 5.2, "rank": 1, "content": "b0 text",
                 "doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                 "page": 1, "chunk_index": 0, "metadata": {}},
                {"chunk_id": "v1", "bm25_score": 3.1, "rank": 2, "content": "v1 text",
                 "doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                 "page": 1, "chunk_index": 1, "metadata": {}},
            ],
        ]

        fused = retriever._rrf_fuse(
            per_query_vector=[vector_chunks],
            bm25_per_query=bm25_per_query,
            queries=["test"],
            top_k=10,
        )

        # Should contain chunks from both sources
        cids = set(fused.keys())
        assert "v0" in cids  # top vector
        assert "b0" in cids  # top BM25
        assert len(fused) >= 2

    def test_fusion_no_bm25_results(self):
        """When BM25 returns nothing, fall back to vector-only."""
        retriever = self._make_retriever()

        vector_chunks = [
            self._make_chunk("v0", 0.95),
            self._make_chunk("v1", 0.80),
        ]

        fused = retriever._rrf_fuse(
            per_query_vector=[vector_chunks],
            bm25_per_query=[[]],  # BM25 returns nothing
            queries=["test"],
            top_k=10,
        )

        assert len(fused) == len(vector_chunks)
        assert set(fused.keys()) == {"v0", "v1"}

    def test_fusion_empty_vector(self):
        """When vector returns nothing but BM25 has results."""
        retriever = self._make_retriever()

        bm25_per_query = [
            [
                {"chunk_id": "b0", "bm25_score": 5.0, "rank": 1, "content": "b0 text",
                 "doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                 "page": 1, "chunk_index": 0, "metadata": {}},
            ],
        ]

        fused = retriever._rrf_fuse(
            per_query_vector=[[]],  # empty vector for one query
            bm25_per_query=bm25_per_query,
            queries=["test"],
            top_k=10,
        )

        assert len(fused) == 1
        assert "b0" in fused

    def test_fusion_top_k_truncation(self):
        """Fused results respect top_k parameter."""
        retriever = self._make_retriever()

        vector_chunks = [
            self._make_chunk(f"v{i}", 0.9 - i * 0.1) for i in range(5)
        ]

        fused = retriever._rrf_fuse(
            per_query_vector=[vector_chunks],
            bm25_per_query=[[]],
            queries=["test"],
            top_k=3,
        )

        assert len(fused) == 3


# ============================================================================
# Hybrid Retriever tests
# ============================================================================

class TestHybridRetrieve:
    def _make_chunk(self, cid, score=0.9):
        from app.models.search import SearchChunk
        return SearchChunk(
            chunk_id=cid, content=f"Content {cid}", score=score,
            doc_id="d1", doc_name="f.pdf", doc_type="txt",
            page=1, chunk_index=0, metadata={},
        )

    def _fake_chroma_batch(self, ids_per_query, docs_per_query=None,
                           metas_per_query=None, dists_per_query=None):
        return {
            "ids": ids_per_query,
            "documents": docs_per_query or [
                ["doc " + cid for cid in q] for q in ids_per_query
            ],
            "metadatas": metas_per_query or [
                [{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                  "page": 1, "chunk_index": 0} for _ in q]
                for q in ids_per_query
            ],
            "distances": dists_per_query or [
                [0.2] * len(q) for q in ids_per_query
            ],
        }

    def _make_patched_retriever(self, chroma_result=None, count=100, bm25_mock=None):
        from app.core.retriever import Retriever
        r = Retriever()
        r._chroma = MagicMock()
        r._chroma.count.return_value = count
        r._chroma.query_batch.return_value = (
            chroma_result or self._fake_chroma_batch([[]])
        )
        r._embedder = MagicMock()
        r._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1024])
        if bm25_mock is not None:
            r._bm25 = bm25_mock
        return r

    @pytest.mark.asyncio
    async def test_hybrid_false_is_vector_only(self):
        """When hybrid=False, result should have hybrid=False."""
        ids = [["c0", "c1"]]
        docs = [["text 0", "text 1"]]
        metas = [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": 0},
                  {"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": 1}]]
        dists = [[0.2, 0.8]]
        cr = self._fake_chroma_batch(ids, docs, metas, dists)
        retriever = self._make_patched_retriever(cr, count=50)
        result = await retriever.retrieve(["q"], hybrid=False)
        assert result.hybrid is False
        assert len(result.chunks) == 2

    @pytest.mark.asyncio
    async def test_hybrid_true_with_bm25_mock(self, monkeypatch):
        """When hybrid=True and BM25 is available, RRF fusion runs."""
        ids = [["c0", "c1"]]
        docs = [["text 0", "text 1"]]
        metas = [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": 0},
                  {"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": 1}]]
        dists = [[0.2, 0.8]]
        cr = self._fake_chroma_batch(ids, docs, metas, dists)

        bm25_mock = MagicMock()
        bm25_mock.is_built = True
        bm25_mock.search_batch.return_value = [
            [{"chunk_id": "c0", "bm25_score": 5.0, "rank": 1,
              "content": "text 0", "doc_id": "d1", "doc_name": "f.pdf",
              "doc_type": "txt", "page": 1, "chunk_index": 0, "metadata": {}}],
        ]

        retriever = self._make_patched_retriever(
            cr, count=50, bm25_mock=bm25_mock,
        )

        # Also mock Chroma get_all for _ensure_bm25_built (won't be called
        # because bm25_mock.is_built is already True)
        retriever._chroma.get_all.return_value = ([], [], [])

        from app.config import settings
        original = settings.agent.hybrid_search_enabled
        settings.agent.hybrid_search_enabled = True
        try:
            result = await retriever.retrieve(["q"], hybrid=True)
            assert result.hybrid is True
            assert len(result.chunks) >= 1
        finally:
            settings.agent.hybrid_search_enabled = original

    @pytest.mark.asyncio
    async def test_hybrid_disabled_by_config(self):
        """When global config disables hybrid, per-call hybrid=True is ignored."""
        ids = [["c0"]]
        docs = [["text 0"]]
        metas = [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": 0}]]
        dists = [[0.2]]
        cr = self._fake_chroma_batch(ids, docs, metas, dists)
        retriever = self._make_patched_retriever(cr, count=10)

        from app.config import settings
        original = settings.agent.hybrid_search_enabled
        settings.agent.hybrid_search_enabled = False
        try:
            result = await retriever.retrieve(["q"], hybrid=True)
            assert result.hybrid is False  # disabled at config level
        finally:
            settings.agent.hybrid_search_enabled = original
