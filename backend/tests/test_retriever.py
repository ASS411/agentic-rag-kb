"""Tests for the multi-query retriever (task 2.3).

Covers: multi-query dedup, top_k_recall, empty queries, empty Chroma, rerank.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.retriever import (
    Retriever,
    RetrievalResult,
    _cosine_similarity_from_distance,
    _chroma_result_to_searchchunks,
)
from app.models.search import SearchChunk


def _make_searchchunk(chunk_id, content, score, doc_id="d1", doc_name="f.pdf"):
    return SearchChunk(
        chunk_id=chunk_id, content=content, score=score,
        doc_id=doc_id, doc_name=doc_name, doc_type="txt",
        page=1, chunk_index=0, metadata={},
    )


def _fake_chroma_batch(ids_per_query, docs_per_query=None,
                       metas_per_query=None, dists_per_query=None):
    n = len(ids_per_query)
    return {
        "ids": ids_per_query,
        "documents": docs_per_query or [["doc " + cid for cid in q] for q in ids_per_query],
        "metadatas": metas_per_query or [[{"doc_id": "d1"} for _ in q] for q in ids_per_query],
        "distances": dists_per_query or [[0.2] * len(q) for q in ids_per_query],
    }


def _make_patched_retriever(chroma_result=None, emb_dim=1024, count=100):
    r = Retriever()
    r._chroma = MagicMock()
    r._chroma.count.return_value = count
    r._chroma.query_batch.return_value = chroma_result or _fake_chroma_batch([[]])
    r._embedder = MagicMock()
    r._embedder.embed_batch = AsyncMock(return_value=[[0.1] * emb_dim])
    return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_cosine_from_distance_zero(self):
        assert _cosine_similarity_from_distance(0.0) == 1.0

    def test_cosine_from_distance_one(self):
        assert _cosine_similarity_from_distance(1.0) == 0.5

    def test_cosine_from_distance_two(self):
        assert _cosine_similarity_from_distance(2.0) == 0.0

    def test_chroma_to_searchchunks_empty(self):
        result = _chroma_result_to_searchchunks("q", [], [], [], [])
        assert result == []

    def test_chroma_to_searchchunks_single(self):
        result = _chroma_result_to_searchchunks(
            "q", ["c0"], ["hello world"],
            [{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
              "page": 1, "chunk_index": 0}], [0.4],
        )
        assert len(result) == 1
        c = result[0]
        assert c.chunk_id == "c0"
        assert c.content == "hello world"
        assert c.doc_id == "d1"
        assert c.score == pytest.approx(0.8, rel=1e-5)


class TestRetrievalResult:
    def test_default_construction(self):
        rr = RetrievalResult()
        assert rr.chunks == []
        assert rr.total_recalled == 0
        assert rr.reranked is False

    def test_with_data(self):
        c = _make_searchchunk("c0", "hi", 0.9)
        rr = RetrievalResult(chunks=[c], total_recalled=15, reranked=True)
        assert len(rr.chunks) == 1
        assert rr.total_recalled == 15
        assert rr.reranked is True


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_empty_queries_returns_empty(self):
        retriever = Retriever()
        result = await retriever.retrieve([])
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_single_query_basic(self):
        cr = _fake_chroma_batch(
            [["c0", "c1"]], [["text 0", "text 1"]],
            [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 1, "chunk_index": 0},
              {"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 1, "chunk_index": 1}]],
            [[0.2, 0.8]],
        )
        retriever = _make_patched_retriever(cr, count=50)
        result = await retriever.retrieve(["test query"], top_k_recall=5)
        assert len(result.chunks) == 2
        assert result.total_recalled == 2
        assert result.reranked is False
        assert result.chunks[0].score > result.chunks[1].score

    @pytest.mark.asyncio
    async def test_multi_query_dedup(self):
        cr = _fake_chroma_batch(
            [["c0", "c1"], ["c0", "c2"]],
            [["text 0", "text 1"], ["text 0", "text 2"]],
            [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 1, "chunk_index": 0},
              {"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 1, "chunk_index": 1}],
             [{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 1, "chunk_index": 0},
              {"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
               "page": 2, "chunk_index": 2}]],
            [[0.2, 0.6], [0.4, 0.9]],
        )
        retriever = _make_patched_retriever(cr, count=10)
        result = await retriever.retrieve(["q1", "q2"], top_k_recall=5)
        assert result.total_recalled == 4
        assert len(result.chunks) == 3
        c0 = [c for c in result.chunks if c.chunk_id == "c0"][0]
        assert c0.score == pytest.approx(0.9, rel=1e-5)

    @pytest.mark.asyncio
    async def test_empty_chroma_collection(self):
        retriever = _make_patched_retriever(count=0)
        result = await retriever.retrieve(["q"])
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_embedder_called_with_all_queries(self):
        retriever = _make_patched_retriever()
        queries = ["q1", "q2", "q3"]
        await retriever.retrieve(queries, top_k_recall=5)
        retriever._embedder.embed_batch.assert_called_once_with(queries)

    @pytest.mark.asyncio
    async def test_n_results_capped_by_collection_size(self):
        retriever = _make_patched_retriever(count=3)
        await retriever.retrieve(["q"], top_k_recall=20)
        kw = retriever._chroma.query_batch.call_args.kwargs
        assert kw["n_results"] == 3


class TestRetrieveWithRerank:
    @pytest.mark.asyncio
    async def test_rerank_flag_true(self):
        ids = [["c%d" % i for i in range(5)]]
        docs = [["text %d" % i for i in range(5)]]
        metas = [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": i} for i in range(5)]]
        dists = [[0.1 * (i + 1) for i in range(5)]]
        cr = _fake_chroma_batch(ids, docs, metas, dists)
        retriever = _make_patched_retriever(cr, count=100)
        mock_rr = MagicMock()
        mock_rr.rerank.return_value = []
        retriever._reranker = mock_rr
        result = await retriever.retrieve(["q"], rerank=True)
        assert result.reranked is True
        mock_rr.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_flag_false(self):
        ids = [["c0", "c1", "c2"]]
        docs = [["t0", "t1", "t2"]]
        metas = [[{"doc_id": "d1", "doc_name": "f.pdf", "doc_type": "txt",
                   "page": 1, "chunk_index": i} for i in range(3)]]
        dists = [[0.2, 0.5, 0.8]]
        cr = _fake_chroma_batch(ids, docs, metas, dists)
        retriever = _make_patched_retriever(cr, count=10)
        result = await retriever.retrieve(["q"], rerank=False)
        assert result.reranked is False
