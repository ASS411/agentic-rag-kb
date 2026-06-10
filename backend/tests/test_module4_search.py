"""Tests for the search API (module 4.1 / 4.2).

Covers:
- Request model validation
- Cosine similarity conversion helpers
- Response assembly from Chroma raw results
- Full endpoint integration (via FastAPI TestClient)
"""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.chunker import Chunk
from app.db.chroma import ChromaStore
from app.models.document import DocType
from app.models.search import SearchChunk, SearchRequest, SearchResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id="doc1_chunk_0",
    content="hello world",
    doc_id="doc1",
    doc_name="test.txt",
    doc_type=DocType.TXT,
    page=1,
    idx=0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=content,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type=doc_type,
        page=page,
        chunk_index=idx,
        char_count=len(content),
        metadata={"doc_name": doc_name, "page": page, "chunk_index": idx},
    )


def _emb(dim: int = 8) -> list[float]:
    return [0.1 * (i % 10 + 1) for i in range(dim)]


def _raw_chroma_result(
    ids: list[str] | None = None,
    documents: list[str] | None = None,
    metadatas: list[dict] | None = None,
    distances: list[float] | None = None,
) -> dict:
    """Build a Chroma QueryResult shaped dict for a single query."""
    return {
        "ids": [ids or []],
        "documents": [documents or []],
        "metadatas": [metadatas or []],
        "distances": [distances or []],
    }


# ---------------------------------------------------------------------------
# Request model validation
# ---------------------------------------------------------------------------


class TestSearchRequest:
    def test_valid_minimal(self):
        req = SearchRequest(query="hello")
        assert req.query == "hello"
        assert req.top_k == 5  # default

    def test_valid_full(self):
        req = SearchRequest(query="什么是RAG？", top_k=10)
        assert req.query == "什么是RAG？"
        assert req.top_k == 10

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            SearchRequest(query="")

    def test_top_k_clamped(self):
        with pytest.raises(ValueError):
            SearchRequest(query="ok", top_k=0)
        with pytest.raises(ValueError):
            SearchRequest(query="ok", top_k=101)

    def test_long_query_accepted(self):
        q = "a" * 4096
        req = SearchRequest(query=q)
        assert len(req.query) == 4096


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_exact_match(self):
        from app.api.search import _cosine_similarity_from_distance

        assert _cosine_similarity_from_distance(0.0) == 1.0

    def test_orthogonal(self):
        from app.api.search import _cosine_similarity_from_distance

        assert _cosine_similarity_from_distance(1.0) == 0.0

    def test_opposite(self):
        from app.api.search import _cosine_similarity_from_distance

        assert _cosine_similarity_from_distance(2.0) == 0.0

    def test_edge_negative(self):
        from app.api.search import _cosine_similarity_from_distance

        assert _cosine_similarity_from_distance(-0.01) == 1.0

    def test_edge_exceeds_two(self):
        from app.api.search import _cosine_similarity_from_distance

        assert _cosine_similarity_from_distance(3.0) == 0.0


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


class TestAssembleResponse:
    def test_single_result(self):
        from app.api.search import _assemble_response

        raw = _raw_chroma_result(
            ids=["abc_chunk_0"],
            documents=["这是第一个chunk。"],
            metadatas=[
                {
                    "doc_id": "abc",
                    "doc_name": "test.pdf",
                    "doc_type": "pdf",
                    "page": 1,
                    "chunk_index": 0,
                }
            ],
            distances=[0.1],
        )
        resp = _assemble_response("测试查询", raw)
        assert resp.query == "测试查询"
        assert resp.total_results == 1
        assert len(resp.results) == 1

        r = resp.results[0]
        assert r.chunk_id == "abc_chunk_0"
        assert r.content == "这是第一个chunk。"
        assert r.score == 0.9  # 1.0 - 0.1
        assert r.doc_id == "abc"
        assert r.doc_name == "test.pdf"
        assert r.doc_type == "pdf"
        assert r.page == 1
        assert r.chunk_index == 0

    def test_multiple_results_ordered(self):
        from app.api.search import _assemble_response

        raw = _raw_chroma_result(
            ids=["c0", "c1", "c2"],
            documents=["zero", "one", "two"],
            metadatas=[
                {"doc_id": "d0", "doc_name": "a.txt", "doc_type": "txt", "page": 1, "chunk_index": 0},
                {"doc_id": "d1", "doc_name": "b.md", "doc_type": "md", "page": 2, "chunk_index": 1},
                {"doc_id": "d2", "doc_name": "c.pdf", "doc_type": "pdf", "page": 3, "chunk_index": 2},
            ],
            distances=[0.0, 0.5, 1.0],
        )
        resp = _assemble_response("q", raw)
        assert resp.total_results == 3
        scores = [r.score for r in resp.results]
        assert scores == [1.0, 0.5, 0.0]

    def test_empty_result(self):
        from app.api.search import _assemble_response

        raw = _raw_chroma_result()
        resp = _assemble_response("nothing", raw)
        assert resp.total_results == 0
        assert resp.results == []

    def test_missing_metadata_fields(self):
        from app.api.search import _assemble_response

        raw = _raw_chroma_result(
            ids=["x"],
            documents=["text"],
            metadatas=[{}],
            distances=[0.3],
        )
        resp = _assemble_response("q", raw)
        r = resp.results[0]
        assert r.doc_id == ""
        assert r.doc_name == ""
        assert r.page == 1  # default
        assert r.chunk_index == 0  # default


# ---------------------------------------------------------------------------
# Search response model
# ---------------------------------------------------------------------------


class TestSearchResponse:
    def test_serialization(self):
        resp = SearchResponse(
            query="hello",
            total_results=1,
            results=[
                SearchChunk(
                    chunk_id="abc_chunk_0",
                    content="hello world",
                    score=0.95,
                    doc_id="abc",
                    doc_name="test.txt",
                    doc_type="txt",
                    page=1,
                    chunk_index=0,
                )
            ],
        )
        d = resp.model_dump()
        assert d["query"] == "hello"
        assert d["total_results"] == 1
        assert len(d["results"]) == 1
        assert d["results"][0]["chunk_id"] == "abc_chunk_0"


# ---------------------------------------------------------------------------
# HTTP endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_store():
    """Isolated ChromaStore with pre-populated test data."""
    d = tempfile.mkdtemp()
    s = ChromaStore(persist_dir=d, collection_name="test_search")

    # Insert three chunks with known content
    chunks = [
        _make_chunk("d1_c0", "Python是一种编程语言", "d1", "python.md", DocType.MARKDOWN, 1, 0),
        _make_chunk("d1_c1", "Python用于数据科学", "d1", "python.md", DocType.MARKDOWN, 1, 1),
        _make_chunk("d2_c0", "机器学习是AI的分支", "d2", "ml.pdf", DocType.PDF, 2, 0),
    ]
    embs = [_emb(8) for _ in chunks]
    s.add(chunks=chunks, embeddings=embs)

    yield s

    s._client.clear_system_cache()
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def client():
    """FastAPI TestClient with the search router mounted."""
    from app.main import app
    return TestClient(app)


class TestSearchEndpointIntegration:
    """Integration tests that exercise the full search endpoint.

    These tests mock out the embedding API (no real LLM calls) but use a
    real local Chroma store.
    """

    def test_search_empty_query(self, client):
        resp = client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422  # validation error

    def test_search_missing_query(self, client):
        resp = client.post("/api/v1/search", json={})
        assert resp.status_code == 422

    def test_search_with_mocked_embedder(self, client, chroma_store):
        """Full flow: embed → chroma query → assemble, with embedder mocked."""
        mock_vec = [0.1] * 8

        with (
            patch("app.api.search.Embedder") as MockEmbedder,
            patch("app.api.search.ChromaStore") as MockChromaStore,
        ):
            # Configure mock embedder
            mock_embedder_instance = AsyncMock()
            mock_embedder_instance.embed.return_value = mock_vec
            MockEmbedder.return_value = mock_embedder_instance

            # Configure mock chroma
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.count.return_value = 3
            mock_chroma_instance.query.return_value = _raw_chroma_result(
                ids=["d1_c0", "d1_c1"],
                documents=["Python是一种编程语言", "Python用于数据科学"],
                metadatas=[
                    {"doc_id": "d1", "doc_name": "python.md", "doc_type": "md", "page": 1, "chunk_index": 0},
                    {"doc_id": "d1", "doc_name": "python.md", "doc_type": "md", "page": 1, "chunk_index": 1},
                ],
                distances=[0.1, 0.3],
            )
            MockChromaStore.return_value = mock_chroma_instance

            resp = client.post(
                "/api/v1/search",
                json={"query": "Python是什么？", "top_k": 3},
            )

            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["query"] == "Python是什么？"
            assert body["data"]["total_results"] == 2
            assert len(body["data"]["results"]) == 2
            assert body["data"]["results"][0]["chunk_id"] == "d1_c0"
            assert body["data"]["results"][0]["score"] == 0.9
            assert body["data"]["results"][0]["doc_name"] == "python.md"

    def test_search_on_empty_collection(self, client):
        """When Chroma has zero chunks, return empty result gracefully."""
        with (
            patch("app.api.search.Embedder") as MockEmbedder,
            patch("app.api.search.ChromaStore") as MockChromaStore,
        ):
            mock_embedder_instance = AsyncMock()
            mock_embedder_instance.embed.return_value = [0.1] * 8
            MockEmbedder.return_value = mock_embedder_instance

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.count.return_value = 0
            MockChromaStore.return_value = mock_chroma_instance

            resp = client.post(
                "/api/v1/search",
                json={"query": "any query", "top_k": 5},
            )

            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["total_results"] == 0
            assert body["data"]["results"] == []

    def test_search_embedder_error(self, client):
        """When the embedding API fails, return 502."""
        with (
            patch("app.api.search.Embedder") as MockEmbedder,
            patch("app.api.search.ChromaStore"),
        ):
            from app.core.embedder import EmbedderError

            mock_embedder_instance = AsyncMock()
            mock_embedder_instance.embed.side_effect = EmbedderError("API timeout")
            MockEmbedder.return_value = mock_embedder_instance

            resp = client.post(
                "/api/v1/search",
                json={"query": "test", "top_k": 5},
            )

            assert resp.status_code == 502
            body = resp.json()
            assert body["success"] is False
            assert body["code"] == 502
            assert "Embedding service error" in body["message"]


# ---------------------------------------------------------------------------
# Model serialization round-trip
# ---------------------------------------------------------------------------


class TestSearchChunkRoundTrip:
    def test_to_dict_and_back(self):
        chunk = SearchChunk(
            chunk_id="abc_chunk_0",
            content="hello",
            score=0.95,
            doc_id="abc",
            doc_name="test.md",
            doc_type="md",
            page=3,
            chunk_index=0,
        )
        d = chunk.model_dump()
        restored = SearchChunk(**d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.content == chunk.content
        assert restored.score == chunk.score
        assert restored.doc_name == chunk.doc_name
