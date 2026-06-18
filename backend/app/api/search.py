"""Search API — semantic retrieval endpoint (module 4.1 / 4.2).

``POST /api/v1/search`` — embed query → Chroma query → assemble results.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.config import settings
from app.core.embedder import Embedder, EmbedderError
from app.db.chroma import ChromaStore
from app.models.response import APIResponse
from app.models.search import SearchChunk, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_from_distance(distance: float) -> float:
    """Convert Chroma cosine distance (0..2) to a similarity score (1..0).

    Cosine distance = 1 - cosine_similarity, so:
        similarity = 1 - distance

    Clamped to [0, 1] to avoid floating-point edge cases.
    """
    sim = 1.0 - distance
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _assemble_response(
    query: str,
    chroma_result: dict,
) -> SearchResponse:
    """Convert a raw Chroma ``QueryResult`` into a ``SearchResponse``.

    Chroma returns lists-of-lists (one inner list per query embedding).
    We sent a single query, so we unwrap index 0 everywhere.

    Parameters
    ----------
    query:
        Original search query string (echoed back).
    chroma_result:
        Raw result dict from ``ChromaStore.query()`` with keys
        ``ids``, ``documents``, ``metadatas``, ``distances``.

    Returns
    -------
    SearchResponse
        Structured response with scored chunks.
    """
    # Unwrap the outer list (single query)
    ids: list[str] = chroma_result.get("ids", [[]])[0] or []
    documents: list[str] = chroma_result.get("documents", [[]])[0] or []
    metadatas: list[dict] = chroma_result.get("metadatas", [[]])[0] or []
    distances: list[float] = chroma_result.get("distances", [[]])[0] or []

    results: list[SearchChunk] = []
    for i in range(len(ids)):
        chunk_id = ids[i] if i < len(ids) else ""
        content = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else 1.0

        score = _cosine_similarity_from_distance(dist)

        results.append(
            SearchChunk(
                chunk_id=chunk_id,
                content=content,
                score=round(score, 6),
                doc_id=meta.get("doc_id", ""),
                doc_name=meta.get("doc_name", ""),
                doc_type=meta.get("doc_type", ""),
                page=meta.get("page", 1),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
        )

    return SearchResponse(
        query=query,
        total_results=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("")
async def search(
    body: SearchRequest,
) -> APIResponse[SearchResponse]:
    """Semantic search: embed a query and retrieve top-k matching chunks.

    Flow:
        1. Validate input (handled by Pydantic model).
        2. Embed the query text via ``Embedder``.
        3. Query Chroma for the *top_k* nearest neighbours (cosine distance).
        4. Assemble results with chunk text, similarity score, and source info.

    Request body (JSON):

    .. code-block:: json

        {
            "query": "什么是知识图谱？",
            "top_k": 5
        }

    Response:

    .. code-block:: json

        {
            "success": true,
            "code": 200,
            "message": "ok",
            "data": {
                "query": "什么是知识图谱？",
                "total_results": 5,
                "results": [
                    {
                        "chunk_id": "abc123_chunk_0",
                        "content": "...",
                        "score": 0.923,
                        "doc_id": "abc123",
                        "doc_name": "knowledge_graph.pdf",
                        "doc_type": "pdf",
                        "page": 3,
                        "chunk_index": 0,
                        "metadata": {...}
                    }
                ]
            }
        }
    """
    # ── 1. Embed query ───────────────────────────────────────────────
    embedder = Embedder()
    try:
        query_embedding = await embedder.embed(body.query)
    except EmbedderError as exc:
        logger.error("Search embedding failed: {}", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service error: {exc}",
        )

    logger.debug(
        "Search query embedded: dims={}, query={!r}",
        len(query_embedding),
        body.query[:100],
    )

    # ── 2. Query Chroma ──────────────────────────────────────────────
    chroma = ChromaStore()

    if chroma.count() == 0:
        logger.info("Search on empty collection — returning zero results")
        return APIResponse.ok(
            data=SearchResponse(
                query=body.query,
                total_results=0,
                results=[],
            ),
            message="No documents indexed yet",
        )

    chroma_result = chroma.query(
        embedding=query_embedding,
        n_results=body.top_k,
    )

    # ── 3. Assemble response ─────────────────────────────────────────
    response = _assemble_response(body.query, chroma_result)

    logger.info(
        "Search: query={!r}, top_k={}, found={}",
        body.query[:100],
        body.top_k,
        response.total_results,
    )

    return APIResponse.ok(data=response)
