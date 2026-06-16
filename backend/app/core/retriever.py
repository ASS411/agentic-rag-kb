"""Multi-query retriever with deduplication and optional re-ranking (module 2.1 / 2.2).

Provides a ``Retriever`` class that accepts a list of query strings,
embeds each one, queries Chroma, merges + deduplicates results, and
optionally re-ranks the candidate pool through a cross-encoder.

Examples::

    retriever = Retriever()
    result = await retriever.retrieve(
        queries=["什么是RAG?", "RAG架构", "检索增强生成"],
        top_k_recall=20,
        rerank=True,
    )
    # result.chunks     -> list[SearchChunk] (top_k_rerank best)
    # result.total_recalled -> int (before dedup)
    # result.reranked   -> True
"""

from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field

from app.config import settings
from app.core.embedder import Embedder
from app.core.bm25_retriever import get_bm25, BM25Retriever, bm25_search_async
from app.db.chroma import ChromaStore
from app.models.search import SearchChunk

from loguru import logger


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """Outcome of a multi-query retrieval call."""

    chunks: list[SearchChunk] = Field(
        default_factory=list,
        description="Deduplicated (and optionally reranked) result chunks",
    )
    total_recalled: int = Field(
        default=0,
        description="Total chunks recalled across all queries *before* deduplication",
    )
    reranked: bool = Field(
        default=False,
        description="Whether the result pool was re-ranked by a cross-encoder",
    )
    hybrid: bool = Field(
        default=False,
        description="Whether BM25+vector hybrid search was used",
    )


# ---------------------------------------------------------------------------
# Helpers — Chunk <-> SearchChunk conversion
# ---------------------------------------------------------------------------


def _searchchunk_to_chunk(sc: SearchChunk):
    """Convert a Pydantic ``SearchChunk`` to the chunker ``Chunk`` dataclass.

    Needed because ``Reranker.rerank()`` expects ``Chunk`` instances.
    """
    from app.core.chunker import Chunk
    from app.models.document import DocType

    dt = DocType.TXT
    if sc.doc_type in ("pdf", "md", "txt"):
        dt = DocType(sc.doc_type)

    return Chunk(
        id=sc.chunk_id,
        content=sc.content,
        doc_id=sc.doc_id,
        doc_name=sc.doc_name,
        doc_type=dt,
        page=sc.page,
        chunk_index=sc.chunk_index,
        char_count=len(sc.content),
        metadata={**sc.metadata, "score": sc.score},
    )


def _chunk_to_searchchunk(c) -> SearchChunk:
    """Convert a chunker ``Chunk`` back to a Pydantic ``SearchChunk``."""
    score = float(
        c.metadata.get("rerank_score", c.metadata.get("score", 0.0))
    )
    return SearchChunk(
        chunk_id=c.id,
        content=c.content,
        score=score,
        doc_id=c.doc_id,
        doc_name=c.doc_name,
        doc_type=c.doc_type.value if hasattr(c.doc_type, "value") else str(c.doc_type),
        page=c.page,
        chunk_index=c.chunk_index,
        metadata=c.metadata,
    )


# ---------------------------------------------------------------------------
# Chroma result → SearchChunk helpers
# ---------------------------------------------------------------------------


def _chroma_result_to_searchchunks(
    query: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> list[SearchChunk]:
    """Convert a single query's Chroma result rows into ``SearchChunk`` objects."""
    chunks: list[SearchChunk] = []
    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else 1.0
        score = _cosine_similarity_from_distance(dist)

        chunks.append(
            SearchChunk(
                chunk_id=ids[i],
                content=documents[i] if i < len(documents) else "",
                score=round(score, 6),
                doc_id=meta.get("doc_id", ""),
                doc_name=meta.get("doc_name", ""),
                doc_type=meta.get("doc_type", ""),
                page=meta.get("page", 1),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
        )
    return chunks


def _cosine_similarity_from_distance(distance: float) -> float:
    """Convert Chroma cosine distance (0..2) to a similarity score (1..0)."""
    return 1.0 - (distance / 2.0)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Multi-query semantic retriever with optional cross-encoder re-rank.

    Accepts multiple query strings, embeds them in one batch, queries
    Chroma for each, merges results by deduplicating on ``chunk_id``,
    and optionally re-ranks the candidate pool through a ``Reranker``.

    Parameters
    ----------
    embedder:
        ``Embedder`` instance.  Created with defaults if omitted.
    chroma:
        ``ChromaStore`` instance.  Created with defaults if omitted.
    reranker:
        ``Reranker`` instance.  Created lazily on first rerank call
        when omitted and *rerank* is requested.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        chroma: ChromaStore | None = None,
        reranker=None,
        bm25: BM25Retriever | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._chroma = chroma or ChromaStore()
        self._reranker = reranker  # None → created on demand
        self._bm25 = bm25  # None → use singleton when hybrid is enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        queries: list[str],
        *,
        top_k_recall: int | None = None,
        top_k_rerank: int | None = None,
        rerank: bool = False,
        hybrid: bool = False,
        bm25_weight: float = 0.5,
    ) -> RetrievalResult:
        """Retrieve and optionally re-rank chunks for *queries*.

        Parameters
        ----------
        queries:
            One or more query strings (e.g. from query-rewrite or the
            original user question alone).
        top_k_recall:
            Number of chunks to fetch per query from Chroma.
            Default: ``settings.agent.top_k_recall`` (20).
        top_k_rerank:
            Number of chunks to return after re-ranking.
            Default: ``settings.agent.top_k_rerank`` (5).  Only used
            when *rerank* is ``True``.
        rerank:
            When ``True``, the deduplicated candidate pool is re-scored
            by the cross-encoder and trimmed to *top_k_rerank*.
        hybrid:
            When ``True``, run BM25 keyword search in parallel with
            vector search and fuse results via Reciprocal Rank Fusion
            (RRF).  Default: ``False``.
        bm25_weight:
            Weight of BM25 in RRF fusion (0–1).  Only used when
            *hybrid* is ``True``.  Default: 0.5.

        Returns
        -------
        RetrievalResult
        """
        if not queries:
            return RetrievalResult()

        if top_k_recall is None:
            top_k_recall = (
                settings.agent.top_k_recall_hybrid
                if hybrid
                else settings.agent.top_k_recall
            )
        if top_k_rerank is None:
            top_k_rerank = settings.agent.top_k_rerank

        # ── 1. Embed all queries ──────────────────────────────────
        logger.debug(
            "Retriever: embedding {} queries, top_k_recall={}, rerank={}, hybrid={}",
            len(queries),
            top_k_recall,
            rerank,
            hybrid,
        )
        query_embeddings = await self._embedder.embed_batch(queries)

        # ── 2. Query Chroma (batch) ───────────────────────────────
        total_count = self._chroma.count()
        if total_count == 0:
            logger.info("Retriever: empty Chroma collection")
            return RetrievalResult()

        # Limit recall to what's actually available
        n_results = min(top_k_recall, total_count)

        chroma_result = self._chroma.query_batch(
            embeddings=query_embeddings,
            n_results=n_results,
        )

        # ── 3. Build candidate pool ───────────────────────────────
        all_ids: list[list[str]] = chroma_result.get("ids", []) or []
        all_docs: list[list[str]] = chroma_result.get("documents", []) or []
        all_metas: list[list[dict]] = chroma_result.get("metadatas", []) or []
        all_dists: list[list[float]] = chroma_result.get("distances", []) or []

        candidates: dict[str, SearchChunk] = {}
        total_recalled = 0

        for qi, query in enumerate(queries):
            ids = all_ids[qi] if qi < len(all_ids) else []
            docs = all_docs[qi] if qi < len(all_docs) else []
            metas = all_metas[qi] if qi < len(all_metas) else []
            dists = all_dists[qi] if qi < len(all_dists) else []

            total_recalled += len(ids)

            chunk_list = _chroma_result_to_searchchunks(query, ids, docs, metas, dists)

            # Deduplicate: keep the highest score for each chunk_id
            for c in chunk_list:
                if c.chunk_id not in candidates or c.score > candidates[c.chunk_id].score:
                    candidates[c.chunk_id] = c

        # ── 3b. BM25 search (hybrid mode, parallel with vector) ───
        if hybrid:
            bm25 = self._bm25 or get_bm25()
            bm25_candidates = await _run_bm25_for_queries(
                bm25, queries, top_k=top_k_recall
            )
            total_recalled += sum(len(v) for v in bm25_candidates)

            # ── RRF fusion ────────────────────────────────────────
            candidates = _rrf_fuse(
                vector_candidates=candidates,
                bm25_candidates=bm25_candidates,
                queries=queries,
                bm25_weight=bm25_weight,
            )
            logger.debug(
                "Retriever hybrid fusion: {} candidates after RRF, "
                "weight_bm25={}",
                len(candidates),
                bm25_weight,
            )

        dedup_chunks = list(candidates.values())

        logger.debug(
            "Retriever: total_recalled={}, deduplicated={}",
            total_recalled,
            len(dedup_chunks),
        )

        # ── 4. Optionally re-rank ─────────────────────────────────
        if rerank and dedup_chunks:
            dedup_chunks = self._do_rerank(
                queries[0],  # use the first (original) query for reranking
                dedup_chunks,
                top_k_rerank,
            )
            return RetrievalResult(
                chunks=dedup_chunks,
                total_recalled=total_recalled,
                reranked=True,
                hybrid=hybrid,
            )

        # ── 5. Sort by score descending when not reranked ─────────
        dedup_chunks.sort(key=lambda c: c.score, reverse=True)

        return RetrievalResult(
            chunks=dedup_chunks[:top_k_rerank] if not rerank else dedup_chunks,
            total_recalled=total_recalled,
            reranked=False,
            hybrid=hybrid,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_rerank(
        self,
        question: str,
        chunks: list[SearchChunk],
        top_k: int,
    ) -> list[SearchChunk]:
        """Re-rank *chunks* with the cross-encoder and return the top *k*."""
        if self._reranker is None:
            from app.core.reranker import Reranker
            self._reranker = Reranker()

        # Convert SearchChunk → Chunk for the reranker
        internal_chunks = [_searchchunk_to_chunk(c) for c in chunks]

        reranked = self._reranker.rerank(question, internal_chunks, top_k=top_k)

        # Convert back to SearchChunk
        return [_chunk_to_searchchunk(c) for c in reranked]


# ---------------------------------------------------------------------------
# BM25 / RRF helpers (hybrid search)
# ---------------------------------------------------------------------------

_RRF_K: int = 60
"""RRF smoothing constant — higher values dampen rank differences."""


async def _run_bm25_for_queries(
    bm25: BM25Retriever,
    queries: list[str],
    top_k: int,
) -> list[list[tuple[str, float]]]:
    """Run BM25 search for all *queries* concurrently in a thread pool.

    Parameters
    ----------
    bm25:
        The ``BM25Retriever`` instance.
    queries:
        One or more search query strings.
    top_k:
        Max results per query.

    Returns
    -------
    list[list[tuple[str, float]]]
        Per-query list of ``(chunk_id, bm25_score)`` pairs.
    """
    if bm25.is_empty() or not queries:
        return [[] for _ in queries]

    tasks = [bm25_search_async(bm25, q, top_k) for q in queries]
    results = await asyncio.gather(*tasks)
    return list(results)


def _rrf_fuse(
    vector_candidates: dict[str, SearchChunk],
    bm25_candidates: list[list[tuple[str, float]]],
    queries: list[str],
    bm25_weight: float = 0.5,
) -> dict[str, SearchChunk]:
    """Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score for chunk *c*::

        rrf(c) = (1 - w) * Ʃ 1/(k + rank_in_vector)
               +    w    * Ʃ 1/(k + rank_in_bm25)

    where *k* = 60 and *w* = *bm25_weight*.

    Parameters
    ----------
    vector_candidates:
        ``chunk_id -> SearchChunk`` mapping from vector search.
    bm25_candidates:
        Per-query BM25 results as ``(chunk_id, score)`` pairs.
    queries:
        Original query list (used to weight BM25 per-query contributions).
    bm25_weight:
        Weight for BM25 in the fusion (0–1).

    Returns
    -------
    dict[str, SearchChunk]
        Fused candidates keyed by chunk_id, with RRF scores stored
        in each ``SearchChunk.score`` field.
    """
    if bm25_weight < 0.0 or bm25_weight > 1.0:
        bm25_weight = max(0.0, min(1.0, bm25_weight))

    vector_weight = 1.0 - bm25_weight

    # ── Build a temporary id -> SearchChunk map for BM25 results ───
    bm25_id_map: dict[str, SearchChunk] = {}
    for qi, per_query_results in enumerate(bm25_candidates):
        for rank_i, (chunk_id, _bm25_score) in enumerate(per_query_results):
            rrf_contrib = 1.0 / (_RRF_K + rank_i + 1)
            if chunk_id not in bm25_id_map:
                # Create a lightweight placeholder — content etc. will
                # come from vector_candidates or remain empty.
                bm25_id_map[chunk_id] = SearchChunk(
                    chunk_id=chunk_id,
                    content="",
                    score=0.0,
                    doc_id="",
                    doc_name="",
                    doc_type="",
                    page=1,
                    chunk_index=0,
                    metadata={},
                )
            bm25_id_map[chunk_id].score += rrf_contrib

    # ── Compute RRF scores ─────────────────────────────────────────
    fused: dict[str, SearchChunk] = {}

    # Collect all chunk_ids from both sources
    all_ids = set(vector_candidates.keys()) | set(bm25_id_map.keys())

    for chunk_id in all_ids:
        rrf_score = 0.0

        # Vector contribution — assign rank based on sorted order
        if chunk_id in vector_candidates:
            # Determine rank among all vector results (sorted by score desc)
            sorted_vector = sorted(
                vector_candidates.values(),
                key=lambda c: c.score,
                reverse=True,
            )
            for rank, sc in enumerate(sorted_vector):
                if sc.chunk_id == chunk_id:
                    rrf_score += vector_weight * (1.0 / (_RRF_K + rank + 1))
                    break

        # BM25 contribution already accumulated above
        if chunk_id in bm25_id_map:
            rrf_score += bm25_weight * bm25_id_map[chunk_id].score

        # Use the vector result's full data if available, otherwise BM25 placeholder
        if chunk_id in vector_candidates:
            fused[chunk_id] = vector_candidates[chunk_id]
            fused[chunk_id].score = round(rrf_score, 6)
        else:
            fused[chunk_id] = bm25_id_map[chunk_id]
            fused[chunk_id].score = round(rrf_score, 6)

    return fused
