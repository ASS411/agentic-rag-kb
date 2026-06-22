"""Multi-query retriever with deduplication, re-ranking, parent-child lookup,
and hybrid BM25+vector search with RRF fusion (module 2.1 / 2.2).

Provides a ``Retriever`` class that accepts a list of query strings,
embeds each one, queries Chroma, merges + deduplicates results, and
optionally re-ranks the candidate pool through a cross-encoder.

Supports parent-child chunk retrieval:
- Search using child chunks (~800 chars) for precise keyword matching
- Retrieve parent chunks (~2000-3000 chars) for complete context during generation

Supports hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)
when ``hybrid=True`` is passed to ``retrieve()``.

Examples::

    retriever = Retriever()

    # Traditional retrieval
    result = await retriever.retrieve(
        queries=["What is RAG?"],
        top_k_recall=20,
        rerank=True,
    )

    # Parent-child retrieval (search child, return parent)
    result = await retriever.retrieve_with_parent_lookup(
        queries=["What is RAG?"],
        top_k_recall=20,
        rerank=True,
    )

    # Hybrid BM25 + vector retrieval
    result = await retriever.retrieve(
        queries=["What is RAG?"],
        top_k_recall=20,
        rerank=True,
        hybrid=True,
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from app.core.embedder import Embedder
from app.core.cache import RedisCacheManager, cache_key_retrieve, _hash_params
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
    parent_lookup: bool = Field(
        default=False,
        description="Whether parent chunks were retrieved for child matches",
    )
    hybrid: bool = Field(
        default=False,
        description="Whether BM25 keyword search was fused with vector results",
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
    """Multi-query semantic retriever with cross-encoder re-rank, parent-child
    lookup, and optional hybrid BM25+vector search.

    Accepts multiple query strings, embeds them in one batch, queries
    Chroma for each, merges results by deduplicating on ``chunk_id``,
    and optionally re-ranks the candidate pool through a ``Reranker``.

    Supports parent-child chunk retrieval:
    - Child chunks (~800 chars) are used for precise keyword matching during search
    - Parent chunks (~2000-3000 chars) are retrieved for complete context during generation

    Supports hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)
    when ``hybrid=True`` is passed to ``retrieve()``.

    Parameters
    ----------
    embedder:
        ``Embedder`` instance.  Created with defaults if omitted.
    chroma:
        ``ChromaStore`` instance.  Created with defaults if omitted.
    reranker:
        ``Reranker`` instance.  Created lazily on first rerank call
        when omitted and *rerank* is requested.
    bm25:
        ``BM25Index`` instance.  Created lazily on first hybrid search
        when omitted and *hybrid* is requested.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        chroma: ChromaStore | None = None,
        reranker=None,
        bm25=None,
        cache: RedisCacheManager | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._chroma = chroma or ChromaStore()
        self._reranker = reranker
        self._bm25 = bm25
        self._bm25_filter: dict | None = None   # tracks filter used for build
        self._cache = cache  # None = deferred lazy creation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_cache(self) -> RedisCacheManager | None:
        if self._cache is None:
            self._cache = RedisCacheManager()
        return self._cache if settings.redis.enabled else None

    async def retrieve(
        self,
        queries: list[str],
        *,
        top_k_recall: int | None = None,
        top_k_rerank: int | None = None,
        rerank: bool = False,
        use_child_chunks: bool = False,
        hybrid: bool = False,
    ) -> RetrievalResult:
        """Retrieve and optionally re-rank chunks for *queries*.

        Parameters
        ----------
        queries:
            One or more query strings.
        top_k_recall:
            Number of chunks to fetch per query from Chroma.
            Default: ``settings.agent.top_k_recall`` (20).
        top_k_rerank:
            Number of chunks to return after re-ranking.
            Default: ``settings.agent.top_k_rerank`` (5).
        rerank:
            When ``True``, the deduplicated candidate pool is re-scored
            by the cross-encoder and trimmed to *top_k_rerank*.
        use_child_chunks:
            When ``True``, filters search to only child chunks for more
            precise retrieval.
        hybrid:
            When ``True``, also runs BM25 keyword search and fuses results
            with vector search via Reciprocal Rank Fusion (RRF).

        Returns
        -------
        RetrievalResult
        """
        if not queries:
            return RetrievalResult()

        if top_k_recall is None:
            top_k_recall = settings.agent.top_k_recall
        if top_k_rerank is None:
            top_k_rerank = settings.agent.top_k_rerank

        _hybrid = hybrid and settings.agent.hybrid_search_enabled

        # DEV-NOTE: cache retrieval results keyed by query+params hash.
        # TTL kept short (1h) so document updates don't serve stale results
        # for too long; explicit invalidation clears on doc upload/delete.
        cache = self._get_cache()
        if cache is not None:
            params_hash = _hash_params(
                queries=sorted(queries),
                top_k_recall=top_k_recall,
                top_k_rerank=top_k_rerank,
                rerank=rerank,
                use_child_chunks=use_child_chunks,
                hybrid=_hybrid,
            )
            query_key = "|".join(sorted(queries))
            rk = cache_key_retrieve(query_key, params_hash)
            cached = await cache.get(rk)
            if cached is not None:
                logger.debug("Retrieve cache hit: queries={}", len(queries))
                return RetrievalResult.model_validate(cached)
            logger.debug("Retrieve cache miss: queries={}", len(queries))

        logger.debug(
            "Retriever: embedding {} queries, top_k_recall={}, rerank={}, "
            "use_child={}, hybrid={}",
            len(queries), top_k_recall, rerank, use_child_chunks, _hybrid,
        )

        # Honour the hybrid toggle; per-call ``hybrid`` is the gate (moved up)
        query_embeddings = await self._embedder.embed_batch(queries)

        total_count = self._chroma.count()
        if total_count == 0:
            logger.info("Retriever: empty Chroma collection")
            return RetrievalResult()

        n_results = min(top_k_recall, total_count)

        where_filter = {"is_child": True} if use_child_chunks else None

        chroma_result = self._chroma.query_batch(
            embeddings=query_embeddings,
            n_results=n_results,
            where=where_filter,
        )

        all_ids: list[list[str]] = chroma_result.get("ids", []) or []
        all_docs: list[list[str]] = chroma_result.get("documents", []) or []
        all_metas: list[list[dict]] = chroma_result.get("metadatas", []) or []
        all_dists: list[list[float]] = chroma_result.get("distances", []) or []

        candidates: dict[str, SearchChunk] = {}
        total_recalled = 0

        # Collect per-query vector results for proper per-query RRF rankings
        per_query_vector: list[list[SearchChunk]] = []

        for qi, query in enumerate(queries):
            ids = all_ids[qi] if qi < len(all_ids) else []
            docs = all_docs[qi] if qi < len(all_docs) else []
            metas = all_metas[qi] if qi < len(all_metas) else []
            dists = all_dists[qi] if qi < len(all_dists) else []

            total_recalled += len(ids)

            chunk_list = _chroma_result_to_searchchunks(query, ids, docs, metas, dists)
            per_query_vector.append(chunk_list)

            for c in chunk_list:
                if c.chunk_id not in candidates or c.score > candidates[c.chunk_id].score:
                    candidates[c.chunk_id] = c

        # --- Hybrid: BM25 keyword search + RRF fusion -----------------------
        if _hybrid:
            self._ensure_bm25_built(where_filter)
            bm25_per_query = self._bm25.search_batch(  # type: ignore[union-attr]
                queries, top_k=top_k_recall,
            )
            candidates = self._rrf_fuse(
                per_query_vector=per_query_vector,
                bm25_per_query=bm25_per_query,
                queries=queries,
                top_k=top_k_recall * 2,
            )
        # -------------------------------------------------------------------

        dedup_chunks = list(candidates.values())

        logger.debug(
            "Retriever: total_recalled={}, deduplicated={}{}",
            total_recalled,
            len(dedup_chunks),
            " (hybrid RRF fused)" if _hybrid else "",
        )

        if rerank and dedup_chunks:
            dedup_chunks = self._do_rerank(
                queries[0],
                dedup_chunks,
                top_k_rerank,
            )
            result = RetrievalResult(
                chunks=dedup_chunks,
                total_recalled=total_recalled,
                reranked=True,
                hybrid=_hybrid,
            )
            if cache is not None:
                await cache.set(
                    rk, result.model_dump(mode="json"),
                    ttl=settings.redis.cache_ttl_retrieve,
                )
            return result

        dedup_chunks.sort(key=lambda c: c.score, reverse=True)

        result = RetrievalResult(
            chunks=dedup_chunks[:top_k_rerank] if not rerank else dedup_chunks,
            total_recalled=total_recalled,
            reranked=False,
            hybrid=_hybrid,
        )
        if cache is not None:
            await cache.set(
                rk, result.model_dump(mode="json"),
                ttl=settings.redis.cache_ttl_retrieve,
            )
        return result

    async def retrieve_with_parent_lookup(
        self,
        queries: list[str],
        *,
        top_k_recall: int | None = None,
        top_k_rerank: int | None = None,
        rerank: bool = False,
        hybrid: bool = False,
    ) -> RetrievalResult:
        """Retrieve using child chunks, then lookup parent chunks for generation.

        This is the parent-child retrieval pattern:
        1. Search using child chunks (~800 chars) for precise keyword matching
        2. Extract parent_chunk_id from matching child chunks
        3. Retrieve parent chunks (~2000-3000 chars) for complete context

        Parameters
        ----------
        queries:
            One or more query strings.
        top_k_recall:
            Number of child chunks to fetch per query from Chroma.
            Default: ``settings.agent.top_k_recall`` (20).
        top_k_rerank:
            Number of parent chunks to return after re-ranking.
            Default: ``settings.agent.top_k_rerank`` (5).
        rerank:
            When ``True``, re-rank child chunks before parent lookup.
        hybrid:
            When ``True``, fuse BM25 keyword search with vector search
            on child chunks before parent lookup.

        Returns
        -------
        RetrievalResult
            Result chunks are parent chunks with complete semantic context.
        """
        if top_k_recall is None:
            top_k_recall = settings.agent.top_k_recall
        if top_k_rerank is None:
            top_k_rerank = settings.agent.top_k_rerank

        logger.info(
            "Parent-child retrieval: {} queries, top_k_recall={}, rerank={}, "
            "hybrid={}",
            len(queries), top_k_recall, rerank, hybrid,
        )

        child_result = await self.retrieve(
            queries=queries,
            top_k_recall=top_k_recall,
            top_k_rerank=top_k_recall,
            rerank=rerank,
            use_child_chunks=True,
            hybrid=hybrid,
        )

        if not child_result.chunks:
            return RetrievalResult(
                chunks=[],
                total_recalled=child_result.total_recalled,
                reranked=child_result.reranked,
                parent_lookup=False,
                hybrid=child_result.hybrid,
            )

        parent_ids = set()
        for chunk in child_result.chunks:
            parent_id = chunk.metadata.get("parent_chunk_id")
            if parent_id:
                parent_ids.add(parent_id)

        if not parent_ids:
            logger.warning("No parent_chunk_id found in child chunks")
            return child_result

        logger.debug(
            "Parent-child retrieval: {} unique parent chunks to fetch",
            len(parent_ids),
        )

        parent_result = self._chroma.get_by_ids(list(parent_ids))

        parent_chunks = self._convert_chroma_get_result(parent_result)

        parent_chunks.sort(key=lambda c: c.score, reverse=True)

        final_chunks = parent_chunks[:top_k_rerank]

        logger.info(
            "Parent-child retrieval: {} child chunks → {} parent chunks",
            len(child_result.chunks),
            len(final_chunks),
        )

        return RetrievalResult(
            chunks=final_chunks,
            total_recalled=child_result.total_recalled,
            reranked=child_result.reranked,
            parent_lookup=True,
            hybrid=child_result.hybrid,
        )

    # ------------------------------------------------------------------
    # Internal — BM25
    # ------------------------------------------------------------------

    def _ensure_bm25_built(self, where_filter: dict | None = None) -> None:
        """Build the BM25 index from Chroma chunks, tracking the filter so
        the index is rebuilt when the filter changes (e.g. a call with
        ``use_child_chunks=True`` after one with ``use_child_chunks=False``).
        """
        if self._bm25 is None:
            from app.core.bm25 import BM25Index
            self._bm25 = BM25Index()

        # Rebuild when filter changes or index has not been built yet
        _filter_changed = self._bm25_filter != where_filter
        if not self._bm25.is_built or _filter_changed:
            ids, docs, metas = self._chroma.get_all(where=where_filter)
            records = [
                {
                    "id": i, "content": d,
                    "doc_id": m.get("doc_id", ""),
                    "doc_name": m.get("doc_name", ""),
                    "doc_type": m.get("doc_type", ""),
                    "page": m.get("page", 1),
                    "chunk_index": m.get("chunk_index", 0),
                    "metadata": m,
                }
                for i, d, m in zip(ids, docs, metas)
            ]
            self._bm25.build_from_metas(records)
            self._bm25_filter = where_filter

    # ------------------------------------------------------------------
    # Internal — RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        *,
        per_query_vector: list[list[SearchChunk]],
        bm25_per_query: list[list[dict]],
        queries: list[str],
        top_k: int = 40,
    ) -> dict[str, SearchChunk]:
        """Fuse vector and BM25 results using Reciprocal Rank Fusion.

        Each query contributes its own independent vector ranking and
        BM25 ranking; the maximum RRF score across queries is kept for
        each chunk.

        Parameters
        ----------
        per_query_vector:
            Per-query vector search results (one list per query), each
            in order of descending similarity score.
        bm25_per_query:
            Per-query BM25 results; each entry is a list of dicts with
            keys ``chunk_id``, ``bm25_score``, ``rank``, and full metadata.
        queries:
            Original query strings (for logging context).
        top_k:
            Number of top fused chunks to retain.

        Returns
        -------
        dict[str, SearchChunk]
            Fused candidate pool keyed by chunk_id, with RRF scores.
        """
        k = settings.agent.hybrid_rrf_k

        # RRF score accumulator: chunk_id → max RRF score
        rrf_scores: dict[str, float] = {}

        # Deduplicated chunk_map for constructing final SearchChunks
        chunk_map: dict[str, SearchChunk] = {}
        for v_list in per_query_vector:
            for c in v_list:
                if c.chunk_id not in chunk_map or c.score > chunk_map[c.chunk_id].score:
                    chunk_map[c.chunk_id] = c

        # Number of queries should match between vector and BM25
        n_queries = max(len(per_query_vector), len(bm25_per_query))

        for qi in range(n_queries):
            # Vector RRF contributions — per-query ranking
            v_list = per_query_vector[qi] if qi < len(per_query_vector) else []
            # Sort by score descending to get per-query ranks
            v_sorted = sorted(v_list, key=lambda c: c.score, reverse=True)
            for rank_1, chunk in enumerate(v_sorted, start=1):
                rrf = 1.0 / (k + rank_1)
                if rrf > rrf_scores.get(chunk.chunk_id, -1.0):
                    rrf_scores[chunk.chunk_id] = rrf

            # BM25 RRF contributions — per-query ranking
            b_list = bm25_per_query[qi] if qi < len(bm25_per_query) else []
            for bm in b_list:
                chunk_id = bm["chunk_id"]
                bm25_rank = bm.get("rank", 999)
                rrf = 1.0 / (k + bm25_rank)
                if rrf > rrf_scores.get(chunk_id, -1.0):
                    rrf_scores[chunk_id] = rrf

                # Register BM25-only chunks not in vector results
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = SearchChunk(
                        chunk_id=chunk_id,
                        content=bm.get("content", ""),
                        score=bm.get("bm25_score", 0.0),
                        doc_id=bm.get("doc_id", ""),
                        doc_name=bm.get("doc_name", ""),
                        doc_type=bm.get("doc_type", ""),
                        page=bm.get("page", 1),
                        chunk_index=bm.get("chunk_index", 0),
                        metadata=bm.get("metadata", {}),
                    )

        # Sort by RRF score, keep top_k
        sorted_ids = sorted(
            rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True,
        )[:top_k]

        result: dict[str, SearchChunk] = {}
        for cid in sorted_ids:
            chunk = chunk_map.get(cid)
            if chunk:
                chunk.score = round(rrf_scores[cid], 6)
                result[cid] = chunk

        logger.debug(
            "RRF fusion: {} vector + {} BM25 across {} queries → {} fused (k={})",
            sum(len(v) for v in per_query_vector),
            sum(len(b) for b in bm25_per_query),
            n_queries,
            len(result),
            k,
        )
        return result

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _convert_chroma_get_result(self, result) -> list[SearchChunk]:
        """Convert Chroma get() result to SearchChunk list."""
        chunks: list[SearchChunk] = []
        ids = result.get("ids", []) or []
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            chunks.append(
                SearchChunk(
                    chunk_id=ids[i],
                    content=documents[i] if i < len(documents) else "",
                    score=meta.get("score", 0.0),
                    doc_id=meta.get("doc_id", ""),
                    doc_name=meta.get("doc_name", ""),
                    doc_type=meta.get("doc_type", ""),
                    page=meta.get("page", 1),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
            )
        return chunks

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

        internal_chunks = [_searchchunk_to_chunk(c) for c in chunks]

        reranked = self._reranker.rerank(question, internal_chunks, top_k=top_k)

        return [_chunk_to_searchchunk(c) for c in reranked]
