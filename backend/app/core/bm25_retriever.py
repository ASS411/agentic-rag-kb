"""BM25 token-based retriever for hybrid search (task 1.1).

Maintains an in-memory BM25 index of all chunk texts, providing
fast keyword-based retrieval that complements vector search.

Provides a module-level singleton via ``get_bm25()``, following the
same pattern used by other services in the codebase.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.chunker import Chunk


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """In-memory BM25 index for keyword-based retrieval.

    The index stores tokenized chunk texts and supports:
    - ``index_chunks`` — partial / incremental index build
    - ``search`` — retrieve top-k chunks by BM25 score
    - ``rebuild_index`` — full reindex from scratch

    All methods are synchronous (CPU-bound).  Callers MUST wrap them
    in ``asyncio.to_thread()`` or a thread-pool executor.

    Thread safety: a re-entrant lock guards index mutations.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Add *chunks* to the BM25 index incrementally.

        Tokenizes the new chunks, appends them to the corpus, and
        rebuilds the underlying ``BM25Okapi`` object.

        Parameters
        ----------
        chunks:
            List of ``Chunk`` objects to index.
        """
        if not chunks:
            return

        with self._lock:
            new_tokenized = [_tokenize(c.content) for c in chunks]
            self._chunks.extend(chunks)
            self._tokenized_corpus.extend(new_tokenized)
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info(
                "BM25 index updated: added {} chunks, total={}",
                len(chunks),
                len(self._chunks),
            )

    def rebuild_index(self, chunks: list[Chunk]) -> None:
        """Rebuild the entire BM25 index from scratch.

        This replaces any previously indexed chunks.  Use after
        document deletions or for full reindexing.

        Parameters
        ----------
        chunks:
            Complete list of ``Chunk`` objects to index.
        """
        with self._lock:
            self._chunks = list(chunks)
            self._tokenized_corpus = [_tokenize(c.content) for c in chunks]
            self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
            logger.info(
                "BM25 index rebuilt: total chunks={}",
                len(self._chunks),
            )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search the BM25 index for the top-*k* chunks matching *query*.

        Parameters
        ----------
        query:
            Free-text search string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Ordered list of ``(chunk_id, score)`` pairs, where *score*
            is the normalised BM25 score (0–1 range).  Returns an empty
            list when the index is empty or *query* is blank.
        """
        if not query or not query.strip():
            return []
        if not self._bm25 or not self._chunks:
            return []

        with self._lock:
            tokenized_query = _tokenize(query)

            # rank_bm25 can be a list of floats or numpy array
            raw_scores: Any = self._bm25.get_scores(tokenized_query)

            # Normalise scores to 0–1 using max-score normalisation
            scores = _to_float_list(raw_scores)

            if not scores:
                return []

            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]

            n = min(top_k, len(scores))
            # Get indices of top-k scores (argsort descending)
            # Use enumerate + sorted for simplicity
            indexed: list[tuple[int, float]] = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_indices = indexed[:n]

            results: list[tuple[str, float]] = []
            for idx, score in top_indices:
                if idx < len(self._chunks):
                    results.append((self._chunks[idx].id, score))

            return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        with self._lock:
            return len(self._chunks)

    def is_empty(self) -> bool:
        """Return ``True`` when no chunks are indexed."""
        with self._lock:
            return len(self._chunks) == 0


# ---------------------------------------------------------------------------
# Helpers — tokenization
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* into lowercase word tokens.

    Uses a simple whitespace-and-punctuation split that works
    reasonably for Chinese (character-level unigrams after punctuation
    splitting) and English (word-level tokens).

    Parameters
    ----------
    text:
        Raw chunk text.

    Returns
    -------
    list[str]
        Lowercase tokens.
    """
    import re

    # Split on non-word characters (keep CJK characters 0x4e00-0x9fff
    # and alphanumerics)
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    return tokens


def _to_float_list(scores: Any) -> list[float]:
    """Convert a numpy array or list of scores to a Python float list."""
    import numpy as np

    if isinstance(scores, np.ndarray):
        return scores.tolist()  # type: ignore[return-type]
    return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bm25_instance: BM25Retriever | None = None
_bm25_lock = threading.Lock()


def get_bm25() -> BM25Retriever:
    """Return the module-level singleton ``BM25Retriever``.

    Lazily creates the instance on first access.  Thread-safe.
    """
    global _bm25_instance
    if _bm25_instance is None:
        with _bm25_lock:
            if _bm25_instance is None:
                _bm25_instance = BM25Retriever()
    return _bm25_instance


def reset_bm25() -> None:
    """Reset the singleton (useful for tests)."""
    global _bm25_instance
    with _bm25_lock:
        _bm25_instance = None


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def bm25_search_async(
    bm25: BM25Retriever,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Run BM25 search in a thread pool (CPU-bound)."""
    return await asyncio.to_thread(bm25.search, query, top_k)