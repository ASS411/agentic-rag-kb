"""BM25 keyword-search index for hybrid retrieval (P1 roadmap §4).

Provides ``BM25Index`` — a thin wrapper around ``rank_bm25.BM25Okapi``
that indexes chunk contents and supports multi-query batch search.

Usage::

    bm25 = BM25Index()
    bm25.build_from_metas(chunk_dicts)   # chunk_dicts = [{id, content}, ...]
    results = bm25.search_batch(["query1", "query2"], top_k=20)
    # -> list of dicts with chunk_id, bm25_score, rank
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from loguru import logger


class BM25Index:
    """In-memory BM25 index over chunk contents.

    Supports rebuild (e.g. after document ingestion) and multi-query
    batch search returning ranked (chunk_id, score) pairs for RRF fusion.
    """

    def __init__(self) -> None:
        self._chunk_records: list[dict] = []          # [{id, content}, ...]
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: object | None = None               # BM25Okapi instance
        self._built: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_metas(self, records: Sequence[dict]) -> None:
        """Build (or rebuild) the BM25 index from chunk records.

        Parameters
        ----------
        records:
            Sequence of dicts, each containing at least ``"id"`` (str) and
            ``"content"`` (str).  Extra keys are preserved for downstream
            lookup after search.
        """
        if not records:
            logger.warning("BM25Index.build_from_metas: empty records, index cleared")
            self._chunk_records = []
            self._tokenized_corpus = []
            self._bm25 = None
            self._built = True
            return

        self._chunk_records = list(records)
        self._tokenized_corpus = [
            self._tokenize(r["content"]) for r in self._chunk_records
        ]

        # Lazy import to avoid hard dependency at module level
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._built = True

        logger.info(
            "BM25Index built: {} chunks indexed, corpus size={} tokens",
            len(self._chunk_records),
            sum(len(t) for t in self._tokenized_corpus),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, *, top_k: int = 20,
    ) -> list[dict]:
        """Search the BM25 index for *query* and return ranked results.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``chunk_id``, ``bm25_score``, ``rank``
            (1-based), plus the original record fields.
        """
        if not self._built or self._bm25 is None or not self._chunk_records:
            return []

        tokenized = self._tokenize(query)
        if not tokenized:
            return []

        scores = self._bm25.get_scores(tokenized)  # type: ignore[union-attr]

        # Build (index, score) pairs, sort descending, take top_k
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True,
        )[:top_k]

        results: list[dict] = []
        for rank_1, (idx, score) in enumerate(indexed, start=1):
            rec = self._chunk_records[idx]
            results.append({
                "chunk_id": rec["id"],
                "bm25_score": float(score),
                "rank": rank_1,
                **{k: v for k, v in rec.items() if k not in ("id",)},
            })
        return results

    def search_batch(
        self, queries: list[str], *, top_k: int = 20,
    ) -> list[list[dict]]:
        """Search for multiple queries; returns per-query result lists."""
        return [self.search(q, top_k=top_k) for q in queries]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_records)

    def reset(self) -> None:
        """Clear the index (useful for tests)."""
        self._chunk_records = []
        self._tokenized_corpus = []
        self._bm25 = None
        self._built = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing.

        Splits on word boundaries (Unicode-aware for CJK support via
        character-level fallback for non-ASCII scripts).
        """
        # Latin / mixed scripts: split on word boundaries
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        if not tokens:
            return []
        # For CJK-heavy content, also add character bigrams as a cheap
        # substitute for proper segmentation
        cjk_tokens: list[str] = []
        for token in tokens:
            if re.search(r"[\u4e00-\u9fff]", token) and len(token) > 1:
                cjk_tokens.extend(
                    token[i:i + 2] for i in range(len(token) - 1)
                )
        return tokens + cjk_tokens
