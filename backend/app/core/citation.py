"""Citation parser — regex extraction of [chunk_N] / [来源 N] markers
and fallback string matching (module 4.4 of DESIGN.md).

Provides ``CitationParser`` which analyses the LLM-generated answer text
against the context pool to produce a structured list of citations with
source metadata (doc name, page, snippet, relevance score).

Usage::

    from app.core.citation import CitationParser

    parser = CitationParser()
    citations = parser.parse(answer_text, context_pool)

    for c in citations:
        print(f"[{c.doc_name} p.{c.page}] {c.content_snippet[:60]}...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from app.models.search import SearchChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex patterns for citation markers in the LLM answer.
# Supports two formats:
#   1. [chunk_N]  — DESIGN.md canonical form
#   2. [来源 N]   — Chinese label used in the current system prompt
_CITATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[chunk[ _-]?(\d+)\]", re.IGNORECASE),
    re.compile(r"\[来源[ _-]?(\d+)\]"),
]

# Lower bound for a "meaningful" character-level match.
# Shorter matches are more likely to be spurious.
_MIN_SNIPPET_MATCH_LEN = 40


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A single verified citation — a chunk referenced by the LLM answer."""

    chunk_id: str
    """Unique chunk identifier (e.g. ``abc123_chunk_5``)."""

    doc_name: str
    """Original filename of the source document."""

    page: int
    """Page number (1-based) within the source document."""

    content_snippet: str
    """First 200 characters of the referenced chunk text."""

    score: float
    """Relevance score from the reranker (or 0.0 if unavailable)."""

    reference_label: str = ""
    """The marker text that was matched (e.g. ``[chunk_2]`` or ``[来源 2]``)."""


@dataclass
class CitationResult:
    """Result of citation parsing — citations + raw answer (unchanged)."""

    citations: list[Citation] = field(default_factory=list)
    """Verified citations sorted by score descending."""

    referenced_count: int = 0
    """Number of unique chunk indices referenced in the answer."""

    orphan_count: int = 0
    """Number of citation markers that did not resolve to a known chunk."""

    fallback_used: bool = False
    """True when regex found no matches and fallback string matching was used."""

    @property
    def cited_chunk_ids(self) -> set[str]:
        """Set of ``chunk_id`` strings that were cited."""
        return {c.chunk_id for c in self.citations}


# ---------------------------------------------------------------------------
# CitationParser
# ---------------------------------------------------------------------------


class CitationParser:
    """Parse and align LLM-generated citations with the context pool.

    Two-phase strategy:

    1. **Regex extraction** — scan the answer for ``[chunk_N]`` or
       ``[来源 N]`` markers, validate each index against the context pool,
       and build a ``Citation`` for every valid match.

    2. **Fallback string matching** — when regex finds *no* valid
       citations, attempt to locate each chunk's text in the answer via
       longest-common-substring heuristics.  This handles cases where
       the LLM paraphrases without using the exact marker format.

    Parameters
    ----------
    min_match_len:
        Minimum character length for a fallback substring match to be
        considered meaningful.  Defaults to ``_MIN_SNIPPET_MATCH_LEN``.
    snippet_len:
        Maximum characters to store in ``Citation.content_snippet``.
        Defaults to 200.
    """

    def __init__(
        self,
        *,
        min_match_len: int = _MIN_SNIPPET_MATCH_LEN,
        snippet_len: int = 200,
    ) -> None:
        self._min_match_len = min_match_len
        self._snippet_len = snippet_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        answer: str,
        context_pool: list[SearchChunk],
    ) -> CitationResult:
        """Parse citations from *answer* using *context_pool* as the source.

        Parameters
        ----------
        answer:
            Full LLM-generated answer text (may contain ``[chunk_N]`` or
            ``[来源 N]`` markers).
        context_pool:
            Ordered list of chunks that were fed into the generation
            prompt.  The 1-based index in this list corresponds to the
            ``N`` in ``[chunk_N]``.

        Returns
        -------
        CitationResult
            Structured result with verified citations and diagnostic counts.
        """
        if not answer or not context_pool:
            return CitationResult()

        # ── Phase 1: regex extraction ──────────────────────────────
        result = self._regex_parse(answer, context_pool)
        # Treat regex as "successful" when it found any markers at all
        # (even when all indices were out-of-range / orphans).
        has_regex_match = bool(result.citations) or result.orphan_count > 0
        if has_regex_match:
            logger.debug(
                "Citation regex: found {} citations ({} orphans)",
                result.referenced_count,
                result.orphan_count,
            )
            return result

        # ── Phase 2: fallback string match ─────────────────────────
        logger.debug("Citation regex found no matches, trying fallback string match")
        result = self._fallback_string_match(answer, context_pool)
        result.fallback_used = True
        logger.debug(
            "Citation fallback: found {} citations",
            result.referenced_count,
        )
        return result

    def parse_with_hybrid(
        self,
        answer: str,
        context_pool: list[SearchChunk],
    ) -> CitationResult:
        """Run regex parse first, then supplement with fallback matches.

        Unlike ``parse()``, this method does **not** short-circuit when
        regex succeeds — it always runs both phases and merges the results,
        deduplicating by ``chunk_id``.

        Returns
        -------
        CitationResult
            Combined citations from both phases.
        """
        regex_result = self._regex_parse(answer, context_pool)
        fallback_result = self._fallback_string_match(answer, context_pool)

        # Merge, de-duplicating by chunk_id
        seen: set[str] = set()
        merged: list[Citation] = []

        for c in regex_result.citations + fallback_result.citations:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)

        return CitationResult(
            citations=merged,
            referenced_count=len(merged),
            orphan_count=regex_result.orphan_count,
        )

    # ------------------------------------------------------------------
    # Phase 1 — regex
    # ------------------------------------------------------------------

    def _regex_parse(
        self,
        answer: str,
        context_pool: list[SearchChunk],
    ) -> CitationResult:
        """Extract citations via regex patterns and validate indices."""
        referenced: set[int] = set()
        orphan_count = 0

        for pattern in _CITATION_PATTERNS:
            for match in pattern.finditer(answer):
                try:
                    idx = int(match.group(1))
                except (ValueError, IndexError):
                    continue

                # 1-based index in the context pool
                if 1 <= idx <= len(context_pool):
                    referenced.add(idx)
                else:
                    orphan_count += 1

        citations: list[Citation] = []
        for idx in sorted(referenced):
            chunk = context_pool[idx - 1]
            citations.append(self._build_citation(idx, chunk))

        # Sort citations by relevance score (best first)
        citations.sort(key=lambda c: c.score, reverse=True)

        return CitationResult(
            citations=citations,
            referenced_count=len(citations),
            orphan_count=orphan_count,
        )

    # ------------------------------------------------------------------
    # Phase 2 — fallback string match
    # ------------------------------------------------------------------

    def _fallback_string_match(
        self,
        answer: str,
        context_pool: list[SearchChunk],
    ) -> CitationResult:
        """Find chunk text substrings in the answer when regex fails.

        For each chunk, extract the 3 longest sentences and check whether
        they appear (even partially) in the answer.  This handles cases
        where the LLM paraphrases content without using citation markers.
        """
        citations: list[Citation] = []

        for chunk in context_pool:
            # Extract key sentences, sorted longest-first
            sentences = _extract_sentences(chunk.content)
            key_sentences = sorted(sentences, key=len, reverse=True)[:3]

            for sentence in key_sentences:
                prefix = sentence[:self._min_match_len]
                if len(prefix) >= self._min_match_len and prefix in answer:
                    citations.append(
                        self._build_citation(None, chunk)  # type: ignore[arg-type]
                    )
                    break  # one citation per chunk

        return CitationResult(
            citations=citations,
            referenced_count=len(citations),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_citation(
        self,
        idx: int | None,
        chunk: SearchChunk,
    ) -> Citation:
        """Build a ``Citation`` from a context-pool chunk.

        Parameters
        ----------
        idx:
            1-based index of the chunk in the context pool (for regex
            matches), or ``None`` for fallback matches.
        chunk:
            The ``SearchChunk`` from the context pool.
        """
        score = float(chunk.metadata.get("rerank_score", chunk.score))
        label = f"[chunk_{idx}]" if idx is not None else ""

        return Citation(
            chunk_id=chunk.chunk_id,
            doc_name=chunk.doc_name,
            page=chunk.page,
            content_snippet=chunk.content[: self._snippet_len],
            score=score,
            reference_label=label,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _extract_sentences(text: str) -> list[str]:
    """Split *text* into sentences on Chinese and English punctuation.

    Keeps the punctuation attached to the sentence it terminates.
    """
    # Split on multiple sentence-ending characters
    parts = re.split(r"(?<=[。！？.!?\n])", text)
    return [p.strip() for p in parts if p.strip()]
