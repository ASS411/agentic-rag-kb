"""Agent loop — query rewrite + replan + quality check (modules 3, 4).

Provides ``AgentLoop`` with ``_rewrite_query``, ``_replan``, and
``_quality_check`` methods that call the LLM to generate alternative /
supplementary search queries and evaluate retrieval sufficiency.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from loguru import logger

from app.config import settings
from app.core.llm import LLMClient, LLMError
from app.core.cache import RedisCacheManager, cache_key_rewrite
from app.core.prompts import (
    build_quality_check_messages,
    build_replan_messages,
    build_rewrite_messages,
    format_sources,
)
from app.core.prompts import RAGPromptBuilder
from app.models.search import SearchChunk
from app.models.sse import (
    SSEDoneEvent,
    SSEAnswerDoneEvent,
    SSEAnswerEvent,
    SSEErrorEvent,
    SSEStepEvent,
    SSESourcesEvent,
)


def _parse_query_list(raw: str, min_expected: int = 1) -> list[str]:
    """Robustly parse an LLM JSON response into a list of query strings.

    Handles:
    - Plain JSON arrays: ``["q1","q2"]``
    - Markdown-fenced JSON: `` ```json [...] ``` ``
    - Trailing commas, single quotes
    - Bare strings split by newlines (fallback)

    Returns the parsed list, or ``[raw]`` truncated to the first 80 chars
    when parsing fails entirely.
    """
    stripped = raw.strip()

    # 1. Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    # 2. Direct JSON parse
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return [str(item) for item in result if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Try with single-quote → double-quote fixup
    try:
        fixed = stripped.replace("'", '"')
        result = json.loads(fixed)
        if isinstance(result, list):
            return [str(item) for item in result if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # 4. Try to find the first [ ... ] block
    bracket_match = re.search(r"\[.*?\]", stripped, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return [str(item) for item in result if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    # 5. Fallback: split by newlines and filter non-empty lines
    lines = [line.strip().strip('"').strip("'").strip("-,;* ")
             for line in stripped.split("\n")
             if line.strip()]
    if len(lines) >= min_expected:
        return lines[:5]

    # 6. Ultimate fallback: return raw text (truncated), or empty if blank
    fallback = stripped[:80].replace("\n", " ")
    if fallback:
        logger.warning("Could not parse query list, using fallback: {}", fallback)
        return [fallback]
    return []


# ---------------------------------------------------------------------------
# JSON object parser for quality-check responses
# ---------------------------------------------------------------------------


def _parse_check_result(raw: str) -> dict:
    """Robustly parse an LLM JSON object response into a dict.

    Handles:
    - Plain JSON objects: ``{"sufficient": true, ...}``
    - Markdown-fenced JSON: `` ```json {...} ``` ``
    - Trailing commas, single quotes

    Returns a dict with keys ``sufficient``, ``reasoning``, ``gap``,
    or a default insufficient dict on parse failure.
    """
    stripped = raw.strip()

    # 1. Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    # 2. Direct JSON parse
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Try with single-quote → double-quote fixup
    try:
        fixed = stripped.replace("'", '"')
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 4. Try to find the first { ... } block
    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 5. Fallback: assume insufficient
    logger.warning("Could not parse quality check result, assuming insufficient")
    return {
        "sufficient": False,
        "reasoning": "Failed to parse LLM response",
        "gap": "quality check JSON parsing error",
    }
# ---------------------------------------------------------------------------
# AgentStep enum (task 5.1)
# ---------------------------------------------------------------------------


class AgentStep(str, Enum):
    """Steps in the agent pipeline state machine."""

    REWRITE = "rewrite"
    SEARCH = "search"
    RERANK = "rerank"
    CHECK = "check"
    REPLAN = "replan"
    GENERATE = "generate"
    DONE = "done"


# ---------------------------------------------------------------------------
# CheckResult model (task 4.2)
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    """Outcome of the LLM-as-Judge quality check.

    Consumed by the Agent loop to decide whether to replan or proceed.
    """

    sufficient: bool = Field(
        ...,
        description="Whether the current context pool contains enough "
                    "information to answer the question",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the evaluation",
    )
    gap: str | None = Field(
        default=None,
        description="Description of the information gap when sufficient=False",
    )


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Thin agent orchestrator for query rewrite and replan.

    This initial version (Phase 2 module 3) only implements the two LLM-
    based query manipulation methods.  The full agent state machine will
    be added in module 5.

    Parameters
    ----------
    llm:
        ``LLMClient`` instance.  Created with defaults if omitted.
    """

    def __init__(self, *, llm: LLMClient | None = None, cache: RedisCacheManager | None = None) -> None:
        self._llm = llm or LLMClient()
        self._cache = cache  # None = deferred lazy creation

    def _get_cache(self) -> RedisCacheManager | None:
        if self._cache is None:
            self._cache = RedisCacheManager()
        return self._cache if settings.redis.enabled else None

    # ------------------------------------------------------------------
    # Query Rewrite (task 3.1)
    # ------------------------------------------------------------------

    async def _rewrite_query(self, question: str) -> list[str]:
        """Analyse *question* and generate 3-5 alternative search queries.

        Uses the query-rewrite prompt template to ask the LLM to rephrase
        the question from different angles, covering synonyms, related
        concepts, and alternative phrasings.

        Parameters
        ----------
        question:
            The user's original natural-language question.

        Returns
        -------
        list[str]
            3-5 rewritten queries.  On LLM failure, falls back to a
            single-item list containing the original question.
        """
        cache = self._get_cache()
        if cache is not None:
            key = cache_key_rewrite(question)
            cached = await cache.get(key)
            if cached is not None:
                logger.debug("Rewrite cache hit: question={}", question[:60])
                return cached
            logger.debug("Rewrite cache miss: question={}", question[:60])

        messages = build_rewrite_messages(question)

        try:
            raw = await self._llm.generate(
                messages,
                temperature=0.7,   # higher temperature for creative rewrites
                max_tokens=512,
            )
        except LLMError as exc:
            logger.error("Query rewrite LLM call failed: {}", exc)
            return [question]

        queries = _parse_query_list(raw, min_expected=2)

        if not queries:
            logger.warning("Query rewrite produced empty list, using original")
            return [question]

        if cache is not None:
            await cache.set(
                cache_key_rewrite(question), queries,
                ttl=settings.redis.cache_ttl_rewrite,
            )

        logger.debug(
            "Query rewrite: {} original -> {} queries: {}",
            question[:60],
            len(queries),
            queries,
        )
        return queries

    # ------------------------------------------------------------------
    # Replan (task 3.2)
    # ------------------------------------------------------------------

    async def _replan(
        self,
        question: str,
        gap_description: str,
    ) -> list[str]:
        """Generate 2-3 supplementary queries to fill a retrieval gap.

        Called when the quality check determines that the current context
        pool is insufficient.  The *gap_description* explains what is
        missing, and the LLM is asked to produce targeted queries.

        Parameters
        ----------
        question:
            The user's original question.
        gap_description:
            Human-readable description of the information gap (e.g. from
            the quality-check step).

        Returns
        -------
        list[str]
            2-3 supplementary search queries.  Falls back to a generic
            re-query if the LLM call fails.
        """
        messages = build_replan_messages(question, gap_description)

        try:
            raw = await self._llm.generate(
                messages,
                temperature=0.5,
                max_tokens=512,
            )
        except LLMError as exc:
            logger.error("Replan LLM call failed: {}", exc)
            return [question]

        queries = _parse_query_list(raw, min_expected=1)

        if not queries:
            logger.warning("Replan produced empty list, using original")
            return [question]

        logger.debug(
            "Replan: gap={}, generated {} queries: {}",
            gap_description[:80],
            len(queries),
            queries,
        )
        return queries

    # ------------------------------------------------------------------
    # Quality Check (task 4.1)
    # ------------------------------------------------------------------

    async def _quality_check(
        self,
        question: str,
        context_pool: list[SearchChunk],
        *,
        max_chunks: int = 20,
        max_chars_per_chunk: int = 800,
    ) -> CheckResult:
        """Evaluate whether *context_pool* is sufficient to answer *question*.

        Uses the LLM as a judge: feeds the question plus the top chunks
        into a structured prompt and asks for a JSON assessment.

        Parameters
        ----------
        question:
            The user's original question.
        context_pool:
            Current deduplicated (and optionally reranked) list of chunks.
        max_chunks:
            Max chunks to show the LLM for evaluation.
        max_chars_per_chunk:
            Truncation length per chunk in the context view.

        Returns
        -------
        CheckResult
            Structured assessment with ``sufficient``, ``reasoning``,
            and optional ``gap``.
        """
        if not context_pool:
            return CheckResult(
                sufficient=False,
                reasoning="No context available",
                gap="Empty context pool — nothing to evaluate",
            )

        messages = build_quality_check_messages(
            question,
            context_pool,
            max_chunks=max_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )

        try:
            raw = await self._llm.generate(
                messages,
                temperature=0.1,  # low temperature for consistent judgment
                max_tokens=512,
            )
        except LLMError as exc:
            logger.error("Quality check LLM call failed: {}", exc)
            return CheckResult(
                sufficient=False,
                reasoning=f"LLM call failed: {exc}",
                gap="quality check error",
            )

        parsed = _parse_check_result(raw)

        result = CheckResult(
            sufficient=bool(parsed.get("sufficient", False)),
            reasoning=str(parsed.get("reasoning", "")),
            gap=parsed.get("gap") or None,
        )

        logger.debug(
            "Quality check: sufficient={}, reasoning={}, gap={}",
            result.sufficient,
            result.reasoning[:80],
            result.gap,
        )
        return result

    # ------------------------------------------------------------------
    # Main agent loop (task 5.2)
    # ------------------------------------------------------------------

    async def run(self, question: str, *, max_rounds: int | None = None):
        """Run the full agent pipeline: rewrite -> search (parent-child) -> check
        -> (replan -> search -> ...) -> generate. Yields SSE event strings.

        Uses parent-child retrieval: searches child chunks for precision,
        then returns parent chunks for complete semantic context. Reranking
        is performed internally by ``retrieve_with_parent_lookup``.
        """
        from app.core.retriever import Retriever

        max_rounds = max_rounds or settings.agent.max_rounds
        top_k_recall = settings.agent.top_k_recall
        top_k_rerank = settings.agent.top_k_rerank
        retriever = Retriever()

        def _evt(step, message="", **extra):
            return SSEStepEvent(
                step=step,
                message=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                **extra,
            ).model_dump_json(exclude_none=True)

        def _ans(content):
            return SSEAnswerEvent(
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump_json(exclude_none=True)

        def _src(content, ids=None):
            return SSESourcesEvent(
                content=content,
                chunk_ids=ids or [],
                source_chunks=[
                    chunk.model_dump(mode="json")
                    for chunk in final_chunks
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump_json(exclude_none=True)

        def _done():
            return SSEAnswerDoneEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump_json(exclude_none=True)

        def _terminal_done():
            return SSEDoneEvent(
                content="",
                conversation_id=None,
                total_rounds=rounds_completed,
                chunks_used=len(final_chunks),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump_json()

        def _error(message: str):
            return SSEErrorEvent(
                content=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump_json(exclude_none=True)

        # REWRITE
        yield _evt("rewrite", message="Query rewriting")
        queries = await self._rewrite_query(question)
        yield _evt(
            "rewrite",
            message=f"{len(queries)} queries generated",
            queries=queries,
            count=len(queries),
        )

        context_pool = {}  # chunk_id -> SearchChunk
        top_pool: list[SearchChunk] = []
        rounds_completed = 0

        for rnd in range(max_rounds):
            rounds_completed = rnd + 1
            # SEARCH (parent-child: search child chunks, return parent chunks)
            yield _evt(
                "search",
                message=f"Searching round {rnd+1} (parent-child)",
                round=rnd + 1,
                query_count=len(queries),
            )
            rr = await retriever.retrieve_with_parent_lookup(
                queries,
                top_k_recall=top_k_recall,
                top_k_rerank=top_k_rerank,
                rerank=True,
                hybrid=True,
            )
            yield _evt(
                "search",
                message=(
                    f"Retrieved {rr.total_recalled} child chunks"
                    f" → {len(rr.chunks)} parent chunks"
                ),
                total_recalled=rr.total_recalled,
                parent_chunks=len(rr.chunks),
                parent_lookup=rr.parent_lookup,
            )

            for c in rr.chunks:
                if c.chunk_id not in context_pool or \
                   c.score > context_pool[c.chunk_id].score:
                    context_pool[c.chunk_id] = c

            # Build top_pool from accumulated parent chunks (already
            # reranked internally by retrieve_with_parent_lookup).
            top_pool = sorted(
                context_pool.values(),
                key=lambda c: c.score,
                reverse=True,
            )[:top_k_rerank]

            if not top_pool:
                yield _evt("check", message="No context found",
                           verdict="insufficient")
                break

            # CHECK
            yield _evt("check", message="Evaluating quality")
            check = await self._quality_check(question, top_pool)
            yield _evt(
                "check",
                message="Quality evaluated",
                verdict="sufficient" if check.sufficient else "insufficient",
                reasoning=check.reasoning,
                gap=check.gap,
            )

            if check.sufficient:
                break

            if rnd < max_rounds - 1:
                gap = check.gap or "需要更多信息"
                yield _evt("replan", message=f"Replanning: {gap[:60]}", gap=gap)
                queries = await self._replan(question, gap)
                yield _evt("replan", message=f"{len(queries)} new queries", queries=queries)

        # GENERATE
        final_chunks = top_pool or sorted(
            context_pool.values(),
            key=lambda c: c.score,
            reverse=True,
        )[:top_k_rerank]
        yield _evt(
            "generate",
            message=f"Generating from {len(final_chunks)} chunks",
            count=len(final_chunks),
        )

        builder = RAGPromptBuilder()
        messages = builder.build(final_chunks, question)
        try:
            async for token in self._llm.generate_stream(messages):
                yield _ans(token)
        except LLMError as exc:
            yield _error(f"LLM error: {exc}")
            return

        # SOURCES + DONE
        sources_text = format_sources(final_chunks)
        chunk_ids = [c.chunk_id for c in final_chunks]
        yield _src(sources_text, chunk_ids)
        yield _done()
        yield _terminal_done()
