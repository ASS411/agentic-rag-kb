"""SSE event models for the agent loop (module 5.3).

Defines structured Pydantic models for every event type emitted by
``AgentLoop.run()`` so the frontend can render step-by-step thinking.

Event types:
- ``agent-step`` — agent state transition with metadata
- ``answer-chunk`` — incremental token from LLM generation
- ``sources`` — final citation list
- ``done`` — terminal sentinel
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from app.models.search import SearchChunk


class CitationResultInfo(BaseModel):
    """Structured citation parsing result embedded in the sources event."""

    has_citations: bool = Field(
        default=False,
        description="Whether the LLM answer contained any verified citations",
    )
    referenced_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that were actually cited by the LLM",
    )
    orphan_count: int = Field(
        default=0,
        description="Number of citation markers that did not resolve to any chunk",
    )


class SSEStepEvent(BaseModel):
    """Emitted when the agent enters a new step in the pipeline."""

    type: Literal["agent-step"] = Field(default="agent-step")
    step: str = Field(
        ...,
        description="Agent step identifier: rewrite | search | rerank | "
                    "check | replan | generate | done",
    )
    message: str | None = Field(
        default="",
        description="Human-readable message for the UI",
    )
    queries: list[str] | None = Field(default=None)
    count: int | None = Field(default=None)
    round: int | None = Field(default=None)
    query_count: int | None = Field(default=None)
    total_recalled: int | None = Field(default=None)
    deduplicated: int | None = Field(default=None)
    verdict: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    gap: str | None = Field(default=None)
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when the step event was emitted",
    )


class SSEAnswerEvent(BaseModel):
    """Emitted for each token chunk during the final LLM generation."""

    type: Literal["answer-chunk"] = Field(default="answer-chunk")
    content: str = Field(
        default="",
        description="Incremental token text (may be empty for control events)",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when the token was emitted",
    )


class SSEAnswerDoneEvent(BaseModel):
    """Optional marker emitted when the answer stream has ended."""

    type: Literal["answer-done"] = Field(default="answer-done")
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when answer generation completed",
    )


class SSESourcesEvent(BaseModel):
    """Emitted after generation completes with the formatted source list."""

    type: Literal["sources"] = Field(default="sources")
    content: str = Field(
        default="",
        description="Pre-formatted Markdown source list",
    )
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks used in the final answer",
    )
    source_chunks: list[SearchChunk] = Field(
        default_factory=list,
        description="Structured chunks used by the answer/source panel",
    )
    citation_result: CitationResultInfo | None = Field(
        default=None,
        description="Citation parsing result with verification info",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when the sources were emitted",
    )


class SSEDoneEvent(BaseModel):
    """Terminal event signalling the end of the stream."""

    type: Literal["done"] = Field(default="done")
    content: str = Field(default="")
    conversation_id: str | None = Field(
        default=None,
        description="Conversation identifier when multi-turn support is enabled",
    )
    total_rounds: int | None = Field(
        default=None,
        description="How many agent rounds were executed",
    )
    chunks_used: int | None = Field(
        default=None,
        description="How many chunks were used in the final answer",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when the stream ended",
    )


class SSEErrorEvent(BaseModel):
    """Terminal error event for stream failures."""

    type: Literal["error"] = Field(default="error")
    content: str = Field(default="")
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp when the error was emitted",
    )
