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

from pydantic import BaseModel, Field

from app.models.search import SearchChunk


class SSEStepEvent(BaseModel):
    """Emitted when the agent enters a new step in the pipeline."""

    type: str = Field(default="agent-step")
    step: str = Field(
        ...,
        description="Agent step identifier: rewrite | search | rerank | "
                    "check | replan | generate | done",
    )
    label: str = Field(
        default="",
        description="Human-readable step label for the UI",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Step-specific metadata (queries, chunk counts, "
                    "scores, evaluation result, etc.)",
    )


class SSEAnswerEvent(BaseModel):
    """Emitted for each token chunk during the final LLM generation."""

    type: str = Field(default="answer-chunk")
    content: str = Field(
        default="",
        description="Incremental token text (may be empty for control events)",
    )


class SSESourcesEvent(BaseModel):
    """Emitted after generation completes with the formatted source list."""

    type: str = Field(default="sources")
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


class SSEDoneEvent(BaseModel):
    """Terminal event signalling the end of the stream."""

    type: str = Field(default="done")
    content: str = Field(default="")
