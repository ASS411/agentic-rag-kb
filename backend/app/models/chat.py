"""Chat request and response models (module 5.3 / 5.4).

Defines the Pydantic schemas for ``POST /api/v1/chat`` in both streaming
and non-streaming modes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.search import SearchChunk


class ChatRequest(BaseModel):
    """Chat request body.

    Example (streaming, default)::

        {"question": "什么是知识图谱？"}

    Example (non-streaming)::

        {"question": "什么是知识图谱？", "stream": false}

    Example (custom top_k)::

        {"question": "什么是知识图谱？", "top_k": 10}
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User question",
        examples=["什么是知识图谱？"],
    )
    stream: bool = Field(
        default=True,
        description="True → SSE streaming; False → full JSON response",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve for context",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Reserved for multi-turn conversations (Phase 4+)",
    )
    use_agent: bool = Field(
        default=True,
        description="When True, use the Agent loop (rewrite → search → "
                    "rerank → check → generate). When False, use the simple "
                    "single-search path (Phase 1 behaviour).",
    )


class ChatResponse(BaseModel):
    """Non-streaming chat response (``stream=false``)."""

    answer: str = Field(
        ...,
        description="Full generated answer text",
    )
    sources: str = Field(
        default="",
        description="Pre-formatted source list (Markdown)",
    )
    source_chunks: list[SearchChunk] = Field(
        default_factory=list,
        description="Structured source chunks used to build the answer",
    )
    question: str = Field(
        ...,
        description="Original question (echoed back)",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Reserved for multi-turn",
    )


class ChatStreamChunk(BaseModel):
    """A single SSE event payload for streaming chat."""

    type: Literal["token", "sources", "done", "error"] = Field(
        ...,
        description="Event type: token | sources | done | error",
    )
    content: str = Field(
        default="",
        description="Token text (type=token) or source list (type=sources) "
        "or error message (type=error)",
    )
    source_chunks: list[SearchChunk] = Field(
        default_factory=list,
        description="Structured source chunks when type=sources",
    )
