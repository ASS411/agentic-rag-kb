"""Chat API — RAG Q&A with SSE streaming and non-streaming fallback.

``POST /api/v1/chat``
    Flow: search → build prompt → LLM generate (streaming or full).

Supports:
- SSE ``text/event-stream`` token-by-token output (``stream=true``, default)
- Full JSON response (``stream=false``)
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.config import settings
from app.core.embedder import Embedder, EmbedderError
from app.core.llm import LLMClient, LLMError, LLMStreamError
from app.core.prompts import RAGPromptBuilder, format_sources
from app.db.chroma import ChromaStore
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
)
from app.models.response import APIResponse
from app.models.search import SearchChunk
from app.api.search import _assemble_response, _cosine_similarity_from_distance
from app.models.sse import SSEStepEvent, SSEAnswerEvent, SSESourcesEvent, SSEDoneEvent

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _source_chunks_payload(chunks: list[SearchChunk]) -> list[dict]:
    """Serialize retrieved chunks for frontend source cards."""
    return [chunk.model_dump(mode="json") for chunk in chunks]


def _sse_event(event_type: str, content: str = "", **extra) -> str:
    """Build a single SSE event string.

    Format: ``data: {json}\n\n``
    """
    payload = {"type": event_type, "content": content}
    payload.update(extra)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> str:
    """Build an SSE error event."""
    payload = {"type": "error", "content": message}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Shared search → prompt pipeline
# ---------------------------------------------------------------------------

async def _retrieve_chunks(
    question: str, top_k: int
) -> list[SearchChunk]:
    """Embed the question and retrieve top-k chunks from Chroma.

    Returns an empty list when the collection is empty or an error occurs.
    """
    embedder = Embedder()
    try:
        query_embedding = await embedder.embed(question)
    except EmbedderError as exc:
        logger.error("Chat embedding failed: {}", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Embedding service error: {exc}",
        )

    chroma = ChromaStore()

    if chroma.count() == 0:
        logger.info("Chat: empty collection — no context available")
        return []

    chroma_result = chroma.query(
        embedding=query_embedding,
        n_results=top_k,
    )

    search_response = _assemble_response(question, chroma_result)
    return search_response.results


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("", response_model=None)
async def chat(
    body: ChatRequest,
):
    """RAG chat with SSE streaming (default) or full JSON response.

    **Streaming mode** (``stream=true``, default)::

        POST /api/v1/chat
        {"question": "什么是知识图谱？"}

        → ``text/event-stream``:

        data: {"type":"token","content":"根据"}\n\n
        data: {"type":"token","content":"已有"}\n\n
        ...
        data: {"type":"sources","content":"1. **doc.pdf** — 第3页\\n..."}\n\n
        data: {"type":"done","content":""}\n\n

    **Non-streaming mode** (``stream=false``)::

        POST /api/v1/chat
        {"question": "什么是知识图谱？", "stream": false}

        → ``application/json``:

        {
            "success": true,
            "code": 200,
            "message": "ok",
            "data": {
                "answer": "根据已有资料...",
                "sources": "1. **doc.pdf** — 第3页\\n...",
                "question": "什么是知识图谱？",
                "conversation_id": null
            }
        }

    Error responses follow the same ``APIResponse`` envelope:

    .. code-block:: json

        {
            "success": false,
            "code": 502,
            "message": "Embedding service error: ...",
            "data": null
        }
    """
    logger.info(
        "Chat: question={!r}, stream={}, top_k={}",
        body.question[:100],
        body.stream,
        body.top_k,
    )

    if body.stream:
        if body.use_agent:
            return StreamingResponse(
                _stream_agent_chat(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            _stream_chat(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )
    else:
        return await _non_stream_chat(body)


# ---------------------------------------------------------------------------
# Agent streaming implementation (module 5.4)
# ---------------------------------------------------------------------------


async def _stream_agent_chat(body: ChatRequest):
    """Run the agent loop and yield SSE events.

    Delegates to ``AgentLoop.run()`` which yields JSON event strings.
    Each string is wrapped as an SSE ``data:`` line.
    """
    from app.core.agent import AgentLoop

    agent = AgentLoop()
    try:
        async for event_json in agent.run(body.question):
            yield f"data: {event_json}\n\n"
    except Exception as exc:
        logger.error("Agent loop error: {}", exc)
        yield _sse_error(f"Agent error: {exc}")


# ---------------------------------------------------------------------------
# Streaming implementation (5.3)
# ---------------------------------------------------------------------------


async def _stream_chat(body: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events for a streaming RAG chat response.

    Event sequence:
        1. ``token`` events — one per content delta
        2. ``sources`` event — the formatted source list
        3. ``done`` event — final marker
        4. ``error`` event — on failure (terminates stream)
    """
    # ── 1. Retrieve chunks ──────────────────────────────────────────
    try:
        chunks = await _retrieve_chunks(body.question, body.top_k)
    except HTTPException as exc:
        yield _sse_error(exc.detail)
        return

    # ── 2. Build prompt ─────────────────────────────────────────────
    builder = RAGPromptBuilder()
    messages = builder.build(chunks, body.question)
    sources_text = format_sources(chunks)

    logger.debug(
        "Chat stream: chunks={}, prompt_msgs={}",
        len(chunks),
        len(messages),
    )

    # ── 3. Stream tokens ────────────────────────────────────────────
    llm = LLMClient()
    try:
        async for token in llm.generate_stream(messages):
            yield _sse_event("token", token)
    except LLMStreamError as exc:
        logger.error("Chat stream LLM error: {}", exc)
        yield _sse_error(f"LLM stream failed: {exc}")
        return
    except LLMError as exc:
        logger.error("Chat stream LLM error: {}", exc)
        yield _sse_error(f"LLM error: {exc}")
        return

    # ── 4. Send sources + done ──────────────────────────────────────
    yield _sse_event(
        "sources",
        sources_text,
        source_chunks=_source_chunks_payload(chunks),
    )
    yield _sse_event("done", "")


# ---------------------------------------------------------------------------
# Non-streaming implementation (5.4)
# ---------------------------------------------------------------------------


async def _non_stream_chat(body: ChatRequest) -> JSONResponse:
    """Generate a full RAG answer and return it as JSON.

    Returns the standard ``APIResponse[ChatResponse]`` envelope.
    """
    # ── 1. Retrieve chunks ──────────────────────────────────────────
    try:
        chunks = await _retrieve_chunks(body.question, body.top_k)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.error(exc.status_code, exc.detail).model_dump(),
        )

    # ── 2. Build prompt ─────────────────────────────────────────────
    builder = RAGPromptBuilder()
    messages = builder.build(chunks, body.question)
    sources_text = format_sources(chunks)

    logger.debug(
        "Chat non-stream: chunks={}, prompt_msgs={}",
        len(chunks),
        len(messages),
    )

    # ── 3. Generate full answer ─────────────────────────────────────
    llm = LLMClient()
    try:
        answer = await llm.generate(messages)
    except LLMError as exc:
        logger.error("Chat non-stream LLM error: {}", exc)
        return JSONResponse(
            status_code=502,
            content=APIResponse.error(502, f"LLM error: {exc}").model_dump(),
        )

    response = ChatResponse(
        answer=answer,
        sources=sources_text,
        source_chunks=chunks,
        question=body.question,
        conversation_id=body.conversation_id,
    )

    logger.info(
        "Chat non-stream OK: answer_len={}, sources={}",
        len(answer),
        len(sources_text),
    )

    return JSONResponse(
        content=APIResponse.ok(data=response).model_dump(),
    )
