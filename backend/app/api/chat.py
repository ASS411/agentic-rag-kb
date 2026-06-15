"""Chat API — RAG Q&A with SSE streaming and non-streaming fallback.

``POST /api/v1/qa/ask``
    Flow: create-pending-record → search → build prompt → LLM generate
    → complete-record.

Two-phase persistence:
  1. INSERT qa_record (status='generating') BEFORE retrieval starts.
  2. UPDATE answer + status='complete' AFTER generation finishes.
  → A ``meta`` SSE event carries conversation_id/record_id early.

Supports:
- SSE ``text/event-stream`` token-by-token output (``stream=true``, default)
- Full JSON response (``stream=false``)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

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
from app.models.sse import (
    SSEStepEvent,
    SSEAnswerEvent,
    SSEAnswerDoneEvent,
    SSESourcesEvent,
    SSEDoneEvent,
)

router = APIRouter(prefix="/qa", tags=["qa"])


class CancelQARecordRequest(BaseModel):
    answer: str | None = None


@router.post("/records/{record_id}/cancel")
async def cancel_qa_record(
    record_id: str,
    body: CancelQARecordRequest | None = None,
) -> APIResponse[dict[str, str]]:
    """Mark an in-flight QA record as interrupted by the user."""
    from app.core.history_store import fail_qa_record

    try:
        answer = body.answer if body else None
        await fail_qa_record(record_id=record_id, answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return APIResponse.ok(data={"record_id": record_id, "status": "error"})


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _source_chunks_payload(chunks: list[SearchChunk]) -> list[dict]:
    """Serialize retrieved chunks for frontend source cards."""
    return [chunk.model_dump(mode="json") for chunk in chunks]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse_event(event_type: str, content: str = "", **extra) -> str:
    """Build a single SSE event string.

    Format: ``data: {json}\n\n``
    """
    payload = {"type": event_type, "content": content, "timestamp": _timestamp()}
    payload.update(extra)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> str:
    """Build an SSE error event."""
    payload = {"type": "error", "content": message, "timestamp": _timestamp()}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _check_concurrent(conversation_id: str | None) -> None:
    """Raise 409 if a generating record already exists for this conversation."""
    if not conversation_id:
        return
    from app.core.history_store import has_generating_record
    if await has_generating_record(conversation_id):
        raise HTTPException(
            status_code=409,
            detail="该对话的回答正在生成中，请等待完成后再发送新问题",
        )


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


@router.post("/ask", response_model=None)
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

    Persistence: creates a pending record before the Agent starts and
    completes it after generation finishes (two-phase pattern).
    """
    from app.core.agent import AgentLoop
    from app.core.history_store import (
        complete_qa_record,
        create_pending_qa_record,
        fail_qa_record,
    )

    # ── 1. Check concurrency + create pending record ──────────────
    await _check_concurrent(body.conversation_id)
    conv_id, record_id = await create_pending_qa_record(
        conversation_id=body.conversation_id,
        question=body.question,
    )
    yield _sse_event("meta", "", conversation_id=conv_id, record_id=record_id)

    agent = AgentLoop()
    answer_parts: list[str] = []
    source_chunks: list[SearchChunk] = []
    total_rounds = 0
    failed = False
    done_seen = False

    try:
        async for event_json in agent.run(
            body.question,
            max_rounds=body.max_rounds,
        ):
            yield f"data: {event_json}\n\n"
            event = json.loads(event_json)
            et = event.get("type")
            if et == "answer-chunk":
                answer_parts.append(str(event.get("content", "")))
            elif et == "sources":
                raw = event.get("source_chunks") or []
                source_chunks = [SearchChunk.model_validate(c) for c in raw]
            elif et == "done":
                total_rounds = int(event.get("total_rounds", 0))
                done_seen = True
    except Exception as exc:
        failed = True
        logger.error("Agent loop error: {}", exc)
        yield _sse_error(f"Agent error: {exc}")
        return
    finally:
        # Always complete the record — catches GeneratorExit,
        # CancelledError, and normal completion in one place.
        full_answer = "".join(answer_parts)
        if full_answer.strip() and not failed and done_seen:
            try:
                await complete_qa_record(
                    record_id=record_id,
                    answer=full_answer,
                    source_chunks=source_chunks,
                    total_rounds=total_rounds,
                )
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise  # re-raise GeneratorExit / CancelledError
                logger.warning("Failed to complete agent QA record: {}", exc)
        else:
            try:
                await asyncio.shield(
                    fail_qa_record(
                        record_id=record_id,
                        answer=full_answer.strip() or None,
                    )
                )
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                logger.warning("Failed to mark agent QA record failed: {}", exc)


async def _non_stream_agent_chat(body: ChatRequest) -> JSONResponse:
    """Run the agent loop and collapse streamed events into a JSON answer.

    Uses the two-phase persistence pattern.
    """
    from app.core.agent import AgentLoop
    from app.core.history_store import (
        complete_qa_record,
        create_pending_qa_record,
        fail_qa_record,
    )

    # ── 1. Check concurrency + create pending record ──────────────
    await _check_concurrent(body.conversation_id)
    conv_id, record_id = await create_pending_qa_record(
        conversation_id=body.conversation_id,
        question=body.question,
    )

    agent = AgentLoop()
    answer_parts: list[str] = []
    sources_text = ""
    source_chunks: list[SearchChunk] = []

    try:
        async for event_json in agent.run(
            body.question,
            max_rounds=body.max_rounds,
        ):
            event = json.loads(event_json)
            event_type = event.get("type")

            if event_type == "answer-chunk":
                answer_parts.append(str(event.get("content", "")))
            elif event_type == "sources":
                sources_text = str(event.get("content", sources_text))
                raw_chunks = event.get("source_chunks") or []
                source_chunks = [
                    SearchChunk.model_validate(chunk) for chunk in raw_chunks
                ]
            elif event_type == "error":
                message = str(event.get("content", "Agent error"))
                await fail_qa_record(record_id=record_id)
                return JSONResponse(
                    status_code=502,
                    content=APIResponse.error(502, message).model_dump(),
                )
    except Exception as exc:
        logger.error("Agent non-stream error: {}", exc)
        await fail_qa_record(record_id=record_id)
        return JSONResponse(
            status_code=502,
            content=APIResponse.error(502, f"Agent error: {exc}").model_dump(),
        )

    # ── 2. Complete the pending record ────────────────────────────
    full_answer = "".join(answer_parts)
    if full_answer.strip():
        try:
            await complete_qa_record(
                record_id=record_id,
                answer=full_answer,
                source_chunks=source_chunks,
            )
        except Exception as exc:
            logger.warning("Failed to complete agent QA record: {}", exc)
    else:
        await fail_qa_record(record_id=record_id)

    response = ChatResponse(
        answer=full_answer,
        sources=sources_text,
        source_chunks=source_chunks,
        question=body.question,
        conversation_id=conv_id,
    )

    return JSONResponse(content=APIResponse.ok(data=response).model_dump())


# ---------------------------------------------------------------------------
# Streaming implementation (5.3)
# ---------------------------------------------------------------------------


async def _stream_chat(body: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events for a streaming RAG chat response.

    Two-phase persistence: creates a pending record first, completes it
    after the LLM stream finishes.

    Event sequence:
        1. ``meta`` event — conversation_id + record_id (early)
        2. ``token`` events — one per content delta
        3. ``sources`` event — the formatted source list
        4. ``done`` event — final marker
        5. ``error`` event — on failure (terminates stream)
    """
    from app.core.history_store import (
        complete_qa_record,
        create_pending_qa_record,
        fail_qa_record,
    )

    # ── 1. Check concurrency + create pending record ──────────────
    await _check_concurrent(body.conversation_id)
    conv_id, record_id = await create_pending_qa_record(
        conversation_id=body.conversation_id,
        question=body.question,
    )
    yield _sse_event("meta", "", conversation_id=conv_id, record_id=record_id)

    # ── 2. Retrieve chunks ──────────────────────────────────────────
    try:
        chunks = await _retrieve_chunks(body.question, body.top_k)
    except HTTPException as exc:
        await fail_qa_record(record_id=record_id)
        yield _sse_error(exc.detail)
        return

    # ── 3. Build prompt ─────────────────────────────────────────────
    builder = RAGPromptBuilder()
    messages = builder.build(chunks, body.question)
    sources_text = format_sources(chunks)

    logger.debug(
        "Chat stream: chunks={}, prompt_msgs={}",
        len(chunks),
        len(messages),
    )

    # ── 4. Stream tokens ────────────────────────────────────────────
    llm = LLMClient()
    full_answer_parts: list[str] = []
    failed = False
    done_seen = False
    try:
        async for token in llm.generate_stream(messages):
            full_answer_parts.append(token)
            yield _sse_event("token", token)
        done_seen = True
    except LLMStreamError as exc:
        failed = True
        logger.error("Chat stream LLM error: {}", exc)
        yield _sse_error(f"LLM stream failed: {exc}")
        return
    except LLMError as exc:
        failed = True
        logger.error("Chat stream LLM error: {}", exc)
        yield _sse_error(f"LLM error: {exc}")
        return
    finally:
        # Always complete the record — catches GeneratorExit,
        # CancelledError, and normal completion in one place.
        full_answer = "".join(full_answer_parts)
        if full_answer.strip() and not failed and done_seen:
            try:
                await complete_qa_record(
                    record_id=record_id,
                    answer=full_answer,
                    source_chunks=chunks,
                    total_rounds=1,
                )
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise  # re-raise GeneratorExit / CancelledError
                logger.warning("Failed to complete QA stream record: {}", exc)
        else:
            try:
                await asyncio.shield(
                    fail_qa_record(
                        record_id=record_id,
                        answer=full_answer.strip() or None,
                    )
                )
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                logger.warning("Failed to mark QA stream record failed: {}", exc)

    # ── 6. Send sources + done ──────────────────────────────────────
    yield _sse_event(
        "sources",
        sources_text,
        source_chunks=_source_chunks_payload(chunks),
    )
    yield _sse_event("answer-done", "")
    yield _sse_event(
        "done",
        "",
        conversation_id=conv_id,
        total_rounds=1,
        chunks_used=len(chunks),
    )


# ---------------------------------------------------------------------------
# Non-streaming implementation (5.4)
# ---------------------------------------------------------------------------


async def _non_stream_chat(body: ChatRequest) -> JSONResponse:
    """Generate a full RAG answer and return it as JSON.

    Uses the two-phase persistence pattern.
    Returns the standard ``APIResponse[ChatResponse]`` envelope.
    """
    if body.use_agent:
        return await _non_stream_agent_chat(body)

    from app.core.history_store import (
        complete_qa_record,
        create_pending_qa_record,
        fail_qa_record,
    )

    # ── 1. Check concurrency + create pending record ──────────────
    await _check_concurrent(body.conversation_id)
    conv_id, record_id = await create_pending_qa_record(
        conversation_id=body.conversation_id,
        question=body.question,
    )

    # ── 2. Retrieve chunks ──────────────────────────────────────────
    try:
        chunks = await _retrieve_chunks(body.question, body.top_k)
    except HTTPException as exc:
        await fail_qa_record(record_id=record_id)
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.error(exc.status_code, exc.detail).model_dump(),
        )

    # ── 3. Build prompt ─────────────────────────────────────────────
    builder = RAGPromptBuilder()
    messages = builder.build(chunks, body.question)
    sources_text = format_sources(chunks)

    logger.debug(
        "Chat non-stream: chunks={}, prompt_msgs={}",
        len(chunks),
        len(messages),
    )

    # ── 4. Generate full answer ─────────────────────────────────────
    llm = LLMClient()
    try:
        answer = await llm.generate(messages)
    except LLMError as exc:
        logger.error("Chat non-stream LLM error: {}", exc)
        await fail_qa_record(record_id=record_id)
        return JSONResponse(
            status_code=502,
            content=APIResponse.error(502, f"LLM error: {exc}").model_dump(),
        )

    # ── 5. Complete the pending record ──────────────────────────────
    if answer.strip():
        try:
            await complete_qa_record(
                record_id=record_id,
                answer=answer,
                source_chunks=chunks,
                total_rounds=1,
            )
        except Exception as exc:
            logger.warning("Failed to complete QA record: {}", exc)
    else:
        await fail_qa_record(record_id=record_id)

    response = ChatResponse(
        answer=answer,
        sources=sources_text,
        source_chunks=chunks,
        question=body.question,
        conversation_id=conv_id,
    )

    logger.info(
        "Chat non-stream OK: answer_len={}, sources={}",
        len(answer),
        len(sources_text),
    )

    return JSONResponse(
        content=APIResponse.ok(data=response).model_dump(),
    )
