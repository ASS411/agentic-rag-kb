"""History persistence helper — save Q&A records to MySQL.

Called by the chat / agent endpoints before and after a turn completes.

Split into two-phase persistence:
  1. create_pending_qa_record  → INSERT with status='generating' (before LLM)
  2. complete_qa_record        → UPDATE with answer + status='complete' (after LLM)

The legacy save_qa_record is kept for backward compatibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import select

from app.config import settings
from app.db.mysql import async_session_factory
from app.models.history import ConversationModel, QARecordModel
from app.models.search import SearchChunk


def _build_sources_json(source_chunks: list[SearchChunk] | None) -> list[dict[str, Any]] | None:
    """Serialize source chunks into the JSON payload stored in the database."""
    if not source_chunks:
        return None
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_name": c.doc_name,
            "page": c.page,
            "content_snippet": c.content[:200],
            "score": c.score,
        }
        for c in source_chunks
    ]


async def _ensure_conversation(
    session,
    conversation_id: str | None,
    question: str,
) -> str:
    """Return an existing *conversation_id* or create a new one.

    Must be called inside an active async session.
    """
    if conversation_id is None:
        conversation_id = uuid.uuid4().hex
        conv = ConversationModel(
            conversation_id=conversation_id,
            title=question[:100],
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(conv)
    else:
        result = await session.execute(
            select(ConversationModel).where(
                ConversationModel.conversation_id == conversation_id
            )
        )
        if result.scalar_one_or_none() is None:
            conv = ConversationModel(
                conversation_id=conversation_id,
                title=question[:100],
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(conv)
    return str(conversation_id)


# ---------------------------------------------------------------------------
# Two-phase persistence (new API)
# ---------------------------------------------------------------------------


async def create_pending_qa_record(
    *,
    conversation_id: str | None,
    question: str,
) -> tuple[str, str]:
    """Insert a pending QA record BEFORE the LLM starts generating.

    Creates a new conversation when *conversation_id* is None.

    Returns ``(conversation_id, record_id)`` — the frontend should store
    *record_id* so it can be passed to :func:`complete_qa_record` later.
    """
    async with async_session_factory() as session:
        conv_id = await _ensure_conversation(session, conversation_id, question)

        record_id = uuid.uuid4().hex
        record = QARecordModel(
            record_id=record_id,
            conversation_id=conv_id,
            question=question,
            answer=None,
            status="generating",
            model=settings.llm.model,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(record)
        await session.commit()

        logger.info(
            "Pending QA record created: record_id={}, conversation_id={}",
            record_id,
            conv_id,
        )

    return str(conv_id), str(record_id)


async def complete_qa_record(
    *,
    record_id: str,
    answer: str,
    source_chunks: list[SearchChunk] | None = None,
    agent_steps: list[dict[str, Any]] | None = None,
    total_rounds: int | None = None,
    tokens_used: int | None = None,
) -> str:
    """Update a pending QA record after the LLM finishes generating.

    The *record_id* must match one returned by :func:`create_pending_qa_record`.
    Returns the ``conversation_id``.
    """
    sources_data = _build_sources_json(source_chunks)

    async with async_session_factory() as session:
        result = await session.execute(
            select(QARecordModel).where(QARecordModel.record_id == record_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ValueError(f"QA record not found: {record_id}")

        record.answer = answer
        record.status = "complete"
        record.sources_json = sources_data
        record.agent_steps_json = agent_steps
        record.total_rounds = total_rounds
        record.tokens_used = tokens_used
        await session.commit()

        logger.info(
            "QA record completed: record_id={}, answer_len={}",
            record_id,
            len(answer),
        )

    return str(record.conversation_id)


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------


async def has_generating_record(conversation_id: str) -> bool:
    """Return True if this conversation already has a record with
    status='generating' (i.e. another request is currently streaming)."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(QARecordModel).where(
                QARecordModel.conversation_id == conversation_id,
                QARecordModel.status == "generating",
            )
        )
        return result.scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# Legacy one-shot API (kept for backward compatibility)
# ---------------------------------------------------------------------------


async def save_qa_record(
    *,
    conversation_id: str | None,
    question: str,
    answer: str,
    source_chunks: list[SearchChunk] | None = None,
    agent_steps: list[dict[str, Any]] | None = None,
    total_rounds: int | None = None,
    tokens_used: int | None = None,
) -> str:
    """Persist a completed Q&A turn to MySQL (one-shot).

    Creates a new conversation when *conversation_id* is None.
    Returns the ``conversation_id`` so the caller can reference it for
    subsequent turns (multi-turn, Phase 4+).
    """
    sources_data = _build_sources_json(source_chunks)

    async with async_session_factory() as session:
        conv_id = await _ensure_conversation(session, conversation_id, question)

        record = QARecordModel(
            record_id=uuid.uuid4().hex,
            conversation_id=conv_id,
            question=question,
            answer=answer,
            sources_json=sources_data,
            agent_steps_json=agent_steps,
            total_rounds=total_rounds,
            model=settings.llm.model,
            tokens_used=tokens_used,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(record)
        await session.commit()

        logger.info(
            "QA record saved: record_id={}, conversation_id={}, answer_len={}",
            record.record_id,
            conv_id,
            len(answer),
        )

    return str(conv_id)
