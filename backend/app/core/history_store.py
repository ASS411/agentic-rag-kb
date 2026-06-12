"""History persistence helper — save Q&A records to MySQL.

Called by the chat / agent endpoints after a turn completes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import select, func

from app.config import settings
from app.db.mysql import async_session_factory
from app.models.history import ConversationModel, QARecordModel
from app.models.search import SearchChunk


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
    """Persist a completed Q&A turn to MySQL.

    Creates a new conversation when *conversation_id* is None.
    Returns the ``conversation_id`` so the caller can reference it for
    subsequent turns (multi-turn, Phase 4+).
    """
    async with async_session_factory() as session:
        # ── Ensure conversation exists ──────────────────────────
        if conversation_id is None:
            conversation_id = uuid.uuid4().hex
            conv = ConversationModel(
                conversation_id=conversation_id,
                title=question[:100],
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(conv)
        else:
            # Verify existing conversation
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

        # ── Build sources JSON ──────────────────────────────────
        sources_data = None
        if source_chunks:
            sources_data = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "content_snippet": c.content[:200],
                    "score": c.score,
                }
                for c in source_chunks
            ]

        # ── Create record ───────────────────────────────────────
        record = QARecordModel(
            record_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
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
            conversation_id,
            len(answer),
        )

    return str(conversation_id)
