"""History API — conversation list and Q&A record retrieval.

Provides:
- GET /api/v1/qa/conversations — paginated conversation list
- GET /api/v1/qa/history — paginated Q&A records for a conversation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.db.mysql import get_db
from app.models.history import (
    ConversationListResponse,
    ConversationModel,
    ConversationSummary,
    QARecordItem,
    QARecordListResponse,
    QARecordModel,
)
from app.models.response import APIResponse

router = APIRouter(prefix="/qa", tags=["history"])

# ── Stale-generating record TTL ─────────────────────────────────────

STALE_GENERATING_MINUTES = 30


def _stale_cutoff() -> datetime:
    """Return the UTC-naive timestamp before which generating records are stale."""
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=STALE_GENERATING_MINUTES
    )


def _not_stale_generating_filter():
    """SQLAlchemy filter: exclude stale generating records."""
    cutoff = _stale_cutoff()
    return ~and_(
        QARecordModel.status == "generating",
        QARecordModel.created_at < cutoff,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/qa/conversations
# ---------------------------------------------------------------------------


@router.get("/conversations")
async def list_conversations(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ConversationListResponse]:
    """List all conversations with record counts, newest first."""
    # Count total
    total_q = select(func.count()).select_from(ConversationModel)
    total_result = await db.execute(total_q)
    total = total_result.scalar() or 0

    offset = (page - 1) * size

    # Fetch conversations ordered by created_at DESC
    conv_q = (
        select(ConversationModel)
        .order_by(desc(ConversationModel.created_at))
        .limit(size)
        .offset(offset)
    )
    conv_result = await db.execute(conv_q)
    conversations = conv_result.scalars().all()

    items: list[ConversationSummary] = []
    for conv in conversations:
        # Count valid records (exclude stale generating)
        valid_filter = _not_stale_generating_filter()
        count_q = (
            select(func.count())
            .select_from(QARecordModel)
            .where(
                QARecordModel.conversation_id == conv.conversation_id,
                valid_filter,
            )
        )
        count_result = await db.execute(count_q)
        record_count = count_result.scalar() or 0

        if record_count == 0:
            continue  # skip conversations with only stale generating records

        # Get last question from valid records
        last_q = (
            select(QARecordModel.question)
            .where(
                QARecordModel.conversation_id == conv.conversation_id,
                valid_filter,
            )
            .order_by(desc(QARecordModel.created_at))
            .limit(1)
        )
        last_result = await db.execute(last_q)
        last_question = last_result.scalar() or ""

        items.append(
            ConversationSummary(
                conversation_id=conv.conversation_id,
                title=conv.title,
                record_count=record_count,
                last_question=last_question,
                created_at=conv.created_at.isoformat() if conv.created_at else None,
            )
        )

    return APIResponse.ok(
        data=ConversationListResponse(items=items, total=total, page=page, size=size)
    )


# ---------------------------------------------------------------------------
# GET /api/v1/qa/history
# ---------------------------------------------------------------------------


@router.get("/history")
async def list_history(
    conversation_id: str = Query(..., min_length=1),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[QARecordListResponse]:
    """List Q&A records for a conversation, newest first.

    Query params:
        conversation_id (required): conversation identifier
        page: 1-indexed page number (default 1)
        size: items per page (default 20, max 100)
    """
    # Verify conversation exists
    conv_q = select(ConversationModel).where(
        ConversationModel.conversation_id == conversation_id
    )
    conv_result = await db.execute(conv_q)
    if conv_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    valid_filter = _not_stale_generating_filter()

    count_q = (
        select(func.count())
        .select_from(QARecordModel)
        .where(
            QARecordModel.conversation_id == conversation_id,
            valid_filter,
        )
    )
    count_result = await db.execute(count_q)
    total = count_result.scalar() or 0

    offset = (page - 1) * size

    records_q = (
        select(QARecordModel)
        .where(
            QARecordModel.conversation_id == conversation_id,
            valid_filter,
        )
        .order_by(desc(QARecordModel.created_at))
        .limit(size)
        .offset(offset)
    )
    records_result = await db.execute(records_q)
    records = records_result.scalars().all()

    items: list[QARecordItem] = []
    for r in records:
        # Deserialize JSON columns (SQLAlchemy may return as dict or JSON string)
        sources = _deserialize_json(r.sources_json)
        steps = _deserialize_json(r.agent_steps_json)

        items.append(
            QARecordItem(
                record_id=r.record_id,
                conversation_id=r.conversation_id,
                question=r.question,
                answer=r.answer,
                status=r.status,
                sources=sources,
                agent_steps=steps,
                total_rounds=r.total_rounds,
                model=r.model,
                tokens_used=r.tokens_used,
                created_at=r.created_at.isoformat() if r.created_at else None,
            )
        )

    return APIResponse.ok(
        data=QARecordListResponse(items=items, total=total, page=page, size=size)
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/qa/conversations/{conversation_id}
# ---------------------------------------------------------------------------


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[str]:
    """Delete a conversation and all its Q&A records."""
    conv_q = select(ConversationModel).where(
        ConversationModel.conversation_id == conversation_id
    )
    conv_result = await db.execute(conv_q)
    conv = conv_result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    records_q = select(QARecordModel).where(
        QARecordModel.conversation_id == conversation_id
    )
    records_result = await db.execute(records_q)
    records = records_result.scalars().all()
    for record in records:
        await db.delete(record)

    await db.delete(conv)
    await db.commit()

    logger.info(
        "Conversation deleted: conv_id={}, records={}",
        conversation_id,
        len(records),
    )
    return APIResponse.ok(data=conversation_id, message="Conversation deleted")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deserialize_json(value: object) -> list[dict]:
    """Safely convert a MySQL JSON column value to a list of dicts."""
    import json

    if value is None:
        return []
    if isinstance(value, list):
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
