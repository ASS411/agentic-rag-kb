"""History data models — SQLAlchemy ORM for conversations / qa_records tables
and Pydantic schemas for the history API.

Tables (defined in db/init.sql):
  - conversations(conversation_id, title, created_at)
  - qa_records(record_id, conversation_id, question, answer, status,
               sources_json, agent_steps_json, total_rounds, model,
               tokens_used, created_at)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.orm import relationship

from app.db.mysql import Base


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class ConversationModel(Base):
    """ORM model for the ``conversations`` table."""

    __tablename__ = "conversations"

    conversation_id = Column(String(64), primary_key=True)
    title = Column(String(512), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationship (one conversation → many records)
    records = relationship("QARecordModel", back_populates="conversation", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Conversation(conversation_id={self.conversation_id!r}, title={self.title!r})>"


class QARecordModel(Base):
    """ORM model for the ``qa_records`` table."""

    __tablename__ = "qa_records"

    record_id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="complete", server_default="complete")
    sources_json = Column(MySQLJSON, nullable=True)
    agent_steps_json = Column(MySQLJSON, nullable=True)
    total_rounds = Column(Integer, nullable=True)
    model = Column(String(64), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationship back to conversation
    conversation = relationship("ConversationModel", back_populates="records")

    def __repr__(self) -> str:
        return f"<QARecord(record_id={self.record_id!r}, question={self.question[:30]!r}...)>"


# ---------------------------------------------------------------------------
# Pydantic API schemas
# ---------------------------------------------------------------------------


class ConversationSummary(BaseModel):
    """Summary of one conversation returned by GET /api/v1/qa/conversations."""

    conversation_id: str
    title: str | None
    record_count: int = 0
    last_question: str = ""
    status: str | None = None
    created_at: str | None = None


class ConversationListResponse(BaseModel):
    """Paginated conversation list."""

    items: list[ConversationSummary] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    size: int = 20


class QARecordItem(BaseModel):
    """A single Q&A record returned by GET /api/v1/qa/history."""

    record_id: str
    conversation_id: str
    question: str
    answer: str | None
    status: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    total_rounds: int | None = None
    model: str | None = None
    tokens_used: int | None = None
    created_at: str | None = None


class QARecordListResponse(BaseModel):
    """Paginated Q&A record list."""

    items: list[QARecordItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    size: int = 20
