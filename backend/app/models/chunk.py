"""Chunk ORM model — maps to MySQL ``chunks`` table (task 3.5).

Stores lightweight metadata for each chunk so downstream consumers
(search UI, analytics) can query chunk information without hitting Chroma.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func

from app.db.mysql import Base


class ChunkModel(Base):
    """SQLAlchemy ORM model for the ``chunks`` table.

    Each row records a single chunk produced by the ingestion pipeline.
    The full text lives in Chroma; this table stores only the metadata
    needed for lookup, counting, and deletion.
    """

    __tablename__ = "chunks"

    chunk_id = Column(String(128), primary_key=True, comment="Unique chunk ID")
    doc_id = Column(
        String(64),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owning document UUID",
    )
    content_hash = Column(
        String(64),
        nullable=False,
        comment="SHA-256 hex digest of chunk content",
    )
    char_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of characters in chunk",
    )
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<ChunkModel(chunk_id={self.chunk_id!r}, "
            f"doc_id={self.doc_id!r}, chars={self.char_count})>"
        )
