"""Document management API — upload, list, and delete endpoints.

``POST /api/v1/documents/upload`` — multipart file upload → file + MySQL.
``GET  /api/v1/documents`` — paginated document list from MySQL.
``DELETE /api/v1/documents/{doc_id}`` — delete from MySQL + file system.

After upload, a background task runs the ingestion pipeline:
parse → chunk → embed → write to Chroma.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.cache import RedisCacheManager
from app.core.pipeline import IngestionPipeline
from app.core.storage import get_storage
from app.db.chroma import ChromaStore
from app.db.mysql import get_db, async_session_factory
from app.models.chunk import ChunkModel
from app.models.document import (
    DocType,
    DocumentListResponse,
    DocumentModel,
    DocumentResponse,
    detect_doc_type,
)
from app.models.response import APIResponse

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _invalidate_retrieval_cache() -> None:
    """Clear all retrieval result caches (called after doc changes)."""
    try:
        cache = RedisCacheManager()
        deleted = await cache.invalidate_by_prefix("retrieve:")
        if deleted:
            logger.info(
                "Invalidated {} retrieval cache entries after doc change",
                deleted,
            )
    except Exception:
        logger.warning("Failed to invalidate retrieval cache")


def _max_size_bytes() -> int:
    """Return the current max upload size in bytes (read live from settings)."""
    return settings.upload.max_upload_size_mb * 1024 * 1024


async def _read_and_validate(
    file: UploadFile,
) -> tuple[bytes, str, int]:
    """Read the uploaded file content and validate type / size.

    Returns ``(content, safe_filename, size)``.
    Raises ``HTTPException`` if validation fails.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    safe_filename = Path(file.filename).name

    try:
        detect_doc_type(file.content_type, safe_filename)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {safe_filename}. "
            f"Supported types: PDF, Markdown (.md), TXT",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(content) > _max_size_bytes():
        max_mb = settings.upload.max_upload_size_mb
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds {max_mb} MB limit",
        )

    return content, safe_filename, len(content)


def _orm_to_response(orm: DocumentModel) -> DocumentResponse:
    """Map a ``DocumentModel`` row to the public ``DocumentResponse``."""
    return DocumentResponse(
        doc_id=orm.doc_id,
        file_name=orm.file_name,
        doc_type=DocType(orm.doc_type),
        size_bytes=orm.size_bytes or 0,
        page_count=orm.page_count or 0,
        chunk_count=orm.chunk_count or 0,
        status=orm.status or "processing",
        error_message=orm.error_message,
        uploaded_at=orm.created_at or datetime.now(timezone.utc).replace(tzinfo=None),
    )


# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------


async def _run_ingestion(doc_id: str, file_path: str) -> None:
    """Background task: run the pipeline and update the MySQL record.

    Creates its own DB session and Chroma store so it does not depend on
    the request-scoped resources.  All errors are caught and logged —
    the background task must never crash the server.
    """
    logger.info("Background ingestion started: doc_id={}", doc_id)

    pipeline = IngestionPipeline()

    try:
        result = await pipeline.run(file_path, doc_id=doc_id)
    except Exception:
        logger.exception("Background ingestion failed: doc_id={}", doc_id)
        await _try_update_status(doc_id, status="error",
                                 error_message="Pipeline execution failed")
        return

    await _try_update_status(
        doc_id,
        status="ready",
        page_count=result.doc.page_count,
        chunk_count=result.chunk_count,
    )

    await _persist_chunk_metadata(doc_id, result.chunks)

    logger.info(
        "Background ingestion complete: doc_id={}, chunks={}, status=ready",
        doc_id,
        result.chunk_count,
    )

    await _invalidate_retrieval_cache()


async def _try_update_status(
    doc_id: str,
    status: str,
    page_count: int | None = None,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> None:
    """Attempt to update the document status in MySQL; silently ignore DB errors."""
    values: dict[str, object] = {"status": status, "error_message": error_message}
    if page_count is not None:
        values["page_count"] = page_count
    if chunk_count is not None:
        values["chunk_count"] = chunk_count

    try:
        async with async_session_factory() as session:
            await session.execute(
                update(DocumentModel)
                .where(DocumentModel.doc_id == doc_id)
                .values(**values)
            )
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to update document status in MySQL: doc_id={}, status={}",
            doc_id,
            status,
        )


async def _persist_chunk_metadata(
    doc_id: str,
    chunks: list,  # list[Chunk] from app.core.chunker
) -> None:
    """Bulk-insert chunk metadata rows into MySQL.

    Each row stores chunk_id, doc_id, content_hash, and char_count.
    Errors are silently ignored (the vector data in Chroma is the source
    of truth; this table is a convenience cache).
    """
    if not chunks:
        return

    from app.core.chunker import Chunk

    rows: list[ChunkModel] = []
    for c in chunks:
        content_hash = hashlib.sha256(c.content.encode("utf-8")).hexdigest()
        rows.append(
            ChunkModel(
                chunk_id=c.id,
                doc_id=doc_id,
                content_hash=content_hash,
                char_count=c.char_count,
            )
        )

    try:
        async with async_session_factory() as session:
            session.add_all(rows)
            await session.commit()
            logger.info(
                "Persisted {} chunk metadata rows for doc_id={}",
                len(rows),
                doc_id,
            )
    except Exception:
        logger.warning(
            "Failed to persist chunk metadata for doc_id={}",
            doc_id,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Document file (PDF, Markdown, or TXT). Max 50 MB.",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[DocumentResponse]:
    """Upload a document file.

    Validates type & size, persists the file via ``FileStorage`` and
    inserts a record into the MySQL ``documents`` table.

    A background task is scheduled to run the ingestion pipeline
    (parse → chunk → embed → write to Chroma).  The response returns
    immediately with ``status="processing"``.
    """
    content, filename, size = await _read_and_validate(file)
    doc_type = detect_doc_type(file.content_type, filename)
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Persist raw file ─────────────────────────────────────────────
    storage = get_storage()
    file_path = storage.save(doc_id, filename, content)

    # ── Persist metadata to MySQL ────────────────────────────────────
    record = DocumentModel(
        doc_id=doc_id,
        file_name=filename,
        doc_type=doc_type.value,
        file_path=str(file_path),
        page_count=0,    # updated by pipeline later
        chunk_count=0,   # updated by pipeline later
        size_bytes=size,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    # ── Schedule background ingestion ────────────────────────────────
    background_tasks.add_task(_run_ingestion, doc_id, str(file_path))

    logger.info(
        "Document uploaded: doc_id={}, file={}, type={}, size={}",
        doc_id,
        filename,
        doc_type.value,
        size,
    )

    return APIResponse.ok(
        data=_orm_to_response(record),
        message="Document uploaded successfully",
    )


@router.get("")
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[DocumentListResponse]:
    """List uploaded documents with pagination (from MySQL)."""
    total_q = select(func.count()).select_from(DocumentModel)
    total_result = await db.execute(total_q)
    total = total_result.scalar() or 0

    offset = (page - 1) * size
    items_q = (
        select(DocumentModel)
        .order_by(DocumentModel.created_at.desc())
        .limit(size)
        .offset(offset)
    )
    items_result = await db.execute(items_q)
    rows = items_result.scalars().all()

    items = [_orm_to_response(r) for r in rows]

    return APIResponse.ok(
        data=DocumentListResponse(items=items, total=total, page=page, size=size)
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[str]:
    """Delete a document — removes Chroma vectors + MySQL record + uploaded files.

    Order: Chroma first (idempotent → safe to retry), then MySQL, then
    filesystem.  If Chroma deletion fails, the MySQL record is kept so
    startup reconciliation (方案A) can retry on next restart.
    """
    # ── Verify document exists ───────────────────────────────────────
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.doc_id == doc_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    # ── 1. Delete from Chroma first (idempotent) ─────────────────────
    chroma = ChromaStore()
    deleted_chunks = chroma.delete_by_doc_id(doc_id)
    logger.info(
        "Chroma vectors removed: doc_id={}, chunks={}",
        doc_id,
        deleted_chunks,
    )

    # ── 2. Delete from MySQL ─────────────────────────────────────────
    await db.delete(record)
    await db.commit()

    # ── 3. Delete uploaded files ─────────────────────────────────────
    storage = get_storage()
    storage.delete(doc_id)

    logger.info(
        "Document fully deleted: doc_id={}, file={}",
        doc_id,
        record.file_name,
    )

    await _invalidate_retrieval_cache()

    return APIResponse.ok(data=doc_id, message="Document deleted")
