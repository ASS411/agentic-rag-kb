"""Document management API — upload, list, and delete endpoints.

``POST /api/v1/documents/upload`` — multipart file upload → file + MySQL.
``GET  /api/v1/documents`` — paginated document list from MySQL.
``DELETE /api/v1/documents/{doc_id}`` — delete from MySQL + file system.

After upload, a background task runs the ingestion pipeline:
parse → chunk → embed → write to Chroma.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.pipeline import IngestionPipeline
from app.core.storage import get_storage
from app.db.chroma import ChromaStore
from app.db.mysql import get_db, async_session_factory
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

    logger.info(
        "Background ingestion complete: doc_id={}, chunks={}, status=ready",
        doc_id,
        result.chunk_count,
    )


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
    """Delete a document — removes MySQL record + uploaded files + Chroma vectors."""
    # ── Delete from MySQL ────────────────────────────────────────────
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.doc_id == doc_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    await db.delete(record)
    await db.commit()

    # ── Delete from Chroma ───────────────────────────────────────────
    chroma = ChromaStore()
    deleted_chunks = chroma.delete_by_doc_id(doc_id)

    # ── Delete uploaded files ────────────────────────────────────────
    storage = get_storage()
    storage.delete(doc_id)

    logger.info(
        "Document deleted: doc_id={}, chroma_chunks_removed={}",
        doc_id,
        deleted_chunks,
    )
    return APIResponse.ok(data=doc_id, message="Document deleted")
