"""Document management API — upload, list, and delete endpoints.

``POST /api/v1/documents/upload`` — multipart file upload → file + MySQL.
``GET  /api/v1/documents`` — paginated document list from MySQL.
``DELETE /api/v1/documents/{doc_id}`` — delete from MySQL + file system.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.mysql import get_db
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


def _ensure_upload_dir(doc_id: str) -> Path:
    """Create and return the per-document upload directory."""
    dir_path = Path(settings.upload.upload_dir) / doc_id
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


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
# Routes
# ---------------------------------------------------------------------------


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Document file (PDF, Markdown, or TXT). Max 50 MB.",
    ),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[DocumentResponse]:
    """Upload a document file.

    Validates type & size, persists the file and inserts a record into
    the MySQL ``documents`` table with ``status='processing'``.
    """
    content, filename, size = await _read_and_validate(file)
    doc_type = detect_doc_type(file.content_type, filename)
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Persist file ─────────────────────────────────────────────────
    upload_dir = _ensure_upload_dir(doc_id)
    file_path = upload_dir / filename
    file_path.write_bytes(content)

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
    # Count total
    total_q = select(func.count()).select_from(DocumentModel)
    total_result = await db.execute(total_q)
    total = total_result.scalar() or 0

    # Fetch page
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
    """Delete a document — removes MySQL record + uploaded file directory."""
    # ── Delete from MySQL ────────────────────────────────────────────
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.doc_id == doc_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    await db.delete(record)
    await db.commit()

    # ── Delete uploaded files ────────────────────────────────────────
    upload_dir = Path(settings.upload.upload_dir) / doc_id
    if upload_dir.exists():
        import shutil

        shutil.rmtree(upload_dir)
        logger.info("Deleted upload directory: {}", upload_dir)

    logger.info("Document deleted: doc_id={}", doc_id)
    return APIResponse.ok(data=doc_id, message="Document deleted")
