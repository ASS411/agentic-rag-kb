"""Document management API — upload, list, and delete endpoints.

``POST /api/v1/documents/upload`` — multipart file upload with type & size validation.
``GET  /api/v1/documents`` — paginated document list.
``DELETE /api/v1/documents/{doc_id}`` — delete a document and its chunks.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from loguru import logger

from app.config import settings
from app.models.document import (
    DocumentListResponse,
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
    # ── Validate filename ────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitise filename (keep only basename to avoid path traversal)
    safe_filename = Path(file.filename).name

    # ── Validate file type ───────────────────────────────────────────
    try:
        detect_doc_type(file.content_type, safe_filename)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {safe_filename}. "
            f"Supported types: PDF, Markdown (.md), TXT",
        )

    # ── Read content and check size ──────────────────────────────────
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Document file (PDF, Markdown, or TXT). Max 50 MB.",
    ),
) -> APIResponse[DocumentResponse]:
    """Upload a document file.

    Validates the file type (PDF / Markdown / TXT) and size,
    generates a unique document ID, persists the file to
    ``data/uploads/{doc_id}/``, and returns document metadata.

    The full processing pipeline (parse → chunk → embed → Chroma) is
    triggered asynchronously after the response is returned (future).

    Returns **201 Created** with the document metadata.
    """
    # ── Read & validate ──────────────────────────────────────────────
    content, filename, size = await _read_and_validate(file)

    doc_type = detect_doc_type(file.content_type, filename)

    # ── Generate doc_id and persist ──────────────────────────────────
    doc_id = uuid.uuid4().hex
    upload_dir = _ensure_upload_dir(doc_id)

    file_path = upload_dir / filename
    file_path.write_bytes(content)

    logger.info(
        "Document uploaded: doc_id={}, file={}, type={}, size={}",
        doc_id,
        filename,
        doc_type.value,
        size,
    )

    # ── Build response ───────────────────────────────────────────────
    doc = DocumentResponse(
        doc_id=doc_id,
        file_name=filename,
        doc_type=doc_type,
        size_bytes=size,
        page_count=0,   # set by parser pipeline (future)
        chunk_count=0,  # set by chunker pipeline (future)
        uploaded_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )

    return APIResponse.ok(data=doc, message="Document uploaded successfully")


@router.get("")
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> APIResponse[DocumentListResponse]:
    """List uploaded documents (stub — will query MySQL later).

    Returns a paginated list.  Currently returns an empty list as the
    MySQL-backed document registry is not yet wired up.
    """
    # TODO: query MySQL documents table once it exists (module 2.6)
    return APIResponse.ok(
        data=DocumentListResponse(items=[], total=0, page=page, size=size)
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
) -> APIResponse[str]:
    """Delete a document and its associated chunks (stub).

    Removes the uploaded file directory, Chroma embeddings, and MySQL
    records.  Not yet fully wired.
    """
    # TODO: full deletion pipeline (module 2.6+3.3)
    upload_dir = Path(settings.upload.upload_dir) / doc_id
    if upload_dir.exists():
        import shutil

        shutil.rmtree(upload_dir)
        logger.info("Deleted upload directory: {}", upload_dir)

    return APIResponse.ok(data=doc_id, message="Document deleted")
