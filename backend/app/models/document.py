"""Document data models — request/response schemas and domain types.

Used by the documents API layer for upload, list, and delete operations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class DocType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    MARKDOWN = "md"
    TXT = "txt"


# MIME → DocType mapping used by the upload endpoint for type validation
_MIME_TO_DOCTYPE: dict[str, DocType] = {
    "application/pdf": DocType.PDF,
    "text/markdown": DocType.MARKDOWN,
    "text/x-markdown": DocType.MARKDOWN,
    "text/plain": DocType.TXT,
    "text/x-python": DocType.TXT,  # .py as plain text
}

# Extension → DocType fallback mapping (when MIME is not recognised)
_EXT_TO_DOCTYPE: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    ".txt": DocType.TXT,
}


def detect_doc_type(content_type: str | None, filename: str) -> DocType:
    """Determine the document type from MIME content-type and filename extension.

    *content_type* is the HTTP ``Content-Type`` header (may be empty or
    ``application/octet-stream``).  *filename* is the original filename sent
    by the client.

    Returns the corresponding ``DocType``.

    Raises ``ValueError`` if the type cannot be determined or is unsupported.
    """
    # 1. Try MIME
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime in _MIME_TO_DOCTYPE:
            return _MIME_TO_DOCTYPE[mime]

    # 2. Fallback to extension
    ext = _get_ext(filename).lower()
    if ext in _EXT_TO_DOCTYPE:
        return _EXT_TO_DOCTYPE[ext]

    raise ValueError(f"Unsupported file type: {filename}")


def _get_ext(filename: str) -> str:
    """Return the file extension (including the leading dot), lowercased."""
    import os

    _base, ext = os.path.splitext(filename)
    return ext.lower()


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Public document information returned by the API."""

    doc_id: str = Field(..., description="Unique document identifier (UUID)")
    file_name: str = Field(..., description="Original filename")
    doc_type: DocType = Field(..., description="Document type: pdf / md / txt")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    page_count: int = Field(default=0, ge=0, description="Number of pages (PDF only)")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks after splitting")
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Upload timestamp (UTC)",
    )


class DocumentListResponse(BaseModel):
    """Paginated document list."""

    items: list[DocumentResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    size: int = Field(..., ge=1)
