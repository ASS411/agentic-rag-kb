"""PDF parser — PyMuPDF-based plain-text extraction.

Exports:
- ``parse_pdf(file_path)`` — synchronous, returns a ``Document``.
- ``PdfParserError`` — raised for parsing failures (corrupted, encrypted, etc.)

Uses PyMuPDF (``fitz``) for fast, reliable text extraction from PDF files.
"""

from __future__ import annotations

from pathlib import Path

from app.models.document import DocType, Document, Page


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PdfParserError(Exception):
    """Raised when PDF parsing fails for any reason."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(file_path: str | Path) -> Document:
    """Extract plain text from a PDF file using PyMuPDF.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A ``Document`` object with one ``Page`` per PDF page, each
        containing the extracted text and page-level metadata.

    Raises:
        PdfParserError: If the file cannot be opened, is encrypted, or
            is otherwise unparseable.
        FileNotFoundError: If *file_path* does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise PdfParserError(
            "PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF",
            file_path=str(file_path),
        ) from exc

    # ── Open PDF ────────────────────────────────────────────────────
    try:
        doc: fitz.Document = fitz.open(str(file_path))
    except Exception as exc:
        raise PdfParserError(
            f"Failed to open PDF: {exc}",
            file_path=str(file_path),
        ) from exc

    try:
        # ── Check encryption ────────────────────────────────────────
        if doc.needs_pass:
            raise PdfParserError(
                "PDF is password-protected. Encrypted documents are not supported.",
                file_path=str(file_path),
            )

        total_pages = doc.page_count
        if total_pages == 0:
            raise PdfParserError(
                "PDF contains 0 pages — nothing to extract.",
                file_path=str(file_path),
            )

        # ── Extract per-page text ──────────────────────────────────
        pages: list[Page] = []
        for page_index in range(total_pages):
            try:
                page: fitz.Page = doc.load_page(page_index)
                text = page.get_text()  # plain text extraction
            except Exception as exc:
                # Individual page failure → record empty page with error
                text = ""
                page = None  # type: ignore[assignment]
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to extract text from page %d of %s: %s",
                    page_index + 1,
                    file_path.name,
                    exc,
                )

            pages.append(
                Page(
                    page_number=page_index + 1,
                    text=text.strip(),
                    metadata=_page_metadata(page),
                )
            )

        # ── Collect document metadata ──────────────────────────────
        metadata = _doc_metadata(doc, file_path)

        return Document(
            file_name=file_path.name,
            doc_type=DocType.PDF,
            pages=pages,
            metadata=metadata,
        )

    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_metadata(doc, file_path: Path) -> dict:
    """Extract PDF-level metadata."""
    meta: dict = {}
    try:
        info = doc.metadata
        if info:
            for key in ("title", "author", "subject", "creator", "producer"):
                val = info.get(key)
                if val:
                    meta[key] = val.strip()
    except Exception:
        pass

    meta["page_count"] = doc.page_count
    meta["file_path"] = str(file_path)
    return meta


def _page_metadata(page) -> dict:
    """Extract page-level metadata (dimensions)."""
    if page is None:
        return {}
    try:
        rect = page.rect
        return {"width": round(rect.width, 1), "height": round(rect.height, 1)}
    except Exception:
        return {}
