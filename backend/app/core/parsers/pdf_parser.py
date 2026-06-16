"""PDF parser — PyMuPDF-based plain-text extraction with pdfplumber table support.

Exports:
- ``parse_pdf(file_path)`` — synchronous, returns a ``Document``.
- ``PdfParserError`` — raised for parsing failures (corrupted, encrypted, etc.)

Uses PyMuPDF (``fitz``) for fast, reliable text extraction from PDF files,
and pdfplumber for table extraction. Tables are converted to Markdown format
and appended to each page's text content.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.models.document import DocType, Document, Page

logger = logging.getLogger(__name__)


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
    """Extract plain text and tables from a PDF file using PyMuPDF + pdfplumber.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A ``Document`` object with one ``Page`` per PDF page, each
        containing the extracted text (with Markdown tables appended)
        and page-level metadata.

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

    # ── Try importing pdfplumber (non-critical) ────────────────────
    _pdfplumber_available = False
    try:
        import pdfplumber  # noqa: F811

        _pdfplumber_available = True
    except ImportError:
        logger.warning(
            "pdfplumber is not installed — table extraction will be skipped. "
            "Run: pip install pdfplumber"
        )

    # ── Open PDF ────────────────────────────────────────────────────
    try:
        doc: fitz.Document = fitz.open(str(file_path))
    except Exception as exc:
        raise PdfParserError(
            f"Failed to open PDF: {exc}",
            file_path=str(file_path),
        ) from exc

    _pdf = None
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

        # ── Open pdfplumber for table extraction ───────────────────
        if _pdfplumber_available:
            try:
                _pdf = pdfplumber.open(str(file_path))
            except Exception as exc:
                logger.warning(
                    "pdfplumber failed to open %s: %s. Table extraction skipped.",
                    file_path.name,
                    exc,
                )
                _pdfplumber_available = False

        # ── Extract per-page text and tables ───────────────────────
        pages: list[Page] = []
        for page_index in range(total_pages):
            page: fitz.Page | None = None
            text = ""

            try:
                page = doc.load_page(page_index)
                text = page.get_text()  # plain text extraction
            except Exception as exc:
                # Individual page failure → record empty page with error
                text = ""
                page = None
                logger.warning(
                    "Failed to extract text from page %d of %s: %s",
                    page_index + 1,
                    file_path.name,
                    exc,
                )

            # ── Extract tables with pdfplumber ──────────────────
            if _pdfplumber_available and _pdf is not None:
                try:
                    pdfplumber_page = _pdf.pages[page_index]
                    tables = pdfplumber_page.extract_tables()
                    if tables:
                        table_md = _tables_to_markdown(tables)
                        if table_md:
                            text = text + "\n\n" + table_md
                except Exception as exc:
                    logger.warning(
                        "Failed to extract tables from page %d of %s: %s",
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
        if _pdf is not None:
            try:
                _pdf.close()
            except Exception:
                pass


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


def _tables_to_markdown(tables: list) -> str:
    """Convert pdfplumber extracted tables to Markdown table format.

    Args:
        tables: List of tables from ``pdfplumber.Page.extract_tables()``.
            Each table is a list of rows, each row is a list of cell strings
            (or None).

    Returns:
        A string containing one or more Markdown tables separated by blank
        lines. Returns an empty string if *tables* is empty or contains no
        valid data.
    """
    if not tables:
        return ""

    output_parts: list[str] = []

    for table_idx, table in enumerate(tables):
        if not table or len(table) == 0:
            continue

        # Normalise: convert every cell to string, empty/None → ""
        rows: list[list[str]] = []
        for row in table:
            rows.append([str(cell).strip() if cell else "" for cell in row])

        # Find the maximum column count (in case rows have different lengths)
        col_count = max((len(row) for row in rows), default=0)
        if col_count == 0:
            continue

        # Pad rows to the same column count
        for i in range(len(rows)):
            while len(rows[i]) < col_count:
                rows[i].append("")

        # Build Markdown table
        lines: list[str] = []

        # Header row (first row)
        header = rows[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator row
        lines.append("| " + " | ".join(["---"] * col_count) + " |")

        # Data rows
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        output_parts.append("\n".join(lines))

    return "\n\n".join(output_parts)
