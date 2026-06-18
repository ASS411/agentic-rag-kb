"""Document parsers package — format detection and routing.

Provides:
- ``parse_document(file_path)`` — factory that detects the document type and
  delegates to the correct parser (PDF, Markdown, TXT).
- Individual parsers imported for direct use:
  - ``parse_pdf`` — PyMuPDF-based PDF extraction
  - ``parse_markdown`` — HTML-augmented Markdown → plain text
  - ``parse_txt`` — chardet encoding detection + text extraction
"""

from __future__ import annotations

from pathlib import Path

from app.core.parsers.md_parser import parse_markdown
from app.core.parsers.pdf_parser import parse_pdf
from app.core.parsers.txt_parser import parse_txt
from app.models.document import DocType, Document, detect_doc_type


# ── DocType → parser mapping ────────────────────────────────────────────────

_PARSERS = {
    DocType.PDF: parse_pdf,
    DocType.MARKDOWN: parse_markdown,
    DocType.TXT: parse_txt,
}


# ── Public factory ───────────────────────────────────────────────────────────


def parse_document(file_path: str | Path) -> Document:
    """Detect document type from file extension and delegate to the correct parser.

    Args:
        file_path: Path to a supported document (.pdf, .md, .txt).

    Returns:
        A ``Document`` with extracted text and metadata.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If *file_path* does not exist.
        PdfParserError | MdParserError | TxtParserError: On format-specific
            parsing failures.
    """
    file_path = Path(file_path)

    # Detect type by extension (no content-type header available here)
    doc_type = detect_doc_type(content_type=None, filename=file_path.name)

    parser = _PARSERS.get(doc_type)
    if parser is None:
        raise ValueError(f"No parser registered for type: {doc_type.value}")

    return parser(file_path)


__all__ = [
    "parse_document",
    "parse_pdf",
    "parse_markdown",
    "parse_txt",
]
