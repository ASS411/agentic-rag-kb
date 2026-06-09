"""Markdown parser — convert .md files to structured plain text.

Exports:
- ``parse_markdown(file_path)`` — synchronous, returns a ``Document``.
- ``MdParserError`` — raised for parsing failures.

Converts Markdown → HTML → plain text via ``markdown`` + ``BeautifulSoup``.
Heading structure is preserved with ``[H1]`` / ``[H2]`` / … prefixes.
Inline formatting markers (bold, italic, links, images) are stripped.
"""

from __future__ import annotations

from pathlib import Path

from app.models.document import DocType, Document, Page


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class MdParserError(Exception):
    """Raised when Markdown parsing fails."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_markdown(file_path: str | Path) -> Document:
    """Parse a Markdown file into a ``Document`` with preserved heading structure.

    The conversion pipeline:

    1. Read raw Markdown text (UTF-8).
    2. Convert to HTML via ``markdown`` with ``tables``, ``fenced_code`` extensions.
    3. Insert ``[H1]`` … ``[H6]`` markers before heading tags so the chunker can
       recognise section boundaries.
    4. Strip remaining HTML tags → plain text via ``BeautifulSoup.get_text()``.

    Args:
        file_path: Path to a ``.md`` or ``.markdown`` file.

    Returns:
        A ``Document`` with a single ``Page`` containing the extracted text.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        MdParserError: If the content cannot be parsed.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    # ── Read raw text ────────────────────────────────────────────────
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MdParserError(
            f"File is not valid UTF-8: {exc}",
            file_path=str(file_path),
        ) from exc

    # ── Markdown → HTML ──────────────────────────────────────────────
    try:
        import markdown
    except ImportError as exc:
        raise MdParserError(
            "markdown library not installed. Run: pip install markdown",
            file_path=str(file_path),
        ) from exc

    html = markdown.markdown(
        raw,
        extensions=["tables", "fenced_code"],
        output_format="html5",
    )

    # ── HTML → structured plain text ─────────────────────────────────
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise MdParserError(
            "beautifulsoup4 not installed. Run: pip install beautifulsoup4",
            file_path=str(file_path),
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    # Inject semantic heading markers *before* each heading element
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = tag.name  # "h1" … "h6"
        marker = f"[{level.upper()}] "
        tag.insert_before(soup.new_string(marker))

    text = soup.get_text()

    # ── Build Document ───────────────────────────────────────────────
    pages = [Page(page_number=1, text=text)]
    if not text.strip():
        pages = [Page(page_number=1, text="")]

    return Document(
        file_name=file_path.name,
        doc_type=DocType.MARKDOWN,
        pages=pages,
        metadata={"source_path": str(file_path)},
    )
