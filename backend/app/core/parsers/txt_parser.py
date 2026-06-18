"""Plain-text parser — encoding-aware .txt file extraction.

Exports:
- ``parse_txt(file_path)`` — synchronous, returns a ``Document``.
- ``TxtParserError`` — raised for decoding failures.

Detects the file encoding (UTF-8, GBK, Shift_JIS, etc.) via ``chardet``,
then decodes and wraps the result in a single-``Page`` ``Document``.
"""

from __future__ import annotations

from pathlib import Path

from app.models.document import DocType, Document, Page


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TxtParserError(Exception):
    """Raised when TXT parsing fails (e.g. undecodable content)."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_txt(file_path: str | Path) -> Document:
    """Parse a plain-text file with automatic encoding detection.

    Args:
        file_path: Path to a ``.txt`` (or any text-based) file.

    Returns:
        A ``Document`` with a single ``Page`` containing the file content.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        TxtParserError: If the file cannot be decoded.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    # ── Read raw bytes ────────────────────────────────────────────────
    raw = file_path.read_bytes()

    if len(raw) == 0:
        return Document(
            file_name=file_path.name,
            doc_type=DocType.TXT,
            pages=[Page(page_number=1, text="")],
            metadata={"encoding": None, "source_path": str(file_path)},
        )

    # ── Detect encoding ───────────────────────────────────────────────
    try:
        import chardet
    except ImportError as exc:
        raise TxtParserError(
            "chardet not installed. Run: pip install chardet",
            file_path=str(file_path),
        ) from exc

    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0.0)

    # ── Decode ────────────────────────────────────────────────────────
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        # Fallback: try UTF-8, then Latin-1 (never fails)
        for fallback in ("utf-8", "latin-1"):
            try:
                text = raw.decode(fallback)
                encoding = fallback
                break
            except UnicodeDecodeError:
                continue
        else:
            raise TxtParserError(
                "Failed to decode file with any supported encoding.",
                file_path=str(file_path),
            )

    # ── Build Document ────────────────────────────────────────────────
    return Document(
        file_name=file_path.name,
        doc_type=DocType.TXT,
        pages=[Page(page_number=1, text=text)],
        metadata={
            "encoding": encoding,
            "confidence": round(confidence, 3),
            "size_bytes": len(raw),
            "source_path": str(file_path),
        },
    )
