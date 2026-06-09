"""Document parsers package.

The parsers module provides format-specific document extraction.
Each parser accepts a file path and returns a ``Document`` domain object.

Sub-modules:
- ``pdf_parser`` — PyMuPDF-based PDF extraction
- ``md_parser`` — Markdown (future)
- ``txt_parser`` — Plain text with encoding detection (future)
"""

from __future__ import annotations

from app.models.document import Document

# Parser function type: (file_path: str) -> Document
# Importable directly for use in the parser factory (task 2.5).
