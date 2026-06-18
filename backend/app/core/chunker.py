"""Text chunker using LangChain RecursiveCharacterTextSplitter (task 3.1).

Splits parsed document text into overlapping chunks suitable for embedding
and retrieval.  Configuration (chunk_size, overlap) comes from
``app.config.settings.agent``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models.document import Document as ParsedDocument
from app.models.document import DocType


# ---------------------------------------------------------------------------
# Chunk domain model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk produced by the chunker.

    Each chunk represents a contiguous segment of text from a source document,
    annotated with the metadata needed for retrieval and citation display.
    """

    id: str
    """Unique chunk identifier: ``{doc_id}_chunk_{index}``."""

    content: str
    """The chunked text content."""

    doc_id: str
    """Owning document ID (UUID hex)."""

    doc_name: str
    """Original filename of the owning document."""

    doc_type: DocType
    """Document type (pdf / md / txt)."""

    page: int
    """Page number within the source document (1-based)."""

    chunk_index: int
    """Zero-based chunk index within the document."""

    char_count: int
    """Number of characters in *content*."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata (content hash, source file, etc.)."""


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

# Separators used by the recursive splitter.
# Attempt to split on natural boundaries first, then degrade to individual
# characters.  The Chinese sentence/paragraph markers help keep semantic
# units together.
_DEFAULT_SEPARATORS: list[str] = [
    "\n\n",   # paragraph break
    "\n",     # line break
    "。",     # Chinese period
    "！",     # Chinese exclamation
    "？",     # Chinese question mark
    ". ",     # English period + space
    "! ",     # English exclamation + space
    "? ",     # English question mark + space
    "；",     # Chinese semicolon
    "; ",     # English semicolon + space
    " ",      # word boundary
    "",       # character-level fallback
]


class Chunker:
    """Splits a parsed ``Document`` into overlapping ``Chunk`` objects.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` with a cascade of
    separators tuned for both English and Chinese text.

    Parameters
    ----------
    chunk_size:
        Target chunk size in characters.  Defaults to ``settings.agent.chunk_size``.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.
        Defaults to ``settings.agent.chunk_overlap``.
    separators:
        Ordered list of separator strings (most natural first).  If not
        provided the module-level ``_DEFAULT_SEPARATORS`` are used.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.agent.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.agent.chunk_overlap
        )
        self.separators = separators if separators is not None else list(_DEFAULT_SEPARATORS)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks.

        Each page is split independently so that page-level metadata is
        preserved on every chunk.  The global chunk index is monotonically
        increasing across pages.

        Parameters
        ----------
        document:
            A parsed ``Document`` produced by one of the parsers in
            ``app.core.parsers``.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.  Returns an empty list for a document
            with no text.
        """
        chunks: list[Chunk] = []
        # Derive a stable doc_id from the file path — the caller will
        # replace this with the real UUID when persisting.
        doc_id = document.metadata.get("doc_id", "")
        doc_name = document.file_name
        doc_type = document.doc_type

        global_index = 0

        for page in document.pages:
            page_text = page.text
            if not page_text:
                continue

            texts = self._splitter.split_text(page_text)

            for local_idx, text in enumerate(texts):
                chunk_id = (
                    f"{doc_id}_chunk_{global_index}"
                    if doc_id
                    else f"_{uuid.uuid4().hex[:8]}_chunk_{global_index}"
                )
                chunk = Chunk(
                    id=chunk_id,
                    content=text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    page=page.page_number,
                    chunk_index=global_index,
                    char_count=len(text),
                    metadata={
                        "doc_name": doc_name,
                        "doc_type": doc_type.value,
                        "page": page.page_number,
                        "chunk_index": global_index,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                )
                chunks.append(chunk)
                global_index += 1

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split raw text into string segments.

        Convenience wrapper around the underlying splitter.  Useful for
        testing or when full ``Chunk`` metadata is not needed.

        Parameters
        ----------
        text:
            Free-form text string.

        Returns
        -------
        list[str]
            Ordered list of text segments.
        """
        return self._splitter.split_text(text)

    # ------------------------------------------------------------------
    # Future: semantic splitting placeholder
    # ------------------------------------------------------------------

    def split_semantic(
        self,
        document: ParsedDocument,
        embedder,  # type: ignore — not yet wired
        similarity_threshold: float = 0.5,
    ) -> list[Chunk]:
        """Split using embedding-based semantic boundary detection.

        **Not yet implemented** — placeholder for a future upgrade.

        Parameters
        ----------
        document:
            Parsed document.
        embedder:
            Embedding client (``app.core.embedder.Embedder``).
        similarity_threshold:
            Adjacent-sentence cosine similarity below which a split is
            inserted.

        Returns
        -------
        list[Chunk]
        """
        raise NotImplementedError("semantic splitting is not yet implemented")
