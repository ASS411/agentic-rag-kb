"""Text chunker with semantic splitting and parent-child structure (task 3.1).

Splits parsed document text into overlapping chunks suitable for embedding
and retrieval. Supports two modes:
1. Traditional fixed-size character splitting (RecursiveCharacterTextSplitter)
2. Semantic splitting using sentence-transformers similarity threshold

When semantic splitting is enabled, produces:
- Parent chunks: Complete semantic units (~2000-3000 chars) for generation
- Child chunks: Smaller slices (~800 chars) for precise retrieval
- Each child chunk references its parent via metadata

Configuration (chunk_size, overlap, semantic_threshold) comes from
``app.config.settings.agent``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import settings
from app.models.document import Document as ParsedDocument
from app.models.document import DocType

if TYPE_CHECKING:
    from app.core.embedder import Embedder

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

    @property
    def parent_id(self) -> str | None:
        """Return the parent chunk ID if this is a child chunk."""
        return self.metadata.get("parent_chunk_id")

    @property
    def is_child(self) -> bool:
        """Return True if this chunk is a child of a parent chunk."""
        return self.parent_id is not None

    @property
    def is_parent(self) -> bool:
        """Return True if this chunk is a parent chunk."""
        return self.metadata.get("is_parent", False)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

# Separators used by the recursive splitter for child chunks.
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

# Regex pattern for sentence splitting in Chinese text
_SENTENCE_PATTERN = r"(?<=[。！？.!?])"


# ---------------------------------------------------------------------------
# Semantic Splitter
# ---------------------------------------------------------------------------


class SemanticSplitter:
    """Splits text into semantic units using embedding similarity.

    Sentences are embedded and cosine similarity between consecutive sentences
    is computed. When similarity falls below a threshold, a split point is
    inserted. This ensures each chunk contains a complete semantic unit.

    Parameters
    ----------
    embedder:
        Embedding client for computing sentence embeddings.
    threshold:
        Cosine similarity threshold below which a split is inserted.
        Default: 0.5
    max_chars:
        Maximum characters per parent chunk. Default: 3000
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.5,
        max_chars: int = 3000,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max_chars = max_chars

    async def split(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks.

        Parameters
        ----------
        text:
            Input text to split.

        Returns
        -------
        list[str]
            Ordered list of semantic chunks.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]

        return await self._semantic_split(sentences)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences using Chinese-aware pattern."""
        import re
        raw_sentences = re.split(_SENTENCE_PATTERN, text)
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences

    async def _semantic_split(self, sentences: list[str]) -> list[str]:
        """Split sentences into semantic chunks based on embedding similarity."""
        if len(sentences) <= 1:
            return ["".join(sentences)]

        texts_to_embed = sentences
        embeddings = await self._embedder.embed_batch(texts_to_embed)

        chunks: list[str] = []
        current_chunk: list[str] = [sentences[0]]
        current_chars = len(sentences[0])

        for i in range(1, len(sentences)):
            prev_embedding = embeddings[i - 1]
            curr_embedding = embeddings[i]
            similarity = self._cosine_similarity(prev_embedding, curr_embedding)

            sentence = sentences[i]
            sentence_chars = len(sentence)

            should_split = (
                similarity < self._threshold
                or (current_chars + sentence_chars > self._max_chars and current_chunk)
            )

            if should_split and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_chars = sentence_chars
            else:
                current_chunk.append(sentence)
                current_chars += sentence_chars

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = (sum(x * x for x in a)) ** 0.5
        norm_b = (sum(x * x for x in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Chunker Main Class
# ---------------------------------------------------------------------------


class Chunker:
    """Splits a parsed ``Document`` into overlapping ``Chunk`` objects.

    Supports two modes:
    1. Traditional fixed-size character splitting (default)
    2. Semantic splitting + parent-child structure (when enabled)

    When semantic splitting is enabled:
    - Parent chunks: Complete semantic units for generation (~2000-3000 chars)
    - Child chunks: Smaller slices for precise retrieval (~800 chars)

    Parameters
    ----------
    chunk_size:
        Target chunk size in characters for child chunks.
        Defaults to ``settings.agent.child_chunk_size``.
    chunk_overlap:
        Number of overlapping characters between consecutive child chunks.
        Defaults to ``settings.agent.child_chunk_overlap``.
    separators:
        Ordered list of separator strings (most natural first).  If not
        provided the module-level ``_DEFAULT_SEPARATORS`` are used.
    semantic_threshold:
        Cosine similarity threshold for semantic splitting.
        Defaults to ``settings.agent.semantic_threshold``.
    parent_chunk_max_chars:
        Maximum characters per parent chunk.
        Defaults to ``settings.agent.parent_chunk_max_chars``.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        semantic_threshold: float | None = None,
        parent_chunk_max_chars: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.agent.child_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.agent.child_chunk_overlap
        )
        self.separators = separators if separators is not None else list(_DEFAULT_SEPARATORS)
        self.semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else settings.agent.semantic_threshold
        )
        self.parent_chunk_max_chars = (
            parent_chunk_max_chars if parent_chunk_max_chars is not None else settings.agent.parent_chunk_max_chars
        )

        self._child_splitter = RecursiveCharacterTextSplitter(
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

        When semantic splitting is enabled in settings and an embedder is
        available, produces both parent and child chunks. Otherwise, uses
        traditional fixed-size splitting.

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
        if settings.agent.semantic_chunking_enabled:
            logger.warning(
                "Semantic chunking is enabled but split() does not have access to an embedder. "
                "Falling back to traditional fixed-size splitting. "
                "Use split_with_embedder() for semantic splitting."
            )

        return self._split_traditional(document)

    async def split_with_embedder(self, document: ParsedDocument, embedder: Embedder) -> list[Chunk]:
        """Split a parsed document using semantic splitting with the provided embedder.

        This method explicitly enables semantic splitting regardless of
        the global settings, using the provided embedder for computing
        sentence embeddings.

        Parameters
        ----------
        document:
            A parsed ``Document``.
        embedder:
            Embedding client for semantic similarity computation.

        Returns
        -------
        list[Chunk]
            Ordered list of parent and child chunks.
        """
        return await self._split_semantic_with_embedder(document, embedder)

    # ------------------------------------------------------------------
    # Traditional Fixed-Size Splitting
    # ------------------------------------------------------------------

    def _split_traditional(self, document: ParsedDocument) -> list[Chunk]:
        """Split document using traditional fixed-size character splitting."""
        chunks: list[Chunk] = []
        doc_id = document.metadata.get("doc_id", "")
        doc_name = document.file_name
        doc_type = document.doc_type

        global_index = 0

        for page in document.pages:
            page_text = page.text
            if not page_text:
                continue

            texts = self._child_splitter.split_text(page_text)

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
                        "is_parent": False,
                        "is_child": False,
                    },
                )
                chunks.append(chunk)
                global_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Semantic Splitting (Lazy)
    # ------------------------------------------------------------------

    def _split_semantic(self, document: ParsedDocument) -> list[Chunk]:
        """Lazy semantic splitting — falls back to traditional.

        The actual semantic splitting requires an embedder. This method
        is a placeholder that triggers semantic splitting when called
        from the pipeline with an embedder.
        """
        raise NotImplementedError(
            "Semantic splitting requires an embedder. "
            "Use split_with_embedder() instead."
        )

    async def _split_semantic_with_embedder(
        self,
        document: ParsedDocument,
        embedder: Embedder,
    ) -> list[Chunk]:
        """Split document using semantic similarity-based splitting.

        Produces both parent chunks (complete semantic units) and child chunks
        (smaller slices for retrieval).

        Parameters
        ----------
        document:
            A parsed ``Document``.
        embedder:
            Embedding client for computing sentence similarity.

        Returns
        -------
        list[Chunk]
            Ordered list of parent chunks followed by child chunks.
        """
        chunks: list[Chunk] = []
        doc_id = document.metadata.get("doc_id", "")
        doc_name = document.file_name
        doc_type = document.doc_type

        global_index = 0
        parent_index = 0

        semantic_splitter = SemanticSplitter(
            embedder=embedder,
            threshold=self.semantic_threshold,
            max_chars=self.parent_chunk_max_chars,
        )

        for page in document.pages:
            page_text = page.text
            if not page_text:
                continue

            parent_texts = await semantic_splitter.split(page_text)

            for parent_text in parent_texts:
                parent_id = (
                    f"{doc_id}_parent_{parent_index}"
                    if doc_id
                    else f"_{uuid.uuid4().hex[:8]}_parent_{parent_index}"
                )

                parent_chunk = Chunk(
                    id=parent_id,
                    content=parent_text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    page=page.page_number,
                    chunk_index=global_index,
                    char_count=len(parent_text),
                    metadata={
                        "doc_name": doc_name,
                        "doc_type": doc_type.value,
                        "page": page.page_number,
                        "chunk_index": global_index,
                        "is_parent": True,
                        "is_child": False,
                        "parent_chunk_id": None,
                        "semantic_threshold": self.semantic_threshold,
                    },
                )
                chunks.append(parent_chunk)
                global_index += 1

                child_texts = self._child_splitter.split_text(parent_text)
                for child_idx, child_text in enumerate(child_texts):
                    child_id = f"{parent_id}_child_{child_idx}"
                    child_chunk = Chunk(
                        id=child_id,
                        content=child_text,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        doc_type=doc_type,
                        page=page.page_number,
                        chunk_index=global_index,
                        char_count=len(child_text),
                        metadata={
                            "doc_name": doc_name,
                            "doc_type": doc_type.value,
                            "page": page.page_number,
                            "chunk_index": global_index,
                            "is_parent": False,
                            "is_child": True,
                            "parent_chunk_id": parent_id,
                            "child_index": child_idx,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                    chunks.append(child_chunk)
                    global_index += 1

                parent_index += 1

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
        return self._child_splitter.split_text(text)
