"""Document ingestion pipeline (task 3.4).

Orchestrates the full processing chain:
parse → chunk → embed → write-to-Chroma.

Composes the parser factory, Chunker, Embedder, and ChromaStore
into a single ``run()`` call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from app.core.chunker import Chunk, Chunker
from app.core.bm25_retriever import get_bm25
from app.core.embedder import Embedder
from app.core.parsers import parse_document
from app.db.chroma import ChromaStore
from app.models.document import Document, DocType


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


class PipelineResult:
    """Outcome of a successful ingestion run."""

    def __init__(
        self,
        doc: Document,
        chunks: list[Chunk],
    ) -> None:
        self.doc = doc
        self.chunks = chunks

    @property
    def doc_id(self) -> str:
        return self.doc.metadata.get("doc_id", "")

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self.chunks)

    def __repr__(self) -> str:
        return (
            f"PipelineResult(doc={self.doc.file_name!r}, "
            f"chunks={self.chunk_count}, chars={self.total_chars})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """Orchestrates document ingestion: parse → chunk → embed → store.

    Parameters
    ----------
    chunker:
        Configured ``Chunker`` instance.  Created with defaults if omitted.
    embedder:
        Configured ``Embedder`` instance.  Created with defaults if omitted.
    chroma:
        Configured ``ChromaStore`` instance.  Created with defaults if omitted.
    """

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        chroma: ChromaStore | None = None,
    ) -> None:
        self._chunker = chunker or Chunker()
        self._embedder = embedder or Embedder()
        self._chroma = chroma or ChromaStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        file_path: str | Path,
        *,
        doc_id: str = "",
    ) -> PipelineResult:
        """Run the full ingestion pipeline for a single document.

        Parameters
        ----------
        file_path:
            Path to a supported document (.pdf, .md, .txt).
        doc_id:
            Stable document identifier (e.g. UUID from the API layer).
            Injected into every chunk's metadata for later retrieval and
            deletion scoping.

        Returns
        -------
        PipelineResult
            The parsed document and the list of chunks that were embedded
            and stored in Chroma.

        Raises
        ------
        FileNotFoundError:
            If *file_path* does not exist.
        ValueError:
            If the document type is unsupported.
        EmbedderError:
            If the embedding API fails after retries.
        """
        file_path = Path(file_path)

        # ── 1. Parse ────────────────────────────────────────────────
        logger.info("Pipeline [{}]: parsing {}", doc_id, file_path.name)
        doc = parse_document(file_path)

        # Attach the doc_id so the chunker can propagate it
        doc.metadata["doc_id"] = doc_id
        doc.metadata["source_path"] = str(file_path)

        logger.info(
            "Pipeline [{}]: parsed {} pages, {} chars, type={}",
            doc_id,
            doc.page_count,
            doc.char_count,
            doc.doc_type.value,
        )

        # ── 2. Chunk ────────────────────────────────────────────────
        chunk_size = getattr(self._chunker, "chunk_size", "?")
        chunk_overlap = getattr(self._chunker, "chunk_overlap", "?")
        logger.info("Pipeline [{}]: chunking (size={}, overlap={})",
                     doc_id, chunk_size, chunk_overlap)
        chunks = self._chunker.split(doc)

        if not chunks:
            logger.warning("Pipeline [{}]: no chunks produced", doc_id)
            return PipelineResult(doc=doc, chunks=[])

        logger.info("Pipeline [{}]: {} chunks produced", doc_id, len(chunks))

        # ── 3. Embed ────────────────────────────────────────────────
        logger.info("Pipeline [{}]: embedding {} chunks", doc_id, len(chunks))
        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed_batch(texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} vectors "
                f"for {len(chunks)} chunks"
            )

        logger.info(
            "Pipeline [{}]: embedding complete, {} vectors "
            "(dim={})",
            doc_id,
            len(embeddings),
            getattr(self._embedder, "_dimensions", "?"),
        )

        # ── 4. Store ────────────────────────────────────────────────
        logger.info("Pipeline [{}]: writing {} chunks to Chroma", doc_id, len(chunks))
        self._chroma.add(chunks=chunks, embeddings=embeddings)

        # ── 5. Rebuild BM25 index ────────────────────────────────
        await _rebuild_bm25_from_chroma(self._chroma)

        logger.info(
            "Pipeline [{}]: ingestion complete — document={!r}, "
            "chunks={}, chars={}",
            doc_id,
            doc.file_name,
            len(chunks),
            sum(c.char_count for c in chunks),
        )

        return PipelineResult(doc=doc, chunks=chunks)

    # ------------------------------------------------------------------
    # Convenience: run for a file already stored by FileStorage
    # ------------------------------------------------------------------

    async def run_from_storage(
        self,
        doc_id: str,
        *,
        upload_dir: str | None = None,
    ) -> PipelineResult | None:
        """Locate a document in the upload directory and ingest it.

        Looks for ``{upload_dir}/{doc_id}/{original_filename}``.

        Parameters
        ----------
        doc_id:
            Document UUID as used by the upload API.
        upload_dir:
            Root upload directory.  Defaults to ``settings.upload.upload_dir``.

        Returns
        -------
        PipelineResult | None
            ``None`` if the document directory or file could not be found.
        """
        from app.config import settings

        base = Path(upload_dir) if upload_dir else Path(settings.upload.upload_dir)
        doc_dir = base / doc_id

        if not doc_dir.is_dir():
            logger.warning(
                "Pipeline [{}]: upload directory not found: {}", doc_id, doc_dir
            )
            return None

        # Pick the first file in the doc directory (there should be exactly one)
        files = list(doc_dir.iterdir())
        if not files:
            logger.warning("Pipeline [{}]: no files in {}", doc_id, doc_dir)
            return None

        target = files[0]
        logger.info(
            "Pipeline [{}]: found file {} in storage", doc_id, target.name
        )

        return await self.run(str(target), doc_id=doc_id)
