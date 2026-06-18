"""Chroma vector store client (task 3.3).

Connection management, collection initialisation, and core operations:
add / query / delete.  Built on ``chromadb.PersistentClient`` for local
persistence.

Configuration is read from ``app.config.settings.chroma``.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.types import (
    Embedding,
    Metadata,
    QueryResult,
    Where,
)
from loguru import logger

from app.config import settings
from app.core.chunker import Chunk


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------


class ChromaStore:
    """Manages a persistent Chroma vector store collection.

    Encapsulates the ``chromadb.PersistentClient`` and exposes the three
    core operations needed by the ingestion pipeline and the retrieval layer:
    ``add``, ``query``, and ``delete``.

    Usage::

        store = ChromaStore()
        store.add(chunks=[...], embeddings=[[...], ...])
        results = store.query(embedding=[0.1, ...], n_results=5)
        store.delete_by_doc_id("abc123")
    """

    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialise the Chroma client and open the collection.

        Parameters
        ----------
        persist_dir:
            Directory for Chroma's persistent storage.
            Defaults to ``settings.chroma.persist_dir``.
        collection_name:
            Name of the Chroma collection.
            Defaults to ``settings.chroma.collection``.
        """
        chroma_cfg = settings.chroma

        self._persist_dir = persist_dir or chroma_cfg.persist_dir
        self._collection_name = collection_name or chroma_cfg.collection

        logger.info(
            "Opening Chroma store: persist_dir={}, collection={}",
            self._persist_dir,
            self._collection_name,
        )

        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Get or create the collection.  If it already exists we simply
        # open it; the metadata / distance function from the first creation
        # is preserved.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Chroma collection ready: name={}, count={}",
            self._collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert chunks and their embeddings into the collection.

        Parameters
        ----------
        chunks:
            ``Chunk`` objects produced by ``app.core.chunker.Chunker``.
        embeddings:
            Pre-computed embedding vectors in the same order as *chunks*.
            Each vector must have the same dimensionality as the collection.

        Raises
        ------
        ValueError:
            If *chunks* and *embeddings* have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatched lengths: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )

        if not chunks:
            return

        ids: list[str] = [c.id for c in chunks]
        documents: list[str] = [c.content for c in chunks]
        metadatas: list[Metadata] = []
        for c in chunks:
            meta: Metadata = {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "doc_type": c.doc_type.value,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "char_count": c.char_count,
                "is_parent": c.metadata.get("is_parent", False),
                "is_child": c.metadata.get("is_child", False),
            }
            parent_id = c.metadata.get("parent_chunk_id")
            if parent_id:
                meta["parent_chunk_id"] = parent_id
            child_idx = c.metadata.get("child_index")
            if child_idx is not None:
                meta["child_index"] = child_idx
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(
            "Chroma add: {} chunks inserted, total={}",
            len(chunks),
            self._collection.count(),
        )

    def query(
        self,
        embedding: list[float],
        *,
        n_results: int = 5,
        where: Where | None = None,
    ) -> QueryResult:
        """Retrieve the top-*k* chunks most similar to *embedding*.

        Parameters
        ----------
        embedding:
            Query embedding vector (same dimensionality as the collection).
        n_results:
            Number of nearest neighbours to return.
        where:
            Optional Chroma metadata filter (e.g. ``{"doc_id": "abc123"}``).

        Returns
        -------
        chromadb.api.types.QueryResult
            Dictionary with keys ``ids``, ``embeddings``, ``documents``,
            ``metadatas``, and ``distances``.
        """
        return self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def query_batch(
        self,
        embeddings: list[list[float]],
        *,
        n_results: int = 5,
        where: Where | None = None,
    ) -> QueryResult:
        """Retrieve results for multiple query embeddings at once.

        Parameters
        ----------
        embeddings:
            List of query embedding vectors.
        n_results:
            Number of nearest neighbours per query.
        where:
            Optional metadata filter applied to all queries.

        Returns
        -------
        chromadb.api.types.QueryResult
        """
        return self._collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Parameters
        ----------
        doc_id:
            Document UUID whose chunks should be removed.

        Returns
        -------
        int
            Number of chunks deleted.  Returns 0 if no matching chunks exist.
        """
        before = self._collection.count()

        self._collection.delete(
            where={"doc_id": doc_id},
        )

        after = self._collection.count()
        deleted = before - after

        if deleted:
            logger.info(
                "Chroma delete: doc_id={}, removed={}, remaining={}",
                doc_id,
                deleted,
                after,
            )

        return deleted

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """Delete specific chunks by their IDs.

        Parameters
        ----------
        chunk_ids:
            List of chunk identifiers to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        before = self._collection.count()

        self._collection.delete(ids=chunk_ids)

        after = self._collection.count()
        deleted = before - after

        logger.debug(
            "Chroma delete: {} ids requested, {} removed",
            len(chunk_ids),
            deleted,
        )

        return deleted

    def get_by_ids(self, ids: list[str]) -> QueryResult:
        """Retrieve chunks by their IDs.

        Parameters
        ----------
        ids:
            List of chunk identifiers to retrieve.

        Returns
        -------
        chromadb.api.types.QueryResult
            Dictionary with keys ``ids``, ``embeddings``, ``documents``,
            ``metadatas``, and ``distances``.
        """
        if not ids:
            return {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

        return self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self._collection.count()

    def collection_info(self) -> dict[str, Any]:
        """Return collection metadata useful for health checks.

        Returns a dict with ``name``, ``count``, and ``metadata`` keys.
        """
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }

    def reset(self) -> None:
        """Delete the entire collection and re-create it empty.

        **Destructive** — intended for tests and development only.
        """
        logger.warning("Resetting Chroma collection: {}", self._collection_name)
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Release resources.

        For the PersistentClient, this is a no-op (data is already on disk),
        but provided as a symmetry with other store implementations.
        """
        logger.debug("Chroma store closed (no-op for PersistentClient)")
