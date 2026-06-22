"""Semantic catalog of uploaded document names (replaces brittle regex
filename extraction).

For every uploaded document, this module embeds the document's
``file_name`` once and stores the resulting vector in a side Chroma
collection (``settings.chroma.catalog_collection``).  At query time
the user's question is embedded and matched against the catalog by
cosine similarity; the matching ``file_name``\s become the
``doc_filter`` for retrieval.

This handles cases the regex-based extractor cannot:

* Filenames referenced without an extension — "agent项目要求"
* Colloquial references — "那个 checklist", "我上传的 agent 列表"
* Typos and partial matches
* Bilingual mixing
"""

from __future__ import annotations

from loguru import logger

from app.config import settings
from app.core.embedder import Embedder, EmbedderError
from app.db.chroma import ChromaStore


def _doc_catalog_id(doc_id: str) -> str:
    """Build a deterministic Chroma id for a catalog entry."""
    return f"catalog_{doc_id}"


class DocCatalog:
    """Embedding-based catalog of uploaded document file names.

    Each document contributes one vector (its ``file_name``) to the
    ``doc_catalog`` Chroma collection.  Queries return the names whose
    vectors are closest to the user's question above a configured
    cosine-similarity threshold.

    Parameters
    ----------
    chroma:
        Reuse an existing ``ChromaStore`` if you have one.  Defaults to
        constructing a new one (cheap — the underlying client is shared
        via the persist directory).
    embedder:
        Reuse an existing ``Embedder`` if you have one.
    threshold:
        Cosine-similarity floor (0..1).  Hits below this are dropped.
        Defaults to ``settings.agent.doc_catalog_threshold``.
    top_k:
        Number of catalog candidates to consider per query.  Defaults
        to ``settings.agent.doc_catalog_top_k``.
    collection_name:
        Override the catalog collection name (default
        ``settings.chroma.catalog_collection``).
    """

    def __init__(
        self,
        *,
        chroma: ChromaStore | None = None,
        embedder: Embedder | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
        min_gap: float | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._chroma = chroma or ChromaStore()
        self._embedder = embedder or Embedder()
        self._threshold = (
            threshold if threshold is not None
            else settings.agent.doc_catalog_threshold
        )
        self._top_k = (
            top_k if top_k is not None
            else settings.agent.doc_catalog_top_k
        )
        self._min_gap = (
            min_gap if min_gap is not None
            else settings.agent.doc_catalog_min_gap
        )
        self._collection_name = (
            collection_name or settings.chroma.catalog_collection
        )
        self._collection = self._chroma.get_collection(self._collection_name)

    # ------------------------------------------------------------------
    # Mutators (called from upload / delete)
    # ------------------------------------------------------------------

    async def add(self, doc_id: str, file_name: str) -> None:
        """Register *doc_id* / *file_name* in the catalog.

        Idempotent: re-adding the same ``doc_id`` overwrites the previous
        entry (handles re-uploads / renames).
        """
        if not file_name:
            logger.warning("DocCatalog.add: empty file_name, skipping: doc_id={}", doc_id)
            return
        try:
            emb = await self._embedder.embed(file_name)
        except EmbedderError as exc:
            logger.error("DocCatalog.add: embedding failed for {}: {}", file_name, exc)
            return

        self._collection.upsert(
            ids=[_doc_catalog_id(doc_id)],
            embeddings=[emb],
            documents=[file_name],
            metadatas=[{"doc_id": doc_id, "file_name": file_name}],
        )
        logger.debug(
            "DocCatalog.add: doc_id={}, file_name={!r}",
            doc_id, file_name,
        )

    def remove(self, doc_id: str) -> None:
        """Remove *doc_id* from the catalog.  No-op if absent."""
        try:
            self._collection.delete(ids=[_doc_catalog_id(doc_id)])
            logger.debug("DocCatalog.remove: doc_id={}", doc_id)
        except Exception as exc:  # chromadb raises on missing ids
            logger.debug(
                "DocCatalog.remove: doc_id={} not present ({})",
                doc_id, exc,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def find_relevant(self, question: str) -> list[str]:
        """Return file_names whose vector is semantically close to *question*.

        Returns an empty list (NOT ``None``) when no document is
        confidently referenced — callers should treat an empty result
        as "no doc_filter, do a full-corpus search".

        Selection rule:

        * The top-1 hit must clear ``self._threshold`` (cosine
          similarity >= threshold, default 0.75).
        * The top-1 hit must beat the top-2 hit by at least
          ``min_gap`` (default 0.03).  This prevents generic
          questions (e.g. "什么是知识图谱") from being mapped to
          whichever document happens to have the highest baseline
          similarity.
        """
        if not question or not question.strip():
            return []
        if self._collection.count() == 0:
            return []

        try:
            emb = await self._embedder.embed(question)
        except EmbedderError as exc:
            logger.warning("DocCatalog.find_relevant: embedding failed: {}", exc)
            return []

        n_results = min(self._top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not distances:
            return []

        sims: list[float] = [1.0 - d for d in distances]
        names: list[str] = [
            (m or {}).get("file_name", "") for m in metadatas
        ]

        top1 = sims[0]
        top2 = sims[1] if len(sims) > 1 else 0.0

        if top1 < self._threshold:
            return []
        # If we have a meaningful gap, we're confident about top-1.
        # Without a gap (e.g. a generic question where every doc scores
        # similar), don't filter.
        if len(sims) > 1 and (top1 - top2) < self._min_gap:
            return []

        return [names[0]] if names[0] else []

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def list_all(self) -> list[dict]:
        """Return every catalog entry as ``{doc_id, file_name}`` dicts."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        ids = result.get("ids", []) or []
        metas = result.get("metadatas", []) or []
        out: list[dict] = []
        for cid, meta in zip(ids, metas):
            meta = meta or {}
            doc_id = meta.get("doc_id", cid.removeprefix("catalog_"))
            out.append({
                "doc_id": doc_id,
                "file_name": meta.get("file_name", ""),
            })
        return out
