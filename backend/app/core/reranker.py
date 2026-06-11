"""Cross-encoder reranker using sentence-transformers (module 1.1).

Provides a ``Reranker`` class that wraps a cross-encoder model
(e.g. ``BAAI/bge-reranker-v2-m3``) to re-score candidate chunks
and return the top-k most relevant results.

Configuration is read from ``app.config.settings.reranker``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from app.core.chunker import Chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of (question, chunk) pairs to score in a single batch.
# Cross-encoders are O(N) per pair and memory-heavy, so we cap batch size.
_MAX_PAIRS_PER_BATCH = 256

# Sentinel value returned when the model has not been loaded yet.
_MODEL_NOT_LOADED = "__RERANKER_MODEL_NOT_LOADED__"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker for document chunks.

    Loads a ``sentence-transformers`` cross-encoder model (default:
    ``BAAI/bge-reranker-v2-m3``) and exposes two scoring methods:

    * ``compute_similarity`` — raw similarity scores for (question, text) pairs
    * ``rerank`` — re-score and sort a list of chunks by relevance

    The model is loaded lazily on first use so that the application
    can start without waiting for a large model download.

    Usage::

        reranker = Reranker()
        scores = reranker.compute_similarity([
            ("什么是RAG?", "RAG是检索增强生成..."),
            ("什么是RAG?", "今天天气不错..."),
        ])
        top_chunks = reranker.rerank("什么是RAG?", chunks, top_k=5)

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path.
        Defaults to ``settings.reranker.model``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
        Defaults to ``settings.reranker.device``.
    batch_size:
        Max pairs per scoring batch.  Defaults to ``_MAX_PAIRS_PER_BATCH``.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        reranker_cfg = settings.reranker

        self._model_name = model_name or reranker_cfg.model
        self._device = device or reranker_cfg.device
        self._batch_size = batch_size or _MAX_PAIRS_PER_BATCH

        # Model is loaded lazily — see _ensure_model()
        self._model: object = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_similarity(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-encoder relevance scores for (question, text) pairs.

        Each pair is ``(question, document_text)``.  The model outputs a
        single relevance score per pair; higher values mean more relevant.

        Parameters
        ----------
        pairs:
            List of ``(question, text)`` tuples.  The same question is
            typically paired with many candidate texts.

        Returns
        -------
        list[float]
            Relevance scores in the same order as *pairs*.

        Raises
        ------
        ValueError:
            If *pairs* is empty.
        RuntimeError:
            If the model fails to load.
        """
        if not pairs:
            return []

        self._ensure_model()

        # sentence-transformers cross-encoder .predict() accepts a list of
        # (premise, hypothesis) pairs and returns a list of scores.
        # The API is synchronous (no async), so we call it directly.
        # For large pair lists we split into sub-batches to control memory.
        all_scores: list[float] = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            batch_scores = self._model.predict(  # type: ignore[union-attr]
                batch,
                show_progress_bar=False,
            )
            # .predict() may return a numpy array or plain list
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            all_scores.extend(batch_scores)

        return all_scores

    def rerank(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """Re-score and sort chunks by their relevance to *question*.

        Each chunk is paired with *question* and scored by the cross-encoder.
        Results are sorted by relevance score in descending order and
        truncated to *top_k*.

        The per-chunk relevance score is stored in ``chunk.metadata["rerank_score"]``
        so downstream consumers (agent, UI) can access it.

        Parameters
        ----------
        question:
            The user's question or search query.
        chunks:
            Candidate chunks (e.g. from a vector-store recall step).
        top_k:
            Number of top results to return.  When ``None`` all chunks are
            returned (reordered).  Default: ``settings.agent.top_k_rerank``.

        Returns
        -------
        list[Chunk]
            Chunks sorted by cross-encoder relevance score (best first),
            truncated to *top_k* (when set).
        """
        if not chunks:
            return []

        if top_k is None:
            top_k = settings.agent.top_k_rerank

        # Build pairs: (question, chunk_content)
        pairs = [(question, c.content) for c in chunks]

        scores = self.compute_similarity(pairs)

        # Attach scores to chunk metadata
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = round(float(score), 6)

        # Sort by score descending
        ranked = sorted(
            zip(chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        result = [c for c, _ in ranked[:top_k]]

        logger.debug(
            "Reranker: model={}, candidates={}, top_k={}, best_score={:.4f}, "
            "worst_score={:.4f}",
            self._model_name,
            len(chunks),
            len(result),
            scores[0] if scores else float("nan"),
            scores[-1] if scores else float("nan"),
        )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the cross-encoder model if it has not been loaded yet.

        Uses lazy initialisation so the application can start quickly
        and the model is only downloaded / loaded on first rerank call.
        """
        if self._model is not None:
            return

        logger.info(
            "Loading reranker model: name={}, device={}",
            self._model_name,
            self._device,
        )

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
        except Exception as exc:
            logger.error(
                "Failed to load reranker model: name={}, error={}",
                self._model_name,
                exc,
            )
            raise RuntimeError(
                f"Failed to load reranker model '{self._model_name}': {exc}"
            ) from exc

        logger.info(
            "Reranker model loaded: name={}, device={}",
            self._model_name,
            self._device,
        )

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when the underlying model has been loaded."""
        return self._model is not None

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier in use."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return the torch device string (``cpu``, ``cuda``, etc.)."""
        return self._device
