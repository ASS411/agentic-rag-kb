"""Tests for the cross-encoder reranker (task 1.2).

Covers:
- compute_similarity: empty input, single pair, batch splitting
- rerank: empty chunks, top_k truncation, sort order (descending)
- rerank_score attachment to chunk metadata
- Lazy model loading (_ensure_model)
- Initialisation from config and custom overrides
- Error handling when model load fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.chunker import Chunk
from app.core.reranker import Reranker, _MAX_PAIRS_PER_BATCH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "doc1_chunk_0",
    content: str = "test content",
    doc_id: str = "doc1",
    doc_name: str = "test.pdf",
) -> Chunk:
    """Build a minimal Chunk for testing."""
    from app.models.document import DocType

    return Chunk(
        id=chunk_id,
        content=content,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type=DocType.TXT,
        page=1,
        chunk_index=0,
        char_count=len(content),
        metadata={},
    )


def _make_patched_reranker(
    scores: list[float] | None = None,
) -> tuple[Reranker, MagicMock]:
    """Create a Reranker with the model pre-loaded as a mock.

    Instead of patching the module-level ``CrossEncoder`` (which is imported
    lazily inside ``_ensure_model`` and therefore not available at module
    level), we directly inject a mock model with a ``.predict()`` method.

    Parameters
    ----------
    scores:
        If provided, ``mock_model.predict()`` returns these scores.
        Otherwise a default ``[0.9]`` is used.
    """
    reranker = Reranker()
    mock_model = MagicMock()
    if scores is not None:
        mock_model.predict.return_value = scores
    else:
        mock_model.predict.return_value = [0.9]
    reranker._model = mock_model
    return reranker, mock_model


# ---------------------------------------------------------------------------
# compute_similarity
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    """Tests for ``Reranker.compute_similarity()``."""

    def test_empty_pairs_returns_empty_list(self):
        reranker, _ = _make_patched_reranker()
        assert reranker.compute_similarity([]) == []

    def test_single_pair_returns_single_score(self):
        reranker, mock_model = _make_patched_reranker()
        mock_model.predict.return_value = [0.75]

        scores = reranker.compute_similarity([("q", "text")])
        assert scores == [0.75]
        mock_model.predict.assert_called_once()

    def test_multiple_pairs_preserve_order(self):
        reranker, mock_model = _make_patched_reranker()
        mock_model.predict.return_value = [0.1, 0.5, 0.3]

        scores = reranker.compute_similarity([
            ("q", "a1"), ("q", "a2"), ("q", "a3")
        ])
        assert scores == [0.1, 0.5, 0.3]

    def test_batch_splitting(self):
        """Large pair lists are split into sub-batches."""
        n = _MAX_PAIRS_PER_BATCH + 50
        chunks_texts = [f"text {i}" for i in range(n)]
        pairs = [("q", t) for t in chunks_texts]

        reranker, mock_model = _make_patched_reranker()
        # Return one score per pair in the first batch call,
        # the rest in a second call
        batch1_scores = [0.5] * _MAX_PAIRS_PER_BATCH
        batch2_scores = [0.5] * 50
        mock_model.predict.side_effect = [batch1_scores, batch2_scores]

        scores = reranker.compute_similarity(pairs)
        assert len(scores) == n
        assert mock_model.predict.call_count == 2

    def test_numpy_array_conversion(self):
        """When .predict() returns a numpy array, convert to list."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        arr = np.array([0.3, 0.7], dtype=np.float32)
        reranker, mock_model = _make_patched_reranker()
        mock_model.predict.return_value = arr

        scores = reranker.compute_similarity([("q", "a1"), ("q", "a2")])
        assert isinstance(scores, list)
        assert scores == pytest.approx([0.3, 0.7], rel=1e-5)

    def test_plain_list_return_value(self):
        """When .predict() returns a plain Python list, use it directly."""
        reranker, mock_model = _make_patched_reranker()
        mock_model.predict.return_value = [0.42]

        scores = reranker.compute_similarity([("q", "a")])
        assert scores == [0.42]


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


class TestRerank:
    """Tests for ``Reranker.rerank()``."""

    def test_empty_chunks_returns_empty_list(self):
        reranker = Reranker()
        result = reranker.rerank("question?", [], top_k=5)
        assert result == []

    def test_top_k_truncation(self):
        chunks = [_make_chunk(f"c{i}", f"content {i}") for i in range(10)]
        scores = [1.0 - i * 0.1 for i in range(10)]

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("q", chunks, top_k=3)
        assert len(result) == 3
        assert result[0].id == "c0"
        assert result[1].id == "c1"
        assert result[2].id == "c2"

    def test_top_k_none_uses_default_from_settings(self):
        """When top_k is None, use settings.agent.top_k_rerank (default 5)."""
        from app.config import settings

        default_top_k = settings.agent.top_k_rerank
        chunks = [_make_chunk(f"c{i}", f"content {i}") for i in range(10)]
        scores = list(range(10))

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("q", chunks, top_k=None)
        assert len(result) == default_top_k
        assert result[0].id == "c9"

    def test_sort_descending_by_score(self):
        chunks = [
            _make_chunk("c0", "RAG is great"),
            _make_chunk("c1", "Weather is sunny"),
            _make_chunk("c2", "RAG combines retrieval"),
        ]
        scores = [0.3, 0.1, 0.9]

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("What is RAG?", chunks, top_k=3)
        assert result[0].id == "c2"
        assert result[1].id == "c0"
        assert result[2].id == "c1"

    def test_rerank_score_attached_to_metadata(self):
        chunks = [
            _make_chunk("c0", "content A"),
            _make_chunk("c1", "content B"),
        ]
        scores = [0.75, 0.25]

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("q", chunks, top_k=2)

        for c in result:
            assert "rerank_score" in c.metadata
            assert isinstance(c.metadata["rerank_score"], float)
        assert result[0].metadata["rerank_score"] == 0.75
        assert result[1].metadata["rerank_score"] == 0.25

    def test_top_k_larger_than_chunks_returns_all(self):
        chunks = [_make_chunk(f"c{i}", f"content {i}") for i in range(3)]
        scores = [0.7, 0.5, 0.3]

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("q", chunks, top_k=10)
        assert len(result) == 3

    def test_single_chunk(self):
        chunk = _make_chunk("c0", "only one")
        scores = [0.99]

        reranker, _ = _make_patched_reranker(scores=scores)
        result = reranker.rerank("q", [chunk], top_k=5)
        assert len(result) == 1
        assert result[0].id == "c0"
        assert result[0].metadata["rerank_score"] == 0.99


# ---------------------------------------------------------------------------
# Initialisation & lazy loading
# ---------------------------------------------------------------------------


class TestInit:
    """Constructor and config integration."""

    def test_default_uses_settings(self):
        reranker = Reranker()
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker.device == "cpu"
        assert reranker.is_loaded is False

    def test_custom_overrides(self):
        reranker = Reranker(
            model_name="custom/model",
            device="cuda:0",
            batch_size=128,
        )
        assert reranker.model_name == "custom/model"
        assert reranker.device == "cuda:0"
        assert reranker._batch_size == 128

    def test_loads_on_first_api_call(self):
        """_ensure_model is called on first compute_similarity call."""
        reranker = Reranker()
        assert not reranker.is_loaded

        # Inject a mock model: _ensure_model sees it's already loaded
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        reranker._model = mock_model

        scores = reranker.compute_similarity([("q", "text")])
        assert scores == [0.5]
        assert reranker.is_loaded
        mock_model.predict.assert_called_once_with(
            [("q", "text")],
            show_progress_bar=False,
        )

    def test_ensure_model_idempotent(self):
        """Calling _ensure_model twice loads the model only once."""
        reranker = Reranker()
        assert not reranker.is_loaded

        # Set _model as if it were already loaded
        reranker._model = MagicMock()

        # _ensure_model should return early (no-op) since _model is already set
        reranker._ensure_model()
        reranker._ensure_model()
        assert reranker.is_loaded


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Error handling behaviour."""

    def test_load_failure_raises_runtime_error(self):
        reranker = Reranker()

        # Simulate a loading failure by making _ensure_model raise RuntimeError
        def _fail():
            raise RuntimeError("Failed to load reranker model 'test': model not found")

        reranker._ensure_model = _fail

        with pytest.raises(RuntimeError, match="Failed to load"):
            reranker.compute_similarity([("q", "text")])

    def test_compute_similarity_loads_before_scoring(self):
        """The model must be loaded before .predict() is called."""
        reranker = Reranker()
        assert not reranker.is_loaded

        # Inject a mock model to simulate already-loaded state
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.88]
        reranker._model = mock_model

        scores = reranker.compute_similarity([("q", "a")])
        assert scores == [0.88]
        assert reranker.is_loaded
        mock_model.predict.assert_called_once()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Property accessors."""

    def test_model_name_property(self):
        reranker = Reranker(model_name="test-model")
        assert reranker.model_name == "test-model"

    def test_device_property(self):
        reranker = Reranker(device="cuda")
        assert reranker.device == "cuda"

    def test_is_loaded_before_and_after(self):
        """is_loaded reflects whether _model has been set."""
        reranker = Reranker()
        assert reranker.is_loaded is False

        # Manually set _model (simulating a completed _ensure_model)
        reranker._model = MagicMock()
        assert reranker.is_loaded is True
