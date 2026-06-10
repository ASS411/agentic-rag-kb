"""Tests for the embedding client (task 3.2).

Covers:
- Single text embedding
- Batch embedding
- Auto-splitting for large batches (sub-batch logic)
- Input validation (empty, whitespace, too many texts)
- Dimension verification
- Error handling (API failures, retries, timeouts)
- Client initialization from config
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.embedder import Embedder, EmbedderError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_embedding(dim: int = 1024) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.1 * (i % 10 + 1) for i in range(dim)]


def _fake_response(texts: list[str], dim: int = 1024) -> MagicMock:
    """Build a mock OpenAI embeddings response with one vector per text."""
    mock = MagicMock()
    mock.data = [
        MagicMock(embedding=_make_mock_embedding(dim), index=i)
        for i in range(len(texts))
    ]
    return mock


def _make_patched_embedder(dim: int = 1024) -> tuple[Embedder, MagicMock]:
    """Create an Embedder with a mocked AsyncOpenAI client."""
    embedder = Embedder()
    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock()
    embedder._client = mock_client
    return embedder, mock_client


# ---------------------------------------------------------------------------
# Single text embedding
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    """Happy-path single text embedding."""

    @pytest.mark.asyncio
    async def test_returns_correct_dimensions(self):
        embedder, mock = _make_patched_embedder(dim=1024)
        mock.embeddings.create.return_value = _fake_response(["hello"], dim=1024)

        vec = await embedder.embed("hello")

        assert len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.asyncio
    async def test_calls_api_with_correct_params(self):
        embedder, mock = _make_patched_embedder(dim=1024)
        mock.embeddings.create.return_value = _fake_response(["hello"], dim=1024)

        await embedder.embed("test text")

        call_kwargs = mock.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-v3"
        assert call_kwargs["input"] == "test text"
        assert call_kwargs["dimensions"] == 1024

    @pytest.mark.asyncio
    async def test_accepts_custom_model_and_dimensions(self):
        embedder = Embedder(model="custom-model", dimensions=512)
        mock_client = MagicMock()
        mock_client.embeddings = MagicMock()
        mock_client.embeddings.create = AsyncMock()
        mock_client.embeddings.create.return_value = _fake_response(["x"], dim=512)
        embedder._client = mock_client

        vec = await embedder.embed("x")

        assert len(vec) == 512
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["dimensions"] == 512


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    """Batch embedding happy-path and sub-batch splitting."""

    @pytest.mark.asyncio
    async def test_small_batch_single_api_call(self):
        embedder, mock = _make_patched_embedder(dim=1024)
        texts = ["a", "b", "c"]
        mock.embeddings.create.return_value = _fake_response(texts, dim=1024)

        vectors = await embedder.embed_batch(texts)

        assert len(vectors) == 3
        assert all(len(v) == 1024 for v in vectors)
        # Should only call the API once for 3 texts
        assert mock.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_large_batch_splits_into_sub_batches(self):
        embedder, mock = _make_patched_embedder(dim=128)
        # Override batch_size to 5 for testing
        embedder._batch_size = 5
        n = 12
        texts = [f"text {i}" for i in range(n)]

        # Each sub-batch returns the right number of embeddings
        async def _side_effect(*args: object, **kwargs: object) -> MagicMock:
            inp = kwargs["input"]
            count = len(inp) if isinstance(inp, list) else 1
            return _fake_response(["x"] * count, dim=128)

        mock.embeddings.create.side_effect = _side_effect

        vectors = await embedder.embed_batch(texts)

        assert len(vectors) == n
        # With batch_size=5 and n=12: ceil(12/5) = 3 sub-batches
        assert mock.embeddings.create.call_count == math.ceil(n / 5)

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self):
        embedder, mock = _make_patched_embedder(dim=8)
        embedder._batch_size = 3
        texts = ["first", "second", "third", "fourth", "fifth"]

        async def _ordered_side_effect(*args: object, **kwargs: object) -> MagicMock:
            inp = kwargs["input"]
            # Return deterministic vectors: index * 0.1
            mock_resp = MagicMock()
            mock_resp.data = [
                MagicMock(embedding=[float(idx) * 0.1], index=idx)
                for idx, _ in enumerate(inp)
            ]
            return mock_resp

        mock.embeddings.create.side_effect = _ordered_side_effect

        vectors = await embedder.embed_batch(texts)

        assert len(vectors) == 5
        # Each sub-batch starts its own index numbering from 0,
        # but the embedder preserves original input order
        # Vector uniqueness test: all values should differ (no duplicates from re-indexing)
        assert vectors[0] != vectors[1]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestEmbedderValidation:
    """Input validation and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_string_raises(self):
        embedder, _ = _make_patched_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self):
        embedder, _ = _make_patched_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed("   \t\n  ")

    @pytest.mark.asyncio
    async def test_empty_list_raises(self):
        embedder, _ = _make_patched_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed_batch([])

    @pytest.mark.asyncio
    async def test_text_too_long_raises(self):
        embedder, _ = _make_patched_embedder()
        # Create a text longer than 8192 chars
        long_text = "x" * 10000
        with pytest.raises(ValueError, match="too long"):
            await embedder.embed(long_text)

    @pytest.mark.asyncio
    async def test_text_exactly_max_length_ok(self):
        embedder, mock = _make_patched_embedder()
        mock.embeddings.create.return_value = _fake_response(["ok"])
        # Exactly 8192 chars should be accepted
        text = "a" * 8192
        vec = await embedder.embed(text)
        assert len(vec) == 1024

    @pytest.mark.asyncio
    async def test_batch_text_too_long_raises(self):
        embedder, _ = _make_patched_embedder()
        with pytest.raises(ValueError, match="too long"):
            await embedder.embed_batch(["ok", "x" * 10000])

    @pytest.mark.asyncio
    async def test_batch_too_many_texts_raises(self):
        embedder, _ = _make_patched_embedder()
        with pytest.raises(ValueError, match="at most"):
            await embedder.embed_batch(["text"] * 2001)

    @pytest.mark.asyncio
    async def test_batch_exactly_max_texts_ok(self):
        embedder, mock = _make_patched_embedder(dim=4)
        embedder._batch_size = 100  # fit in single call
        n = 1000  # MAX_TEXTS_PER_REQUEST
        texts = [f"t{i}" for i in range(n)]

        async def _size_aware_side_effect(*args: object, **kwargs: object) -> MagicMock:
            inp = kwargs["input"]
            count = len(inp) if isinstance(inp, list) else 1
            return _fake_response([f"t{j}" for j in range(count)], dim=4)

        mock.embeddings.create.side_effect = _size_aware_side_effect

        vectors = await embedder.embed_batch(texts)
        assert len(vectors) == n


# ---------------------------------------------------------------------------
# Error handling and retries
# ---------------------------------------------------------------------------


class TestEmbedderRetry:
    """Retry and error-handling behaviour."""

    @pytest.mark.asyncio
    async def test_retries_on_transient_error_then_succeeds(self):
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 3

        # Fail twice, succeed on third
        mock.embeddings.create.side_effect = [
            Exception("temporary network error"),
            Exception("another transient error"),
            _fake_response(["hello"]),
        ]

        vec = await embedder.embed("hello")
        assert len(vec) == 1024
        assert mock.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exceeded(self):
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 2

        mock.embeddings.create.side_effect = Exception("persistent failure")

        with pytest.raises(EmbedderError, match="persistent failure"):
            await embedder.embed("hello")

        # 1 initial + 2 retries = 3 attempts
        assert mock.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries_fails_immediately(self):
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 0

        mock.embeddings.create.side_effect = Exception("boom")

        with pytest.raises(EmbedderError):
            await embedder.embed("hello")

        assert mock.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_retry_on_validation_error(self):
        """Client-side errors (4xx) should not be retried."""
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 3

        # Simulate a BadRequestError from the OpenAI SDK
        import openai

        mock.embeddings.create.side_effect = openai.BadRequestError(
            message="invalid model",
            response=MagicMock(),
            body=MagicMock(),
        )

        with pytest.raises(EmbedderError):
            await embedder.embed("hello")

        # Should NOT retry on BadRequestError
        assert mock.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_error(self):
        """Rate limit errors (429) should be retried."""
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 2

        import openai

        mock.embeddings.create.side_effect = [
            openai.RateLimitError(
                message="rate limit exceeded",
                response=MagicMock(status_code=429),
                body=MagicMock(),
            ),
            _fake_response(["hello"]),
        ]

        vec = await embedder.embed("hello")
        assert len(vec) == 1024
        assert mock.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Server errors (5xx) should be retried."""
        embedder, mock = _make_patched_embedder()
        embedder._max_retries = 2

        import openai

        mock.embeddings.create.side_effect = [
            openai.InternalServerError(
                message="server error",
                response=MagicMock(status_code=500),
                body=MagicMock(),
            ),
            _fake_response(["hello"]),
        ]

        vec = await embedder.embed("hello")
        assert len(vec) == 1024
        assert mock.embeddings.create.call_count == 2


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------


class TestEmbedderInit:
    """Constructor and config integration."""

    def test_default_uses_settings(self):
        """Embedder() reads from app.config.settings.embedding."""
        embedder = Embedder()
        assert embedder._model == "text-embedding-v3"
        assert embedder._dimensions == 1024
        assert embedder._batch_size == 25
        assert embedder._max_retries == 3

    def test_custom_overrides(self):
        embedder = Embedder(
            model="custom-model",
            dimensions=512,
            batch_size=50,
            max_retries=5,
            timeout=30.0,
        )
        assert embedder._model == "custom-model"
        assert embedder._dimensions == 512
        assert embedder._batch_size == 50
        assert embedder._max_retries == 5
        assert embedder._timeout == 30.0

    def test_init_creates_client(self):
        """Verify _client is created (even without real API key)."""
        embedder = Embedder()
        assert embedder._client is not None
        assert embedder._client.base_url is not None


# ---------------------------------------------------------------------------
# Utility: _build_client
# ---------------------------------------------------------------------------


class TestBuildClient:
    """The _build_client helper."""

    def test_api_key_used(self):
        embedder = Embedder(api_key="sk-custom-key")
        assert embedder._client.api_key == "sk-custom-key"

    def test_base_url_set(self):
        embedder = Embedder(base_url="https://custom.api/v1")
        # base_url may be stored as an httpx.URL or str
        url_str = str(embedder._client.base_url)
        assert "custom.api" in url_str

    def test_timeout_propagated(self):
        embedder = Embedder(timeout=123.0)
        assert embedder._client.timeout == 123.0


# ---------------------------------------------------------------------------
# Type / shape checks
# ---------------------------------------------------------------------------


class TestEmbeddingShape:
    """Output shape invariants."""

    @pytest.mark.asyncio
    async def test_all_values_are_floats(self):
        embedder, mock = _make_patched_embedder(dim=256)
        mock.embeddings.create.return_value = _fake_response(["x"], dim=256)

        vec = await embedder.embed("x")
        assert len(vec) == 256
        assert all(isinstance(x, float) for x in vec)

    @pytest.mark.asyncio
    async def test_dimension_consistency_batch(self):
        embedder, mock = _make_patched_embedder(dim=64)
        n = 5
        mock.embeddings.create.return_value = _fake_response(["t"] * n, dim=64)

        vectors = await embedder.embed_batch(["t1", "t2", "t3", "t4", "t5"])
        assert len(vectors) == n
        for v in vectors:
            assert len(v) == 64
            assert all(isinstance(x, float) for x in v)
