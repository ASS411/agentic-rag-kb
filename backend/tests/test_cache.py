"""Tests for Redis cache manager.

Covers:
- Cache key helpers (embedding, rewrite, retrieve)
- Graceful degradation when Redis is unavailable
- JSON serialization / deserialization round-trip
- TTL support
- Key existence check (has)
- Pattern-based invalidation
- Connection failure does not raise exceptions
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.cache import (
    RedisCacheManager,
    _hash_params,
    cache_key_embedding,
    cache_key_rewrite,
    cache_key_retrieve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis_client():
    """Return a MagicMock that mimics an aioredis.Redis client."""
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=0)
    client.keys = AsyncMock(return_value=[])
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def cache_with_client(mock_redis_client):
    """Create a RedisCacheManager with a pre-injected mock client."""
    cache = RedisCacheManager()
    cache._client = mock_redis_client
    cache._enabled = True
    return cache, mock_redis_client


# ---------------------------------------------------------------------------
# Key helper tests
# ---------------------------------------------------------------------------


class TestCacheKeyHelpers:
    def test_embedding_key_is_deterministic(self):
        k1 = cache_key_embedding("hello world")
        k2 = cache_key_embedding("hello world")
        assert k1 == k2
        assert k1.startswith("emb:")

    def test_embedding_key_different_for_different_texts(self):
        k1 = cache_key_embedding("hello")
        k2 = cache_key_embedding("world")
        assert k1 != k2

    def test_rewrite_key_deterministic(self):
        k1 = cache_key_rewrite("What is RAG?")
        k2 = cache_key_rewrite("What is RAG?")
        assert k1 == k2
        assert k1.startswith("rewrite:")

    def test_retrieve_key_deterministic(self):
        k1 = cache_key_retrieve("query", "abc123")
        k2 = cache_key_retrieve("query", "abc123")
        assert k1 == k2
        assert k1.startswith("retrieve:")

    def test_hash_params_deterministic(self):
        h1 = _hash_params(top_k=20, rerank=True)
        h2 = _hash_params(top_k=20, rerank=True)
        assert h1 == h2
        assert len(h1) == 12


# ---------------------------------------------------------------------------
# Graceful degradation — Redis unavailable
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """When Redis is disabled or the import fails, no exceptions are raised."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        result = await cache.get("any_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_noop_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        await cache.set("any_key", {"value": 1})

    @pytest.mark.asyncio
    async def test_has_returns_false_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        assert await cache.has("any_key") is False

    @pytest.mark.asyncio
    async def test_delete_noop_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        await cache.delete("any_key")

    @pytest.mark.asyncio
    async def test_invalidate_pattern_returns_zero_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        deleted = await cache.invalidate_pattern("retrieve:*")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_invalidate_by_prefix_returns_zero_when_disabled(self):
        cache = RedisCacheManager()
        cache._enabled = False
        deleted = await cache.invalidate_by_prefix("retrieve:")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_no_exception_on_connect_failure(self):
        cache = RedisCacheManager(url="redis://does-not-exist:9999/0")
        cache._enabled = True
        result = await cache.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_exception_when_redis_module_missing(self):
        with patch.dict("sys.modules", {"redis": None}):
            cache = RedisCacheManager()
            cache._enabled = True
            result = await cache.get("test_key")
            assert result is None


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    @pytest.mark.asyncio
    async def test_set_and_get_simple_value(self, cache_with_client):
        cache, mock = cache_with_client
        data = {"answer": "hello", "score": 0.95}
        mock.get.return_value = json.dumps(data).encode("utf-8")

        result = await cache.get("k1")
        assert result == data

    @pytest.mark.asyncio
    async def test_set_and_get_list(self, cache_with_client):
        cache, mock = cache_with_client
        data = [1, 2, 3, "four"]
        mock.get.return_value = json.dumps(data).encode("utf-8")

        result = await cache.get("k1")
        assert result == data

    @pytest.mark.asyncio
    async def test_set_and_get_nested_dict(self, cache_with_client):
        cache, mock = cache_with_client
        data = {"outer": {"inner": [1, 2, 3], "flag": True}}
        mock.get.return_value = json.dumps(data).encode("utf-8")

        result = await cache.get("k1")
        assert result == data

    @pytest.mark.asyncio
    async def test_set_and_get_float_list_embedding(self, cache_with_client):
        cache, mock = cache_with_client
        data = [0.1, 0.2, 0.3, 0.4]
        mock.get.return_value = json.dumps(data).encode("utf-8")

        result = await cache.get("emb:abc")
        assert result == data

    @pytest.mark.asyncio
    async def test_get_returns_none_on_corrupt_json(self, cache_with_client):
        cache, mock = cache_with_client
        mock.get.return_value = b"not valid json {{{"

        result = await cache.get("k1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, cache_with_client):
        cache, mock = cache_with_client
        mock.get.return_value = None

        result = await cache.get("k1")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_serializes_and_sends_ttl(self, cache_with_client):
        cache, mock = cache_with_client

        await cache.set("k1", {"a": 1}, ttl=3600)

        mock.set.assert_called_once()
        call_key = mock.set.call_args[0][0]
        call_value = mock.set.call_args[0][1]
        call_kwargs = mock.set.call_args.kwargs

        assert call_key == "k1"
        decoded = json.loads(call_value)
        assert decoded == {"a": 1}
        assert call_kwargs.get("ex") == 3600

    @pytest.mark.asyncio
    async def test_set_without_ttl(self, cache_with_client):
        cache, mock = cache_with_client

        await cache.set("k1", "value")

        mock.set.assert_called_once()
        call_kwargs = mock.set.call_args.kwargs
        assert call_kwargs.get("ex") is None

    @pytest.mark.asyncio
    async def test_set_handles_non_serializable_via_default_str(self, cache_with_client):
        cache, mock = cache_with_client

        await cache.set("k1", {"set_val": {1, 2, 3}})

        mock.set.assert_called_once()
        call_value = mock.set.call_args[0][1]
        decoded = json.loads(call_value)
        assert "set_val" in decoded


# ---------------------------------------------------------------------------
# Key existence check
# ---------------------------------------------------------------------------


class TestHasKey:
    @pytest.mark.asyncio
    async def test_has_returns_true_when_exists(self, cache_with_client):
        cache, mock = cache_with_client
        mock.exists.return_value = 1

        assert await cache.has("k1") is True
        mock.exists.assert_called_once_with("k1")

    @pytest.mark.asyncio
    async def test_has_returns_false_when_not_exists(self, cache_with_client):
        cache, mock = cache_with_client
        mock.exists.return_value = 0

        assert await cache.has("k1") is False


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


class TestDeletion:
    @pytest.mark.asyncio
    async def test_delete_calls_redis(self, cache_with_client):
        cache, mock = cache_with_client

        await cache.delete("k1")

        mock.delete.assert_called_once_with("k1")

    @pytest.mark.asyncio
    async def test_delete_no_exception_on_error(self, cache_with_client):
        cache, mock = cache_with_client
        mock.delete.side_effect = ConnectionError("boom")

        await cache.delete("k1")


# ---------------------------------------------------------------------------
# Pattern-based invalidation
# ---------------------------------------------------------------------------


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_pattern_deletes_matching_keys(self, cache_with_client):
        cache, mock = cache_with_client
        mock.keys.return_value = [b"retrieve:abc", b"retrieve:def"]
        mock.delete.return_value = 2

        deleted = await cache.invalidate_pattern("retrieve:*")

        assert deleted == 2
        mock.keys.assert_called_once_with("retrieve:*")
        mock.delete.assert_called_once_with(b"retrieve:abc", b"retrieve:def")

    @pytest.mark.asyncio
    async def test_invalidate_pattern_no_matching_keys(self, cache_with_client):
        cache, mock = cache_with_client
        mock.keys.return_value = []

        deleted = await cache.invalidate_pattern("retrieve:*")

        assert deleted == 0
        mock.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_by_prefix(self, cache_with_client):
        cache, mock = cache_with_client
        mock.keys.return_value = [b"retrieve:xyz"]
        mock.delete.return_value = 1

        deleted = await cache.invalidate_by_prefix("retrieve:")

        assert deleted == 1
        mock.keys.assert_called_once_with("retrieve:*")

    @pytest.mark.asyncio
    async def test_invalidate_pattern_no_exception_on_error(self, cache_with_client):
        cache, mock = cache_with_client
        mock.keys.side_effect = ConnectionError("boom")

        deleted = await cache.invalidate_pattern("retrieve:*")
        assert deleted == 0


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        cache = RedisCacheManager()
        mock = MagicMock()
        mock.aclose = AsyncMock()
        cache._client = mock

        await cache.close()

        mock.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_client(self):
        cache = RedisCacheManager()
        await cache.close()

    @pytest.mark.asyncio
    async def test_close_does_not_raise_on_error(self):
        cache = RedisCacheManager()
        mock = MagicMock()
        mock.aclose = AsyncMock(side_effect=Exception("boom"))
        cache._client = mock

        await cache.close()
