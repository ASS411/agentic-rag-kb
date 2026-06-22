"""Redis cache manager with graceful degradation.

Provides an async ``RedisCacheManager`` for caching embeddings, query
rewrites, and retrieval results.  When Redis is unavailable the manager
silently degrades — all operations return ``None`` / default values and
log warnings without raising exceptions.

Usage::

    from app.core.cache import RedisCacheManager

    cache = RedisCacheManager()
    await cache.set("key", {"foo": 42}, ttl=3600)
    value = await cache.get("key")
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from loguru import logger

from app.config import settings


# ---------------------------------------------------------------------------
# Cache key helpers (public — reused by embedder / agent / retriever)
# ---------------------------------------------------------------------------


def cache_key_embedding(text: str) -> str:
    """Build a deterministic cache key for a single text embedding."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"emb:{digest}"


def cache_key_rewrite(question: str) -> str:
    """Build a deterministic cache key for query rewrite results."""
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()
    return f"rewrite:{digest}"


def cache_key_retrieve(query_str: str, params_hash: str) -> str:
    """Build a deterministic cache key for retrieval results."""
    digest = hashlib.md5(
        f"{query_str}|{params_hash}".encode("utf-8")
    ).hexdigest()
    return f"retrieve:{digest}"


def _hash_params(**kwargs) -> str:
    """Build a short deterministic hash for retrieval parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# RedisCacheManager
# ---------------------------------------------------------------------------


class RedisCacheManager:
    """Async Redis cache with graceful fallback when Redis is down.

    All public methods are no-fail — when the Redis connection is
    unavailable, ``get`` returns ``None``, ``set`` is a no-op, and
    ``has`` returns ``False``.  A warning is logged on the first
    failure; subsequent calls are silent until the next health-check
    window.

    Parameters
    ----------
    url:
        Redis connection URL.  Defaults to ``settings.redis.url``.
    """

    def __init__(self, *, url: str | None = None) -> None:
        self._url = url or settings.redis.url
        self._enabled = settings.redis.enabled
        self._client: Any = None
        self._failed = False  # tracks whether the last attempt failed

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> bool:
        """Lazily create and test the Redis client.  Returns ``True`` on
        success, ``False`` when disabled or unavailable."""
        if not self._enabled:
            return False

        if self._client is not None:
            return True

        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.warning(
                "redis package not installed — caching disabled"
            )
            self._enabled = False
            return False

        try:
            self._client = aioredis.from_url(
                self._url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            await self._client.ping()
            logger.info("Redis cache connected: url={}", self._url)
            self._failed = False
            return True
        except Exception as exc:
            logger.warning(
                "Redis cache unavailable (url={}): {} — caching disabled",
                self._url,
                exc,
            )
            self._client = None
            self._enabled = False
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value by *key*.  Returns ``None`` on miss or
        when Redis is unavailable."""
        if not await self._ensure_client():
            return None

        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.debug("Redis GET failed for key={}: {}", key, exc)
            self._failed = True
            return None

        if raw is None:
            logger.debug("Redis cache miss: key={}", key)
            return None

        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Redis cache value corrupt for key={}: {}", key, exc)
            return None

        logger.debug("Redis cache hit: key={}", key)
        return value

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        """Store *value* in cache under *key* with an optional *ttl* in seconds."""
        if not await self._ensure_client():
            return

        try:
            raw = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to serialize cache value for key={}: {}", key, exc)
            return

        try:
            await self._client.set(key, raw, ex=ttl)
            logger.debug(
                "Redis cache set: key={}, ttl={}s",
                key,
                ttl if ttl else "none",
            )
        except Exception as exc:
            logger.debug("Redis SET failed for key={}: {}", key, exc)
            self._failed = True

    async def delete(self, key: str) -> None:
        """Remove *key* from cache."""
        if not await self._ensure_client():
            return

        try:
            await self._client.delete(key)
            logger.debug("Redis cache delete: key={}", key)
        except Exception as exc:
            logger.debug("Redis DELETE failed for key={}: {}", key, exc)
            self._failed = True

    async def has(self, key: str) -> bool:
        """Check whether *key* exists in cache."""
        if not await self._ensure_client():
            return False

        try:
            exists = await self._client.exists(key)
            return bool(exists)
        except Exception as exc:
            logger.debug("Redis EXISTS failed for key={}: {}", key, exc)
            self._failed = True
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob *pattern* (e.g. ``retrieve:*``).

        Returns the number of keys deleted (0 if Redis is unavailable)."""
        if not await self._ensure_client():
            return 0

        try:
            keys = await self._client.keys(pattern)
            if not keys:
                return 0
            deleted = await self._client.delete(*keys)
            logger.info(
                "Redis cache invalidated: pattern={}, deleted={}",
                pattern,
                deleted,
            )
            return deleted
        except Exception as exc:
            logger.debug(
                "Redis invalidate_pattern failed for pattern={}: {}",
                pattern,
                exc,
            )
            self._failed = True
            return 0

    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Convenience wrapper for ``invalidate_pattern(f"{prefix}*")``."""
        return await self.invalidate_pattern(f"{prefix}*")

    async def close(self) -> None:
        """Gracefully close the Redis connection (no-op if never connected)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
