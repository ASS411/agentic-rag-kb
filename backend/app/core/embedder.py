"""Embedding client — Qwen text-embedding-v3 via OpenAI-compatible API (task 3.2).

Provides async single and batched embedding calls with automatic retry
on transient failures.  Configuration is read from ``app.config.settings.embedding``.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CHARS_PER_TEXT = 8192       # per-text character limit (guard)
_MAX_TEXTS_PER_REQUEST = 1000    # max texts in a single batch request


# ---------------------------------------------------------------------------
# Helpers — retry predicate
# ---------------------------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that are safe to retry.

    Retryable: everything except known client/validation errors.
    Non-retryable: BadRequestError, AuthenticationError, PermissionDeniedError,
    and our own ValueError/TypeError validations.
    """
    import openai

    # Never retry client-side / validation errors (4xx)
    if isinstance(exc, openai.BadRequestError):
        return False
    if isinstance(exc, openai.AuthenticationError):
        return False
    if isinstance(exc, openai.PermissionDeniedError):
        return False
    if isinstance(exc, openai.NotFoundError):
        return False
    # Also skip retry for input validation errors from our own layer
    if isinstance(exc, (ValueError, TypeError)):
        return False

    # Everything else (network, timeout, server errors, rate limits,
    # and unknown/generic exceptions) is considered transient.
    return True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbedderError(Exception):
    """Raised when an embedding operation fails after all retries."""


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Async embedding client for Qwen text-embedding-v3 and compatible models.

    Wraps the OpenAI-compatible ``/v1/embeddings`` endpoint with automatic
    batching, retry, and per-request dimension selection.

    Usage::

        embedder = Embedder()
        vec = await embedder.embed("你好世界")
        vecs = await embedder.embed_batch(["文本1", "文本2", "文本3"])
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialise the embedding client.

        All parameters default to the corresponding ``settings.embedding.*``
        values when not provided, allowing per-instance overrides in tests.
        """
        emb = settings.embedding

        self._api_key = api_key if api_key is not None else emb.api_key.get_secret_value()
        self._base_url = str(base_url if base_url is not None else emb.base_url)
        self._model = model or emb.model
        self._dimensions = dimensions or emb.dimensions
        self._batch_size = batch_size if batch_size is not None else emb.batch_size
        self._max_retries = max_retries if max_retries is not None else emb.max_retries
        self._timeout = timeout if timeout is not None else float(emb.timeout_seconds)

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,  # we handle retries ourselves for fine-grained control
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Compute the embedding vector for a single text.

        Parameters
        ----------
        text:
            Input text.  Must be non-empty after stripping whitespace.
            Maximum length: {_MAX_CHARS_PER_TEXT} characters.

        Returns
        -------
        list[float]
            Embedding vector with ``self._dimensions`` elements.

        Raises
        ------
        ValueError
            If *text* is empty, whitespace-only, or too long.
        EmbedderError
            If the API call fails after all retries.
        """
        _validate_single(text)
        result = await self._call_api(inputs=[text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of texts.

        Large input lists are automatically split into sub-batches of
        ``self._batch_size`` to stay within API limits and reduce per-
        request latency.

        Parameters
        ----------
        texts:
            List of input texts.  Must be non-empty and contain at most
            {_MAX_TEXTS_PER_REQUEST} items.

        Returns
        -------
        list[list[float]]
            Embedding vectors in the same order as *texts*.
            Each vector has ``self._dimensions`` elements.

        Raises
        ------
        ValueError
            If *texts* is empty, any text is too long, or the list is
            too large.
        EmbedderError
            If any API call fails after all retries.
        """
        _validate_batch(texts)

        if len(texts) <= self._batch_size:
            return await self._call_api(inputs=texts)

        # Split into sub-batches and send sequentially to respect API rate limits
        sub_batches = _chunk_list(texts, self._batch_size)

        all_results: list[list[float]] = []
        for i, batch in enumerate(sub_batches):
            logger.debug(
                "Embedding batch {}/{}: {} texts",
                i + 1,
                len(sub_batches),
                len(batch),
            )
            batch_result = await self._call_api(inputs=batch)
            all_results.extend(batch_result)

        return all_results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_api(self, inputs: list[str]) -> list[list[float]]:
        """Execute a single embeddings API call with retry.

        Parameters
        ----------
        inputs:
            List of texts to embed (1 – *batch_size*).  Validation is
            performed by the caller (``embed`` / ``embed_batch``).

        Returns
        -------
        list[list[float]]
            Embedding vectors in the same order as *inputs*.
        """
        if not inputs:
            return []

        retry_limit = self._max_retries + 1  # total attempts = retries + 1

        decorated = retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(retry_limit),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )(self._do_call)

        try:
            return await decorated(inputs)
        except Exception as exc:
            raise EmbedderError(str(exc)) from exc

    async def _do_call(self, inputs: list[str]) -> list[list[float]]:
        """Raw API call — no retry wrapper (retry is applied by ``_call_api``)."""

        # When there's only one input, pass it as a plain string (OpenAI compat)
        api_input = inputs[0] if len(inputs) == 1 else inputs

        t0 = time.monotonic()
        logger.debug(
            "Embedding API call: model={}, inputs={}, dimensions={}",
            self._model,
            len(inputs),
            self._dimensions,
        )

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=api_input,
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except Exception as exc:
            logger.error(
                "Embedding API call failed: model={}, inputs={}, error={}",
                self._model,
                len(inputs),
                exc,
            )
            raise

        elapsed = time.monotonic() - t0
        vectors = [d.embedding for d in response.data]

        # Sort by index to preserve input order (API guarantees order but be safe)
        vectors_sorted = sorted(
            zip(response.data, vectors),
            key=lambda pair: pair[0].index,
        )
        result = [v for _, v in vectors_sorted]

        logger.debug(
            "Embedding API call OK: model={}, inputs={}, dims={}, elapsed={:.2f}s, "
            "tokens={}",
            self._model,
            len(inputs),
            len(result[0]) if result else 0,
            elapsed,
            getattr(response.usage, "total_tokens", "?") if response.usage else "?",
        )
        return result


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate_single(text: str) -> None:
    """Validate a single text for ``embed()``."""
    if not text or not text.strip():
        raise ValueError("empty")
    if len(text) > _MAX_CHARS_PER_TEXT:
        raise ValueError("too long")


def _validate_batch(texts: list[str]) -> None:
    """Validate a batch for ``embed_batch()``."""
    if not texts:
        raise ValueError("empty")
    if len(texts) > _MAX_TEXTS_PER_REQUEST:
        raise ValueError(f"at most {_MAX_TEXTS_PER_REQUEST} texts per request")
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("all items must be strings")
        if len(text) > _MAX_CHARS_PER_TEXT:
            raise ValueError("too long")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split *items* into sub-lists of at most *chunk_size* elements."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
