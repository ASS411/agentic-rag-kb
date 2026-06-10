"""LLM client — Qwen-Plus via OpenAI-compatible API (task 5.1).

Provides async chat completion (streaming and non-streaming) with automatic
retry on transient failures.  Configuration is read from ``app.config.settings.llm``.

Usage::

    from app.core.llm import LLMClient

    llm = LLMClient()

    # Non-streaming
    answer = await llm.generate([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ])

    # Streaming
    async for chunk in llm.generate_stream(messages):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
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

_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 2048
_STREAMING_CHUNK_TIMEOUT = 60  # seconds between streamed chunks


# ---------------------------------------------------------------------------
# Helpers — retry predicate
# ---------------------------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that are safe to retry.

    Retryable: network errors, timeouts, server errors (5xx), rate limits.
    Non-retryable: client errors (BadRequest, Auth, Permission, NotFound),
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


class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call times out."""


class LLMStreamError(LLMError):
    """Raised when a streaming response fails mid-stream."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Async chat-completion client for Qwen-Plus and compatible models.

    Wraps the OpenAI-compatible ``/v1/chat/completions`` endpoint with
    automatic retry, timeout, and streaming support.

    Usage::

        llm = LLMClient()
        answer = await llm.generate(messages)
        async for token in llm.generate_stream(messages):
            ...
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialise the LLM client.

        All parameters default to the corresponding ``settings.llm.*``
        values when not provided, allowing per-instance overrides in tests.
        """
        llm_cfg = settings.llm

        self._api_key = (
            api_key if api_key is not None else llm_cfg.api_key.get_secret_value()
        )
        self._base_url = str(base_url if base_url is not None else llm_cfg.base_url)
        self._model = model or llm_cfg.model
        self._max_retries = max_retries if max_retries is not None else 3
        self._timeout = timeout if timeout is not None else 120.0

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,  # we handle retries ourselves for fine-grained control
        )

    # ------------------------------------------------------------------
    # Public API — non-streaming
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat-completion request and return the full generated text.

        Parameters
        ----------
        messages:
            List of dicts with ``role`` and ``content`` keys (OpenAI format).
            Example: ``[{"role": "user", "content": "Hello"}]``.
        temperature:
            Sampling temperature (0.0 – 2.0).  Default: 0.3.
        max_tokens:
            Maximum tokens to generate.  Default: 2048.
        top_p:
            Nucleus sampling parameter (optional).
        stop:
            Stop sequence(s) (optional).
        extra_body:
            Extra parameters passed to the API body (optional).

        Returns
        -------
        str
            The generated response text.

        Raises
        ------
        ValueError
            If *messages* is empty or malformed.
        LLMError
            If the API call fails after all retries.
        LLMTimeoutError
            If the API call times out.
        """
        _validate_messages(messages)

        t0 = time.monotonic()
        logger.debug(
            "LLM generate: model={}, msgs={}, max_tokens={}",
            self._model,
            len(messages),
            max_tokens,
        )

        try:
            response = await self._call_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                stream=False,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM generate timeout: model={}, elapsed={:.2f}s",
                self._model,
                elapsed,
            )
            raise LLMTimeoutError(
                f"LLM call timed out after {elapsed:.1f}s"
            )

        elapsed = time.monotonic() - t0
        content = response.choices[0].message.content or ""

        usage_info = ""
        if response.usage:
            usage_info = (
                f", prompt_tokens={response.usage.prompt_tokens}, "
                f"completion_tokens={response.usage.completion_tokens}"
            )

        logger.info(
            "LLM generate OK: model={}, msgs={}, elapsed={:.2f}s{}, "
            "response_len={}",
            self._model,
            len(messages),
            elapsed,
            usage_info,
            len(content),
        )
        return content

    # ------------------------------------------------------------------
    # Public API — streaming
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Send a chat-completion request and stream tokens as they arrive.

        Yields each content delta as a string.  Use this for SSE endpoints.

        Parameters
        ----------
        messages:
            Same format as ``generate()``.
        temperature:
            Sampling temperature (0.0 – 2.0).  Default: 0.3.
        max_tokens:
            Maximum tokens to generate.  Default: 2048.
        top_p:
            Nucleus sampling parameter (optional).
        stop:
            Stop sequence(s) (optional).
        extra_body:
            Extra parameters passed to the API body (optional).

        Yields
        ------
        str
            Content delta chunks from the streaming response.

        Raises
        ------
        ValueError
            If *messages* is empty or malformed.
        LLMError
            If the API call fails after all retries.
        LLMStreamError
            If the stream is interrupted.
        """
        _validate_messages(messages)

        t0 = time.monotonic()
        logger.debug(
            "LLM generate_stream: model={}, msgs={}, max_tokens={}",
            self._model,
            len(messages),
            max_tokens,
        )

        try:
            stream = await self._call_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                stream=True,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM stream init timeout: model={}, elapsed={:.2f}s",
                self._model,
                elapsed,
            )
            raise LLMTimeoutError(
                f"LLM stream initiation timed out after {elapsed:.1f}s"
            )

        total_tokens = 0
        full_text: list[str] = []

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_text.append(delta.content)
                    total_tokens += 1
                    yield delta.content

                # Safety: if no delta for too long, stream may be stuck
                # (relies on httpx timeout to break)

        except asyncio.TimeoutError:
            logger.warning(
                "LLM stream interrupted by timeout after {} tokens",
                total_tokens,
            )
            raise LLMStreamError(
                f"Stream timed out after {total_tokens} tokens"
            )
        except Exception as exc:
            logger.error(
                "LLM stream error: model={}, tokens={}, error={}",
                self._model,
                total_tokens,
                exc,
            )
            raise LLMStreamError(f"Stream failed: {exc}") from exc

        elapsed = time.monotonic() - t0
        total_text = "".join(full_text)
        logger.info(
            "LLM stream OK: model={}, msgs={}, elapsed={:.2f}s, "
            "tokens={}, response_len={}",
            self._model,
            len(messages),
            elapsed,
            total_tokens,
            len(total_text),
        )

    # ------------------------------------------------------------------
    # Internal — API call with retry
    # ------------------------------------------------------------------

    async def _call_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        stop: str | list[str] | None,
        extra_body: dict[str, Any] | None,
        stream: bool,
    ):
        """Execute a chat completion API call with retry.

        Parameters
        ----------
        (All parameters match the public API; validation is done by the caller.)

        Returns
        -------
        For ``stream=False``: the full ChatCompletion response.
        For ``stream=True``: an async iterator over streamed chunks.
        """
        retry_limit = self._max_retries + 1  # total attempts = retries + 1

        decorated = retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(retry_limit),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )(self._do_chat_completion)

        try:
            return await decorated(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                stream=stream,
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    async def _do_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        stop: str | list[str] | None,
        extra_body: dict[str, Any] | None,
        stream: bool,
    ):
        """Raw API call — no retry wrapper (retry is applied by ``_call_chat_completion``)."""

        # Build kwargs dict, omitting None values
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = stop
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error(
                "LLM API call failed: model={}, msgs={}, stream={}, error={}",
                self._model,
                len(messages),
                stream,
                exc,
            )
            raise


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate_messages(messages: list[dict[str, str]]) -> None:
    """Validate the messages list before sending to the API."""
    if not messages:
        raise ValueError("messages must not be empty")
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"messages[{i}] must be a dict, got {type(msg).__name__}")
        if "role" not in msg:
            raise ValueError(f"messages[{i}] missing required key 'role'")
        if "content" not in msg:
            raise ValueError(f"messages[{i}] missing required key 'content'")
        role = msg["role"]
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"messages[{i}].role must be a non-empty string")
        if not isinstance(msg["content"], str):
            raise TypeError(f"messages[{i}].content must be a string")
