"""Token counting utility using tiktoken.

Provides character-level fallback when tiktoken is unavailable and
conservative estimates for non-OpenAI models (Qwen series).

Usage::

    from app.utils.token_counter import count_tokens, fits_context_window

    tokens = count_tokens("你好 world")
    if fits_context_window("long text...", max_tokens=8000):
        ...

Design note (DESIGN.md §11.3):
    tiktoken is calibrated for OpenAI tokenizers; estimates for Qwen
    may differ by ±15%.  Use conservative ``max_tokens`` settings and
    prefer DashScope's returned ``usage.total_tokens`` for actual counts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default encoding — cl100k_base is the best match for Qwen models
# (Qwen uses a BPE tokenizer similar to GPT-4's).
_DEFAULT_ENCODING = "cl100k_base"

# Conservative multiplier for non-OpenAI models.
# Multiply tiktoken estimates by this factor to account for tokenizer
# differences.  1.15 means we assume Qwen may use up to 15% more tokens.
_SAFETY_FACTOR = 1.15

# Maximum context window for common models (tokens).
_CONTEXT_LIMITS: dict[str, int] = {
    "qwen-plus": 32_000,
    "qwen-max": 32_000,
    "qwen-turbo": 8_000,
    "gpt-4o": 128_000,
    "gpt-4": 8_000,
    "gpt-3.5-turbo": 4_096,
    "deepseek-v3": 64_000,
}


# ---------------------------------------------------------------------------
# Lazy encoder
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_encoder():
    """Return a tiktoken encoder, or None when unavailable."""
    try:
        import tiktoken
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except ImportError:
        logger.warning(
            "tiktoken not installed — falling back to character estimate "
            "(install with: pip install tiktoken>=0.7.0)"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to load tiktoken encoding: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Return the approximate token count for *text*.

    Uses tiktoken ``cl100k_base`` when available; falls back to a simple
    character-length estimate (÷4 for CJK, ÷3.5 for mixed).
    """
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return _char_estimate(text)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count total tokens across a chat-messages list.

    Each message adds ~4 tokens of framing overhead on top of the content.
    """
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # role + separator overhead
    return total


def context_limit(model: str | None = None, default: int = 8_000) -> int:
    """Return the context window size for *model*.

    Falls back to ``default`` when the model is unknown.
    """
    if model and model in _CONTEXT_LIMITS:
        return _CONTEXT_LIMITS[model]
    return default


def fits_context_window(
    text: str,
    *,
    max_tokens: int,
    safety_factor: float | None = None,
) -> bool:
    """Check whether *text* fits within *max_tokens*.

    Applies a conservative safety factor for non-OpenAI tokenizers.
    """
    if safety_factor is None:
        safety_factor = _SAFETY_FACTOR
    estimate = count_tokens(text)
    safe_estimate = int(estimate * safety_factor)
    return safe_estimate <= max_tokens


def truncate_to_window(
    text: str,
    *,
    max_tokens: int,
    safety_factor: float | None = None,
) -> str:
    """Truncate *text* so it fits within *max_tokens*.

    Uses a character-level heuristic for the fallback, and tiktoken's
    ``decode(encode()[:max])`` truncation when the encoder is available.
    """
    if safety_factor is None:
        safety_factor = _SAFETY_FACTOR

    safe_limit = int(max_tokens / safety_factor)

    encoder = _get_encoder()
    if encoder is not None:
        tokens = encoder.encode(text)
        if len(tokens) <= safe_limit:
            return text
        truncated = tokens[:safe_limit]
        return encoder.decode(truncated)

    # Fallback: character slicing estimate
    if _char_estimate(text) <= safe_limit:
        return text
    # ~2.5 chars per token for mixed CJK/ASCII text
    char_limit = int(safe_limit * 2.5)
    return text[:char_limit] + "…"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _char_estimate(text: str) -> int:
    """Rough token estimate from character count.

    CJK characters typically map 1:1 with tokens; ASCII words average
    ~0.75 tokens per char.  We use a blended heuristic.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
    ascii_chars = len(text) - cjk
    return cjk + int(ascii_chars / 3.5) + 1
