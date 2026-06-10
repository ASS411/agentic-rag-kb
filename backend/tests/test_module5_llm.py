"""Tests for the LLM client (task 5.1).

Covers:
- Non-streaming chat completion (generate)
- Streaming chat completion (generate_stream)
- Client initialization from config and overrides
- Input validation (empty messages, missing roles, etc.)
- Error handling (API failures, retries, timeouts)
- Retry behaviour on transient vs non-transient errors
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm import (
    LLMClient,
    LLMError,
    LLMStreamError,
    LLMTimeoutError,
    _validate_messages,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_messages() -> list[dict[str, str]]:
    """Return a standard set of messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"},
    ]


def _fake_chat_completion(content: str) -> MagicMock:
    """Build a mock non-streaming chat completion response."""
    mock = MagicMock()
    mock.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    mock.usage = MagicMock(
        prompt_tokens=50,
        completion_tokens=len(content.split()),
        total_tokens=50 + len(content.split()),
    )
    return mock


def _fake_stream_chunks(text: str) -> list[MagicMock]:
    """Build mock streaming chunks with delta content."""
    chunks = []
    for token in text.split():
        chunk = MagicMock()
        chunk.choices = [
            MagicMock(delta=MagicMock(content=token + " "))
        ]
        chunks.append(chunk)
    return chunks


def _make_patched_client() -> tuple[LLMClient, MagicMock]:
    """Create an LLMClient with a mocked AsyncOpenAI client."""
    client = LLMClient()
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    client._client = mock_client
    return client, mock_client


# ---------------------------------------------------------------------------
# Non-streaming generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Happy-path non-streaming generation."""

    @pytest.mark.asyncio
    async def test_returns_string_content(self):
        client, mock = _make_patched_client()
        mock.chat.completions.create.return_value = _fake_chat_completion(
            "Retrieval-Augmented Generation combines search with LLMs."
        )

        result = await client.generate(_make_messages())

        assert isinstance(result, str)
        assert "Retrieval-Augmented" in result

    @pytest.mark.asyncio
    async def test_calls_api_with_correct_params(self):
        client, mock = _make_patched_client()
        mock.chat.completions.create.return_value = _fake_chat_completion("ok")

        await client.generate(
            _make_messages(),
            temperature=0.7,
            max_tokens=512,
        )

        call_kwargs = mock.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "qwen-plus"
        assert call_kwargs["messages"] == _make_messages()
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_default_params(self):
        client, mock = _make_patched_client()
        mock.chat.completions.create.return_value = _fake_chat_completion("ok")

        await client.generate(_make_messages())

        call_kwargs = mock.chat.completions.create.call_args.kwargs
        # Default temperature = 0.3, max_tokens = 2048
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_optional_params_passed(self):
        client, mock = _make_patched_client()
        mock.chat.completions.create.return_value = _fake_chat_completion("ok")

        await client.generate(
            _make_messages(),
            top_p=0.9,
            stop=["END"],
            extra_body={"enable_search": True},
        )

        call_kwargs = mock.chat.completions.create.call_args.kwargs
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["extra_body"] == {"enable_search": True}

    @pytest.mark.asyncio
    async def test_handles_empty_content(self):
        """API returns None content — should be converted to empty string."""
        client, mock = _make_patched_client()
        resp = _fake_chat_completion("ok")
        resp.choices[0].message.content = None
        mock.chat.completions.create.return_value = resp

        result = await client.generate(_make_messages())
        assert result == ""


# ---------------------------------------------------------------------------
# Streaming generate_stream
# ---------------------------------------------------------------------------


class TestGenerateStream:
    """Streaming generation tests."""

    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        client, mock = _make_patched_client()

        async def _fake_stream(*args, **kwargs):
            for chunk in _fake_stream_chunks("RAG is very useful"):
                yield chunk

        mock.chat.completions.create.return_value = _fake_stream()

        tokens: list[str] = []
        async for token in client.generate_stream(_make_messages()):
            tokens.append(token)

        assert len(tokens) == 4
        assert "RAG" in tokens[0]

    @pytest.mark.asyncio
    async def test_stream_param_is_true(self):
        client, mock = _make_patched_client()

        async def _fake_stream(*args, **kwargs):
            yield _fake_stream_chunks("ok")[0]

        mock.chat.completions.create.return_value = _fake_stream()

        async for _ in client.generate_stream(_make_messages()):
            pass

        call_kwargs = mock.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_skips_empty_deltas(self):
        """Chunks without delta.content should be skipped."""
        client, mock = _make_patched_client()

        chunk_with_content = MagicMock()
        chunk_with_content.choices = [
            MagicMock(delta=MagicMock(content="hello "))
        ]
        chunk_empty_delta = MagicMock()
        chunk_empty_delta.choices = [MagicMock(delta=MagicMock(content=None))]
        chunk_no_choices = MagicMock()
        chunk_no_choices.choices = []

        async def _fake_stream(*args, **kwargs):
            yield chunk_empty_delta
            yield chunk_with_content
            yield chunk_no_choices
            yield chunk_with_content

        mock.chat.completions.create.return_value = _fake_stream()

        tokens: list[str] = []
        async for token in client.generate_stream(_make_messages()):
            tokens.append(token)

        # Only chunks with content should be yielded
        assert len(tokens) == 2
        assert all(t == "hello " for t in tokens)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidateMessages:
    """Input validation for the messages list."""

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_messages([])

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            _validate_messages("not a list")  # type: ignore[arg-type]

    def test_non_dict_item_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _validate_messages(["not a dict"])  # type: ignore[list-item]

    def test_missing_role_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required key 'role'"):
            _validate_messages([{"content": "no role"}])

    def test_missing_content_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required key 'content'"):
            _validate_messages([{"role": "user"}])

    def test_empty_role_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_messages([{"role": "", "content": "x"}])
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_messages([{"role": "  ", "content": "x"}])

    def test_content_not_string_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            _validate_messages([{"role": "user", "content": 123}])  # type: ignore[dict-item]

    def test_valid_messages_pass(self):
        # Should not raise
        _validate_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])


# ---------------------------------------------------------------------------
# Error handling and retries
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """Retry and error-handling behaviour."""

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        client, mock = _make_patched_client()
        client._max_retries = 2

        mock.chat.completions.create.side_effect = [
            Exception("temporary network error"),
            Exception("another transient error"),
            _fake_chat_completion("finally ok"),
        ]

        result = await client.generate(_make_messages())
        assert result == "finally ok"
        assert mock.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_bad_request(self):
        """Client errors (4xx) should NOT be retried."""
        client, mock = _make_patched_client()
        client._max_retries = 2

        import openai

        mock.chat.completions.create.side_effect = openai.BadRequestError(
            message="invalid request",
            response=MagicMock(status_code=400),
            body=MagicMock(),
        )

        with pytest.raises(LLMError):
            await client.generate(_make_messages())

        assert mock.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_authentication_error(self):
        client, mock = _make_patched_client()
        client._max_retries = 2

        import openai

        mock.chat.completions.create.side_effect = openai.AuthenticationError(
            message="invalid api key",
            response=MagicMock(status_code=401),
            body=MagicMock(),
        )

        with pytest.raises(LLMError):
            await client.generate(_make_messages())

        assert mock.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Server errors (5xx) should be retried."""
        client, mock = _make_patched_client()
        client._max_retries = 2

        import openai

        mock.chat.completions.create.side_effect = [
            openai.InternalServerError(
                message="server error",
                response=MagicMock(status_code=500),
                body=MagicMock(),
            ),
            _fake_chat_completion("ok after retry"),
        ]

        result = await client.generate(_make_messages())
        assert result == "ok after retry"
        assert mock.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_llm_error(self):
        client, mock = _make_patched_client()
        client._max_retries = 2  # total 3 attempts

        mock.chat.completions.create.side_effect = Exception("persistent failure")

        with pytest.raises(LLMError):
            await client.generate(_make_messages())

        # 1 initial + 2 retries = 3 total
        assert mock.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    """Constructor and config integration."""

    def test_default_uses_settings(self):
        """LLMClient() reads from app.config.settings.llm."""
        client = LLMClient()
        assert client._model == "qwen-plus"
        assert client._max_retries == 3

    def test_custom_overrides(self):
        client = LLMClient(
            model="gpt-4o",
            max_retries=5,
            timeout=60.0,
        )
        assert client._model == "gpt-4o"
        assert client._max_retries == 5
        assert client._timeout == 60.0

    def test_init_creates_client(self):
        """Verify _client is created (even without real API key)."""
        client = LLMClient()
        assert client._client is not None
        assert client._client.base_url is not None

    def test_api_key_override(self):
        client = LLMClient(api_key="sk-custom-key")
        assert client._api_key == "sk-custom-key"

    def test_base_url_override(self):
        client = LLMClient(base_url="https://custom.api/v1")
        url_str = str(client._client.base_url)
        assert "custom.api" in url_str

    def test_timeout_propagated(self):
        client = LLMClient(timeout=123.0)
        assert client._client.timeout == 123.0


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Timeout scenarios."""

    @pytest.mark.asyncio
    async def test_generate_timeout_raises_llm_timeout_error(self):
        client, mock = _make_patched_client()

        import asyncio

        async def _slow_response(*args, **kwargs):
            await asyncio.sleep(99)  # will timeout
            return _fake_chat_completion("too late")

        mock.chat.completions.create.side_effect = asyncio.TimeoutError()

        with pytest.raises(LLMTimeoutError):
            await client.generate(_make_messages())

    @pytest.mark.asyncio
    async def test_stream_init_timeout_raises_llm_timeout_error(self):
        client, mock = _make_patched_client()

        import asyncio

        mock.chat.completions.create.side_effect = asyncio.TimeoutError()

        with pytest.raises(LLMTimeoutError):
            async for _ in client.generate_stream(_make_messages()):
                pass


# ---------------------------------------------------------------------------
# Streaming error during iteration
# ---------------------------------------------------------------------------


class TestStreamErrors:
    """Errors during stream iteration."""

    @pytest.mark.asyncio
    async def test_stream_mid_iteration_error(self):
        client, mock = _make_patched_client()

        async def _broken_stream(*args, **kwargs):
            yield _fake_stream_chunks("start")[0]
            raise Exception("connection reset")

        mock.chat.completions.create.return_value = _broken_stream()

        with pytest.raises(LLMStreamError, match="Stream failed"):
            async for _ in client.generate_stream(_make_messages()):
                pass
