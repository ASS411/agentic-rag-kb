"""Tests for the Chat API (tasks 5.3 / 5.4).
Covers streaming SSE, non-streaming JSON, error cases, and event format.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Non-streaming (5.4)
# ---------------------------------------------------------------------------

class TestNonStreamChat:

    def test_returns_json_response(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.generate = AsyncMock(return_value="answer text")
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "test", "stream": False},
                )

                assert response.status_code == 200
                body = response.json()
                assert body["success"] is True
                assert body["data"]["answer"] == "answer text"

    def test_question_too_long_returns_422(self):
        response = client.post(
            "/api/v1/chat",
            json={"question": "x" * 5000, "stream": False},
        )
        assert response.status_code == 422

    def test_missing_question_returns_422(self):
        response = client.post(
            "/api/v1/chat",
            json={"stream": False},
        )
        assert response.status_code == 422

    def test_llm_error_returns_502(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:
                from app.core.llm import LLMError

                mock_llm = MagicMock()
                mock_llm.generate = AsyncMock(side_effect=LLMError("api down"))
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "test", "stream": False},
                )

                assert response.status_code == 502
                assert response.json()["success"] is False

    def test_empty_collection_still_works(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.generate = AsyncMock(return_value="cannot answer")
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "test", "stream": False},
                )

                assert response.status_code == 200
                assert response.json()["data"]["sources"] != ""


# ---------------------------------------------------------------------------
# Streaming (5.3)
# ---------------------------------------------------------------------------

class TestStreamChat:

    def test_content_type_is_sse(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "a"
                    yield "b"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post("/api/v1/chat", json={"question": "hi"})

                assert response.status_code == 200
                ct = response.headers.get("content-type", "")
                assert "text/event-stream" in ct

    def test_stream_has_token_sources_done_events(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "hello"
                    yield "world"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post("/api/v1/chat", json={"question": "hello"})

                body = response.text
                assert "data:" in body

                events = []
                for line in body.strip().split("\n"):
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

                types = [e["type"] for e in events]
                assert "token" in types
                assert "sources" in types
                assert "done" in types

    def test_stream_event_structure(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "token1"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post("/api/v1/chat", json={"question": "test"})
                body = response.text

                for line in body.strip().split("\n"):
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        assert "type" in event
                        assert "content" in event
                        assert event["type"] in ("token", "sources", "done", "error")

    def test_stream_default_when_no_stream_param(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "x"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post("/api/v1/chat", json={"question": "default"})
                ct = response.headers.get("content-type", "")
                assert "text/event-stream" in ct


# ---------------------------------------------------------------------------
# SSE event format helpers
# ---------------------------------------------------------------------------

class TestSSEEventFormat:

    def test_sse_event_token(self):
        from app.api.chat import _sse_event

        result = _sse_event("token", "hello")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        parsed = json.loads(result[6:].strip())
        assert parsed == {"type": "token", "content": "hello"}

    def test_sse_event_sources(self):
        from app.api.chat import _sse_event

        result = _sse_event("sources", "1. **doc.pdf**")
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "sources"

    def test_sse_event_done(self):
        from app.api.chat import _sse_event

        result = _sse_event("done", "")
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "done"

    def test_sse_error(self):
        from app.api.chat import _sse_error

        result = _sse_error("something went wrong")
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "error"
        assert "something went wrong" in parsed["content"]

    def test_sse_chinese_preserved(self):
        from app.api.chat import _sse_event

        result = _sse_event("token", "hello world")
        parsed = json.loads(result[6:].strip())
        assert parsed["content"] == "hello world"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class TestChatRequestValidation:

    def test_empty_question_returns_422(self):
        response = client.post(
            "/api/v1/chat",
            json={"question": "", "stream": False},
        )
        assert response.status_code == 422

    def test_valid_request_accepted(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.generate = AsyncMock(return_value="ok")
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "valid", "stream": False, "top_k": 10},
                )
                assert response.status_code == 200
