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


def _source_chunk(chunk_id: str = "chunk_1"):
    from app.models.search import SearchChunk

    return SearchChunk(
        chunk_id=chunk_id,
        content="source text",
        score=0.91,
        doc_id="doc_1",
        doc_name="notes.md",
        doc_type="md",
        page=1,
        chunk_index=0,
        metadata={"section": "intro"},
    )


# ---------------------------------------------------------------------------
# Non-streaming (5.4)
# ---------------------------------------------------------------------------

class TestNonStreamChat:

    def test_returns_json_response(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = [_source_chunk()]
            with patch("app.api.chat.LLMClient") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.generate = AsyncMock(return_value="answer text")
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "test", "stream": False, "use_agent": False},
                )

                assert response.status_code == 200
                body = response.json()
                assert body["success"] is True
                assert body["data"]["answer"] == "answer text"
                assert body["data"]["source_chunks"][0]["chunk_id"] == "chunk_1"

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
                    json={"question": "test", "stream": False, "use_agent": False},
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
                    json={"question": "test", "stream": False, "use_agent": False},
                )

                assert response.status_code == 200
                assert response.json()["data"]["sources"] != ""

    def test_agent_non_stream_returns_json_response(self):
        source_chunk = _source_chunk().model_dump(mode="json")

        async def _fake_run(*args, **kwargs):
            yield json.dumps({
                "type": "answer-chunk",
                "content": "hello ",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "answer-chunk",
                "content": "world",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "sources",
                "content": "1. source",
                "source_chunks": [source_chunk],
                "chunk_ids": ["chunk_1"],
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "done",
                "content": "",
                "conversation_id": None,
                "total_rounds": 1,
                "chunks_used": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
            })

        with patch("app.core.agent.AgentLoop") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = MagicMock(return_value=_fake_run())
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/chat",
                json={"question": "agent json", "stream": False, "use_agent": True},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["answer"] == "hello world"
        assert body["data"]["sources"] == "1. source"
        assert body["data"]["source_chunks"][0]["chunk_id"] == "chunk_1"

    def test_agent_non_stream_error_event_returns_502(self):
        async def _fake_run(*args, **kwargs):
            yield json.dumps({
                "type": "error",
                "content": "Agent boom",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })

        with patch("app.core.agent.AgentLoop") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = MagicMock(return_value=_fake_run())
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/chat",
                json={"question": "agent json", "stream": False, "use_agent": True},
            )

        assert response.status_code == 502
        body = response.json()
        assert body["success"] is False
        assert "Agent boom" in body["message"]

    def test_agent_non_stream_passes_request_max_rounds(self):
        async def _fake_run(*args, **kwargs):
            yield json.dumps({
                "type": "done",
                "content": "",
                "conversation_id": None,
                "total_rounds": 2,
                "chunks_used": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
            })

        with patch("app.core.agent.AgentLoop") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = MagicMock(return_value=_fake_run())
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/chat",
                json={
                    "question": "agent json",
                    "stream": False,
                    "use_agent": True,
                    "max_rounds": 2,
                },
            )

        assert response.status_code == 200
        mock_agent.run.assert_called_once_with("agent json", max_rounds=2)


# ---------------------------------------------------------------------------
# Streaming (5.3)
# ---------------------------------------------------------------------------

class TestStreamChat:

    @staticmethod
    def _events(response):
        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

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

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "hi", "use_agent": False},
                )

                assert response.status_code == 200
                ct = response.headers.get("content-type", "")
                assert "text/event-stream" in ct

    def test_stream_has_token_sources_done_events(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = [_source_chunk()]
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "hello"
                    yield "world"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "hello", "use_agent": False},
                )

                body = response.text
                assert "data:" in body

                events = self._events(response)

                types = [e["type"] for e in events]
                assert "token" in types
                assert "sources" in types
                assert "answer-done" in types
                assert "done" in types
                assert types.index("sources") < types.index("answer-done")
                assert types.index("answer-done") < types.index("done")
                sources_event = [e for e in events if e["type"] == "sources"][0]
                assert sources_event["source_chunks"][0]["chunk_id"] == "chunk_1"
                done_event = [e for e in events if e["type"] == "done"][0]
                assert done_event["timestamp"]
                assert "total_rounds" in done_event
                assert "chunks_used" in done_event

    def test_agent_stream_has_answer_done_before_done(self):
        async def _fake_run(*args, **kwargs):
            yield json.dumps({
                "type": "agent-step",
                "step": "generate",
                "message": "Generating",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "answer-chunk",
                "content": "hello",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "sources",
                "content": "1. source",
                "source_chunks": [],
                "chunk_ids": [],
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "answer-done",
                "timestamp": "2026-01-01T00:00:00+00:00",
            })
            yield json.dumps({
                "type": "done",
                "content": "",
                "conversation_id": None,
                "total_rounds": 1,
                "chunks_used": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
            })

        with patch("app.core.agent.AgentLoop") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = MagicMock(return_value=_fake_run())
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/chat",
                json={"question": "agent stream", "use_agent": True},
            )

        assert response.status_code == 200
        events = self._events(response)
        types = [event["type"] for event in events]

        assert "answer-done" in types
        assert types.index("sources") < types.index("answer-done")
        assert types.index("answer-done") < types.index("done")

    def test_stream_event_structure(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "token1"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "test", "use_agent": False},
                )
                body = response.text

                for line in body.strip().split("\n"):
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        assert "type" in event
                        assert "content" in event
                        assert "timestamp" in event
                        assert event["type"] in ("token", "sources", "answer-done", "done", "error")

    def test_stream_default_when_no_stream_param(self):
        with patch("app.api.chat._retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = []
            with patch("app.api.chat.LLMClient") as mock_llm_class:

                async def _fake_stream(*args, **kwargs):
                    yield "x"

                mock_llm = MagicMock()
                mock_llm.generate_stream = MagicMock(return_value=_fake_stream())
                mock_llm_class.return_value = mock_llm

                response = client.post(
                    "/api/v1/chat",
                    json={"question": "default", "use_agent": False},
                )
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
        assert parsed["type"] == "token"
        assert parsed["content"] == "hello"
        assert parsed["timestamp"]

    def test_sse_event_sources(self):
        from app.api.chat import _sse_event

        result = _sse_event(
            "sources",
            "1. **doc.pdf**",
            source_chunks=[_source_chunk().model_dump(mode="json")],
        )
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "sources"
        assert parsed["source_chunks"][0]["doc_name"] == "notes.md"

    def test_sse_event_done(self):
        from app.api.chat import _sse_event

        result = _sse_event("done", "")
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "done"
        assert parsed["timestamp"]

    def test_sse_event_answer_done(self):
        from app.api.chat import _sse_event

        result = _sse_event("answer-done", "")
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "answer-done"
        assert parsed["timestamp"]

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
                    json={
                        "question": "valid",
                        "stream": False,
                        "top_k": 10,
                        "use_agent": False,
                    },
                )
                assert response.status_code == 200

    def test_invalid_max_rounds_returns_422(self):
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "valid",
                "stream": False,
                "use_agent": True,
                "max_rounds": 0,
            },
        )
        assert response.status_code == 422
