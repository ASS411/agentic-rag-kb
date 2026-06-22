"""Integration tests for AgentLoop.run() state machine (task 5.5).

Covers: full pipeline flow, max_rounds limit, early stop on sufficient check,
SSE event shape, error fallback.  Updated for parent-child retrieval (task fix).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.agent import AgentLoop, AgentStep
from app.core.llm import LLMError


def _make_patched_agent():
    """Create an AgentLoop with all external dependencies mocked."""
    agent = AgentLoop()
    agent._llm.generate = AsyncMock(return_value='["q1","q2","q3"]')

    async def _fake_stream(*args, **kwargs):
        for token in ["Hello ", "World"]:
            yield token
    agent._llm.generate_stream = _fake_stream
    return agent


def _mock_retriever(monkeypatch, total_recalled=6, dedup_count=4):
    """Patch Retriever to return controlled RetrievalResult via
    retrieve_with_parent_lookup (the only method the new agent loop calls)."""
    from app.core import retriever as ret_mod
    from app.models.search import SearchChunk

    def _fake_chunk(cid):
        return SearchChunk(
            chunk_id=cid, content=f"Content {cid}", score=0.9,
            doc_id="d1", doc_name="doc.pdf", doc_type="txt",
            page=1, chunk_index=int(cid[1:]) if cid[1:].isdigit() else 0,
            metadata={},
        )

    class FakeRetriever:
        async def retrieve_with_parent_lookup(
            self, queries, *, top_k_recall=20, top_k_rerank=5, rerank=True,
            hybrid=False,
        ):
            from app.core.retriever import RetrievalResult
            return RetrievalResult(
                chunks=[_fake_chunk(f"c{i}") for i in range(dedup_count)],
                total_recalled=total_recalled,
                reranked=True,
                parent_lookup=True,
                hybrid=hybrid and True,
            )

    monkeypatch.setattr(ret_mod, "Retriever", FakeRetriever)


class TestAgentLoopRun:
    @pytest.mark.asyncio
    async def test_emits_rewrite_step_first(self):
        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["rewritten q"]'

        events = []
        async for evt_json in agent.run("test question?"):
            events.append(evt_json)
            if len(events) >= 2:
                break  # only need first events

        # First event should be agent-step with step=rewrite
        import json
        first = json.loads(events[0])
        assert first["type"] == "agent-step"
        assert first["step"] == "rewrite"

    @pytest.mark.asyncio
    async def test_sufficient_breaks_early(self, monkeypatch):
        """When quality check returns sufficient=True, the loop exits after
        one round (no replan)."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        # First LLM call: rewrite
        agent._llm.generate.return_value = '["q1"]'

        # Override _quality_check to return sufficient immediately
        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="good")
        agent._quality_check = _fake_check

        events = []
        async for evt_json in agent.run("test?"):
            events.append(evt_json)

        # Should NOT have a replan step
        import json
        steps = [json.loads(e)["step"] for e in events
                 if json.loads(e).get("type") == "agent-step"]
        assert "replan" not in steps

    @pytest.mark.asyncio
    async def test_insufficient_triggers_replan(self, monkeypatch):
        """When check returns insufficient, a replan step is emitted."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(
                sufficient=False, reasoning="missing", gap="缺少信息")
        agent._quality_check = _fake_check

        events = []
        async for evt_json in agent.run("test?"):
            events.append(evt_json)

        import json
        steps = [json.loads(e)["step"] for e in events
                 if json.loads(e).get("type") == "agent-step"]
        assert "replan" in steps

    @pytest.mark.asyncio
    async def test_request_max_rounds_limits_replan(self, monkeypatch):
        """A per-request max_rounds override should cap the search loop."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(
                sufficient=False,
                reasoning="missing",
                gap="缺少信息",
            )

        agent._quality_check = _fake_check

        import json

        events = [json.loads(evt) async for evt in agent.run("test?", max_rounds=1)]
        steps = [
            event["step"]
            for event in events
            if event.get("type") == "agent-step"
        ]
        done_evt = next(event for event in events if event["type"] == "done")

        assert "replan" not in steps
        assert done_evt["total_rounds"] == 1

    @pytest.mark.asyncio
    async def test_generates_answer_chunks(self, monkeypatch):
        """The stream should contain answer-chunk events before done."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="ok")
        agent._quality_check = _fake_check

        events = []
        async for evt_json in agent.run("test?"):
            events.append(evt_json)

        import json
        types = [json.loads(e)["type"] for e in events]
        assert "answer-chunk" in types
        assert "answer-done" in types
        assert "sources" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_sources_come_from_parent_child_retrieval(self, monkeypatch):
        """Final sources should reflect parent chunks returned by
        retrieve_with_parent_lookup, sorted by score and capped at
        top_k_rerank."""
        from app.config import settings
        from app.core import retriever as ret_mod
        from app.models.search import SearchChunk

        original_top_k = settings.agent.top_k_rerank
        settings.agent.top_k_rerank = 2
        try:
            # Create a custom mock retriever that returns chunks with
            # known scores so we can verify ordering.
            class CustomFakeRetriever:
                async def retrieve_with_parent_lookup(
                    self, queries, *, top_k_recall=20, top_k_rerank=5,
                    rerank=True, hybrid=False,
                ):
                    from app.core.retriever import RetrievalResult
                    chunks = [
                        SearchChunk(
                            chunk_id=f"p{i}", content=f"Content p{i}",
                            score=0.5 + i * 0.1,
                            doc_id="d1", doc_name="doc.pdf", doc_type="txt",
                            page=1, chunk_index=i, metadata={},
                        )
                        for i in range(4)
                    ]
                    return RetrievalResult(
                        chunks=chunks,
                        total_recalled=10,
                        reranked=True,
                        parent_lookup=True,
                    )

            monkeypatch.setattr(ret_mod, "Retriever", CustomFakeRetriever)

            agent = _make_patched_agent()
            agent._llm.generate.return_value = '["q1"]'

            async def _fake_check(question, context_pool, **kw):
                from app.core.agent import CheckResult
                return CheckResult(sufficient=True, reasoning="ok")

            agent._quality_check = _fake_check

            import json

            events = [json.loads(evt) async for evt in agent.run("test?")]
            sources_evt = next(
                evt for evt in events if evt["type"] == "sources")

            source_chunks = sources_evt["source_chunks"]
            # With top_k_rerank=2, should keep only top 2 by score:
            # p3(0.8), p2(0.7)
            assert len(source_chunks) == 2
            assert [chunk["chunk_id"] for chunk in source_chunks] == [
                "p3", "p2"]
            assert source_chunks[0]["score"] > source_chunks[1]["score"]
        finally:
            settings.agent.top_k_rerank = original_top_k

    @pytest.mark.asyncio
    async def test_search_sse_shows_parent_child_info(self, monkeypatch):
        """Search step SSE events should include parent_lookup flag and
        parent_chunks count."""
        _mock_retriever(monkeypatch, total_recalled=6, dedup_count=4)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="ok")

        agent._quality_check = _fake_check

        import json

        events = [json.loads(evt) async for evt in agent.run("test?")]

        # Find the second search event (the result one, not the start one)
        search_events = [
            evt for evt in events
            if evt.get("type") == "agent-step" and evt.get("step") == "search"
        ]
        # At least one search event should have parent_lookup info
        result_events = [
            evt for evt in search_events
            if "parent_lookup" in evt
        ]
        assert result_events, "Expected search event with parent_lookup field"
        result_evt = result_events[0]
        assert result_evt["parent_lookup"] is True
        assert result_evt["parent_chunks"] == 4

    @pytest.mark.asyncio
    async def test_done_event_contains_summary_fields(self, monkeypatch):
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="ok")

        agent._quality_check = _fake_check

        import json

        events = [json.loads(evt) async for evt in agent.run("test?")]
        done_evt = next(evt for evt in events if evt["type"] == "done")
        assert done_evt["timestamp"]
        assert done_evt["conversation_id"] is None
        assert done_evt["total_rounds"] == 1
        assert done_evt["chunks_used"] > 0

    @pytest.mark.asyncio
    async def test_event_json_format(self, monkeypatch):
        """Each yield is valid JSON with expected keys."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="ok")
        agent._quality_check = _fake_check

        import json
        async for evt_json in agent.run("test?"):
            evt = json.loads(evt_json)
            assert "type" in evt
            # agent-step events must have step, message
            if evt["type"] == "agent-step":
                assert "step" in evt
                assert "message" in evt
            elif evt["type"] == "answer-chunk":
                assert "content" in evt
            elif evt["type"] == "answer-done":
                assert "timestamp" in evt
            elif evt["type"] == "sources":
                assert "content" in evt
                assert "chunk_ids" in evt
            elif evt["type"] == "done":
                assert "timestamp" in evt
                assert "total_rounds" in evt
                assert "chunks_used" in evt
                pass  # minimal
            elif evt["type"] == "error":
                pass
            else:
                pytest.fail(f"Unknown event type: {evt['type']}")

    @pytest.mark.asyncio
    async def test_all_agent_sse_events_include_iso_timestamp(self, monkeypatch):
        """Every Agent SSE event should carry a parseable timestamp."""
        _mock_retriever(monkeypatch)

        agent = _make_patched_agent()
        agent._llm.generate.return_value = '["q1"]'

        async def _fake_check(question, context_pool, **kw):
            from app.core.agent import CheckResult
            return CheckResult(sufficient=True, reasoning="ok")

        agent._quality_check = _fake_check

        import json

        events = [json.loads(evt) async for evt in agent.run("test?")]

        assert events
        for event in events:
            timestamp = event.get("timestamp")
            assert timestamp, event
            datetime.fromisoformat(timestamp)
