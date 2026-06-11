"""Integration tests for AgentLoop.run() state machine (task 5.5).

Covers: full pipeline flow, max_rounds limit, early stop on sufficient check,
SSE event shape, error fallback.
"""

from __future__ import annotations

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
    """Patch Retriever to return a controlled RetrievalResult."""
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
        async def retrieve(self, queries, top_k_recall=20, rerank=False):
            from app.core.retriever import RetrievalResult
            return RetrievalResult(
                chunks=[_fake_chunk(f"c{i}") for i in range(dedup_count)],
                total_recalled=total_recalled,
                reranked=False,
            )

    monkeypatch.setattr(ret_mod, "Retriever", FakeRetriever)


def _mock_reranker(monkeypatch, top_k=3):
    """Patch Reranker to return a subset of chunks."""
    from app.core import reranker as rnk_mod
    from app.models.document import DocType

    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, question, chunks, top_k=top_k):
            # Return the last top_k chunks with fake rerank scores so the
            # test can detect whether generation uses reranked output.
            for c in chunks:
                c.metadata["rerank_score"] = 0.1
            result = chunks[-top_k:]
            for idx, c in enumerate(result):
                c.metadata["rerank_score"] = 0.9 - idx * 0.1
            return result

        def compute_similarity(self, pairs):
            return [0.9] * len(pairs)

    monkeypatch.setattr(rnk_mod, "Reranker", FakeReranker)


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
        _mock_reranker(monkeypatch)

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
        _mock_reranker(monkeypatch)

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
    async def test_generates_answer_chunks(self, monkeypatch):
        """The stream should contain answer-chunk events before done."""
        _mock_retriever(monkeypatch)
        _mock_reranker(monkeypatch)

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
        assert "sources" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_sources_come_from_reranked_top_pool(self, monkeypatch):
        """Final sources should reflect reranked chunks, not raw recall order."""
        from app.config import settings

        original_top_k = settings.agent.top_k_rerank
        settings.agent.top_k_rerank = 2
        try:
            _mock_retriever(monkeypatch, dedup_count=4)
            _mock_reranker(monkeypatch, top_k=2)

            agent = _make_patched_agent()
            agent._llm.generate.return_value = '["q1"]'

            async def _fake_check(question, context_pool, **kw):
                from app.core.agent import CheckResult
                return CheckResult(sufficient=True, reasoning="ok")

            agent._quality_check = _fake_check

            import json

            events = [json.loads(evt) async for evt in agent.run("test?")]
            sources_evt = next(evt for evt in events if evt["type"] == "sources")

            source_chunks = sources_evt["source_chunks"]
            assert [chunk["chunk_id"] for chunk in source_chunks] == ["c2", "c3"]
            assert source_chunks[0]["score"] > source_chunks[1]["score"]
        finally:
            settings.agent.top_k_rerank = original_top_k

    @pytest.mark.asyncio
    async def test_event_json_format(self, monkeypatch):
        """Each yield is valid JSON with expected keys."""
        _mock_retriever(monkeypatch)
        _mock_reranker(monkeypatch)

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
            elif evt["type"] == "sources":
                assert "content" in evt
                assert "chunk_ids" in evt
            elif evt["type"] == "done":
                pass  # minimal
            elif evt["type"] == "error":
                pass
            else:
                pytest.fail(f"Unknown event type: {evt['type']}")
