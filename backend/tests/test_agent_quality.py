"""Tests for quality check (LLM-as-Judge) — task 4.4.

Covers:
- CheckResult model construction
- _parse_check_result: JSON object, fences, single quotes, fallback
- build_quality_check_messages structure
- AgentLoop._quality_check: sufficient, insufficient, empty ctx, LLM error
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.agent import AgentLoop, CheckResult, _parse_check_result
from app.core.llm import LLMError
from app.core.prompts import build_quality_check_messages
from app.models.search import SearchChunk


def _make_chunk(content, doc_name="test.pdf"):
    return SearchChunk(
        chunk_id="c0", content=content, score=0.9, doc_id="d1",
        doc_name=doc_name, doc_type="txt", page=1, chunk_index=0, metadata={},
    )


def _make_patched_agent(llm_return):
    agent = AgentLoop()
    agent._llm = MagicMock()
    agent._llm.generate = AsyncMock(return_value=llm_return)
    return agent


class TestCheckResult:
    def test_sufficient_without_gap(self):
        cr = CheckResult(sufficient=True, reasoning="good")
        assert cr.sufficient is True
        assert cr.reasoning == "good"
        assert cr.gap is None

    def test_insufficient_with_gap(self):
        cr = CheckResult(
            sufficient=False,
            reasoning="missing info",
            gap="缺少架构信息",
        )
        assert cr.sufficient is False
        assert cr.gap == "缺少架构信息"

    def test_gap_none_when_not_provided(self):
        cr = CheckResult(sufficient=True, reasoning="ok")
        assert cr.gap is None


class TestParseCheckResult:
    def test_plain_json(self):
        d = _parse_check_result('{"sufficient":true,"reasoning":"ok","gap":null}')
        assert d["sufficient"] is True
        assert d["reasoning"] == "ok"
        assert d["gap"] is None

    def test_fenced_json(self):
        raw = '```json\n{"sufficient":false,"reasoning":"no","gap":"missing"}\n```'
        d = _parse_check_result(raw)
        assert d["sufficient"] is False
        assert d["gap"] == "missing"

    def test_single_quotes(self):
        d = _parse_check_result("{'sufficient':true,'reasoning':'yes','gap':null}")
        assert d["sufficient"] is True

    def test_unparseable_fallback(self):
        d = _parse_check_result("not valid json at all")
        assert d["sufficient"] is False
        assert "Failed to parse" in d["reasoning"]


class TestQualityCheckMessages:
    def test_has_system_and_user(self):
        ctx = [_make_chunk("RAG is..."), _make_chunk("retrieval means...")]
        msgs = build_quality_check_messages("What is RAG?", ctx)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_contains_question_and_chunks(self):
        ctx = [_make_chunk("RAG is retrieval augmented generation")]
        msgs = build_quality_check_messages("What is RAG?", ctx)
        user = msgs[1]["content"]
        assert "What is RAG?" in user
        assert "retrieval augmented generation" in user


class TestAgentQualityCheck:
    @pytest.mark.asyncio
    async def test_sufficient_result(self):
        agent = _make_patched_agent(
            '{"sufficient":true,"reasoning":"covers everything","gap":null}'
        )
        ctx = [_make_chunk("RAG is a technique")]
        result = await agent._quality_check("What is RAG?", ctx)
        assert result.sufficient is True
        assert "covers" in result.reasoning
        assert result.gap is None

    @pytest.mark.asyncio
    async def test_insufficient_result(self):
        agent = _make_patched_agent(
            '{"sufficient":false,"reasoning":"missing details","gap":"no architecture info"}'
        )
        ctx = [_make_chunk("RAG is a thing")]
        result = await agent._quality_check("RAG architecture?", ctx)
        assert result.sufficient is False
        assert result.gap == "no architecture info"

    @pytest.mark.asyncio
    async def test_empty_context_pool(self):
        agent = _make_patched_agent("")
        result = await agent._quality_check("q", [])
        assert result.sufficient is False
        assert "No context" in result.reasoning

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self):
        agent = _make_patched_agent("")
        agent._llm.generate.side_effect = LLMError("timeout")
        ctx = [_make_chunk("RAG")]
        result = await agent._quality_check("q", ctx)
        assert result.sufficient is False
        assert "LLM call failed" in result.reasoning
