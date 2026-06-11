"""Tests for agent prompt templates + AgentLoop (task 3.4).

Covers: build_rewrite_messages, build_replan_messages, _parse_query_list,
AgentLoop._rewrite_query, AgentLoop._replan, LLM error fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.agent import AgentLoop, _parse_query_list
from app.core.llm import LLMError
from app.core.prompts import build_rewrite_messages, build_replan_messages


def _make_patched_agent(llm_return):
    agent = AgentLoop()
    agent._llm = MagicMock()
    agent._llm.generate = AsyncMock(return_value=llm_return)
    return agent


class TestRewritePrompt:
    def test_has_system_and_user(self):
        msgs = build_rewrite_messages("test?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_question(self):
        msgs = build_rewrite_messages("什么是RAG?")
        assert "什么是RAG?" in msgs[1]["content"]

    def test_contains_json_instruction(self):
        msgs = build_rewrite_messages("anything")
        full = msgs[0]["content"] + msgs[1]["content"]
        assert "JSON" in full
        assert "[" in msgs[1]["content"]


class TestReplanPrompt:
    def test_has_system_and_user(self):
        msgs = build_replan_messages("q", "missing info")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_question_and_gap(self):
        msgs = build_replan_messages("什么是RAG?", "缺少架构信息")
        content = msgs[1]["content"]
        assert "什么是RAG?" in content
        assert "缺少架构信息" in content

    def test_contains_json_instruction(self):
        msgs = build_replan_messages("q", "gap")
        full = msgs[0]["content"] + msgs[1]["content"]
        assert "JSON" in full


class TestParseQueryList:
    def test_plain_json_array(self):
        assert _parse_query_list('["a","b","c"]') == ["a", "b", "c"]

    def test_fenced_json(self):
        raw = '```json\n["x","y"]\n```'
        assert _parse_query_list(raw) == ["x", "y"]

    def test_single_quotes(self):
        assert _parse_query_list("['a','b']") == ["a", "b"]

    def test_bare_lines(self):
        assert _parse_query_list("line1\nline2\nline3") == ["line1", "line2", "line3"]

    def test_empty_string(self):
        result = _parse_query_list("")
        assert result == []

    def test_unparseable_returns_fallback(self):
        result = _parse_query_list("  ")
        assert result == []


class TestAgentRewriteQuery:
    @pytest.mark.asyncio
    async def test_returns_parsed_queries(self):
        agent = _make_patched_agent('["q1","q2","q3","q4"]')
        result = await agent._rewrite_query("original?")
        assert result == ["q1", "q2", "q3", "q4"]

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_original(self):
        agent = _make_patched_agent("")
        agent._llm.generate.side_effect = LLMError("down")
        result = await agent._rewrite_query("fallback?")
        assert result == ["fallback?"]

    @pytest.mark.asyncio
    async def test_empty_parse_falls_back_to_original(self):
        agent = _make_patched_agent("   ")
        result = await agent._rewrite_query("original?")
        assert result == ["original?"]


class TestAgentReplanQuery:
    @pytest.mark.asyncio
    async def test_returns_supplementary_queries(self):
        agent = _make_patched_agent('["sup1","sup2"]')
        result = await agent._replan("q", "missing X")
        assert result == ["sup1", "sup2"]

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_original(self):
        agent = _make_patched_agent("")
        agent._llm.generate.side_effect = LLMError("timeout")
        result = await agent._replan("original", "gap")
        assert result == ["original"]

    @pytest.mark.asyncio
    async def test_empty_parse_falls_back_to_original(self):
        agent = _make_patched_agent("  ")
        result = await agent._replan("original", "gap")
        assert result == ["original"]
