"""Tests for the RAG prompt templates (task 5.2).

Covers:
- Context formatting with search chunks
- Source list formatting
- RAGPromptBuilder: build() and build_with_sources()
- Message structure validation
- Custom system prompt and answer template
- Edge cases: empty chunks, many chunks, long content
"""

from __future__ import annotations

import pytest

from app.core.prompts import (
    RAGPromptBuilder,
    format_context,
    format_sources,
)
from app.models.search import SearchChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    content: str,
    doc_name: str = "test.pdf",
    page: int = 1,
    score: float = 0.95,
    chunk_index: int = 0,
    doc_id: str = "abc123",
    doc_type: str = "pdf",
    chunk_id: str | None = None,
) -> SearchChunk:
    """Build a minimal SearchChunk for testing."""
    return SearchChunk(
        chunk_id=chunk_id or f"{doc_id}_chunk_{chunk_index}",
        content=content,
        score=score,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type=doc_type,
        page=page,
        chunk_index=chunk_index,
        metadata={},
    )


def _make_chunks(n: int = 3) -> list[SearchChunk]:
    """Build a list of test chunks."""
    return [
        _make_chunk(
            content=f"Chunk {i} content about knowledge graphs.",
            doc_name=f"doc_{i}.pdf",
            page=i + 1,
            chunk_index=i,
            doc_id=f"doc_{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    """Tests for the format_context helper."""

    def test_single_chunk(self):
        chunks = [_make_chunk("知识图谱是一种语义网络。")]
        result = format_context(chunks)
        assert "[来源 1]" in result
        assert "知识图谱是一种语义网络" in result
        assert "test.pdf" in result

    def test_multiple_chunks(self):
        chunks = _make_chunks(3)
        result = format_context(chunks)
        assert "[来源 1]" in result
        assert "[来源 2]" in result
        assert "[来源 3]" in result
        # Should include separators between chunks
        assert "---" in result

    def test_empty_chunks_returns_placeholder(self):
        result = format_context([])
        assert "无" in result

    def test_max_chunks_truncates(self):
        chunks = _make_chunks(10)
        result = format_context(chunks, max_chunks=3)
        assert "[来源 1]" in result
        assert "[来源 3]" in result
        assert "[来源 4]" not in result

    def test_long_content_truncated(self):
        long_text = "A" * 2000
        chunks = [_make_chunk(content=long_text)]
        result = format_context(chunks, max_chars_per_chunk=100)
        # Truncated + "…"
        assert len(result) < len(long_text) + 200  # generous margin
        assert "…" in result

    def test_show_score_includes_score(self):
        chunks = [_make_chunk("test", score=0.8742)]
        result = format_context(chunks, show_score=True)
        assert "0.8742" in result

    def test_no_score_by_default(self):
        chunks = [_make_chunk("test", score=0.9999)]
        result = format_context(chunks)
        assert "0.9999" not in result

    def test_page_info_included(self):
        chunks = [_make_chunk("test", page=7)]
        result = format_context(chunks)
        assert "第7页" in result

    def test_page_zero_omitted(self):
        chunks = [_make_chunk("test", page=0)]
        result = format_context(chunks)
        assert "页" not in result  # no page info when page=0

    def test_doc_name_included(self):
        chunks = [_make_chunk("test", doc_name="研究报告.pdf")]
        result = format_context(chunks)
        assert "研究报告.pdf" in result


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------


class TestFormatSources:
    """Tests for the format_sources helper."""

    def test_single_source(self):
        chunks = [_make_chunk("x", doc_name="a.pdf", page=3)]
        result = format_sources(chunks)
        assert "1." in result
        assert "a.pdf" in result
        assert "第3页" in result

    def test_multiple_sources(self):
        chunks = [
            _make_chunk("a", doc_name="x.pdf", page=1),
            _make_chunk("b", doc_name="y.pdf", page=2),
            _make_chunk("c", doc_name="z.pdf", page=3),
        ]
        result = format_sources(chunks)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "x.pdf" in lines[0]
        assert "y.pdf" in lines[1]
        assert "z.pdf" in lines[2]

    def test_empty_returns_placeholder(self):
        result = format_sources([])
        assert "无" in result

    def test_page_zero_omitted(self):
        chunks = [_make_chunk("x", page=0)]
        result = format_sources(chunks)
        assert "页" not in result

    def test_respects_max_chunks(self):
        chunks = _make_chunks(10)
        result = format_sources(chunks, max_chunks=3)
        lines = result.split("\n")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# RAGPromptBuilder.build()
# ---------------------------------------------------------------------------


class TestRAGPromptBuilder:
    """Tests for the RAGPromptBuilder class."""

    def test_build_returns_messages_list(self):
        builder = RAGPromptBuilder()
        chunks = _make_chunks(2)
        messages = builder.build(chunks, "什么是知识图谱？")

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_has_content(self):
        builder = RAGPromptBuilder()
        messages = builder.build(_make_chunks(1), "test question")
        assert len(messages[0]["content"]) > 10

    def test_user_message_contains_context_and_question(self):
        builder = RAGPromptBuilder()
        chunks = _make_chunks(2)
        messages = builder.build(chunks, "什么是知识图谱？")

        user_content = messages[1]["content"]
        assert "什么是知识图谱？" in user_content
        assert "Chunk 0" in user_content
        assert "Chunk 1" in user_content

    def test_empty_chunks_still_produces_valid_messages(self):
        builder = RAGPromptBuilder()
        messages = builder.build([], "什么是知识图谱？")
        assert len(messages) == 2
        assert "什么是知识图谱？" in messages[1]["content"]

    def test_custom_system_prompt(self):
        custom = "你是一个测试助手。"
        builder = RAGPromptBuilder(system_prompt=custom)
        messages = builder.build(_make_chunks(1), "q")
        assert messages[0]["content"] == custom

    def test_custom_answer_template(self):
        custom_template = "上下文: {context}\n问题: {question}\n请回答。"
        builder = RAGPromptBuilder(answer_template=custom_template)
        chunks = [_make_chunk("知识图谱定义")]
        messages = builder.build(chunks, "测试问题")

        user_content = messages[1]["content"]
        assert "上下文:" in user_content
        assert "问题:" in user_content
        assert "知识图谱定义" in user_content
        assert "测试问题" in user_content

    def test_per_call_max_chunks_override(self):
        builder = RAGPromptBuilder(max_chunks=10)
        chunks = _make_chunks(10)
        messages = builder.build(chunks, "q", max_chunks=3)
        # Only 3 sources should appear
        user_content = messages[1]["content"]
        assert "[来源 3]" in user_content
        assert "[来源 4]" not in user_content

    def test_per_call_max_chars_override(self):
        builder = RAGPromptBuilder(max_chars_per_chunk=2000)
        long_text = "X" * 2000
        chunks = [_make_chunk(content=long_text)]
        messages = builder.build(chunks, "q", max_chars_per_chunk=100)
        user_content = messages[1]["content"]
        assert "…" in user_content

    def test_system_prompt_property(self):
        builder = RAGPromptBuilder()
        assert len(builder.system_prompt) > 10

    def test_answer_template_property(self):
        builder = RAGPromptBuilder()
        assert "{context}" in builder.answer_template
        assert "{question}" in builder.answer_template

    def test_default_max_chunks(self):
        builder = RAGPromptBuilder(max_chunks=5)
        chunks = _make_chunks(10)
        messages = builder.build(chunks, "q")
        user_content = messages[1]["content"]
        assert "[来源 5]" in user_content
        assert "[来源 6]" not in user_content

    def test_builder_is_reusable(self):
        builder = RAGPromptBuilder()
        m1 = builder.build(_make_chunks(1), "q1")
        m2 = builder.build(_make_chunks(1), "q2")
        assert m1[0]["content"] == m2[0]["content"]  # system prompt reused
        assert "q1" in m1[1]["content"]
        assert "q2" in m2[1]["content"]

    def test_chinese_content_preserved(self):
        builder = RAGPromptBuilder()
        chunks = [_make_chunk("中文内容：知识图谱是语义网络的一种。")]
        messages = builder.build(chunks, "什么是知识图谱？")
        assert "中文内容" in messages[1]["content"]
        assert "语义网络" in messages[1]["content"]


# ---------------------------------------------------------------------------
# RAGPromptBuilder.build_with_sources()
# ---------------------------------------------------------------------------


class TestBuildWithSources:
    """Tests for build_with_sources()."""

    def test_returns_tuple(self):
        builder = RAGPromptBuilder()
        result = builder.build_with_sources(_make_chunks(2), "question")
        assert isinstance(result, tuple)
        assert len(result) == 2
        messages, sources = result
        assert isinstance(messages, list)
        assert isinstance(sources, str)

    def test_sources_string_not_empty(self):
        builder = RAGPromptBuilder()
        _, sources = builder.build_with_sources(_make_chunks(3), "q")
        assert len(sources) > 0
        # Should have 3 lines
        assert len(sources.split("\n")) == 3

    def test_sources_consistent_with_messages(self):
        builder = RAGPromptBuilder()
        chunks = _make_chunks(3)
        messages, sources = builder.build_with_sources(chunks, "test")

        # Sources should reference the same docs as messages
        for chunk in chunks:
            assert chunk.doc_name in sources

    def test_empty_chunks_handled(self):
        builder = RAGPromptBuilder()
        messages, sources = builder.build_with_sources([], "q")
        assert len(messages) == 2
        assert "无" in sources
