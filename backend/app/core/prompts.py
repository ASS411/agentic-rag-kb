"""RAG prompt templates — system prompt + context assembly (task 5.2).

Provides structured prompt builders that format retrieved chunks into a
context block, combine them with a system prompt and user query, and
return a messages list ready for ``LLMClient.generate()`` and
``LLMClient.generate_stream()``.

Usage::

    from app.core.prompts import RAGPromptBuilder
    from app.core.llm import LLMClient

    builder = RAGPromptBuilder()
    chunks = search_response.results   # list[SearchChunk]

    messages = builder.build(chunks, "什么是知识图谱？")
    answer = await LLMClient().generate(messages)
"""

from __future__ import annotations

from app.models.search import SearchChunk

# ---------------------------------------------------------------------------
# Default prompt texts (Chinese — project-facing language)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。"
    "你的回答必须严格基于用户提供的上下文信息，不得凭空编造。"
    "如果上下文中没有足够的信息来回答问题，请诚实地说明"
    "「根据已有资料，无法回答此问题」，"
    "并建议用户补充相关文档。"
)

_DEFAULT_ANSWER_TEMPLATE = """## 上下文

{context}

## 用户问题

{question}

## 回答要求

1. 严格基于上述上下文回答，不编造信息
2. 如果上下文中没有相关信息，诚实说明「根据已有资料，无法回答此问题」
3. 引用上下文时标注来源编号，格式：[来源 N]
4. 答案应结构化，适当使用标题和列表提高可读性
5. 在答案末尾列出所有引用来源（来源编号 + 文档名 + 页码）

## 回答"""


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(
    chunks: list[SearchChunk],
    *,
    max_chunks: int = 20,
    max_chars_per_chunk: int = 1200,
    show_score: bool = False,
) -> str:
    """Format a list of search-result chunks into a single context string.

    Each chunk is rendered as::

        [来源 N] (文档: xxx.pdf, 第3页)
        内容...

    Parameters
    ----------
    chunks:
        Retrieved chunks from ``POST /api/v1/search``.
    max_chunks:
        Maximum number of chunks to include (safeguard against context
        overflow).  Chunks beyond this are silently dropped.
    max_chars_per_chunk:
        Truncate each chunk's content to this many characters if longer.
    show_score:
        If True, append the similarity score to each chunk header.

    Returns
    -------
    str
        Assembled context string suitable for insertion into the
        answer template.
    """
    if not chunks:
        return "（无可用上下文）"

    parts: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        source_id = i + 1  # 1-based for human-readable output
        content = chunk.content or ""

        # Truncate long chunks
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "…"

        header = f"[来源 {source_id}] (文档: {chunk.doc_name}"
        if chunk.page and chunk.page > 0:
            header += f", 第{chunk.page}页"
        if show_score:
            header += f", 相关度: {chunk.score:.4f}"
        header += ")"

        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Source list builder
# ---------------------------------------------------------------------------

def format_sources(
    chunks: list[SearchChunk],
    *,
    max_chunks: int = 20,
) -> str:
    """Build a human-readable source list from retrieved chunks.

    Parameters
    ----------
    chunks:
        The same chunks passed to ``format_context``.
    max_chunks:
        Must match the value used in ``format_context``.

    Returns
    -------
    str
        A Markdown-formatted source list, e.g.::

            1. knowledge_graph.pdf — 第3页
            2. 知识图谱入门.md — 第1页
    """
    if not chunks:
        return "（无引用来源）"

    lines: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        source_id = i + 1
        line = f"{source_id}. **{chunk.doc_name}**"
        if chunk.page and chunk.page > 0:
            line += f" — 第{chunk.page}页"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class RAGPromptBuilder:
    """Assembles RAG prompt messages from chunks and a user query.

    The builder is stateless; you can reuse a single instance across
    requests.  System prompt and answer template are customisable via
    constructor arguments.

    Usage::

        builder = RAGPromptBuilder()
        messages = builder.build(search_response.results, user_question)
        answer = await llm.generate(messages)
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        answer_template: str | None = None,
        max_chunks: int = 20,
        max_chars_per_chunk: int = 1200,
        show_score: bool = False,
    ) -> None:
        """Initialise the prompt builder.

        Parameters
        ----------
        system_prompt:
            Override the default system prompt.  Use ``None`` to keep the
            built-in default.
        answer_template:
            Override the answer template.  Must contain ``{context}`` and
            ``{question}`` placeholders.
        max_chunks:
            Default maximum chunks per context (can be overridden per
            ``build()`` call).
        max_chars_per_chunk:
            Default truncation length per chunk (can be overridden per call).
        show_score:
            Whether to include similarity scores in the context header.
        """
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._answer_template = answer_template or _DEFAULT_ANSWER_TEMPLATE
        self._max_chunks = max_chunks
        self._max_chars_per_chunk = max_chars_per_chunk
        self._show_score = show_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: list[SearchChunk],
        question: str,
        *,
        max_chunks: int | None = None,
        max_chars_per_chunk: int | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages list for a RAG chat completion.

        Parameters
        ----------
        chunks:
            Retrieved chunks from ``POST /api/v1/search``.  An empty list
            is valid — the LLM will be told that no context is available.
        question:
            The user's natural-language question.
        max_chunks:
            Override the instance default for this call.
        max_chars_per_chunk:
            Override the instance default for this call.

        Returns
        -------
        list[dict[str, str]]
            Messages in OpenAI chat format, ready for
            ``LLMClient.generate()`` or ``generate_stream()``::

                [
                    {"role": "system", "content": "…"},
                    {"role": "user", "content": "…"},
                ]
        """
        n_chunks = max_chunks if max_chunks is not None else self._max_chunks
        n_chars = (
            max_chars_per_chunk
            if max_chars_per_chunk is not None
            else self._max_chars_per_chunk
        )

        context_text = format_context(
            chunks,
            max_chunks=n_chunks,
            max_chars_per_chunk=n_chars,
            show_score=self._show_score,
        )

        user_message = self._answer_template.format(
            context=context_text,
            question=question,
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

    def build_with_sources(
        self,
        chunks: list[SearchChunk],
        question: str,
        *,
        max_chunks: int | None = None,
        max_chars_per_chunk: int | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        """Like ``build()`` but also returns a pre-formatted source list.

        The source list can be sent to the frontend alongside the answer
        for the source-panel display.

        Returns
        -------
        tuple[list[dict[str, str]], str]
            (messages, source_list_string)
        """
        n_chunks = max_chunks if max_chunks is not None else self._max_chunks
        n_chars = (
            max_chars_per_chunk
            if max_chars_per_chunk is not None
            else self._max_chars_per_chunk
        )

        messages = self.build(
            chunks,
            question,
            max_chunks=n_chunks,
            max_chars_per_chunk=n_chars,
        )
        sources = format_sources(chunks, max_chunks=n_chunks)

        return messages, sources

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._system_prompt

    @property
    def answer_template(self) -> str:
        """Return the active answer template."""
        return self._answer_template
