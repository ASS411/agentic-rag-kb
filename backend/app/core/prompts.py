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


# ---------------------------------------------------------------------------
# Agent prompt templates (Phase 2 — module 3.3)
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = (
    "你是一个专业的检索查询改写助手。"
    "你的任务是将用户的原始问题改写为 3-5 条不同角度的检索查询，"
    "以便从知识库中全面召回相关信息。"
    "\n\n"
    "要求：\n"
    "1. 每条查询应从不同角度或使用不同关键词来表述\n"
    "2. 包含原问题的核心语义，但避免简单重复\n"
    "3. 考虑可能的同义词、缩写、中英文对照等变化\n"
    "4. 查询应为自然语言短语或问句，不要过长\n"
    "5. 必须严格输出 JSON 数组格式"
)

_REWRITE_USER_TEMPLATE = """用户原始问题：
{question}

请为上述问题生成 3-5 条不同角度的检索查询，以 JSON 字符串数组格式输出：
```json
["查询1", "查询2", "查询3"]
```

只输出 JSON 数组，不要包含其他文字。"""

_REPLAN_SYSTEM_PROMPT = (
    "你是一个专业的检索策略补充助手。"
    "当已有检索结果不足以回答用户问题时，"
    "你需要分析缺失的信息并生成 2-3 条针对性的补充检索查询。"
    "\n\n"
    "要求：\n"
    "1. 每条查询针对已知的信息缺口\n"
    "2. 用不同的关键词或角度来覆盖缺失内容\n"
    "3. 查询应简洁、精准、可直接用于向量检索\n"
    "4. 必须严格输出 JSON 数组格式"
)

_REPLAN_USER_TEMPLATE = """用户原始问题：
{question}

当前检索覆盖的缺口（缺失或不充分的信息）：
{gap_description}

请为补充上述缺口生成 2-3 条检索查询，以 JSON 字符串数组格式输出：
```json
["补充查询1", "补充查询2"]
```

只输出 JSON 数组，不要包含其他文字。"""


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    """Build messages for the query-rewrite LLM call.

    Parameters
    ----------
    question:
        The user's original natural-language question.

    Returns
    -------
    list[dict[str, str]]
        Messages list ready for ``LLMClient.generate()``.
    """
    return [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": _REWRITE_USER_TEMPLATE.format(question=question)},
    ]


def build_replan_messages(question: str, gap_description: str) -> list[dict[str, str]]:
    """Build messages for the replan (gap-filling) LLM call.

    Parameters
    ----------
    question:
        The user's original natural-language question.
    gap_description:
        Human-readable description of what information is missing or
        insufficient in the current context pool.

    Returns
    -------
    list[dict[str, str]]
        Messages list ready for ``LLMClient.generate()``.
    """
    return [
        {"role": "system", "content": _REPLAN_SYSTEM_PROMPT},
        {"role": "user", "content": _REPLAN_USER_TEMPLATE.format(
            question=question,
            gap_description=gap_description,
        )},
    ]


# ---------------------------------------------------------------------------
# Document-name filter extraction
# ---------------------------------------------------------------------------

import re

_FILENAME_PATTERN = re.compile(
    r"""
    # Bracket / book-mark styles (e.g. 【x.txt】 / 《x.txt》 / [x.txt] / "x.txt")
    [\[【《「『\"“'](?P<n1>[^\]】》」』\"”'\s\u3000-\u303f\uff00-\uffef]+?\.(?:txt|md|pdf|docx?|xlsx?|pptx?|csv|json|yml|yaml))
    [\]】》」』\"”']
    |
    # Verb-prefixed filename: optionally absorb a CJK verb (根据/在/从/...)
    # that is glued directly to the filename with no whitespace separator,
    # so that "根据xxx.txt" is captured as "xxx.txt" rather than
    # "根据xxx.txt" (the old behaviour, which silently filtered Chroma
    # to zero results because no document had such a name).  The leading
    # character class allows ASCII alnum / underscore / hyphen / CJK
    # ideographs (so "agent项目要求.txt" still matches); the body
    # character class additionally rejects CJK punctuation, ASCII
    # brackets and parens so the match cannot bleed into surrounding
    # bracket book-marks.
    (?:根据|在|从|查看|参考|阅读|查询)?
    (?P<n2>[A-Za-z0-9_\-一-鿿][^\s\u3000-\u303f\uff00-\uffef\[\]【】《》「」『』\"”'\(\)]*?\.(?:txt|md|pdf|docx?|xlsx?|pptx?|csv|json|yml|yaml))
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_target_doc_names(question: str) -> list[str] | None:
    """Extract referenced document names from a user question.

    Supports patterns like::

        [agent项目要求.txt]
        《agent项目要求.txt》
        根据 agent项目要求.txt 的内容
        agent项目要求.txt

    Returns ``None`` when no filename is detected (no filter should be
    applied in that case).
    """
    if not question:
        return None

    seen = set()
    names: list[str] = []
    for m in _FILENAME_PATTERN.finditer(question):
        g = m.group("n1") or m.group("n2")
        if not g:
            continue
        name = g.strip().strip(".\"'").strip("[]【】《》「』\"“”'")
        # Accept only reasonable filename shapes
        if " " in name or len(name) < 3:
            continue
        if name.lower() not in {n.lower() for n in seen}:
            seen.add(name)
            names.append(name)

    return names if names else None


async def resolve_doc_filter(
    question: str,
    catalog: "DocCatalog | None" = None,
) -> list[str] | None:
    """Resolve which document(s) the user is referring to in *question*.

    Strategy (in order):

    1. **Semantic catalog match** (preferred).  When a ``DocCatalog`` is
       available and ``settings.agent.doc_catalog_enabled`` is True, the
       question is embedded and compared against every uploaded
       file_name.  Filenames with cosine similarity above the
       configured threshold become the doc_filter.  This handles
       colloquial references such as "那个 checklist", partial names
       like "agent项目要求" (no extension), and typos.

    2. **Regex fallback**.  When the catalog is empty / disabled /
       unavailable, fall back to :func:`extract_target_doc_names` which
       matches bracketed / verb-prefixed filenames literally.

    Returns ``None`` when no document reference is detected.
    """
    from app.config import settings as _settings
    from app.core.doc_catalog import DocCatalog

    if not question or not question.strip():
        return None

    if _settings.agent.doc_catalog_enabled:
        cat = catalog or DocCatalog()
        names = await cat.find_relevant(question)
        if names:
            return names

    # Fallback: regex-based extraction (preserves old behaviour when the
    # catalog is empty or the feature is disabled).
    return extract_target_doc_names(question)


# ---------------------------------------------------------------------------
# Quality Check prompt (Phase 2 — module 4.3)
# ---------------------------------------------------------------------------

_QUALITY_CHECK_SYSTEM_PROMPT = (
    "你是一个严格的检索质量评估助手。"
    "你的任务是评估给定的上下文片段是否足以回答用户的问题。"
    "\n\n"
    "评估标准：\n"
    "1. **关键信息覆盖**：上下文中是否包含回答问题的核心事实、定义或数据\n"
    "2. **完整性**：是否涵盖了问题的所有重要方面（如原因、过程、结果等）\n"
    "3. **时效性**：信息是否与问题的时间背景一致（如需要最新数据时）\n"
    "4. **可靠性**：上下文来源是否可靠、内容是否连贯\n"
    "\n\n"
    "输出要求：\n"
    "- 必须严格输出 JSON 对象格式\n"
    "- sufficient 为 true 时 gap 字段为 null\n"
    "- sufficient 为 false 时 gap 字段描述具体缺失的信息"
)

_QUALITY_CHECK_USER_TEMPLATE = """## 用户问题

{question}

## 检索到的上下文（共 {chunk_count} 条）

{context}

## 评估任务

请评估上述上下文是否足以完整、准确地回答用户问题。

输出 JSON 对象格式（不要包含其他文字）：
```json
{{
  "sufficient": true,
  "reasoning": "上下文包含了RAG的定义、架构组件和应用场景，足以回答用户问题。",
  "gap": null
}}
```

或：
```json
{{
  "sufficient": false,
  "reasoning": "上下文仅提及RAG的基本概念，未涉及用户询问的具体实现细节。",
  "gap": "缺失RAG系统的具体实现步骤和代码示例"
}}
```

只输出 JSON 对象，不要包含其他文字。"""


def build_quality_check_messages(
    question: str,
    context_pool: list,
    *,
    max_chunks: int = 20,
    max_chars_per_chunk: int = 800,
) -> list[dict[str, str]]:
    """Build messages for the quality-check (LLM-as-Judge) LLM call.

    Parameters
    ----------
    question:
        The user's original natural-language question.
    context_pool:
        List of ``SearchChunk`` objects representing the current
        retrieval context pool.
    max_chunks:
        Maximum chunks to include in the context view.
    max_chars_per_chunk:
        Truncate each chunk's content to this many characters.

    Returns
    -------
    list[dict[str, str]]
        Messages list ready for ``LLMClient.generate()``.
    """
    from app.core.prompts import format_context

    context_text = format_context(
        context_pool,
        max_chunks=max_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
        show_score=False,
    )

    user_content = _QUALITY_CHECK_USER_TEMPLATE.format(
        question=question,
        chunk_count=len(context_pool),
        context=context_text,
    )

    return [
        {"role": "system", "content": _QUALITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
