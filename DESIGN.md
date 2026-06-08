# Agentic RAG 个人知识库问答系统 — 完整技术方案

---

## 一、项目定位

**Agentic RAG Personal Knowledge Base Q&A System**

不是简单的"上传文档 → 检索 → 回答"，而是一个具备**自主决策能力**的知识库 Agent：

- 自动改写用户问题，生成多条检索 query
- 多路检索 + 精排，召回最优上下文
- 自评估检索质量，不足时自动补充检索
- 带引用溯源的结构化答案输出
- **Agent 思考过程**全程可见（流式）

---

## 二、完整技术栈

### 2.1 后端

| 模块 | 技术 | 版本/说明 |
|------|------|-----------|
| Web 框架 | FastAPI | ≥0.110 |
| 异步服务器 | Uvicorn | ≥0.27 |
| 配置管理 | pydantic-settings | 环境变量 + `.env` |
| 文档解析 — PDF | pdfplumber + PyPDF2 | 双解析器 fallback：pdfplumber 优先（表格/中文好），失败降级 PyPDF2 |
| 文档解析 — Markdown | Python markdown 库 | 转纯文本后切分 |
| 文档解析 — TXT | 内置 `open()` | 自动检测编码（chardet） |
| 文件上传 | python-multipart | FastAPI UploadFile |
| 文本切分 | LangChain RecursiveCharacterTextSplitter | chunk_size=800, overlap=150 |
| 语义切分（可选升级） | 自实现 semantic chunking | 基于 embedding 相似度边界检测 |
| 向量模型 | Qwen text-embedding-v3 | 1024 维, 阿里云 DashScope API |
| 向量数据库 | Chroma | 本地持久化 `./chroma_data/`, HNSW 索引 |
| LLM — 生成 | Qwen-Plus | OpenAI 兼容 API (DashScope), 可插拔 |
| LLM — Agent 决策 | Qwen-Plus | 同上, 轻量任务也用 Plus |
| Rerank | BAAI/bge-reranker-v2-m3 | Cross-encoder, 本地部署 via `sentence-transformers` |
| 流式输出 | FastAPI StreamingResponse + SSE | `text/event-stream` |
| 数据验证 | Pydantic v2 | 请求/响应模型 |
| 日志 | loguru | 结构化日志 |
| 异步 HTTP | httpx | LLM API 调用 |
| Token 计数 | tiktoken | 上下文长度控制 |

### 2.2 前端

| 模块 | 技术 | 说明 |
|------|------|------|
| 框架 | React 18+ | TypeScript |
| 构建工具 | Vite | |
| UI 组件 | shadcn/ui | 基于 Radix UI |
| 样式 | Tailwind CSS | |
| 服务端状态 | TanStack Query (React Query v5) | 文档列表、历史记录 |
| UI 状态 | Zustand | 上传进度、侧栏、选中来源 |
| 文件上传 | react-dropzone | 拖拽上传 |
| Markdown 渲染 | react-markdown + remark-gfm | 答案渲染 |
| 代码高亮 | rehype-highlight | 答案中代码块 |
| 流式答案 | EventSource (原生) | SSE 消费 |
| 来源高亮 | 自定义 SourceHighlight 组件 | 点击/悬停联动 |
| 图表/统计 | recharts（可选） | 文档统计仪表盘 |

### 2.3 部署

| 模块 | 技术 |
|------|------|
| 容器化 | Docker + Docker Compose |
| 后端镜像 | `python:3.12-slim` + FastAPI |
| 前端镜像 | Node 多阶段构建 → Nginx |
| 持久化卷 | `chroma_data/`, `uploads/`, 模型缓存 |
| 环境变量 | `.env`（API Key, 模型名, 路径） |
| 开发代理 | Vite proxy → `localhost:8000` |

---

## 三、项目目录结构

```
agentic-rag-kb/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 入口
│   │   ├── config.py                # pydantic-settings 配置
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py         # 文档上传/列表/删除 API
│   │   │   ├── qa.py                # 问答 API（SSE 流式）
│   │   │   └── history.py           # 问答历史 API
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # Agent 决策循环（核心）
│   │   │   ├── retriever.py         # 多路检索 + rerank
│   │   │   ├── parser.py            # 文档解析器（PDF/MD/TXT）
│   │   │   ├── chunker.py           # 文本切分
│   │   │   ├── embedder.py          # embedding 生成
│   │   │   ├── generator.py         # LLM 答案生成
│   │   │   └── citation.py          # 引用解析与对齐
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── chroma_store.py      # Chroma 操作封装
│   │   │   └── sqlite_store.py      # SQLite 操作（文档/历史）
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── document.py          # Pydantic 文档模型
│   │   │   ├── qa.py                # 问答请求/响应模型
│   │   │   └── sse.py               # SSE 事件模型
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logger.py            # loguru 配置
│   │       └── token_counter.py     # tiktoken 工具
│   ├── data/
│   │   ├── chroma_data/             # Chroma 持久化目录
│   │   ├── uploads/                 # 上传文件存储
│   │   └── cache/                   # 模型缓存（reranker）
│   ├── tests/
│   │   ├── test_agent.py
│   │   ├── test_retriever.py
│   │   ├── test_parser.py
│   │   └── test_citation.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── api/
│   │   │   ├── client.ts            # axios/ky 实例
│   │   │   ├── documents.ts         # 文档 API hooks
│   │   │   ├── qa.ts                # 问答 + SSE hooks
│   │   │   └── history.ts           # 历史 API hooks
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx       # 侧边栏（文档列表）
│   │   │   │   ├── Header.tsx
│   │   │   │   └── Layout.tsx
│   │   │   ├── upload/
│   │   │   │   ├── DropZone.tsx      # 拖拽上传区
│   │   │   │   ├── UploadProgress.tsx
│   │   │   │   └── DocCard.tsx       # 文档卡片
│   │   │   ├── qa/
│   │   │   │   ├── QuestionInput.tsx  # 问答输入框
│   │   │   │   ├── AnswerPanel.tsx    # 答案展示
│   │   │   │   ├── SourceCard.tsx     # 来源卡片
│   │   │   │   ├── SourceHighlight.tsx# 来源高亮联动
│   │   │   │   └── ThinkingPanel.tsx  # Agent 思考过程面板
│   │   │   └── common/
│   │   │       ├── Loading.tsx
│   │   │       └── Empty.tsx
│   │   ├── stores/
│   │   │   ├── uiStore.ts           # Zustand: UI 状态
│   │   │   └── qaStore.ts           # Zustand: 当前问答状态
│   │   ├── hooks/
│   │   │   ├── useSSE.ts             # SSE 连接 hook
│   │   │   └── useSourceScroll.ts    # 来源滚动联动
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript 类型
│   │   └── lib/
│   │       └── utils.ts
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
├── docker-compose.yml
├── .env.example
├── README.md
└── DESIGN.md                        # 本文档
```

---

## 四、核心模块设计

### 4.1 Agent 决策循环（`core/agent.py`）— **项目心脏**

这是整个系统最核心的模块。Agent 不是简单的"搜一次→回答"，而是一个**状态机驱动的决策循环**：

```
                    ┌─────────────┐
                    │  用户提问    │
                    └──────┬──────┘
                           ↓
                  ┌────────────────┐
                  │  1. Query      │  LLM 分析问题，生成 N 条检索 query
                  │     Rewrite    │  (改写、拆解子问题、多角度)
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  2. Multi-     │  每条 query 向量检索 top-K
                  │     Search     │  合并去重，召回 ~20 条候选
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  3. Rerank     │  Cross-encoder 精排
                  │                │  取 top-5 加入上下文池
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  4. Quality    │  LLM-as-Judge 评估：
                  │     Check      │  当前上下文能否回答问题？
                  └───┬───────┬────┘
                      │       │
              足够    │       │  不足（且未超最大轮次）
                      │       ↓
                      │  ┌────────────────┐
                      │  │  5. Re-plan    │  基于当前缺口，
                      │  │                │  改写 query，回到步骤 2
                      │  └───────┬────────┘
                      │          │
                      ↓          ↓
                  ┌────────────────┐
                  │  6. Generate   │  LLM 综合上下文，生成答案
                  │                │  要求标注引用 [chunk_N]
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  7. Citation   │  正则解析引用标记
                  │      Parse     │  对齐 chunk_id → 文档信息
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  最终答案 + 来源 │
                  └────────────────┘
```

#### Agent Loop 伪代码

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class AgentStep(str, Enum):
    REWRITE = "rewrite"
    SEARCH = "search"
    RERANK = "rerank"
    CHECK = "check"
    REPLAN = "replan"
    GENERATE = "generate"
    DONE = "done"

@dataclass
class AgentState:
    question: str
    queries: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    context_pool: list[Chunk] = field(default_factory=list)  # 所有轮次的累积
    round: int = 0
    max_rounds: int = 3
    sufficient: bool = False
    gap_description: str = ""  # 本轮不足时, LLM 描述缺口

class AgentLoop:
    def __init__(self, llm, embedder, chroma_store, reranker):
        self.llm = llm
        self.embedder = embedder
        self.chroma = chroma_store
        self.reranker = reranker

    async def run(self, question: str) -> AgentResult:
        state = AgentState(question=question)

        # Step 1: Query Rewrite
        yield SSEStepEvent(step="rewrite", message="正在分析问题...")
        state.queries = await self._rewrite_query(question)
        yield SSEStepEvent(step="rewrite", queries=state.queries)

        # Step 2-5: Search → Rerank → Check → Replan loop
        while state.round < state.max_rounds:
            state.round += 1

            # Step 2: Multi-Query Search
            yield SSEStepEvent(step="search", message=f"第{state.round}轮检索...")
            raw_chunks = await self._multi_search(state.queries, top_k=20)
            yield SSEStepEvent(step="search", count=len(raw_chunks))

            # Step 3: Rerank
            yield SSEStepEvent(step="rerank", message="正在精排...")
            ranked = await self._rerank(question, raw_chunks, top_k=5)
            state.context_pool.extend(ranked)

            # Step 4: Quality Check (LLM-as-Judge)
            yield SSEStepEvent(step="check", message="评估检索质量...")
            check_result = await self._quality_check(question, state.context_pool)

            if check_result.sufficient:
                state.sufficient = True
                yield SSEStepEvent(step="check", verdict="sufficient")
                break
            else:
                state.gap_description = check_result.gap
                yield SSEStepEvent(
                    step="check",
                    verdict="insufficient",
                    gap=check_result.gap
                )
                # Step 5: Re-plan
                yield SSEStepEvent(step="replan", message="补充检索...")
                state.queries = await self._replan(question, check_result.gap)

        # Step 6: Generate
        yield SSEStepEvent(step="generate", message="生成答案...")
        collected_chunks: list[str] = []
        async for chunk in self._generate_stream(question, state.context_pool):
            collected_chunks.append(chunk)
            yield SSEAnswerEvent(text=chunk)

        # Step 7: Citation Parse & Align
        full_answer = "".join(collected_chunks)
        sources = self._parse_citations(full_answer, state.context_pool)
        yield SSESourcesEvent(sources=sources)

    async def _rewrite_query(self, question: str) -> list[str]:
        """LLM 改写问题为多条检索 query"""
        prompt = f"""你是一个检索专家。分析用户问题，生成 3-5 条不同角度的检索 query。

用户问题：{question}

要求：
1. 如果问题是复合问题，拆解为子问题分别生成 query
2. 用不同的措辞和关键词覆盖同一概念
3. 考虑问题可能涉及的不同方面
4. 每条 query 简洁、关键词密集

返回 JSON 格式：{{"queries": ["query1", "query2", ...]}}"""

        response = await self.llm.complete(prompt, response_format="json")
        return json.loads(response)["queries"]

    async def _multi_search(self, queries: list[str], top_k: int = 20) -> list[Chunk]:
        """多 query 检索 + 去重"""
        all_chunks = []
        seen_ids = set()

        for query in queries:
            vec = await self.embedder.embed(query)
            results = self.chroma.query(vec, n_results=top_k)
            for chunk in results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_chunks.append(chunk)

        return all_chunks

    async def _rerank(self, question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Cross-encoder 精排"""
        pairs = [[question, chunk.content] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        # 按分数排序，返回 top_k
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]

    async def _quality_check(self, question: str, context: list[Chunk]) -> CheckResult:
        """LLM-as-Judge 评估检索质量"""
        context_text = "\n\n---\n\n".join(
            f"[chunk_{i}] {c.content[:500]}" for i, c in enumerate(context)
        )
        prompt = f"""你是一个检索质量评估器。判断以下上下文是否足够回答用户问题。

用户问题：{question}

检索到的上下文：
{context_text}

判断标准：
1. 上下文中是否包含回答所需的关键信息？
2. 信息是否完整，有没有重要缺口？
3. 不同角度的信息是否都已覆盖？

返回 JSON：
{{
    "sufficient": true/false,
    "reasoning": "简要判断理由",
    "gap": "如果不足，描述缺失了什么信息（否则 null）"
}}"""
        response = await self.llm.complete(prompt, response_format="json")
        result = json.loads(response)
        return CheckResult(
            sufficient=result["sufficient"],
            reasoning=result["reasoning"],
            gap=result.get("gap")
        )

    async def _replan(self, question: str, gap: str) -> list[str]:
        """基于缺口生成补充检索 query"""
        prompt = f"""用户问题：{question}
当前检索存在以下缺口：{gap}

请生成 2-3 条新的检索 query，专门针对这些缺失的信息。

返回 JSON：{{"queries": ["query1", ...]}}"""
        response = await self.llm.complete(prompt, response_format="json")
        return json.loads(response)["queries"]

    async def _generate_stream(self, question: str, context: list[Chunk]) -> AsyncIterator[str]:
        """流式生成答案"""
        context_text = "\n\n---\n\n".join(
            f"[chunk_{i}] (来源: {c.doc_name}, 第{c.page}页)\n{c.content}"
            for i, c in enumerate(context)
        )
        prompt = f"""基于以下上下文回答用户问题。

## 上下文
{context_text}

## 用户问题
{question}

## 回答要求
1. 基于上下文回答，不要编造信息
2. 如果上下文中没有相关信息，诚实说明
3. 在引用上下文时标注来源编号，格式：[chunk_N]
4. 答案结构化，使用标题和列表提高可读性
5. 最后列出所有引用来源

## 回答"""
        async for token in self.llm.stream(prompt):
            yield token
```

### 4.2 检索模块（`core/retriever.py`）

```python
class Retriever:
    """多路检索 + Rerank 封装"""

    def __init__(self, embedder: Embedder, chroma: ChromaStore, reranker: Reranker):
        self.embedder = embedder
        self.chroma = chroma
        self.reranker = reranker

    async def retrieve(
        self,
        queries: list[str],
        top_k_recall: int = 20,
        top_k_rerank: int = 5,
    ) -> list[Chunk]:
        # 1. 多路向量检索
        all_chunks = []
        seen = set()
        for query in queries:
            vec = await self.embedder.embed(query)
            results = self.chroma.query(vec, n_results=top_k_recall)
            for r in results:
                if r.id not in seen:
                    seen.add(r.id)
                    all_chunks.append(r)

        # 2. Cross-encoder 精排
        if len(all_chunks) <= top_k_rerank:
            return all_chunks

        # 注意：Retriever 不持有原始 question，此处用第一条 query 近似；
        # 实际使用时 Agent._rerank() 会传入原始 question 以获得最佳精排效果
        pairs = [[queries[0], c.content] for c in all_chunks]
        scores = self.reranker.compute_similarity(pairs)
        ranked = sorted(zip(all_chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k_rerank]]
```

### 4.3 文档解析模块（`core/parser.py`）

```python
from enum import Enum
import pdfplumber
import PyPDF2
from pathlib import Path

class DocType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    TXT = "txt"

class Document:
    """解析后的文档"""
    file_name: str
    doc_type: DocType
    pages: list[Page]  # 每页: page_number, text, metadata

class Page:
    page_number: int
    text: str
    metadata: dict  # {bbox, tables, images...}

class DocumentParser:
    """多格式解析器，PDF 双引擎 fallback"""

    async def parse(self, file_path: Path) -> Document:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return await self._parse_pdf(file_path)
        elif ext in [".md", ".markdown"]:
            return await self._parse_markdown(file_path)
        elif ext == ".txt":
            return await self._parse_txt(file_path)
        else:
            raise UnsupportedFormatError(f"不支持的文件格式: {ext}")

    async def _parse_pdf(self, file_path: Path) -> Document:
        """pdfplumber 优先，失败降级 PyPDF2"""
        try:
            return self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.warning(f"pdfplumber 解析失败: {e}, 降级到 PyPDF2")
            return self._parse_with_pypdf2(file_path)

    def _parse_with_pdfplumber(self, file_path: Path) -> Document:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                pages.append(Page(
                    page_number=i + 1,
                    text=self._merge_table_text(text, tables),
                    metadata={"width": page.width, "height": page.height}
                ))
        return Document(file_name=file_path.name, doc_type=DocType.PDF, pages=pages)

    def _parse_with_pypdf2(self, file_path: Path) -> Document:
        pages = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(Page(page_number=i + 1, text=text, metadata={}))
        return Document(file_name=file_path.name, doc_type=DocType.PDF, pages=pages)

    def _merge_table_text(self, text: str, tables: list) -> str:
        """将表格数据合并到文本中"""
        for table in tables:
            if table:
                table_str = "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in table
                )
                text += f"\n\n[表格]\n{table_str}\n"
        return text

    async def _parse_markdown(self, file_path: Path) -> Document:
        """Markdown → 结构化文本（保留标题层级标记供 chunker 利用）"""
        import markdown
        from bs4 import BeautifulSoup

        raw = file_path.read_text(encoding="utf-8")
        html = markdown.markdown(raw, extensions=["tables", "fenced_code", "toc"])
        soup = BeautifulSoup(html, "html.parser")
        # 为标题插入语义标记，方便后续 chunker 识别段落边界
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_before(soup.new_string(f"\n[{tag.name.upper()}] "))
        text = soup.get_text()
        return Document(
            file_name=file_path.name,
            doc_type=DocType.MARKDOWN,
            pages=[Page(page_number=1, text=text, metadata={})]
        )

    async def _parse_txt(self, file_path: Path) -> Document:
        """TXT 解析（自动检测编码）"""
        import chardet

        raw = file_path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected["encoding"] or "utf-8"
        text = raw.decode(encoding)
        return Document(
            file_name=file_path.name,
            doc_type=DocType.TXT,
            pages=[Page(page_number=1, text=text, metadata={})]
        )
```

### 4.4 文本切分模块（`core/chunker.py`）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str           # 唯一 ID: {doc_id}_chunk_{index}
    content: str      # 切分后的文本
    doc_id: str       # 文档 ID
    doc_name: str     # 文档名
    page: int         # 页码
    chunk_index: int  # chunk 在文档中的序号
    metadata: dict    # 额外元数据

class Chunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
            length_function=len,
        )

    def split(self, document: Document) -> list[Chunk]:
        chunks = []
        for page in document.pages:
            texts = self.splitter.split_text(page.text)
            for idx, text in enumerate(texts):
                chunk = Chunk(
                    id=f"{document.file_name}_p{page.page_number}_c{idx}",
                    content=text,
                    doc_id=document.file_name,
                    doc_name=document.file_name,
                    page=page.page_number,
                    chunk_index=idx,
                    metadata={
                        "doc_name": document.file_name,
                        "page": page.page_number,
                        "char_count": len(text),
                    }
                )
                chunks.append(chunk)
        return chunks

    # ----- 可选升级：语义切分 -----
    def split_semantic(
        self,
        document: Document,
        embedder,
        similarity_threshold: float = 0.5
    ) -> list[Chunk]:
        """基于 embedding 相似度的语义边界切分"""
        # 1. 先按句子切分
        sentences = self._split_sentences(document)
        # 2. 逐句计算 embedding
        embeddings = [embedder.embed(s) for s in sentences]
        # 3. 相邻句相似度 < 阈值 → 切分点
        boundaries = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < similarity_threshold:
                boundaries.append(i)
        # 4. 按边界合并句子为 chunks
        return self._merge_at_boundaries(document, sentences, boundaries)
```

### 4.5 引用解析模块（`core/citation.py`）

```python
import re
from dataclasses import dataclass

@dataclass
class Citation:
    chunk_id: str
    doc_name: str
    page: int
    content_snippet: str  # 引用的原文片段
    score: float          # 相似度分数

class CitationParser:
    CITATION_PATTERN = re.compile(r'\[chunk_(\d+)\]')

    def parse(self, answer: str, context_pool: list[Chunk]) -> list[Citation]:
        """
        两步解析：
        1. 正则提取 [chunk_N] 标记
        2. 验证 chunk_id 存在，收集引用信息
        """
        referenced_ids = set()
        for match in self.CITATION_PATTERN.finditer(answer):
            idx = int(match.group(1))
            if 0 <= idx < len(context_pool):
                referenced_ids.add(idx)

        citations = []
        for idx in referenced_ids:
            chunk = context_pool[idx]
            citations.append(Citation(
                chunk_id=chunk.id,
                doc_name=chunk.doc_name,
                page=chunk.page,
                content_snippet=chunk.content[:200],
                score=chunk.metadata.get("rerank_score", 0.0),
            ))

        return sorted(citations, key=lambda c: c.score, reverse=True)

    def fallback_string_match(
        self,
        answer: str,
        context_pool: list[Chunk],
        min_match_len: int = 50
    ) -> list[Citation]:
        """
        LLM 没按格式标注时的回退方案：
        在答案中搜索每个 chunk 的子串，找到最长的连续匹配。
        """
        citations = []
        for chunk in context_pool:
            # 从 chunk 中提取特征句子（取最长的几句）
            sentences = re.split(r'[。.！？\n]', chunk.content)
            key_sentences = sorted(sentences, key=len, reverse=True)[:3]

            for sentence in key_sentences:
                if len(sentence) >= min_match_len and sentence[:30] in answer:
                    citations.append(Citation(
                        chunk_id=chunk.id,
                        doc_name=chunk.doc_name,
                        page=chunk.page,
                        content_snippet=sentence[:200],
                        score=chunk.metadata.get("rerank_score", 0.0),
                    ))
                    break

        return citations
```

---

## 五、API 设计

### 5.1 文档上传

```
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Body:
  file: binary (PDF/MD/TXT, max 50MB)

Response 201:
{
  "doc_id": "abc123",
  "file_name": "agent-design.pdf",
  "doc_type": "pdf",
  "page_count": 15,
  "chunk_count": 42,
  "size_bytes": 2048576,
  "uploaded_at": "2026-06-08T10:30:00Z"
}
```

**处理流程**：
1. 接收文件 → 保存到 `data/uploads/`
2. 解析文档 → Document 对象
3. 切分 → list[Chunk]
4. 批量 embed → 写入 Chroma
5. 写入 SQLite 文档表
6. 返回文档信息

### 5.2 文档列表

```
GET /api/v1/documents?page=1&size=20

Response 200:
{
  "items": [
    {
      "doc_id": "abc123",
      "file_name": "agent-design.pdf",
      "doc_type": "pdf",
      "chunk_count": 42,
      "uploaded_at": "2026-06-08T10:30:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "size": 20
}
```

### 5.3 删除文档

```
DELETE /api/v1/documents/{doc_id}

Response 200:
{
  "deleted": true,
  "doc_id": "abc123"
}
```

同时删除本地文件和 Chroma 中的向量。

### 5.4 问答（核心 API，SSE 流式）

```
POST /api/v1/qa/ask
Content-Type: application/json

Body:
{
  "question": "如何将 RAG 系统升级为 Agentic RAG？",
  "conversation_id": null,    // 可选，多轮对话
  "max_rounds": 3,            // 可选，Agent 最大检索轮次
  "top_k": 5                  // 可选，每轮保留 chunk 数
}

Response: text/event-stream
```

**SSE 事件流**：

```
event: agent-step
data: {"step": "rewrite", "message": "正在分析问题...", "timestamp": "..."}

event: agent-step
data: {"step": "rewrite", "queries": ["RAG to Agentic RAG upgrade", "Agentic RAG architecture design", "RAG system agent layer implementation"], "timestamp": "..."}

event: agent-step
data: {"step": "search", "message": "第1轮检索...", "timestamp": "..."}

event: agent-step
data: {"step": "search", "count": 17, "timestamp": "..."}

event: agent-step
data: {"step": "rerank", "message": "正在精排...", "timestamp": "..."}

event: agent-step
data: {"step": "check", "message": "评估检索质量...", "timestamp": "..."}

event: agent-step
data: {"step": "check", "verdict": "sufficient", "reasoning": "上下文涵盖了 Agent 架构设计和实现步骤", "timestamp": "..."}

event: answer-chunk
data: {"text": "基于你的文档，将 RAG", "timestamp": "..."}

event: answer-chunk
data: {"text": "升级为 Agentic RAG 需要以下步骤...", "timestamp": "..."}

event: answer-done
data: {"timestamp": "..."}

event: sources
data: {"sources": [{"chunk_id": "...", "doc_name": "agent-design.pdf", "page": 3, "content_snippet": "...", "score": 0.92}, ...], "timestamp": "..."}

event: done
data: {"conversation_id": "conv_xyz", "total_rounds": 1, "chunks_used": 5, "timestamp": "..."}
```

### 5.5 问答历史

```
GET /api/v1/qa/history?conversation_id=conv_xyz&page=1&size=20

GET /api/v1/qa/conversations  // 会话列表
```

### 5.6 健康检查

```
GET /api/v1/health

Response 200:
{
  "status": "ok",
  "chroma_docs": 142,
  "model": "qwen-plus",
  "embedding_model": "text-embedding-v3"
}
```

---

## 六、数据模型

### 6.1 SQLite 表设计

```sql
-- 文档表
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,       -- pdf, md, txt
    file_path TEXT NOT NULL,      -- 本地存储路径
    page_count INTEGER DEFAULT 1,
    chunk_count INTEGER DEFAULT 0,
    size_bytes INTEGER,
    status TEXT DEFAULT 'ready',  -- processing, ready, error
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 会话表
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,                   -- 首条问题的摘要
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 问答记录表
CREATE TABLE qa_records (
    record_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT,            -- JSON: [{chunk_id, doc_name, page, snippet, score}]
    agent_steps_json TEXT,        -- JSON: Agent 各步骤的记录
    total_rounds INTEGER,
    model TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
```

### 6.2 Pydantic 模型

```python
# models/document.py
from pydantic import BaseModel
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    page_count: int
    chunk_count: int
    size_bytes: int
    uploaded_at: datetime

class DocumentItem(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    chunk_count: int
    uploaded_at: datetime

class DocumentListResponse(BaseModel):
    items: list[DocumentItem]
    total: int
    page: int
    size: int

# models/qa.py
class QuestionRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    max_rounds: int = 3
    top_k: int = 5

# models/sse.py
from enum import Enum

class SSEEventType(str, Enum):
    AGENT_STEP = "agent-step"
    ANSWER_CHUNK = "answer-chunk"
    ANSWER_DONE = "answer-done"
    SOURCES = "sources"
    ERROR = "error"
    DONE = "done"

class SSEStepEvent(BaseModel):
    type: SSEEventType = SSEEventType.AGENT_STEP
    step: str           # rewrite, search, rerank, check, replan, generate
    message: str | None = None
    queries: list[str] | None = None
    count: int | None = None
    verdict: str | None = None
    reasoning: str | None = None
    gap: str | None = None
    timestamp: str

class SSEAnswerEvent(BaseModel):
    type: SSEEventType = SSEEventType.ANSWER_CHUNK
    text: str
    timestamp: str

class SSESourcesEvent(BaseModel):
    type: SSEEventType = SSEEventType.SOURCES
    sources: list[Citation]
    timestamp: str

class SSEDoneEvent(BaseModel):
    type: SSEEventType = SSEEventType.DONE
    conversation_id: str
    total_rounds: int
    chunks_used: int
    timestamp: str
```

---

## 七、前端组件设计

### 7.1 组件树

```
App
├── Layout
│   ├── Header
│   │   ├── Logo + 标题
│   │   └── 新建对话按钮
│   ├── Sidebar
│   │   ├── 对话历史列表
│   │   └── 文档管理入口
│   └── Main
│       ├── (无文档时) EmptyState → DropZone
│       │   └── "拖拽 PDF/MD/TXT 到此处开始"
│       └── (有文档时)
│           ├── QuestionInput
│           │   ├── TextArea（自动增高）
│           │   └── SendButton
│           ├── ThinkingPanel（Agent 思考过程，可折叠）
│           │   ├── StepItem: "🔍 分析问题 → 3 条检索 query"
│           │   ├── StepItem: "📚 第1轮检索 → 17 条候选"
│           │   ├── StepItem: "🎯 精排 → top-5"
│           │   └── StepItem: "✅ 上下文充足，生成答案"
│           ├── AnswerPanel
│           │   ├── MarkdownRenderer（流式渲染）
│           │   └── SourceCards
│           │       ├── SourceCard 1 [hover → 原文高亮]
│           │       ├── SourceCard 2
│           │       └── ...
│           └── (右侧) SourceHighlight 面板
│               └── 点击来源 → 滚动到对应原文，高亮相关段落
│
├── UploadModal
│   ├── DropZone
│   └── UploadProgress（批量）
└── DocManagementPage
    └── DocCard[]
```

### 7.2 状态管理设计

**Zustand (UI 状态)**：
```typescript
// stores/uiStore.ts
interface UIState {
  sidebarOpen: boolean;
  thinkingExpanded: boolean;
  selectedSourceId: string | null;
  uploadModalOpen: boolean;
  // actions
  toggleSidebar: () => void;
  setSelectedSource: (id: string | null) => void;
  // ...
}
```

**TanStack Query (服务端状态)**：
```typescript
// api/documents.ts
export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: () => fetch('/api/v1/documents').then(r => r.json()),
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append('file', file);
      return fetch('/api/v1/documents/upload', { method: 'POST', body: form });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['documents'] }),
  });
}
```

### 7.3 SSE Hook 设计

```typescript
// hooks/useSSE.ts
interface SSEState {
  thinkingSteps: ThinkingStep[];
  answerText: string;
  sources: Source[];
  isStreaming: boolean;
  isDone: boolean;
}

function useAskQuestion() {
  const [state, setState] = useState<SSEState>({
    thinkingSteps: [],
    answerText: '',
    sources: [],
    isStreaming: false,
    isDone: false,
  });

  const ask = useCallback((question: string) => {
    setState(prev => ({ ...prev, isStreaming: true, thinkingSteps: [], answerText: '', sources: [], isDone: false }));

    const eventSource = new EventSource(`/api/v1/qa/ask`, { /* POST via fetch + ReadableStream */ });

    // 使用 fetch + ReadableStream 实现 POST SSE
    fetch('/api/v1/qa/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }).then(async (response) => {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // 解析 SSE 格式，更新 state
        parseSSEAndUpdate(text, setState);
      }
    });

    // 返回 abort 函数
    return () => { /* abort controller */ };
  }, []);

  return { ...state, ask };
}
```

### 7.4 来源高亮联动

```
用户交互流程：

1. 用户在 AnswerPanel 看到答案，文字中有标注 [chunk_2]
2. 用户点击 [chunk_2] 标签
   ↓
3. uiStore.setSelectedSource("chunk_2")
   ↓
4. SourceCard 高亮
5. ThinkingPanel 中对应 chunk 高亮
6. (如果打开) SourceHighlight 面板滚动到原文位置
   ↓
7. 用户悬停 SourceCard → AnswerPanel 中引用该来源的文字背景变色
```

---

## 八、配置设计

### 8.1 `.env` 文件

```bash
# ===== LLM =====
LLM_PROVIDER=dashscope            # dashscope (阿里云) | openai | deepseek | custom
LLM_API_KEY=sk-xxxxxxxx           # DashScope API Key
LLM_MODEL=qwen-plus
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ===== Embedding =====
# Qwen Embedding 与 LLM 使用同一 DashScope 端点，统一 API Key
EMBEDDING_PROVIDER=dashscope
EMBEDDING_API_KEY=sk-xxxxxxxx
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSIONS=1024

# ===== Reranker =====
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu               # cpu | cuda

# ===== Chroma =====
CHROMA_PERSIST_DIR=./data/chroma_data
CHROMA_COLLECTION=knowledge_base

# ===== Database =====
SQLITE_PATH=./data/app.db

# ===== Upload =====
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50

# ===== Agent =====
AGENT_MAX_ROUNDS=3
AGENT_TOP_K_RECALL=20
AGENT_TOP_K_RERANK=5
AGENT_CHUNK_SIZE=800
AGENT_CHUNK_OVERLAP=150

# ===== Server =====
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173
LOG_LEVEL=INFO
```

### 8.2 `config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # LLM
    llm_provider: str = "dashscope"
    llm_api_key: str
    llm_model: str = "qwen-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding（Qwen Embedding，与 LLM 同一端点和 API Key）
    embedding_provider: str = "dashscope"
    embedding_api_key: str
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # Chroma
    chroma_persist_dir: str = "./data/chroma_data"
    chroma_collection: str = "knowledge_base"

    # SQLite
    sqlite_path: str = "./data/app.db"

    # Upload
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    # Agent
    agent_max_rounds: int = 3
    agent_top_k_recall: int = 20
    agent_top_k_rerank: int = 5
    agent_chunk_size: int = 800
    agent_chunk_overlap: int = 150

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

---

## 九、Docker 部署

### 9.1 `docker-compose.yml`

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend/data:/app/data          # Chroma + SQLite + uploads 持久化
      - ./backend/.env:/app/.env:ro
    environment:
      - CUDA_VISIBLE_DEVICES=             # 使用 CPU
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

  # 可选：Ollama 本地模型
  # ollama:
  #   image: ollama/ollama
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ollama_data:/root/.ollama

volumes:
  # ollama_data:
```

### 9.2 后端 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 reranker 模型（避免首次运行等待）
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.3 前端 Dockerfile

```dockerfile
# Stage 1: Build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm install
COPY . .
RUN npm run build

# Stage 2: Serve
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 十、开发路线图

### Phase 1: MVP 核心链路（~10 天）

```
目标：文档上传 → 切分 → 单次检索 → 流式答案 → 来源展示
不含：Agent 循环、Rerank、质量评估
```

| 任务 | 估时 |
|------|------|
| 项目初始化 + 配置 + FastAPI 骨架 | 1d |
| 文档解析（PDF/MD/TXT）+ 文件上传 API | 2d |
| 文本切分 + Embedding + Chroma 写入 | 2d |
| 单 query 检索 API | 1d |
| LLM 生成 + SSE 流式输出 | 2d |
| 前端：上传 + 输入框 + 流式答案 | 2d |
| **MVP 完成** | **10d** |

### Phase 2: Agent 增强（~7 天）

```
目标：Query Rewrite + Multi-Search + Rerank + Quality Check + Replan
```

| 任务 | 估时 |
|------|------|
| Query Rewrite 模块 | 1d |
| Multi-Query 检索 + 去重 | 1d |
| BGE Reranker 集成 | 2d |
| Quality Check（LLM-as-Judge）| 1.5d |
| Replan + Agent 循环集成 | 1.5d |
| 前端：ThinkingPanel（Agent 思考过程）| 2d |
| **Phase 2 完成** | **7d** |

### Phase 3: 引用与打磨（~5 天）

```
目标：引用解析、来源高亮联动、效果评测
```

| 任务 | 估时 |
|------|------|
| 引用解析（正则 + 回退匹配）| 1d |
| 前端：SourceCard + SourceHighlight 联动 | 2d |
| 小规模效果评测（有/无 Rerank 对比）| 1d |
| Docker 部署 + README | 1d |
| **Phase 3 完成** | **5d** |

### Phase 4: 可选增强

- [ ] 多轮对话（conversation history 管理）
- [ ] 语义切分对比实验
- [ ] Ollama 本地模型支持
- [ ] 文档标签/分类
- [ ] 用户认证
- [ ] 文档统计仪表盘

---

## 十一、已知限制与后续改进

本节列出当前设计中的已知限制，Phase 1-3 MVP 阶段可接受，后续逐步解决。

### 11.1 安全与访问控制
- **无用户认证**：当前无登录/鉴权机制，任何能访问服务的人均可上传文档和提问。个人单机使用可接受，部署到局域网/公网前需引入（建议 JWT + OAuth2）。
- **无速率限制**：LLM API 调用无限流保护，恶意或异常循环可能快速消耗 API 配额。建议引入 `slowapi` 或自实现 token-bucket 限流。

### 11.2 容错与恢复
- **SSE 断流无重连**：前端 SSE 连接中断后当前无自动重连逻辑，用户需手动重试。建议在 `useSSE` hook 中实现指数退避重连。
- **Agent 循环无事务回滚**：若 Agent 在检索/生成中途异常退出（LLM 超时、OOM 等），已产生的上下文和 SSE 事件无持久化，无法恢复。

### 11.3 Embedding 与 Token 计数
- **Token 计数精度**：`tiktoken` 库为 OpenAI 模型设计，对 Qwen 系列 tokenizer 计数有偏差。上下文长度控制建议保守设置 `max_tokens`，后续可替换为 DashScope 返回的实际 token 数。
- **无 Embedding 缓存**：同一文档重复上传或 chunk 内容未变时会重新计算 embedding，浪费 API 费用。建议基于 `hash(content)` 实现缓存层，命中则跳过 API 调用。
- **批量 embedding 未提及**：当前伪代码逐条 embed，实际应使用 DashScope 的批量 API 以降低网络开销。

### 11.4 Markdown 解析
- **标题层级未充分利用**：`_parse_markdown` 已改进为插入 `[H1]`/`[H2]` 等标记，但 `chunker.py` 尚未配合利用这些标记做语义切分边界。需要后续在 `RecursiveCharacterTextSplitter` 的 `separators` 或语义切分中利用这些标记。

### 11.5 多轮对话
- **当前仅支持单轮问答**：`conversation_id` 已在 API 和数据库模型中预留，但 Agent 循环未注入历史对话上下文。多轮对话需要管理消息历史 + 上下文窗口裁剪，Phase 4 实现。

### 11.6 Reranker 性能
- **CPU 推理延迟**：BGE-Reranker-v2-m3 在 CPU 上每次 rerank 约 1-3 秒（取决于候选数量），高并发下会成为瓶颈。建议：
  - Phase 1-3 使用 CPU 足够
  - 后续可考虑 GPU 加速、结果缓存（相同 question 复用）、或切换到 API-based reranker

### 11.7 前端 SSE 实现
- **EventSource 不支持 POST**：浏览器原生 `EventSource` 仅支持 GET 请求，而问答 API 使用 POST。当前设计用 `fetch + ReadableStream` 替代，需确保流式解析的健壮性（chunk 边界处理、重连逻辑）。

---

## 附录 A: `requirements.txt`

```
# Web
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9

# Config
pydantic-settings>=2.1.0

# LLM
openai>=1.30.0
httpx>=0.27.0
tiktoken>=0.7.0

# Document Parsing
pdfplumber>=0.10.0
PyPDF2>=3.0.0
markdown>=3.5.0
beautifulsoup4>=4.12.0
chardet>=5.2.0

# Text Splitting
langchain-text-splitters>=0.2.0

# Vector DB
chromadb>=0.5.0

# Reranker
sentence-transformers>=2.7.0

# Database
aiosqlite>=0.20.0

# Logging
loguru>=0.7.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0  # for TestClient
```

---

## 附录 B: `package.json` 核心依赖

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "@tanstack/react-query": "^5.40.0",
    "zustand": "^4.5.0",
    "react-dropzone": "^14.2.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0",
    "rehype-highlight": "^7.0.0",
    "lucide-react": "^0.378.0",
    "tailwind-merge": "^2.3.0",
    "clsx": "^2.1.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.4.0",
    "vite": "^5.3.0",
    "tailwindcss": "^3.4.0",
    "shadcn-ui": "^0.9.0"
  }
}
```
