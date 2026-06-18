# Agentic RAG 个人知识库问答系统

> 不只是"上传文档 → 检索 → 回答"，而是一个具备**自主决策能力**的知识库 Agent。

## 核心能力

- 🔍 **自动改写问题** — LLM 分析用户问题，生成多条检索 query
- 🔄 **多轮检索 + 自评估** — Agent 自主判断上下文是否充足，不足则补充检索（最多 3 轮）
- 📚 **多路召回 + 精排** — 多 query 召回 → Cross-encoder 精排，取最优上下文
- 📎 **带引用溯源** — 结构化答案输出，精确到文档、页码、原文片段
- 👁️ **思考过程可见** — Query Rewrite → Search → Rerank → Quality Check 全程流式展示
- 💬 **对话管理** — 多轮对话上下文记忆，对话历史可折叠、可删除
- 🗑️ **文档管理** — 上传 / 列表 / 删除文档，支持 PDF、Markdown、TXT

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| 关系数据库 | MySQL 8.0 + SQLAlchemy 2.0 (async) |
| 向量数据库 | Chroma (HNSW) |
| 向量模型 | Qwen text-embedding-v3 (1024d) |
| LLM | Qwen-Plus / DeepSeek / OpenAI (OpenAI 兼容协议) |
| Reranker | BAAI/bge-reranker-v2-m3 (Cross-encoder) |
| 前端 | React 18 + TypeScript + Vite |
| UI 组件 | shadcn/ui + Radix UI + Tailwind CSS |
| 状态管理 | Zustand |
| 数据请求 | TanStack Query (React Query) |
| 流式通信 | SSE (Server-Sent Events) |
| 部署 | Docker Compose |

## API 总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/documents/upload` | 上传文档 (multipart) |
| `GET` | `/api/v1/documents` | 文档列表 (分页) |
| `DELETE` | `/api/v1/documents/{doc_id}` | 删除文档及索引 |
| `POST` | `/api/v1/search` | 语义检索 |
| `POST` | `/api/v1/ask` | 流式问答 (SSE) |
| `POST` | `/api/v1/records/{record_id}/cancel` | 取消正在生成的回答 |
| `GET` | `/api/v1/qa/conversations` | 对话列表 (分页) |
| `GET` | `/api/v1/qa/history?conversation_id=` | 对话历史记录 |
| `DELETE` | `/api/v1/qa/conversations/{id}` | 删除对话 |
| `GET` | `/api/v1/qa/suggestions` | 获取推荐问题 (基于已上传文档) |

## 快速开始

### 1. 环境准备

```bash
git clone <repo-url> agentic-rag-kb
cd agentic-rag-kb

# 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入：
#   LLM_API_KEY、EMBEDDING_API_KEY、MYSQL_PASSWORD
```

### 2. 启动 MySQL

```bash
docker run -d --name mysql-rag \
  -e MYSQL_ROOT_PASSWORD=your_password \
  -e MYSQL_DATABASE=agentic_rag \
  -p 3306:3306 mysql:8.0
```

### 3. 后端启动

```bash
cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 确保 MySQL 已运行，.env 已配置
uvicorn app.main:app --reload --port 8000
```

### 4. 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`

### 5. Docker 一键部署

```bash
docker compose up -d
```

服务端口：
- 前端：`http://localhost:3000`
- 后端：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

## 项目结构

```
agentic-rag-kb/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口 + 生命周期
│   │   ├── config.py            # pydantic-settings 配置 (12 个配置组)
│   │   ├── api/                 # API 路由
│   │   │   ├── chat.py          # 流式问答 + 取消
│   │   │   ├── documents.py     # 文档 CRUD
│   │   │   ├── health.py        # 健康检查
│   │   │   ├── history.py       # 对话历史 + 建议
│   │   │   └── search.py        # 语义检索
│   │   ├── core/                # 核心逻辑
│   │   │   ├── agent.py         # Agent 循环 (rewrite → search → rerank → check)
│   │   │   ├── pipeline.py      # 文档摄入流水线
│   │   │   ├── parser/          # PDF / MD / TXT 解析器
│   │   │   ├── chunker/         # 文本分块器
│   │   │   ├── embedder.py      # 向量化
│   │   │   └── storage.py       # 文件存储
│   │   ├── db/                  # 数据库封装
│   │   │   ├── mysql.py         # MySQL 连接 + 会话
│   │   │   ├── chroma.py        # Chroma 向量存储
│   │   │   └── migrate.py       # 自动建表
│   │   ├── models/              # Pydantic + SQLAlchemy 模型
│   │   └── utils/               # 日志 / Token 计数 / 请求 ID
│   ├── db/                      # 数据库脚本
│   │   └── init.sql             # MySQL 初始化
│   ├── data/                    # 持久化数据 (chroma/ + uploads/)
│   ├── tests/                   # 单元测试
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 主应用组件
│   │   ├── api/                 # API 调用层
│   │   │   ├── documents.ts     # 文档接口
│   │   │   ├── history.ts       # 对话历史接口
│   │   │   ├── chat.ts          # SSE 流式问答
│   │   │   └── client.ts        # HTTP 客户端
│   │   ├── components/          # UI 组件
│   │   │   ├── layout/          # Sidebar 布局
│   │   │   ├── qa/              # 问答面板 (Welcome / Answer / Source)
│   │   │   ├── upload/          # 上传 + 文档卡片
│   │   │   ├── ui/              # shadcn/ui 组件 (Button / AlertDialog / Collapsible)
│   │   │   └── common/          # 通用组件
│   │   ├── stores/              # Zustand 状态 (qaStore / uiStore)
│   │   ├── hooks/               # useSSE / useSourceScroll
│   │   ├── lib/                 # cn() / utils
│   │   └── types/               # TypeScript 类型定义
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── docs/                        # 设计文档
├── docker-compose.yml
├── .env.example
└── README.md
```

## 配置说明

完整配置项见 `.env.example`。关键配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商: dashscope / openai / deepseek / custom | dashscope |
| `LLM_MODEL` | 模型名称 | qwen-plus |
| `EMBEDDING_MODEL` | 向量模型 | text-embedding-v3 |
| `RERANKER_DEVICE` | Reranker 运行设备 | cpu |
| `MYSQL_DATABASE` | 数据库名 | agentic_rag |
| `AGENT_MAX_ROUNDS` | 最大检索轮数 | 3 |
| `AGENT_TOP_K_RERANK` | 精排后保留片段数 | 5 |
| `CORS_ORIGINS` | 允许的前端域名 (逗号分隔) | http://localhost:5173 |

## 工作流程

```
用户上传文档 → 解析 (PDF/MD/TXT) → 分块 → 向量化 → 存入 Chroma + MySQL
                                                              ↓
用户提问 → Agent 改写 Query → 多路检索 → Cross-encoder 精排
    ↓                                                          ↓
LLM 评估上下文质量 ← 不足则补充检索（最多3轮）
    ↓
上下文充足 → LLM 生成带引用的答案 → SSE 流式返回前端
```

## License

MIT
