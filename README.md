# Agentic RAG 个人知识库问答系统

不是简单的"上传文档 → 检索 → 回答"，而是一个具备**自主决策能力**的知识库 Agent。

## 核心能力

- 🔍 **自动改写问题**：LLM 分析用户问题，生成多条检索 query
- 📚 **多路检索 + 精排**：多 query 召回 → Cross-encoder 精排，取最优上下文
- 🧠 **自评估检索质量**：LLM-as-Judge 评估上下文是否充足，不足自动补充检索
- 📎 **带引用溯源**：结构化的答案输出，精确到文档、页码、原文片段
- 👁️ **Agent 思考过程可见**：Query Rewrite → Search → Rerank → Quality Check 全程流式展示

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| 关系数据库 | MySQL 8.0 |
| 向量数据库 | Chroma (HNSW) |
| 向量模型 | Qwen text-embedding-v3 (1024d) |
| LLM | Qwen-Plus (OpenAI 兼容) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 前端 | React 18 + TypeScript + Vite |
| UI | shadcn/ui + Tailwind CSS |
| 部署 | Docker Compose |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url> agentic-rag-kb
cd agentic-rag-kb

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、MYSQL_PASSWORD 等

# 启动 MySQL（Docker 方式）
docker run -d --name mysql-rag \
  -e MYSQL_ROOT_PASSWORD=your_password \
  -e MYSQL_DATABASE=agentic_rag \
  -p 3306:3306 mysql:8.0
```

### 2. 后端启动

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`

### 4. Docker 部署（可选）

```bash
docker compose up -d
```

## 项目结构

```
agentic-rag-kb/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # pydantic-settings 配置
│   │   ├── api/                 # API 路由
│   │   ├── core/                # Agent / Retriever / Parser / Chunker / Embedder
│   │   ├── db/                  # Chroma + MySQL 封装
│   │   ├── models/              # Pydantic 模型
│   │   └── utils/               # 日志 / Token 计数
│   ├── data/                    # 持久化数据
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/                 # API hooks (TanStack Query)
│   │   ├── components/          # UI 组件
│   │   ├── stores/              # Zustand 状态
│   │   ├── hooks/               # useSSE 等自定义 hook
│   │   └── types/               # TypeScript 类型
│   └── package.json
├── docker-compose.yml
├── DESIGN.md                    # 完整技术方案
└── README.md
```
