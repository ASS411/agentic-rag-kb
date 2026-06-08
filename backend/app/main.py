"""FastAPI application entry point.

Creates the app instance, configures CORS middleware, and manages
startup / shutdown lifecycle events.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# Ensure emoji-safe output on Windows terminals that default to GBK.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle startup and shutdown events.

    Startup:
        - Log server info
        - (Future) Initialise DB connection pool, Chroma client, etc.

    Shutdown:
        - Log shutdown
        - (Future) Close DB pool, Chroma client, etc.
    """
    # ── Startup ──────────────────────────────────────────────────────
    print(f"\U0001f680 {settings.server.host}:{settings.server.port} 启动中...")
    print(f"   LLM: {settings.llm.provider}/{settings.llm.model}")
    print(f"   Embedding: {settings.embedding.provider}/{settings.embedding.model}")
    print(f"   CORS: {settings.server.cors_origins}")

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────
    print("\U0001f6d1 服务已关闭")


app = FastAPI(
    title="Agentic RAG Knowledge Base",
    description="个人知识库问答系统 — 文档上传 → 检索 → 流式答案",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
