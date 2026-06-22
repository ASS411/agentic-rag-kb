"""FastAPI application entry point.

Creates the app instance, configures CORS middleware, request-ID
middleware, exception handlers, and manage startup / shutdown lifecycle.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.evaluate import router as evaluate_router
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.search import router as search_router
from app.config import settings
from app.db.mysql import dispose_engine
from app.models.response import (
    AppException,
    app_exception_handler,
    global_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.utils.logging import RequestIDMiddleware, setup_logging
from db.migrate import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle startup and shutdown events.

    Startup:
        - Initialise loguru
        - Log server info
        - (Future) Initialise DB connection pool, Chroma client, etc.

    Shutdown:
        - Log shutdown
        - (Future) Close DB pool, Chroma client, etc.
    """
    # ── Startup ──────────────────────────────────────────────────────
    setup_logging(level=settings.server.log_level)

    logger.info(
        "\U0001f680 {host}:{port} 启动中...",
        host=settings.server.host,
        port=settings.server.port,
    )
    logger.info("   LLM: {}/{}", settings.llm.provider, settings.llm.model)
    logger.info("   Embedding: {}/{}", settings.embedding.provider, settings.embedding.model)
    logger.info("   CORS: {}", settings.server.cors_origins)
    logger.info("   MySQL: {}:{}/{}", settings.mysql.host, settings.mysql.port, settings.mysql.database)

    # ── Ensure database tables exist ─────────────────────────────────
    await init_db()
    logger.info("   Database tables verified")

    # ── Reconcile Chroma with MySQL (remove stale chunks) ───────────
    try:
        from app.core.storage import reconcile_chroma
        removed = await reconcile_chroma()
        if removed > 0:
            logger.warning("   Chroma sync: removed {} stale chunks", removed)
        else:
            logger.info("   Chroma sync: no stale chunks found")
    except Exception as exc:
        logger.warning("   Chroma sync skipped: {}", exc)

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────
    await dispose_engine()
    logger.info("\U0001f6d1 服务已关闭")


app = FastAPI(
    title="Agentic RAG Knowledge Base",
    description="个人知识库问答系统 — 文档上传 → 检索 → 流式答案",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Request-ID middleware (first, so every request gets an ID) ─────────
app.add_middleware(RequestIDMiddleware)

# ── CORS middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ────────────────────────────────────────────────
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# ── API routers ───────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(history_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(evaluate_router, prefix="/api/v1")
