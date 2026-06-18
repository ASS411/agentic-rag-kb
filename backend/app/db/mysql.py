"""MySQL async connection pool via SQLAlchemy + aiomysql.

Provides:
- Module-level ``async_engine`` singleton.
- ``async_session_factory`` for per-request sessions.
- ``get_db()`` FastAPI dependency that yields an ``AsyncSession``.
- ``dispose_engine()`` for graceful shutdown.
- ``Base`` declarative base for ORM models.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ---------------------------------------------------------------------------
# Engine (module-level singleton — created once, reused across requests)
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    settings.mysql.async_dsn,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connections before use
    echo=False,
)

# Session factory — callable that produces AsyncSession instances.
async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` and close it after the request.

    Usage (FastAPI route)::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


async def dispose_engine() -> None:
    """Dispose the connection pool.  Call during application shutdown."""
    await async_engine.dispose()
