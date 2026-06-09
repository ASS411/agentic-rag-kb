"""Database migration / initialisation helper.

Creates all ORM-mapped tables via SQLAlchemy ``Base.metadata.create_all()``.

Usage (standalone):::

    python -m db.migrate

Usage (programmatic):::

    from db.migrate import init_db
    await init_db()
"""

from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import text

from app.db.mysql import async_engine, Base


async def init_db() -> None:
    """Create all tables defined by SQLAlchemy ORM models.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``
    semantics under the hood.
    """
    async with async_engine.begin() as conn:
        # Ensure utf8mb4 is used for new tables
        await conn.execute(text("SET NAMES utf8mb4"))

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables initialised via SQLAlchemy create_all()")


# ── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(init_db())
