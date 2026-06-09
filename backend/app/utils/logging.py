"""Loguru-based logging configuration.

Provides:
- ``setup_logging()`` — configure loguru with console + daily-rotating file sinks.
- ``RequestIDMiddleware`` — inject a unique ``X-Request-ID`` into every request.
- ``request_id()`` — context-aware helper to retrieve the current request ID.
"""

from __future__ import annotations

import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable shared across the request lifecycle.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def request_id() -> str:
    """Return the current request ID (or ``"-"`` when outside a request)."""
    return _request_id_ctx.get()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that assigns a unique request ID and exposes it in
    the response header ``X-Request-ID``.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = _request_id_ctx.set(rid)
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            _request_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _formatter(record: dict) -> str:
    """Custom log format with request ID when available."""
    rid = _request_id_ctx.get("-")
    if rid != "-":
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>[{extra[rid]}]</cyan> | "
            "<level>{message}</level>\n"
        ).format(rid=rid, **record)
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>\n"
    ).format(**record)


def setup_logging(*, log_dir: str = "./data/logs", level: str = "INFO") -> None:
    """Initialise loguru with console + daily-rotating file sinks.

    Call once at application startup.
    """
    # Remove the default handler (stderr).
    logger.remove()

    # Console sink — coloured, compact.
    logger.add(
        sys.stderr,
        format=_formatter,
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # File sink — daily rotation, 30-day retention, gzip compression.
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        format=_formatter,
        level="DEBUG",  # File keeps everything for debugging.
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="gz",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging initialised — console={level}, file=DEBUG, dir={log_path}")
