"""Health-check endpoint — ``GET /api/v1/health``.

Returns overall application health along with per-component status.
"""

from __future__ import annotations

from sqlalchemy import text

from fastapi import APIRouter

from app.config import settings
from app.db.mysql import async_session_factory
from app.models.response import APIResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> APIResponse[dict]:
    """Return application health status.

    Checks:
    - **database** — MySQL connectivity via ``SELECT 1``.
    - **chroma** — placeholder (not yet configured).

    Returns an ``APIResponse`` with per-component details.
    """
    components: dict[str, dict] = {}

    # ── Database check ───────────────────────────────────────────────
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        components["database"] = {"status": "healthy", "message": "connected"}
    except Exception as exc:
        components["database"] = {"status": "unhealthy", "message": str(exc)}

    # ── Chroma check (placeholder) ───────────────────────────────────
    components["chroma"] = {"status": "not_configured", "message": "Chroma client not initialised"}

    # ── Aggregate status ─────────────────────────────────────────────
    statuses = {c["status"] for c in components.values()}
    if "unhealthy" in statuses:
        overall = "degraded"
    elif "not_configured" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    return APIResponse.ok(
        data={
            "status": overall,
            "version": settings.server.version,
            "components": components,
        }
    )
