"""Unified API response model and global exception handlers.

Provides:
- ``APIResponse[T]`` — generic success/error wrapper consumed by every endpoint.
- App-level exception classes with HTTP status codes.
- Exception handlers that convert exceptions into structured JSON responses.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Unified response model
# ---------------------------------------------------------------------------


class APIResponse(BaseModel, Generic[T]):
    """Every endpoint returns this envelope.

    Success example::

        APIResponse.ok(data=doc)

    Error example::

        APIResponse.error(code=404, message="Document not found")
    """

    success: bool = True
    code: int = 200
    message: str = "ok"
    data: T | None = None

    @classmethod
    def ok(cls, data: T | None = None, message: str = "ok") -> "APIResponse[T]":
        """Build a 200 success response."""
        return cls(success=True, code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int, message: str) -> "APIResponse[None]":
        """Build an error response (success=False)."""
        return cls(success=False, code=code, message=message, data=None)


# ---------------------------------------------------------------------------
# Business exception hierarchy
# ---------------------------------------------------------------------------


class AppException(Exception):
    """Base application exception.

    Raise subclasses in service / core layers; the global handler turns
    them into ``APIResponse`` with the appropriate HTTP status.
    """

    def __init__(self, code: int = 500, message: str = "Internal server error") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class NotFoundException(AppException):
    """Resource not found (HTTP 404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(code=404, message=message)


class ValidationException(AppException):
    """Business-level validation error (HTTP 422)."""

    def __init__(self, message: str = "Validation error") -> None:
        super().__init__(code=422, message=message)


class ConflictException(AppException):
    """Resource conflict (HTTP 409)."""

    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(code=409, message=message)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

from starlette.exceptions import HTTPException as StarletteHTTPException


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Convert a Starlette ``HTTPException`` (404, 405, etc.) into structured error."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse.error(exc.status_code, exc.detail).model_dump(),
    )


async def app_exception_handler(
    request: Request, exc: AppException
) -> JSONResponse:
    """Convert an ``AppException`` (or subclass) into a structured error."""
    return JSONResponse(
        status_code=exc.code,
        content=APIResponse.error(exc.code, exc.message).model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert a Pydantic ``RequestValidationError`` into a structured error."""
    detail_parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        detail_parts.append(f"{loc}: {error['msg']}")
    detail = "; ".join(detail_parts)

    return JSONResponse(
        status_code=422,
        content=APIResponse.error(422, detail).model_dump(),
    )


async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for any unhandled exception (HTTP 500)."""
    return JSONResponse(
        status_code=500,
        content=APIResponse.error(500, "Internal server error").model_dump(),
    )
