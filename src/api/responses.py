"""Shared response helper for building uniform ErrorEnvelope responses."""

from __future__ import annotations

from typing import Any

from litestar import Response

from src.api.schemas.errors import ErrorEnvelope


def error_response(
    error: str, message: str, status_code: int, detail: dict[str, Any] | None = None
) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(error=error, message=message, status_code=status_code, detail=detail),
        status_code=status_code,
    )
