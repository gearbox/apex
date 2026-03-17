"""Request logging middleware for structured per-request context."""

from __future__ import annotations

import time

import structlog
from litestar.enums import ScopeType
from litestar.middleware import AbstractMiddleware
from litestar.types import Receive, Scope, Send
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.core.uid import new_id

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(AbstractMiddleware):
    """Binds per-request structured context into structlog context vars.

    Generates or extracts a ``request_id``, binds ``method``, ``path``,
    and ``client_ip`` for the duration of the request, then logs
    ``request.started`` and ``request.finished`` events with timing.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle an ASGI request with request-scoped logging context.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != ScopeType.HTTP:
            await self.app(scope, receive, send)
            return

        # Skip SSE stream — it's long-lived; request.finished would fire only on disconnect
        if scope.get("path", "").startswith("/v1/events/stream"):
            await self.app(scope, receive, send)
            return

        clear_contextvars()

        headers = dict(scope.get("headers", []))
        rid_bytes = headers.get(b"x-request-id")
        request_id = rid_bytes.decode() if rid_bytes else str(new_id())

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        bind_contextvars(
            request_id=request_id,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
            client_ip=client_ip,
        )

        start = time.perf_counter()
        logger.info("request.started", request_id=request_id)

        try:
            await self.app(scope, receive, send)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info("request.finished", duration_ms=duration_ms)
            clear_contextvars()
