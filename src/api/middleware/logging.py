"""Request logging middleware for structured per-request context."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog
from litestar.enums import ScopeType
from litestar.middleware import AbstractMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.core.uid import new_id

if TYPE_CHECKING:
    from litestar.types import Receive, Scope, Send

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

        # Store in scope state for downstream access (guards, dependencies, services)
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        bind_contextvars(
            request_id=request_id,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
            client_ip=client_ip,
        )

        start = time.perf_counter()
        logger.info("request.started")

        # Wrap send to inject X-Request-Id into the response and capture the status code.
        rid_header_value = request_id.encode()
        status_code: int | None = None

        async def send_with_request_id(message: dict) -> None:  # type: ignore[type-arg]
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-request-id", rid_header_value))
                message = {**message, "headers": headers_list}
            await send(message)  # type: ignore[arg-type]

        completed = False
        try:
            await self.app(scope, receive, send_with_request_id)  # type: ignore[arg-type]
            completed = True
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            # 5xx are server faults → warning; everything else (incl. expected 4xx) stays info.
            # status_code is None only if the app never sent http.response.start (e.g. disconnect).
            if status_code is not None and status_code >= 500:
                logger.warning("request.finished", duration_ms=duration_ms, status_code=status_code)
            else:
                logger.info("request.finished", duration_ms=duration_ms, status_code=status_code)
            if completed:
                # Success path only: clear context so it doesn't leak to the next request.
                # On exception we intentionally preserve context — Litestar's
                # ExceptionHandlerMiddleware runs after this block, so exception handler
                # logs will still carry request_id, method, path, client_ip.
                # The next request's clear_contextvars() call (top of __call__) handles cleanup.
                clear_contextvars()
