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

        # Wrap send to inject X-Request-Id into the response
        rid_header_value = request_id.encode()

        async def send_with_request_id(message: dict) -> None:  # type: ignore[type-arg]
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-request-id", rid_header_value))
                message = {**message, "headers": headers_list}
            await send(message)  # type: ignore[arg-type]

        try:
            await self.app(scope, receive, send_with_request_id)  # type: ignore[arg-type]
        except BaseException:
            # Do NOT clear contextvars on exception. Litestar's
            # ExceptionHandlerMiddleware sits outside user middleware, so
            # exception handlers run AFTER this block. Preserving context
            # lets handler logs include request_id, method, path, client_ip.
            # Cleanup happens via clear_contextvars() at the top of the
            # next request.
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info("request.finished", duration_ms=duration_ms)
            raise
        else:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info("request.finished", duration_ms=duration_ms)
            clear_contextvars()
