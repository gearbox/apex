"""Product resolution middleware."""

from __future__ import annotations

from urllib.parse import urlparse

import structlog
from litestar.middleware.base import AbstractMiddleware
from litestar.types import Receive, Scope, Send

from src.core.config import get_settings
from src.core.product_registry import resolve_product_by_domain, resolve_product_by_slug

logger = structlog.get_logger()


class ProductMiddleware(AbstractMiddleware):
    """Resolve product context from request headers.

    Resolution order:
    1. Origin header (preferred — set by browsers on cross-origin requests)
    2. Host header
    3. X-Product-Id header (development fallback)
    4. localhost/dev fallback to DEFAULT_PRODUCT env var

    Sets scope["state"]["product_config"] and scope["state"]["product_id"].
    Adds X-Product-Id response header.
    Returns 400 if no product can be resolved.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))

        product_config = None

        # 1. Try Origin header
        origin = headers.get(b"origin", b"").decode("utf-8", errors="ignore")
        if origin:
            hostname = urlparse(origin).hostname or ""
            if hostname:
                product_config = resolve_product_by_domain(hostname)

        # 2. Try Host header
        if product_config is None:
            host = headers.get(b"host", b"").decode("utf-8", errors="ignore")
            if host:
                hostname = host.split(":")[0].lower()
                product_config = resolve_product_by_domain(hostname)

        # 3. Try X-Product-Id header
        if product_config is None:
            x_product_id = headers.get(b"x-product-id", b"").decode("utf-8", errors="ignore")
            if x_product_id:
                product_config = resolve_product_by_slug(x_product_id)

        # 4. Try localhost/dev fallback
        if product_config is None:
            host = headers.get(b"host", b"").decode("utf-8", errors="ignore").split(":")[0]
            if host in ("localhost", "127.0.0.1", "0.0.0.0"):
                settings = get_settings()
                product_config = resolve_product_by_slug(settings.default_product)

        if product_config is None:
            import msgspec

            body = msgspec.json.encode(
                {
                    "error": "unknown_product",
                    "message": "Cannot resolve product from request origin",
                }
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})  # type: ignore[arg-type]
            return

        # Set in scope state
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["product_config"] = product_config
        scope["state"]["product_id"] = product_config.slug

        # Bind to structlog context
        structlog.contextvars.bind_contextvars(product_id=product_config.slug)

        # Wrap send to add X-Product-Id header
        product_id_bytes = product_config.slug.encode()

        async def send_with_product_header(message: dict) -> None:  # type: ignore[type-arg]
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-product-id", product_id_bytes))
                message = {**message, "headers": headers_list}
            await send(message)  # type: ignore[arg-type]

        await self.app(scope, receive, send_with_product_header)  # type: ignore[arg-type]
