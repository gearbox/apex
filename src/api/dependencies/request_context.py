"""Request-context DI provider (client IP + user agent evidence)."""

from __future__ import annotations

from typing import Any

from litestar import Request  # noqa: TC002 — resolved at runtime by Litestar DI

from src.api.middleware.rate_limit import get_real_ip
from src.api.services.legal.acceptance import RequestContext
from src.core.config import Settings  # noqa: TC001 — resolved at runtime by Litestar DI


def provide_request_context(request: Request[Any, Any, Any], settings: Settings) -> RequestContext:
    """Build the evidence context stored with legal acceptance events.

    Uses ``get_real_ip`` (trusted-header aware), never the client-controlled
    leftmost ``X-Forwarded-For`` entry.
    """
    return RequestContext(
        ip_address=get_real_ip(request, settings),
        user_agent=request.headers.get("user-agent"),
    )
