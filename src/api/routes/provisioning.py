"""Public provisioning endpoints for Vast.ai GPU nodes (D3, Change 1 + Change 3).

Both handlers carry NO Litestar guard, like InternalGpuSessionController — the caller
is Vast's own provisioner process, which sends no configurable headers, so auth is a
per-session token that rides in the query string (D4/D7). Redact query strings in
every log line and never surface the raw URL to a client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import structlog
from limits import parse
from limits.aio.strategies import MovingWindowRateLimiter
from litestar import Controller, Request, Response, get, post
from litestar.params import Body, Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_304_NOT_MODIFIED,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.rate_limit import get_rate_limiter_storage, get_real_ip
from src.api.responses import error_response as _error
from src.api.schemas.provisioning import ProvisionerFailureWebhookBody
from src.api.services.provisioning_script import ProvisioningScriptService
from src.api.services.provisioning_webhook import ProvisioningWebhookService
from src.core.config import Settings
from src.core.enums import ScriptServeOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


async def _script_rate_limit_ok(request: Request[Any, Any, Any], settings: Settings) -> bool:
    """Per-IP moving-window check for the script-serving route.

    Self-contained rather than routed through RateLimitMiddleware: that middleware
    keys on the exact concrete request path (`request.url.path`), which can never
    match a route with path params like {variant}/{ref}. Fails OPEN on storage
    trouble, matching the middleware's own documented posture — rate limiting is
    abuse control, not authorization.
    """
    try:
        storage = get_rate_limiter_storage()
    except RuntimeError:
        return True
    limiter = MovingWindowRateLimiter(storage)
    ip = get_real_ip(request, settings)
    key = f"rate_limit:provisioning_script:{ip}"
    limit_item = parse(settings.rate_limit_provisioning_script)
    try:
        return await limiter.hit(limit_item, key)
    except (RedisError, OSError) as exc:
        logger.warning("provisioning.script.rate_limit_storage_error", error=str(exc))
        return True


class ProvisioningController(Controller):
    """Bootstrap-script delivery + provisioner failure webhook. No JWT auth guard."""

    path = "/v1/provisioning"
    tags: Sequence[str] | None = ("Provisioning",)

    @get("/scripts/{variant:str}/{ref:str}", status_code=HTTP_200_OK)
    async def get_script(
        self,
        variant: str,
        ref: str,
        request: Request[Any, Any, Any],
        session: AsyncSession,
        provisioning_script_service: ProvisioningScriptService,
        settings: Settings,
        token: str | None = None,
        session_id: Annotated[UUID | None, Parameter(query="session")] = None,
    ) -> Response[Any]:
        """Serve the pinned bootstrap script for one GPU session's node to fetch."""
        if not await _script_rate_limit_ok(request, settings):
            logger.warning("provisioning.script.rate_limited", session_id=str(session_id))
            return _error("rate_limited", "Too many requests", HTTP_429_TOO_MANY_REQUESTS)

        result = await provisioning_script_service.serve_for_session(
            db=session, session_id=session_id, token=token, variant=variant, ref=ref
        )

        if result.outcome == ScriptServeOutcome.bad_request:
            return _error("bad_request", "Invalid variant or ref", HTTP_400_BAD_REQUEST)
        if result.outcome == ScriptServeOutcome.unauthorized:
            return _error("unauthorized", "Invalid or missing token", HTTP_401_UNAUTHORIZED)
        if result.outcome == ScriptServeOutcome.not_found:
            return _error(
                "provisioning_script_ref_not_found", "Script ref not found", HTTP_404_NOT_FOUND
            )
        if result.outcome == ScriptServeOutcome.unavailable:
            return _error(
                "provisioning_script_unavailable",
                "Script temporarily unavailable",
                HTTP_502_BAD_GATEWAY,
            )

        script = result.script
        if script is None:  # unreachable given ScriptServeResult's construction
            return _error(
                "internal_error", "Missing script content", HTTP_500_INTERNAL_SERVER_ERROR
            )

        etag = f'"{script.sha256}"'
        if request.headers.get("if-none-match") == etag:
            return Response(content=b"", status_code=HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
        return Response(
            content=script.content,
            status_code=HTTP_200_OK,
            media_type="text/plain; charset=utf-8",
            headers={"ETag": etag},
        )

    @post("/webhook/{session_id:uuid}", status_code=HTTP_200_OK)
    async def webhook(
        self,
        session_id: UUID,
        data: Annotated[ProvisionerFailureWebhookBody, Body()],
        provisioning_webhook_service: ProvisioningWebhookService,
        token: str | None = None,
    ) -> Response[dict[str, Any]]:
        """Receive the provisioner's terminal failure callback for one session."""
        status = await provisioning_webhook_service.handle_failure(
            session_id=session_id, token=token, payload=data
        )
        if status == HTTP_401_UNAUTHORIZED:
            return _error("unauthorized", "Invalid callback token", HTTP_401_UNAUTHORIZED)
        return Response(content={"ok": True}, status_code=HTTP_200_OK)
