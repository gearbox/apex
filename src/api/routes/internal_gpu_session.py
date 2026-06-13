"""Internal GPU session endpoints — no user auth; node bearer auth validated per-request.

URL contract (must match what aisha's ProvisioningReporter constructs):
  POST /v1/internal/gpu-sessions/{session_id}/provisioning
  Authorization: Bearer {ACS_APEX_CALLBACK_TOKEN}

This controller has NO guards — authentication is performed in the handler
by comparing a SHA-256 hash of the presented bearer token against the stored
callback_token_hash on the session row (constant-time compare).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

import msgspec
import structlog
from litestar import Controller, Request, Response, post
from litestar.params import Body
from litestar.status_codes import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.gpu_session import DownloadProgressBody
from src.api.services.gpu_session.provisioning_callback_service import ProvisioningCallbackService

logger = structlog.get_logger(__name__)


class ProvisioningCallbackBody(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """Mirrors the payload emitted by aisha's ProvisioningReporter."""

    session_id: UUID
    phase: str
    message: str = ""
    download: DownloadProgressBody | None = None
    elapsed_seconds: int = 0
    error: str | None = None
    ts: datetime


def _error(status_code: int, error: str, message: str) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(error=error, message=message, status_code=status_code),
        status_code=status_code,
    )


class InternalGpuSessionController(Controller):
    """Internal endpoints for GPU node callbacks — no JWT auth guard."""

    path = "/v1/internal/gpu-sessions"
    tags: Sequence[str] | None = ["Internal"]
    # No guards: node bearer auth is validated in the handler.

    @post("/{session_id:uuid}/provisioning", status_code=HTTP_200_OK)
    async def provisioning_callback(
        self,
        session_id: UUID,
        request: Request[Any, Any, Any],
        data: Annotated[ProvisioningCallbackBody, Body()],
        provisioning_callback_service: ProvisioningCallbackService,
    ) -> Response[dict[str, Any]]:
        """Receive a provisioning progress callback from a GPU node.

        Returns 401 for missing/invalid bearer token.
        Returns 400 if path and body session_id disagree.
        Returns 200 for all other outcomes (status-gated, stale-ts, and successful writes)
        so the node never retries on a benign race.
        """
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _error(HTTP_401_UNAUTHORIZED, "unauthorized", "Missing Bearer token")
        bearer_token = auth_header[len("Bearer ") :]

        if not bearer_token:
            return _error(HTTP_401_UNAUTHORIZED, "unauthorized", "Empty Bearer token")

        if data.session_id != session_id:
            logger.warning(
                "gpu_session.callback.session_id_mismatch",
                path_session_id=str(session_id),
                body_session_id=str(data.session_id),
            )
            return _error(HTTP_400_BAD_REQUEST, "bad_request", "session_id mismatch")

        authorized, _status = await provisioning_callback_service.handle_callback(
            session_id=session_id,
            bearer_token=bearer_token,
            phase=data.phase,
            message=data.message,
            download=data.download,
            error=data.error,
            elapsed_seconds=data.elapsed_seconds,
            ts=data.ts,
        )

        if not authorized:
            return _error(HTTP_401_UNAUTHORIZED, "unauthorized", "Invalid callback token")

        return Response(content={"ok": True}, status_code=HTTP_200_OK)
