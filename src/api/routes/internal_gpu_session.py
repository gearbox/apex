"""Internal GPU session endpoints — no user auth; node bearer auth validated per-request.

URL contract (must match Aisha telemetry v2 and the P3 command-queue claim contract):
  POST /v1/internal/gpu-sessions/{session_id}/operations/{operation_id}/events
  POST /v1/internal/gpu-sessions/{session_id}/commands/claim
  Authorization: Bearer {ACS_APEX_CALLBACK_TOKEN}

This controller has NO guards — authentication is performed in the handler
by comparing a SHA-256 hash of the presented bearer token against the stored
callback_token_hash on the session row (constant-time compare).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import structlog
from litestar import Controller, Request, Response, post
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
)

from src.api.responses import error_response as _error
from src.api.schemas.gpu_session import ClaimCommandRequest, OperationEventBody
from src.api.services.gpu_session.command_service import GpuSessionCommandService
from src.api.services.gpu_session.operation_event_service import OperationEventService

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


def _extract_bearer_token(request: Request[Any, Any, Any]) -> str | None:
    """Extract and validate presence of a Bearer token. None on missing/empty."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[len("Bearer ") :]
    return token or None


class InternalGpuSessionController(Controller):
    """Internal endpoints for GPU node callbacks — no JWT auth guard."""

    path = "/v1/internal/gpu-sessions"
    tags: Sequence[str] | None = ("Internal",)
    # No guards: node bearer auth is validated in the handler.

    @post("/{session_id:uuid}/operations/{operation_id:uuid}/events", status_code=HTTP_200_OK)
    async def operation_event(
        self,
        session_id: UUID,
        operation_id: UUID,
        request: Request[Any, Any, Any],
        data: Annotated[OperationEventBody, Body()],
        operation_event_service: OperationEventService,
    ) -> Response[dict[str, Any]]:
        """Receive one v2 operation event from a GPU node."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "gpu_session.operation.rejected",
                session_id=str(session_id),
                reason="missing_bearer",
            )
            return _error("unauthorized", "Missing Bearer token", HTTP_401_UNAUTHORIZED)
        bearer_token = auth_header[len("Bearer ") :]
        if not bearer_token:
            logger.warning(
                "gpu_session.operation.rejected",
                session_id=str(session_id),
                reason="empty_bearer",
            )
            return _error("unauthorized", "Empty Bearer token", HTTP_401_UNAUTHORIZED)

        if data.session_id != session_id:
            logger.warning(
                "gpu_session.operation.rejected",
                session_id=str(session_id),
                reason="session_id_mismatch",
                body_session_id=str(data.session_id),
            )
            return _error("bad_request", "session_id mismatch", HTTP_400_BAD_REQUEST)
        if data.operation_id != operation_id:
            logger.warning(
                "gpu_session.operation.rejected",
                session_id=str(session_id),
                operation_id=str(operation_id),
                reason="operation_id_mismatch",
                body_operation_id=str(data.operation_id),
            )
            return _error("bad_request", "operation_id mismatch", HTTP_400_BAD_REQUEST)
        if data.schema_version != 2:
            return _error(
                "bad_request",
                "Unsupported schema_version; expected 2",
                HTTP_400_BAD_REQUEST,
            )

        authorized, status = await operation_event_service.handle_event(
            session_id=session_id,
            bearer_token=bearer_token,
            event=data,
        )
        if not authorized:
            return _error("unauthorized", "Invalid callback token", HTTP_401_UNAUTHORIZED)
        if status == HTTP_404_NOT_FOUND:
            return _error("not_found", "Unknown operation", HTTP_404_NOT_FOUND)
        return Response(content={"ok": True}, status_code=HTTP_200_OK)

    @post("/{session_id:uuid}/commands/claim", status_code=HTTP_200_OK)
    async def claim_command(
        self,
        session_id: UUID,
        request: Request[Any, Any, Any],
        data: Annotated[ClaimCommandRequest, Body()],
        gpu_session_command_service: GpuSessionCommandService,
    ) -> Response[Any]:
        """Claim one queued command for a node's provisioning agent.

        Status codes are chosen for the agent's reaction, not REST aesthetics (D29):
        204 for no work AND for a terminal session (never 404/409 — the node is
        simply shutting down), 401 for bad/missing auth, 400 for a schema mismatch,
        200 + envelope for a successful claim. Never 5xx on a normal path — that
        costs the agent a full tenacity cycle. Responds immediately (D30, no
        long-polling): the agent's own HTTP timeout is 30s and its poll interval 5s.
        """
        bearer_token = _extract_bearer_token(request)
        if bearer_token is None:
            logger.warning(
                "gpu_session.command.claim_rejected",
                session_id=str(session_id),
                reason="missing_or_empty_bearer",
            )
            return _error("unauthorized", "Missing Bearer token", HTTP_401_UNAUTHORIZED)
        if data.schema_version != 2:
            return _error(
                "bad_request",
                "Unsupported schema_version; expected 2",
                HTTP_400_BAD_REQUEST,
            )
        if not data.agent_id:
            return _error("bad_request", "agent_id must be non-empty", HTTP_400_BAD_REQUEST)

        status, envelope = await gpu_session_command_service.claim(
            session_id=session_id,
            bearer_token=bearer_token,
            agent_id=data.agent_id,
        )
        if status == HTTP_401_UNAUTHORIZED:
            logger.warning(
                "gpu_session.command.claim_rejected",
                session_id=str(session_id),
                reason="invalid_token_or_unknown_session",
            )
            return _error("unauthorized", "Invalid callback token", HTTP_401_UNAUTHORIZED)
        if status == HTTP_204_NO_CONTENT:
            # Explicit empty body — Litestar's own 204 handling isn't relied on here.
            return Response(content=b"", status_code=HTTP_204_NO_CONTENT)
        return Response(content=envelope, status_code=HTTP_200_OK)
