"""Receiver for the Vast.ai provisioner failure webhook (Change 3).

The provisioner POSTs here after max_retries are exhausted for the node's onstart
script (e.g. Phase 9 fetching the bootstrap script over an authorization it doesn't
have). A reachable ComfyUI is not evidence of successful provisioning (2026-09-13
staging incident) — this receiver is Apex's own signal, independent of its probe,
that a node has already given up.

SECURITY: never log the callback token.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from litestar.status_codes import HTTP_200_OK, HTTP_401_UNAUTHORIZED

from src.api.services.gpu_session.operation_event_service import _validate_token
from src.core.enums import STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.schemas.provisioning import ProvisionerFailureWebhookBody
    from src.api.services.gpu_session.service import GpuSessionService

logger = structlog.get_logger(__name__)

# Distinct from GpuProvisioningWorker's own _REASON_* constants (a different module's
# failure taxonomy) but following the same naming convention.
_REASON_NODE_PROVISION_SCRIPT_FAILED = "node_provision_script_failed"

# Reason strings are persisted as error_message (String, effectively unbounded in
# Postgres but kept short for logs/UI); the upstream `error` field is free text.
_MAX_REASON_LENGTH = 500


class ProvisioningWebhookService:
    """Validates and applies one provisioner failure callback.

    Delegates the actual teardown + refund to GpuSessionService.fail_pre_active_session
    (D8) — this class is only auth, idempotency, and the mismatch-log policy (D9).
    """

    def __init__(
        self,
        *,
        gpu_session_service: GpuSessionService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._gpu_session_service = gpu_session_service
        self._session_factory = session_factory

    async def handle_failure(
        self,
        *,
        session_id: UUID,
        token: str | None,
        payload: ProvisionerFailureWebhookBody,
    ) -> int:
        """Apply one webhook call. Returns the HTTP status the route should send.

        Ordering matches the spec exactly: not-found/already-terminal is a cheap,
        unauthenticated 200 no-op (D9 — a 404 would make the provisioner's own
        retry loop spam this endpoint); only then is the token checked.
        """
        async with self._session_factory() as db:
            session_row = await GpuSessionRepository(db).get_by_id(session_id)

        if session_row is None:
            logger.info("provisioning.webhook.noop", session_id=str(session_id), reason="not_found")
            return HTTP_200_OK
        if session_row.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
            logger.info(
                "provisioning.webhook.noop",
                session_id=str(session_id),
                status=str(session_row.status),
            )
            return HTTP_200_OK

        if not token or not _validate_token(token, session_row.callback_token_hash):
            logger.warning(
                "provisioning.webhook.rejected", session_id=str(session_id), reason="invalid_token"
            )
            return HTTP_401_UNAUTHORIZED

        # The token is the authority, not the container id (D9) — mismatch is only
        # ever a warning-level observability signal, never a reason to skip the fail.
        if (
            session_row.vastai_instance_id is not None
            and str(session_row.vastai_instance_id) != payload.container_id
        ):
            logger.warning(
                "provisioning.webhook.instance_mismatch",
                session_id=str(session_id),
                expected_instance_id=session_row.vastai_instance_id,
                container_id=payload.container_id,
            )

        reason = f"{_REASON_NODE_PROVISION_SCRIPT_FAILED}: {payload.error}"[:_MAX_REASON_LENGTH]
        await self._gpu_session_service.fail_pre_active_session(session_id, reason=reason)
        return HTTP_200_OK
