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

from src.api.security.callback_token import validate_callback_token
from src.api.utils.redaction import redact_secrets
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

_MAX_UPSTREAM_DETAIL_LENGTH = 500


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

        A callback token is validated before revealing whether a session exists or
        is terminal. Valid callbacks still receive a 200 no-op for terminal rows,
        so the provisioner does not retry a completed failure forever.
        """
        async with self._session_factory() as db:
            session_row = await GpuSessionRepository(db).get_by_id(session_id)

        if session_row is None:
            logger.warning(
                "provisioning.webhook.rejected", session_id=str(session_id), reason="invalid_token"
            )
            return HTTP_401_UNAUTHORIZED

        if not token or not validate_callback_token(token, session_row.callback_token_hash):
            logger.warning(
                "provisioning.webhook.rejected", session_id=str(session_id), reason="invalid_token"
            )
            return HTTP_401_UNAUTHORIZED

        if session_row.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
            logger.info(
                "provisioning.webhook.noop",
                session_id=str(session_id),
                status=str(session_row.status),
            )
            return HTTP_200_OK

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

        # Upstream fields can include this callback's own query-string token. They
        # are observability-only: never persist, publish, or bill against them.
        logger.warning(
            "provisioning.webhook.failure_detail",
            session_id=str(session_id),
            upstream_error=redact_secrets(payload.error, max_length=_MAX_UPSTREAM_DETAIL_LENGTH),
            upstream_manifest=redact_secrets(
                payload.manifest, max_length=_MAX_UPSTREAM_DETAIL_LENGTH
            ),
        )
        await self._gpu_session_service.fail_pre_active_session(
            session_id, reason=_REASON_NODE_PROVISION_SCRIPT_FAILED
        )
        return HTTP_200_OK
