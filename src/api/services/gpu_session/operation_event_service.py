"""Receiver for Aisha telemetry v2 operation events.

The receiver is a pure writer: it validates node bearer auth, atomically updates
the latest state of one operation, and updates the bootstrap stall projection.
GpuProvisioningWorker remains the owner of all GPU session state transitions.

SECURITY: never log tokens. This is also the trust boundary for every node-supplied
free-text field on the wire envelope (``message``, ``error``, and the open-ended
``progress``/``plan``/``summary`` bodies, where a URL carrying the callback token can
hide at any depth): each is passed through ``redact_secrets``/``redact_secrets_mapping``
here, once, before it reaches a repository write — never re-derive this at an
individual call site. D3 put the callback token into the node's own environment
(``PROVISIONING_SCRIPT``/``PROVISIONER_WEBHOOK_URL`` query strings), so any node-side
code path that echoes a failing command or an env dump into telemetry can otherwise
leak it straight into ``gpu_session_operations``, ``gpu_session_commands.error``, and
the client's SSE stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.security.callback_token import validate_callback_token
from src.api.services.gpu_session.failure_reasons import MAX_GPU_SESSION_FAILURE_REASON_LENGTH
from src.api.utils.redaction import redact_secrets, redact_secrets_mapping
from src.core.enums import (
    TERMINAL_GPU_SESSION_STATUSES,
    TERMINAL_OPERATION_STATUSES,
    CommandStatus,
    OperationStatus,
)
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.gpu_session import OperationEventBody
    from src.db.models.gpu_session_operation import GpuSessionOperation
    from src.db.repositories.gpu_session_operation import EventOutcome

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OperationEventResult:
    """Writer outcome consumed by the callback route after its transaction commits."""

    authorized: bool
    status: int
    outcome: EventOutcome | None = None
    operation: GpuSessionOperation | None = None


class OperationEventService:
    """Validate and persist operation telemetry using a caller-owned transaction."""

    async def handle_event(
        self,
        *,
        session_id: UUID,
        bearer_token: str,
        event: OperationEventBody,
        db: AsyncSession,
    ) -> OperationEventResult:
        """Apply one envelope without committing or publishing anything."""
        return await self._handle_event(
            session_id=session_id,
            bearer_token=bearer_token,
            event=event,
            db=db,
        )

    async def _handle_event(
        self,
        *,
        session_id: UUID,
        bearer_token: str,
        event: OperationEventBody,
        db: AsyncSession,
    ) -> OperationEventResult:
        """Validate and apply one envelope without committing or publishing anything."""
        session_repo = GpuSessionRepository(db)
        session = await session_repo.get_by_id(session_id)
        if session is None:
            logger.warning(
                "gpu_session.operation.rejected",
                session_id=str(session_id),
                reason="session_not_found",
            )
            return OperationEventResult(authorized=False, status=401)

        if not validate_callback_token(bearer_token, session.callback_token_hash):
            logger.warning(
                "gpu_session.operation.rejected", session_id=str(session_id), reason="invalid_token"
            )
            return OperationEventResult(authorized=False, status=401)

        if session.status in TERMINAL_GPU_SESSION_STATUSES:
            logger.info(
                "gpu_session.operation.ignored_terminal_session",
                session_id=str(session_id),
                session_status=str(session.status),
                operation_id=str(event.operation_id),
            )
            return OperationEventResult(authorized=True, status=200)

        operation_repo = GpuSessionOperationRepository(db)
        operation = await operation_repo.get(event.operation_id)
        if operation is None or operation.session_id != session_id:
            logger.error(
                "gpu_session.operation.unknown",
                session_id=str(session_id),
                operation_id=str(event.operation_id),
                reason="not_found" if operation is None else "cross_session",
            )
            return OperationEventResult(authorized=True, status=404)

        # SECURITY: the trust boundary — see module docstring. Every node-supplied
        # free-text field is redacted here, before it reaches a repository write.
        redacted_message = redact_secrets(
            event.message, max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH
        )
        redacted_error = (
            redact_secrets(event.error, max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH)
            if event.error is not None
            else None
        )
        redacted_progress = redact_secrets_mapping(
            event.progress, max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH
        )
        redacted_plan = redact_secrets_mapping(
            event.plan, max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH
        )
        redacted_summary = redact_secrets_mapping(
            event.summary, max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH
        )

        outcome = await operation_repo.apply_event(
            operation_id=event.operation_id,
            session_id=session_id,
            sequence=event.sequence,
            event_id=event.event_id,
            status=event.status,
            phase=event.phase.value if event.phase is not None else None,
            node_started_at=event.started_at,
            event_at=event.ts,
            message=redacted_message,
            progress=redacted_progress,
            plan=redacted_plan,
            summary=redacted_summary,
            error=redacted_error,
            target_bundle_version=(
                event.target.bundle_version if event.target is not None else None
            ),
        )
        if not outcome.applied:
            log = (
                logger.warning
                if outcome.reason in {"sequence_collision", "terminal_after_terminal"}
                else logger.debug
            )
            log(
                "gpu_session.operation.not_applied",
                session_id=str(session_id),
                operation_id=str(event.operation_id),
                sequence=event.sequence,
                reason=outcome.reason,
            )
            return OperationEventResult(
                authorized=True,
                status=200,
                outcome=outcome,
                operation=operation,
            )

        if event.operation_id == session.bootstrap_operation_id:
            await session_repo.touch_last_progress(session.id, datetime.now(UTC))

        # P3/D27: terminal telemetry closes its command in the same transaction.
        if event.status in TERMINAL_OPERATION_STATUSES and operation.command_id is not None:
            closed = await GpuSessionCommandRepository(db).mark_terminal(
                operation.command_id,
                status=(
                    CommandStatus.succeeded
                    if event.status == OperationStatus.succeeded
                    else CommandStatus.failed
                ),
                at=event.ts,
                error=redacted_error,
            )
            if closed:
                logger.info(
                    "gpu_session.command.closed",
                    session_id=str(session_id),
                    command_id=str(operation.command_id),
                    operation_id=str(event.operation_id),
                    status=event.status.value,
                )

        refreshed = await operation_repo.get(event.operation_id)
        if refreshed is None:
            raise RuntimeError(
                f"Operation {event.operation_id} disappeared after its event was applied"
            )
        logger.info(
            "gpu_session.operation.applied",
            session_id=str(session_id),
            operation_id=str(event.operation_id),
            sequence=event.sequence,
        )
        return OperationEventResult(
            authorized=True,
            status=200,
            outcome=outcome,
            operation=refreshed,
        )
