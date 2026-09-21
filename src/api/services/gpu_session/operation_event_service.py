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

T1/T11 (round-3 remediation): Layer 1 of the redactor (exact known-secret
replacement — see ``src/api/utils/redaction.py``'s module docstring) needs the
plaintext values apex actually holds; ``_known_secrets`` is built once from
``Settings`` at construction. It intentionally does not include the per-session
callback token or tunnel token — apex only ever stores the callback token's hash,
and the tunnel token is never persisted at all, so neither is in scope here (see
T8 in the round-3 remediation notes; this is a documented limitation, not an
oversight). ``event.event_id`` and ``event.target.bundle_version`` are also routed
through Layer 1 (cheap — a ``str.replace`` pass) even though they are structurally
simple, not free text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.security.callback_token import validate_callback_token
from src.api.services.gpu_session.failure_reasons import MAX_GPU_SESSION_FAILURE_REASON_LENGTH
from src.api.utils.redaction import redact_known_secrets, redact_secrets, redact_secrets_mapping
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
    from src.core.config import Settings
    from src.db.models.gpu_session_operation import GpuSessionOperation
    from src.db.repositories.gpu_session_operation import EventOutcome

logger = structlog.get_logger(__name__)


def _known_secrets_from_settings(settings: Settings) -> frozenset[str]:
    """Layer 1 exact-match set: every plaintext secret apex holds at startup."""
    return frozenset(
        value
        for value in (settings.github_content_token, settings.hf_token, settings.civitai_api_token)
        if value
    )


@dataclass(frozen=True, slots=True)
class OperationEventResult:
    """Writer outcome consumed by the callback route after its transaction commits."""

    authorized: bool
    status: int
    outcome: EventOutcome | None = None
    operation: GpuSessionOperation | None = None


class OperationEventService:
    """Validate and persist operation telemetry using a caller-owned transaction."""

    def __init__(self, *, settings: Settings) -> None:
        self._known_secrets = _known_secrets_from_settings(settings)

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
            # X2, round-5 remediation: distinct from "session_not_found" above —
            # the session exists but the token doesn't match its current hash,
            # which is expected (not a bug) for a node a provisioning retry just
            # abandoned during its callback-token rotation window.
            logger.warning(
                "gpu_session.callback.stale_token",
                session_id=str(session_id),
                reason="invalid_token",
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
            event.message,
            max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH,
            known_secrets=self._known_secrets,
        )
        redacted_error = (
            redact_secrets(
                event.error,
                max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH,
                known_secrets=self._known_secrets,
            )
            if event.error is not None
            else None
        )
        redacted_progress = redact_secrets_mapping(
            event.progress,
            max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH,
            known_secrets=self._known_secrets,
        )
        redacted_plan = redact_secrets_mapping(
            event.plan,
            max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH,
            known_secrets=self._known_secrets,
        )
        redacted_summary = redact_secrets_mapping(
            event.summary,
            max_length=MAX_GPU_SESSION_FAILURE_REASON_LENGTH,
            known_secrets=self._known_secrets,
        )
        # T11: not free text, but still routed through the cheap Layer-1-only
        # pass — an exact known-secret match must never survive even here.
        redacted_event_id = redact_known_secrets(event.event_id, self._known_secrets)

        outcome = await operation_repo.apply_event(
            operation_id=event.operation_id,
            session_id=session_id,
            sequence=event.sequence,
            event_id=redacted_event_id,
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
                redact_known_secrets(event.target.bundle_version, self._known_secrets)
                if event.target is not None and event.target.bundle_version is not None
                else None
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
