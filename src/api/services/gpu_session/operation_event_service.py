"""Receiver for Aisha telemetry v2 operation events.

The receiver is a pure writer: it validates node bearer auth, atomically updates
the latest state of one operation, and updates the bootstrap stall projection.
GpuProvisioningWorker remains the owner of all GPU session state transitions.

SECURITY: never log tokens.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.core.enums import TERMINAL_GPU_SESSION_STATUSES
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.schemas.gpu_session import OperationEventBody

logger = structlog.get_logger(__name__)


def _validate_token(presented: str, stored_hash: str | None) -> bool:
    """Constant-time comparison of presented bearer token against stored SHA-256 hash."""
    if not stored_hash:
        return False
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    return hmac.compare_digest(presented_hash, stored_hash)


class OperationEventService:
    """Validate and persist operation telemetry from GPU session nodes."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def handle_event(
        self,
        *,
        session_id: UUID,
        bearer_token: str,
        event: OperationEventBody,
    ) -> tuple[bool, int]:
        """Validate and apply one v2 envelope, returning (authorized, HTTP status)."""
        async with self._session_factory() as db, db.begin():
            session_repo = GpuSessionRepository(db)
            session = await session_repo.get_by_id(session_id)
            if session is None:
                logger.warning(
                    "gpu_session.operation.rejected",
                    session_id=str(session_id),
                    reason="session_not_found",
                )
                return False, 401

            if not _validate_token(bearer_token, session.callback_token_hash):
                logger.warning(
                    "gpu_session.operation.rejected",
                    session_id=str(session_id),
                    reason="invalid_token",
                )
                return False, 401

            if session.status in TERMINAL_GPU_SESSION_STATUSES:
                logger.info(
                    "gpu_session.operation.ignored_terminal_session",
                    session_id=str(session_id),
                    session_status=str(session.status),
                    operation_id=str(event.operation_id),
                )
                return True, 200

            operation_repo = GpuSessionOperationRepository(db)
            operation = await operation_repo.get(event.operation_id)
            if operation is None or operation.session_id != session_id:
                logger.error(
                    "gpu_session.operation.unknown",
                    session_id=str(session_id),
                    operation_id=str(event.operation_id),
                    reason="not_found" if operation is None else "cross_session",
                )
                return True, 404

            outcome = await operation_repo.apply_event(
                operation_id=event.operation_id,
                session_id=session_id,
                sequence=event.sequence,
                event_id=event.event_id,
                status=event.status,
                phase=event.phase.value if event.phase is not None else None,
                node_started_at=event.started_at,
                event_at=event.ts,
                message=event.message,
                progress=event.progress,
                plan=event.plan,
                summary=event.summary,
                error=event.error,
                target_bundle_version=(
                    event.target.bundle_version if event.target is not None else None
                ),
            )
            if not outcome.applied:
                log = (
                    logger.warning
                    if outcome.reason
                    in {
                        "sequence_collision",
                        "terminal_after_terminal",
                    }
                    else logger.debug
                )
                log(
                    "gpu_session.operation.not_applied",
                    session_id=str(session_id),
                    operation_id=str(event.operation_id),
                    sequence=event.sequence,
                    reason=outcome.reason,
                )
                return True, 200

            if event.operation_id == session.bootstrap_operation_id:
                await session_repo.touch_last_progress(session.id, datetime.now(UTC))

        logger.info(
            "gpu_session.operation.applied",
            session_id=str(session_id),
            operation_id=str(event.operation_id),
            sequence=event.sequence,
        )
        return True, 200
