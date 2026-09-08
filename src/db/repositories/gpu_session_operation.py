"""Repository for latest-state GPU session operation telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, or_, select, update

from src.core.enums import TERMINAL_OPERATION_STATUSES, OperationKind, OperationStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.gpu_session_operation import GpuSessionOperation

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class EventOutcome:
    """Result of attempting one monotonic operation-event update."""

    applied: bool
    reason: str


class GpuSessionOperationRepository:
    """Persist and atomically advance operation rows; callers own transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        id: UUID,
        session_id: UUID,
        product_id: str,
        user_id: UUID | None = None,
        kind: OperationKind | str,
        deployment_id: UUID | None = None,
        target_bundle: str | None = None,
        target_bundle_version: str | None = None,
        target_mode: str | None = None,
        batch_id: str | None = None,
        batch_index: int | None = None,
        batch_total: int | None = None,
        command_id: UUID | None = None,
    ) -> GpuSessionOperation:
        """Insert an Apex-owned operation in its initial queued state."""
        if user_id is None:
            user_id = await self._session.scalar(
                select(GpuSession.user_id).where(GpuSession.id == session_id)
            )
        if user_id is None:
            raise ValueError(f"Cannot create operation for unknown GPU session {session_id}")
        operation = GpuSessionOperation(
            id=id,
            session_id=session_id,
            deployment_id=deployment_id,
            product_id=product_id,
            user_id=user_id,
            command_id=command_id,
            kind=kind,
            status=OperationStatus.queued,
            target_bundle=target_bundle,
            target_bundle_version=target_bundle_version,
            target_mode=target_mode,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
            last_sequence=-1,
            revision=0,
        )
        self._session.add(operation)
        await self._session.flush()
        return operation

    async def get(self, operation_id: UUID) -> GpuSessionOperation | None:
        """Return one operation by its Apex-owned identifier."""
        return await self._session.get(GpuSessionOperation, operation_id, populate_existing=True)

    async def get_many(self, operation_ids: set[UUID]) -> dict[UUID, GpuSessionOperation]:
        """Fetch response projections in one query, keyed by operation id."""
        if not operation_ids:
            return {}
        result = await self._session.execute(
            select(GpuSessionOperation).where(GpuSessionOperation.id.in_(operation_ids))
        )
        operations = result.scalars().all()
        return {operation.id: operation for operation in operations}

    async def get_for_user(
        self, operation_id: UUID, session_id: UUID, user_id: UUID, product_id: str
    ) -> GpuSessionOperation | None:
        """Return an operation only when its session belongs to this user/product."""
        result = await self._session.execute(
            select(GpuSessionOperation)
            .join(GpuSession, GpuSessionOperation.session_id == GpuSession.id)
            .where(
                GpuSessionOperation.id == operation_id,
                GpuSessionOperation.session_id == session_id,
                GpuSession.user_id == user_id,
                GpuSession.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def latest_by_deployment(self, session_id: UUID) -> dict[UUID, GpuSessionOperation]:
        """Fetch the newest operation for every deployment in a session in one query.

        Cohort restarts are session-scoped operations, so their fan-out lives on
        the deployment restart pointer rather than the operation's singular
        ``deployment_id``. Resolve both relationships and pick the newest
        operation per deployment; a stale, terminal pointer must not hide a
        later deployment-scoped removal or provision operation.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment.id, GpuSessionOperation)
            .join(
                GpuSessionOperation,
                or_(
                    GpuSessionOperation.deployment_id == GpuSessionDeployment.id,
                    GpuSessionOperation.id == GpuSessionDeployment.restart_operation_id,
                ),
            )
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionOperation.session_id == session_id,
            )
            .distinct(GpuSessionDeployment.id)
            .order_by(
                GpuSessionDeployment.id,
                GpuSessionOperation.created_at.desc(),
                GpuSessionOperation.id.desc(),
            )
        )
        operations_by_deployment: dict[UUID, GpuSessionOperation] = {}
        for deployment_id, operation in result.all():
            operations_by_deployment[deployment_id] = operation
        return operations_by_deployment

    async def latest_for_deployment_and_kind(
        self, deployment_id: UUID, kind: OperationKind | str
    ) -> GpuSessionOperation | None:
        """Return a deployment's newest operation of one kind."""
        kind_value = kind.value if isinstance(kind, OperationKind) else kind
        result = await self._session.execute(
            select(GpuSessionOperation)
            .where(
                GpuSessionOperation.deployment_id == deployment_id,
                GpuSessionOperation.kind == kind_value,
            )
            .order_by(GpuSessionOperation.created_at.desc(), GpuSessionOperation.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def apply_event(
        self,
        *,
        operation_id: UUID,
        session_id: UUID,
        sequence: int,
        event_id: str,
        status: OperationStatus,
        phase: str | None,
        node_started_at: datetime,
        event_at: datetime,
        message: str,
        progress: dict[str, Any] | None,
        plan: dict[str, Any] | None,
        summary: dict[str, Any] | None,
        error: str | None,
        target_bundle_version: str | None = None,
    ) -> EventOutcome:
        """Apply an event using one guarded update rather than a read/write race.

        Terminal rows are immutable: the terminal guard rejects every later event,
        including a higher-sequence non-terminal best-effort update.
        """
        is_terminal = status in TERMINAL_OPERATION_STATUSES
        values: dict[str, object] = {
            "last_sequence": sequence,
            "last_event_id": event_id,
            "last_event_at": event_at,
            "node_started_at": node_started_at,
            "status": status,
            "phase": phase,
            "message": message,
            "updated_at": func.now(),
        }
        if progress is not None:
            values["progress"] = progress
        if plan is not None:
            values["plan"] = plan
        if summary is not None:
            values["summary"] = summary
        if error is not None:
            values["error"] = error
        if target_bundle_version is not None:
            # A node resolves ``current`` to a concrete version. Preserve any version
            # Apex pinned when creating the operation rather than silently replacing it.
            values["target_bundle_version"] = func.coalesce(
                GpuSessionOperation.target_bundle_version, target_bundle_version
            )
        if is_terminal:
            values["terminal_at"] = event_at

        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionOperation)
                .where(
                    GpuSessionOperation.id == operation_id,
                    GpuSessionOperation.session_id == session_id,
                    GpuSessionOperation.last_sequence < sequence,
                    # Once terminal, no late best-effort event may overwrite the durable result.
                    GpuSessionOperation.terminal_at.is_(None),
                )
                .values(revision=GpuSessionOperation.revision + 1, **values)
            ),
        )
        await self._session.flush()
        if result.rowcount == 1:
            return EventOutcome(applied=True, reason="applied")

        # This follow-up is only on the guarded-update miss. It classifies logs;
        # correctness remains exclusively in the conditional UPDATE above.
        current = await self.get(operation_id)
        if current is None or current.session_id != session_id:
            return EventOutcome(applied=False, reason="unknown")
        if sequence == current.last_sequence:
            if event_id == current.last_event_id:
                return EventOutcome(applied=False, reason="duplicate")
            return EventOutcome(applied=False, reason="sequence_collision")
        if sequence < current.last_sequence:
            return EventOutcome(applied=False, reason="stale")
        if current.terminal_at is not None:
            if is_terminal:
                return EventOutcome(applied=False, reason="terminal_after_terminal")
            return EventOutcome(applied=False, reason="after_terminal")
        return EventOutcome(applied=False, reason="stale")

    async def close_failed(self, operation_id: UUID, *, at: datetime, error: str) -> bool:
        """Guarded out-of-band terminal write for a command that never got a real event.

        Used by the P3 command-expiry sweep and the D31 teardown cascade to fail an
        operation whose paired command timed out or was cancelled rather than
        completing through a node-reported apply_event. Same terminal_at IS NULL
        guard as apply_event, so a real event that already closed the operation is
        never overwritten — whichever of the two arrives first wins.
        """
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionOperation)
                .where(
                    GpuSessionOperation.id == operation_id,
                    GpuSessionOperation.terminal_at.is_(None),
                )
                .values(
                    status=OperationStatus.failed,
                    terminal_at=at,
                    error=error,
                    message=error,
                    revision=GpuSessionOperation.revision + 1,
                )
            ),
        )
        await self._session.flush()
        return result.rowcount == 1
