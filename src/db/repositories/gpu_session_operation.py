"""Repository for latest-state GPU session operation telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, select, update

from src.core.enums import TERMINAL_OPERATION_STATUSES, OperationKind, OperationStatus
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
        kind: OperationKind | str,
        target_bundle: str | None = None,
        target_bundle_version: str | None = None,
        target_mode: str | None = None,
        batch_id: str | None = None,
        batch_index: int | None = None,
        batch_total: int | None = None,
        command_id: UUID | None = None,
    ) -> GpuSessionOperation:
        """Insert an Apex-owned operation in its initial queued state."""
        operation = GpuSessionOperation(
            id=id,
            session_id=session_id,
            product_id=product_id,
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
                .values(**values)
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
