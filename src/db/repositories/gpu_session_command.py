"""Repository for the GPU session command queue (P3).

Callers own transactions — no ``commit()`` here, matching every other repository in
this codebase.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import case, func, select, update

from src.core.enums import CommandStatus, OperationKind
from src.db.models.gpu_session_command import GpuSessionCommand

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


class GpuSessionCommandRepository:
    """Persist and atomically claim/close GPU session command-queue rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        id: UUID,
        session_id: UUID,
        product_id: str,
        operation_id: UUID,
        kind: OperationKind | str,
        payload: dict[str, Any],
        deployment_id: UUID | None = None,
        batch_id: str | None = None,
        batch_index: int | None = None,
        batch_total: int | None = None,
    ) -> GpuSessionCommand:
        """Insert a queued command. Caller has already validated payload/kind."""
        command = GpuSessionCommand(
            id=id,
            session_id=session_id,
            product_id=product_id,
            operation_id=operation_id,
            deployment_id=deployment_id,
            kind=kind,
            payload=payload,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
            status=CommandStatus.queued,
        )
        self._session.add(command)
        await self._session.flush()
        return command

    async def get_by_operation(self, operation_id: UUID) -> GpuSessionCommand | None:
        """Look up the command paired with one operation, if any."""
        result = await self._session.execute(
            select(GpuSessionCommand).where(GpuSessionCommand.operation_id == operation_id)
        )
        return result.scalar_one_or_none()

    async def claim(
        self,
        session_id: UUID,
        agent_id: str,
        *,
        now: datetime,
        deadline_for: Callable[[str], int],
    ) -> GpuSessionCommand | None:
        """D22+D23 in one guarded statement — no read-then-write.

        The target row is either (a) a command this exact ``agent_id`` already holds
        claimed — a lost-response retry lands back on it, ``claimed_at``/``deadline_at``
        untouched (D23) — or (b) the oldest queued command, gated by NOT EXISTS on any
        claimed row for the session (D9's one-in-flight rule; the partial unique index
        on the table is the backstop, not the mechanism). The two branches are
        mutually exclusive by construction, so exactly one can ever match. FOR UPDATE
        SKIP LOCKED inside the same statement is what makes two concurrent claims on
        the same session race-free without an application lock.

        Returns None (→ 204) when nothing matches: a different agent already holds
        the session's one in-flight command, or the queue is empty.
        """
        already_claimed = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.session_id == session_id,
                GpuSessionCommand.status == CommandStatus.claimed,
            )
            .exists()
        )
        candidate = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.session_id == session_id,
                (
                    (
                        (GpuSessionCommand.status == CommandStatus.claimed)
                        & (GpuSessionCommand.agent_id == agent_id)
                    )
                    | ((GpuSessionCommand.status == CommandStatus.queued) & ~already_claimed)
                ),
            )
            # created_at ties within one enqueue_batch transaction (Postgres evaluates
            # CURRENT_TIMESTAMP once per transaction) — batch_index breaks the tie so
            # batch members are handed out in order.
            .order_by(
                GpuSessionCommand.created_at.asc(),
                GpuSessionCommand.batch_index.asc().nulls_first(),
            )
            .with_for_update(skip_locked=True)
            .limit(1)
            .scalar_subquery()
        )

        # Per-kind deadline (D26), computed once in Python per kind (the kind space is
        # exactly these three) and embedded as a SQL CASE so the whole claim — target
        # row, status flip, and deadline — is still one statement.
        deadline_case = case(
            (
                GpuSessionCommand.kind == OperationKind.bundle_provision.value,
                now + timedelta(seconds=deadline_for(OperationKind.bundle_provision.value)),
            ),
            (
                GpuSessionCommand.kind == OperationKind.bundle_removal.value,
                now + timedelta(seconds=deadline_for(OperationKind.bundle_removal.value)),
            ),
            else_=now + timedelta(seconds=deadline_for(OperationKind.comfyui_restart.value)),
        )

        result = await self._session.execute(
            update(GpuSessionCommand)
            .where(GpuSessionCommand.id == candidate)
            .values(
                status=CommandStatus.claimed,
                # coalesce: a self-reclaim (branch a) must not move these fields.
                agent_id=func.coalesce(GpuSessionCommand.agent_id, agent_id),
                claimed_at=func.coalesce(GpuSessionCommand.claimed_at, now),
                deadline_at=func.coalesce(GpuSessionCommand.deadline_at, deadline_case),
            )
            .returning(GpuSessionCommand)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalar_one_or_none()

    async def mark_terminal(
        self,
        command_id: UUID,
        *,
        status: CommandStatus,
        at: datetime,
        error: str | None = None,
    ) -> bool:
        """Guarded terminal write. A second terminal call on the same command no-ops."""
        values: dict[str, object] = {"status": status, "terminal_at": at}
        if error is not None:
            values["error"] = error
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionCommand)
                .where(
                    GpuSessionCommand.id == command_id,
                    GpuSessionCommand.terminal_at.is_(None),
                )
                .values(**values)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def expire_overdue(self, now: datetime) -> Sequence[GpuSessionCommand]:
        """Move claimed commands past their deadline to expired.

        Data-only: the caller (the sweep worker) is responsible for failing each
        returned command's paired operation — this repo doesn't touch operations.
        """
        result = await self._session.execute(
            update(GpuSessionCommand)
            .where(
                GpuSessionCommand.status == CommandStatus.claimed,
                GpuSessionCommand.deadline_at.is_not(None),
                GpuSessionCommand.deadline_at < now,
            )
            .values(status=CommandStatus.expired, terminal_at=now, error="deadline exceeded")
            .returning(GpuSessionCommand)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()

    async def cancel_for_session(
        self, session_id: UUID, *, at: datetime, reason: str
    ) -> Sequence[GpuSessionCommand]:
        """D31: move every queued/claimed command for a session to cancelled.

        Called from the two D15 lifecycle chokepoints, alongside the deployment
        cascade, when a session transitions to stopped/failed.
        """
        result = await self._session.execute(
            update(GpuSessionCommand)
            .where(
                GpuSessionCommand.session_id == session_id,
                GpuSessionCommand.status.in_((CommandStatus.queued, CommandStatus.claimed)),
            )
            .values(status=CommandStatus.cancelled, terminal_at=at, error=reason)
            .returning(GpuSessionCommand)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()
