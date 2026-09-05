"""Background worker: expires claimed GPU session commands past their deadline (P3/P6).

A claimed command with no terminal event by deadline_at means the node vanished, the
claim response was processed but the node then died mid-run, or the node is simply
stuck. Either way, the session must not wait forever for a command that will never
report back — this worker frees it so the agent's next claim can move on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.core.config import Settings

logger = structlog.get_logger(__name__)


class GpuSessionCommandSweepWorker(PeriodicWorker):
    """Expires overdue claimed commands and fails their paired operations."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="gpu_command_sweep",
            interval_seconds=settings.gpu_command_sweep_interval_seconds,
            initial_delay_seconds=15.0,
            jitter_seconds=5.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._session_factory = session_factory

    async def run_once(self) -> None:
        """Expire overdue claims and cancel queued commands orphaned by a terminal session."""
        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            command_repo = GpuSessionCommandRepository(db)
            operation_repo = GpuSessionOperationRepository(db)
            expired = await command_repo.expire_overdue(now)
            for command in expired:
                elapsed_seconds = (
                    (now - command.claimed_at).total_seconds()
                    if command.claimed_at is not None
                    else 0.0
                )
                message = (
                    f"command timed out after {elapsed_seconds:.0f}s "
                    f"(kind={command.kind}, deadline_at="
                    f"{command.deadline_at.isoformat() if command.deadline_at else 'unknown'})"
                )
                await operation_repo.close_failed(command.operation_id, at=now, error=message)
                logger.warning(
                    "gpu_session.command.expired",
                    command_id=str(command.id),
                    operation_id=str(command.operation_id),
                    session_id=str(command.session_id),
                    kind=command.kind,
                    elapsed_seconds=elapsed_seconds,
                )

            orphaned = await command_repo.cancel_queued_for_terminal_sessions(at=now)
            for command in orphaned:
                message = "session reached a terminal state before command could be claimed"
                await operation_repo.close_failed(command.operation_id, at=now, error=message)
                logger.warning(
                    "gpu_session.command.orphaned",
                    command_id=str(command.id),
                    operation_id=str(command.operation_id),
                    session_id=str(command.session_id),
                    kind=command.kind,
                )
