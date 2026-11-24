"""Periodic worker that fails and refunds orphaned studio chat replies and memory drafts.

Each tick (leader only) runs the :class:`~src.studio.chat.sweep.ChatSweeper`: every artist message
and memory draft that has been ``pending`` for longer than the reply deadline plus a grace is
failed (``reply_timeout``) and its reservation refunded, one transaction per row. The replies
themselves are written by asyncio tasks (``ChatTaskSupervisor``), not by this worker; it exists
for the task that died with its process.

Started only when ``Settings.studio_chat_active`` and workers are on.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Final

import structlog

from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

    from src.studio.chat.sweep import ChatSweeper

__all__ = ["StudioChatSweepError", "StudioChatSweepWorker"]

logger = structlog.get_logger(__name__)

WORKER_NAME: Final = "studio_chat_sweep"
"""Feeds the leader-lease key; keep it stable across releases."""


class StudioChatSweepError(RuntimeError):
    """A tick failed. Carries only the cause's *type*: a database error's message can quote the
    parameters of the statement."""


class StudioChatSweepWorker(PeriodicWorker):
    """Sweeps orphaned chat replies and memory drafts."""

    def __init__(
        self,
        *,
        sweeper: ChatSweeper,
        interval_seconds: int,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        """Create the worker.

        Args:
            sweeper: Fails and refunds the orphans.
            interval_seconds: Seconds between ticks.
            redis_enabled: Whether Redis is configured (enables the leader lease).
            redis_client_factory: Factory for the operational Redis client.
        """
        super().__init__(
            name=WORKER_NAME,
            interval_seconds=interval_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._sweeper = sweeper

    async def run_once(self) -> None:
        """One tick: sweep.

        Intentionally lets a failure escape (the base loop logs ``worker.tick_error`` and leaves
        the heartbeat stale), so a persistently failing sweep is visible to the health checker - but
        as a :class:`StudioChatSweepError`, whose message never carries data.

        Raises:
            StudioChatSweepError: The sweep failed.
        """
        try:
            swept = await self._sweeper.sweep()
        except Exception as exc:  # the tick guard: re-raised below, minus the message
            logger.error(  # noqa: TRY400 - an exception message can quote user data
                "studio.chat.sweep_failed",
                exc_type=type(exc).__name__,
                frames="".join(traceback.format_tb(exc.__traceback__)),
            )
            raise StudioChatSweepError(type(exc).__name__) from None
        if swept:
            logger.info("studio.chat.swept", count=swept)
