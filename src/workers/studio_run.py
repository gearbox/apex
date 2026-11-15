"""Periodic worker that drives studio planner runs.

Each tick (leader only) sweeps runs that outlived their lease, claims as many queued runs as there
are free slots, and **starts** one asyncio task per claimed run - tasks are not awaited by the tick,
so a slow model call never delays the next sweep. The task set is bounded by
``studio_run_max_concurrent``.

Losing the leader lease does not stop in-flight runs: every write a run makes is guarded by its
status. On shutdown the worker waits for in-flight runs up to the drain budget; whatever is left is
failed (``run_timeout``) and refunded by the next process's sweep, never executed twice.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING, Final

import structlog

from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

    from src.studio.services.run_executor import ClaimedRun, StudioRunExecutor

__all__ = ["StudioRunTickError", "StudioRunWorker"]

logger = structlog.get_logger(__name__)

WORKER_NAME: Final = "studio_run"
"""Feeds the leader-lease key; keep it stable across releases."""


class StudioRunTickError(RuntimeError):
    """A tick failed. Carries only the cause's *type*: a database error's message can quote the
    parameters of the statement (a context snapshot holds the artist and the brief)."""


class StudioRunWorker(PeriodicWorker):
    """Sweeps, claims and launches studio runs."""

    def __init__(
        self,
        *,
        executor: StudioRunExecutor,
        interval_seconds: int,
        max_concurrent: int,
        drain_timeout_seconds: float = 30.0,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        """Create the worker.

        Args:
            executor: Claims, executes and sweeps runs.
            interval_seconds: Seconds between ticks.
            max_concurrent: Most runs this process executes at once.
            drain_timeout_seconds: How long shutdown waits for in-flight runs.
            redis_enabled: Whether Redis is configured (enables the leader lease).
            redis_client_factory: Factory for the operational Redis client.
        """
        super().__init__(
            name=WORKER_NAME,
            interval_seconds=interval_seconds,
            drain_timeout_seconds=drain_timeout_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._executor = executor
        self._max_concurrent = max_concurrent
        self._drain = drain_timeout_seconds
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def in_flight(self) -> int:
        """Runs this process is executing right now."""
        return len(self._tasks)

    async def run_once(self) -> None:
        """One tick: sweep, then claim into the free slots and start the claimed runs.

        Intentionally lets a failure escape (the base loop logs ``worker.tick_error`` and leaves the
        heartbeat stale), so a persistently failing tick is visible to the health checker - but as
        a :class:`StudioRunTickError`, whose message never carries data.

        Raises:
            StudioRunTickError: The sweep or the claim failed.
        """
        try:
            await self._executor.sweep()
            for claimed in await self._executor.claim(self._max_concurrent - len(self._tasks)):
                self._launch(claimed)
        except Exception as exc:  # the tick guard: re-raised below, minus the message
            logger.error(  # noqa: TRY400 - an exception message can quote user data
                "studio.run.tick_failed",
                exc_type=type(exc).__name__,
                frames="".join(traceback.format_tb(exc.__traceback__)),
            )
            raise StudioRunTickError(type(exc).__name__) from None

    def _launch(self, claimed: ClaimedRun) -> None:
        task = asyncio.create_task(
            self._executor.execute(claimed), name=f"studio-run-{claimed.run_id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:  # execute() never raises; this is a bug
            logger.error(
                "studio.run.task_failed", task=task.get_name(), exc_type=type(exc).__name__
            )

    async def stop(self) -> None:
        """Stop ticking, then give in-flight runs the drain budget before cancelling them."""
        await super().stop()
        if not self._tasks:
            return
        pending = set(self._tasks)
        _, still_running = await asyncio.wait(pending, timeout=self._drain)
        for task in still_running:
            task.cancel()
        if still_running:
            logger.warning("studio.run.drain_timeout", abandoned=len(still_running))
            await asyncio.gather(*still_running, return_exceptions=True)
