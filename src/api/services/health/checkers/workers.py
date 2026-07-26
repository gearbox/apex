"""Health checker for background PeriodicWorker liveness."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.api.services.health.base import ComponentHealth
from src.core.enums import ComponentCategory, ComponentStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.workers.base import PeriodicWorker

# Staleness threshold floor — protects short-interval workers (e.g. the 1s
# Aisha poller) from flagging on every minor scheduling jitter.
_MIN_STALE_SECONDS = 120.0


class WorkerHeartbeatChecker:
    """Reports staleness of registered PeriodicWorkers.

    A running, leader worker is stale if its last successful tick is older
    than ``max(3 * interval_seconds, 120s)``. A worker that hasn't ticked
    yet is healthy within a startup grace period of
    ``2 * interval_seconds + initial_delay_seconds`` measured from its
    ``started_at``. Non-leader workers (running but never expected to tick
    because another process holds the lease) are never reported stale.
    Stopped workers are excluded entirely.
    """

    name = "background_workers"
    category = ComponentCategory.workers
    product_id = None
    gates_readiness = True  # not in _READINESS_CATEGORIES anyway (issue #142 G2)

    def __init__(self, workers: Sequence[PeriodicWorker]) -> None:
        self._workers = workers

    async def check(self) -> ComponentHealth:
        now = datetime.now(UTC)
        metadata: dict[str, object] = {}
        any_stale = False

        for worker in self._workers:
            if not worker.is_running:
                continue

            stale = self._is_stale(worker, now)
            metadata[worker.name] = {
                "last_tick_at": worker.last_tick_at.isoformat() if worker.last_tick_at else None,
                "stale": stale,
                "duration_ms": worker.last_tick_duration_ms,
            }
            any_stale = any_stale or stale

        status = ComponentStatus.degraded if any_stale else ComponentStatus.healthy
        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=status,
            latency_ms=0.0,
            metadata=metadata,
        )

    @staticmethod
    def _is_stale(worker: PeriodicWorker, now: datetime) -> bool:
        # A non-leader worker is running but intentionally never ticking —
        # not stale, regardless of how long ago (or whether) it last ticked.
        if not worker.is_leader:
            return False

        last_tick_at = worker.last_tick_at
        if last_tick_at is None:
            grace_seconds = 2 * worker.interval_seconds + worker.initial_delay_seconds
            if worker.started_at is None:
                return False
            age_seconds = (now - worker.started_at).total_seconds()
            return age_seconds > grace_seconds

        threshold_seconds = max(3 * worker.interval_seconds, _MIN_STALE_SECONDS)
        age_seconds = (now - last_tick_at).total_seconds()
        return age_seconds > threshold_seconds
