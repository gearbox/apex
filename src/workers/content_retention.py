"""Periodic background task that enforces content retention (expires_at)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.product_registry import PRODUCT_REGISTRY
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

    from src.api.services.content_retention import ContentRetentionService


class ContentRetentionWorker(PeriodicWorker):
    """Worker that periodically sweeps expired outputs/uploads (DB + R2).

    No leader lock (D5): sweep deletes are idempotent — a row already gone
    is simply not selected again, and an R2 delete of a missing key is a
    no-op — so duplicate ticks across instances are harmless. Revisit if
    Apex goes multi-replica and duplicate R2 delete traffic becomes a
    concern worth avoiding.
    """

    def __init__(
        self,
        *,
        service: ContentRetentionService,
        interval: int,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="content_retention",
            interval_seconds=interval,
            jitter_seconds=5.0,
            use_leader_lease=False,
            redis_client_factory=redis_client_factory,
        )
        self._service = service

    async def run_once(self) -> None:
        """Execute a single content retention sweep.

        Intentionally does not catch exceptions: the PeriodicWorker base loop
        logs `worker.tick_error` and, critically, leaves `_last_tick_at`
        unadvanced on failure, so a persistently failing sweep goes stale and
        WorkerHeartbeatChecker reports it unhealthy. Swallowing here would keep
        the heartbeat green while expired content silently re-accumulates.
        R2-side failures are already downgraded to a flag inside
        ContentRetentionService.sweep(), so anything propagating here is a
        DB-level failure that should surface.
        """
        await self._service.sweep()
        for product_config in PRODUCT_REGISTRY.values():
            await self._service.reconcile_storage_artifacts(product_id=product_config.slug)
