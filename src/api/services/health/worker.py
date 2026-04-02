"""Background worker: periodic health checks → DB snapshot + Redis publish.

Runs as an asyncio.Task started in the app lifespan. In multi-worker
Granian deployments, a Redis leader lock ensures only one process runs
the snapshot loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import msgspec
import structlog

from src.db.repositories.health import HealthSnapshotRepository

if TYPE_CHECKING:
    from src.api.services.health.service import HealthService
    from src.db.session import DatabaseManager

logger = structlog.get_logger()

_encoder = msgspec.json.Encoder()

# Redis leader lock key
_LEADER_LOCK_KEY = "health:snapshot_worker:lock"


class HealthSnapshotWorker:
    """Periodic health snapshot persistence + SSE publish.

    Lifecycle: start() creates an asyncio.Task; stop() cancels it.
    The worker is started in the app lifespan and stopped on shutdown.
    """

    def __init__(
        self,
        health_service: HealthService,
        db_manager: DatabaseManager,
        interval_seconds: int,
        retention_days: int,
        redis_url: str | None = None,
    ) -> None:
        self._health_service = health_service
        self._db_manager = db_manager
        self._interval = interval_seconds
        self._retention_days = retention_days
        self._redis_url = redis_url
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the snapshot worker and cleanup task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "health.snapshot_worker.started",
            interval_s=self._interval,
            retention_days=self._retention_days,
        )

    async def stop(self) -> None:
        """Stop the worker and cleanup task."""
        if not self._running:
            return
        self._running = False
        for task in (self._task, self._cleanup_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._task = None
        self._cleanup_task = None
        logger.info("health.snapshot_worker.stopped")

    # ------------------------------------------------------------------
    # Snapshot loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main periodic loop: check → persist → publish."""
        # Brief initial delay to let the app finish starting
        await asyncio.sleep(5)

        while self._running:
            try:
                if not await self._try_acquire_leader_lock():
                    await asyncio.sleep(self._interval)
                    continue

                await self._run_once()
            except Exception:
                logger.exception("health.snapshot_worker.tick_error")

            if self._running:
                await asyncio.sleep(self._interval)

    async def _run_once(self) -> None:
        """Execute a single snapshot cycle."""
        start = time.monotonic()

        detailed = await self._health_service.check_all_and_build()

        # Persist to DB
        try:
            await self._health_service.persist_snapshot(detailed)
        except Exception:
            logger.exception("health.snapshot_worker.persist_error")

        # Publish to Redis for SSE consumers
        try:
            await self._publish_to_redis(detailed)
        except Exception:
            logger.exception("health.snapshot_worker.publish_error")

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.debug(
            "health.snapshot_worker.tick",
            status=detailed["status"],
            duration_ms=duration_ms,
        )

    async def _publish_to_redis(self, detailed: dict[str, Any]) -> None:
        """Publish snapshot to Redis Pub/Sub for SSE stream consumers.

        Uses a dedicated channel (not the EventBus user channels) because
        health snapshots are admin-only system events, not per-user.
        """
        if self._redis_url is None:
            return

        from src.core.redis import get_redis_client

        client = get_redis_client()
        data = _encoder.encode(detailed).decode()
        await client.publish("health:stream", data)

    # ------------------------------------------------------------------
    # Leader lock (Granian multi-worker safety)
    # ------------------------------------------------------------------

    async def _try_acquire_leader_lock(self) -> bool:
        """Try to acquire the Redis leader lock.

        Returns True if this process should run the snapshot loop.
        Without Redis, always returns True (single-process assumption).

        Lock TTL is derived from the snapshot interval with headroom
        to prevent expiry during a slow check cycle.
        """
        if self._redis_url is None:
            return True

        # TTL = 2x interval, minimum 90s — ensures the lock survives
        # one full cycle even if checks are slow
        lock_ttl = max(self._interval * 2, 90)

        try:
            from src.core.redis import get_redis_client

            client = get_redis_client()
            acquired = await client.set(
                _LEADER_LOCK_KEY,
                "1",
                nx=True,
                ex=lock_ttl,
            )
            return bool(acquired)
        except Exception:
            logger.warning("health.snapshot_worker.lock_error")
            # If Redis is down, proceed anyway (degraded but better than no checks)
            return True

    # ------------------------------------------------------------------
    # Retention cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Daily cleanup of old health snapshots."""
        # Wait until after initial startup + first snapshot cycle
        await asyncio.sleep(60)

        while self._running:
            try:
                await self._cleanup_once()
            except Exception:
                logger.exception("health.snapshot_worker.cleanup_error")

            if self._running:
                # Run daily
                await asyncio.sleep(86400)

    async def _cleanup_once(self) -> None:
        """Delete snapshots older than retention period."""
        async with self._db_manager.session() as session:
            repo = HealthSnapshotRepository(session)
            count = await repo.cleanup(retention_days=self._retention_days)
            await session.commit()

        if count > 0:
            logger.info(
                "health.snapshot_worker.cleanup",
                deleted=count,
                retention_days=self._retention_days,
            )
