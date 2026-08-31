"""Background workers: periodic health checks -> DB snapshot + Redis publish,
and daily retention cleanup of old snapshots.

Two separate PeriodicWorker subclasses (D7): snapshot persistence and
retention cleanup run on very different cadences (60s vs daily), so they
each get their own tick loop and their own leader lease key rather than
sharing one class with two internal tasks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import msgspec
import structlog

from src.api.schemas.ops_events import PLATFORM_PRODUCT_ID, OpsEventType
from src.api.services.health import HEALTH_STREAM_CHANNEL
from src.api.services.health.transitions import BAD_STATUSES, detect_transitions
from src.db.repositories.health import HealthSnapshotRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

    from src.api.services.gpu_session.credit_guard import SessionCreditGuard
    from src.api.services.health.service import HealthService
    from src.api.services.ops_event_bus import OpsEventBus
    from src.db.session import DatabaseManager

logger = structlog.get_logger()

_encoder = msgspec.json.Encoder()


class HealthSnapshotWorker(PeriodicWorker):
    """Periodic health snapshot persistence + SSE publish."""

    def __init__(
        self,
        health_service: HealthService,
        db_manager: DatabaseManager,
        interval_seconds: int,
        redis_client_factory: Callable[[], Redis],
        redis_url: str | None = None,
        session_credit_guard: SessionCreditGuard | None = None,
        ops_event_bus: OpsEventBus | None = None,
    ) -> None:
        super().__init__(
            name="health_snapshot",
            interval_seconds=interval_seconds,
            initial_delay_seconds=5.0,
            redis_enabled=redis_url is not None,
            redis_client_factory=redis_client_factory,
        )
        self._health_service = health_service
        self._db_manager = db_manager
        self._redis_url = redis_url
        self._session_credit_guard = session_credit_guard
        self._ops_event_bus = ops_event_bus

    # ------------------------------------------------------------------
    # Snapshot tick
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Execute a single snapshot cycle: check → load previous → persist →
        detect transitions → publish."""
        start = time.monotonic()

        detailed = await self._health_service.check_all_and_build()

        # Load the previous cycle's snapshot BEFORE persisting the new one —
        # comparing against the DB (not in-memory state) survives restarts
        # and leader failover without duplicate or missed edges (D5).
        previous_snapshot_data: dict[str, Any] | None = None
        try:
            previous_snapshot_data = await self._load_previous_snapshot_data()
        except Exception:
            logger.exception("health.snapshot_worker.load_previous_error")

        # Persist to DB
        persisted = False
        try:
            await self._persist_snapshot(detailed)
            persisted = True
        except Exception:
            logger.exception("health.snapshot_worker.persist_error")

        # Publish to Redis for SSE consumers
        try:
            await self._publish_to_redis(detailed)
        except Exception:
            logger.exception("health.snapshot_worker.publish_error")

        # Detect per-subsystem transitions and publish ops events. Never
        # propagates — a Redis blip must not affect the already-persisted
        # snapshot. Gated on persist success: publishing against
        # previous_snapshot_data when this cycle's snapshot never landed
        # would re-detect the same edge next cycle (get_latest() still
        # returns the older row) and fire a duplicate alert — deferring to
        # the next successful cycle instead.
        if persisted:
            try:
                await self._detect_and_publish_transitions(previous_snapshot_data, detailed)
            except Exception:
                logger.exception("health.snapshot_worker.transition_detect_error")
        else:
            logger.warning("health.snapshot_worker.transitions_skipped_unpersisted")

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.debug(
            "health.snapshot_worker.tick",
            status=detailed["status"],
            duration_ms=duration_ms,
        )

        # Run credit guard after health checks — outside the timed section so
        # its latency doesn't inflate the reported health check duration.
        if self._session_credit_guard is not None:
            try:
                await self._session_credit_guard.run_cycle()
            except Exception:
                logger.exception("health.snapshot_worker.credit_guard_error")

    async def _load_previous_snapshot_data(self) -> dict[str, Any] | None:
        """Fetch the previous cycle's persisted snapshot_data, or None on first run."""
        async with self._db_manager.session() as session:
            repo = HealthSnapshotRepository(session)
            previous = await repo.get_latest()
            return previous.snapshot_data if previous is not None else None

    async def _persist_snapshot(self, detailed: dict[str, Any]) -> None:
        """Write health snapshot to the database."""
        from datetime import datetime

        async with self._db_manager.session() as session:
            repo = HealthSnapshotRepository(session)
            await repo.insert(
                checked_at=datetime.fromisoformat(detailed["checked_at"]),
                overall_status=detailed["status"],
                snapshot_data=detailed,
            )
            await session.commit()

    async def _detect_and_publish_transitions(
        self, previous_snapshot_data: dict[str, Any] | None, detailed: dict[str, Any]
    ) -> None:
        """Diff the current cycle against the previous one and publish ops events."""
        if self._ops_event_bus is None:
            return

        transitions = detect_transitions(previous_snapshot_data, detailed)
        for transition in transitions:
            event_type = (
                OpsEventType.HEALTH_SUBSYSTEM_DEGRADED
                if transition.current_status in BAD_STATUSES
                else OpsEventType.HEALTH_SUBSYSTEM_RESTORED
            )
            logger.info(
                "health.transition_detected",
                subsystem=transition.subsystem,
                previous_status=transition.previous_status,
                current_status=transition.current_status,
            )
            await self._ops_event_bus.publish(
                event_type=event_type,
                product_id=PLATFORM_PRODUCT_ID,
                payload=transition,
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
        await client.publish(HEALTH_STREAM_CHANNEL, data)


class HealthSnapshotCleanupWorker(PeriodicWorker):
    """Daily retention cleanup of old health snapshots."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        retention_days: int,
        *,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="health_snapshot_cleanup",
            interval_seconds=86400,
            initial_delay_seconds=60.0,
            jitter_seconds=300.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._db_manager = db_manager
        self._retention_days = retention_days

    async def run_once(self) -> None:
        """Delete snapshots older than the retention period."""
        async with self._db_manager.session() as session:
            repo = HealthSnapshotRepository(session)
            count = await repo.cleanup(retention_days=self._retention_days)
            await session.commit()

        if count > 0:
            logger.info(
                "health.snapshot_cleanup_worker.cleanup",
                deleted=count,
                retention_days=self._retention_days,
            )
