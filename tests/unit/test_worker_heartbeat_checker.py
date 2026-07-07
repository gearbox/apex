"""Tests for WorkerHeartbeatChecker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from src.api.services.health.checkers.workers import WorkerHeartbeatChecker
from src.core.enums import ComponentStatus

if TYPE_CHECKING:
    from src.api.services.health.base import ComponentHealth


def _meta(result: ComponentHealth, worker_name: str) -> dict[str, object]:
    return cast("dict[str, object]", result.metadata[worker_name])


def _make_worker(
    *,
    name: str = "test_worker",
    is_running: bool = True,
    is_leader: bool = True,
    interval_seconds: float = 60.0,
    initial_delay_seconds: float = 0.0,
    started_at: datetime | None = None,
    last_tick_at: datetime | None = None,
    last_tick_duration_ms: int | None = 5,
) -> MagicMock:
    worker = MagicMock()
    worker.name = name
    worker.is_running = is_running
    worker.is_leader = is_leader
    worker.interval_seconds = interval_seconds
    worker.initial_delay_seconds = initial_delay_seconds
    worker.started_at = started_at or datetime.now(UTC)
    worker.last_tick_at = last_tick_at
    worker.last_tick_duration_ms = last_tick_duration_ms
    return worker


class TestAllFreshHealthy:
    async def test_all_fresh_healthy(self) -> None:
        w1 = _make_worker(name="w1", last_tick_at=datetime.now(UTC) - timedelta(seconds=5))
        w2 = _make_worker(name="w2", last_tick_at=datetime.now(UTC) - timedelta(seconds=10))

        checker = WorkerHeartbeatChecker(workers=[w1, w2])
        result = await checker.check()

        assert result.status == ComponentStatus.healthy
        assert _meta(result, "w1")["stale"] is False
        assert _meta(result, "w2")["stale"] is False


class TestStaleWorkerDegrades:
    async def test_stale_worker_degrades(self) -> None:
        fresh = _make_worker(name="fresh", last_tick_at=datetime.now(UTC) - timedelta(seconds=5))
        stale = _make_worker(
            name="stale",
            interval_seconds=60.0,
            last_tick_at=datetime.now(UTC) - timedelta(seconds=500),  # > max(180, 120)
        )

        checker = WorkerHeartbeatChecker(workers=[fresh, stale])
        result = await checker.check()

        assert result.status == ComponentStatus.degraded
        assert _meta(result, "fresh")["stale"] is False
        assert _meta(result, "stale")["stale"] is True

    async def test_short_interval_uses_120s_floor(self) -> None:
        """A 1s-interval worker isn't flagged stale just past 3x interval (3s)."""
        worker = _make_worker(
            interval_seconds=1.0,
            last_tick_at=datetime.now(UTC) - timedelta(seconds=30),
        )

        checker = WorkerHeartbeatChecker(workers=[worker])
        result = await checker.check()

        assert result.status == ComponentStatus.healthy


class TestNeverTickedWithinGrace:
    async def test_never_ticked_within_grace_is_healthy(self) -> None:
        worker = _make_worker(
            interval_seconds=60.0,
            initial_delay_seconds=5.0,
            started_at=datetime.now(UTC) - timedelta(seconds=10),
            last_tick_at=None,
        )

        checker = WorkerHeartbeatChecker(workers=[worker])
        result = await checker.check()

        assert result.status == ComponentStatus.healthy
        assert _meta(result, worker.name)["stale"] is False

    async def test_never_ticked_past_grace_is_stale(self) -> None:
        worker = _make_worker(
            interval_seconds=10.0,
            initial_delay_seconds=0.0,
            started_at=datetime.now(UTC) - timedelta(seconds=1000),  # grace = 2*10 = 20s
            last_tick_at=None,
        )

        checker = WorkerHeartbeatChecker(workers=[worker])
        result = await checker.check()

        assert result.status == ComponentStatus.degraded
        assert _meta(result, worker.name)["stale"] is True


class TestNonLeaderNotStale:
    async def test_non_leader_not_stale_even_when_never_ticked(self) -> None:
        worker = _make_worker(
            is_leader=False,
            started_at=datetime.now(UTC) - timedelta(hours=1),
            last_tick_at=None,
        )

        checker = WorkerHeartbeatChecker(workers=[worker])
        result = await checker.check()

        assert result.status == ComponentStatus.healthy
        assert _meta(result, worker.name)["stale"] is False

    async def test_non_leader_not_stale_even_with_old_last_tick(self) -> None:
        worker = _make_worker(
            is_leader=False,
            last_tick_at=datetime.now(UTC) - timedelta(hours=1),
        )

        checker = WorkerHeartbeatChecker(workers=[worker])
        result = await checker.check()

        assert result.status == ComponentStatus.healthy
        assert _meta(result, worker.name)["stale"] is False


class TestStoppedWorkerExcluded:
    async def test_stopped_worker_excluded(self) -> None:
        stopped = _make_worker(
            name="stopped",
            is_running=False,
            last_tick_at=datetime.now(UTC) - timedelta(hours=1),
        )
        running = _make_worker(name="running", last_tick_at=datetime.now(UTC))

        checker = WorkerHeartbeatChecker(workers=[stopped, running])
        result = await checker.check()

        assert "stopped" not in result.metadata
        assert "running" in result.metadata
        assert result.status == ComponentStatus.healthy


class TestCheckerIdentity:
    async def test_name_category_product_id(self) -> None:
        from src.core.enums import ComponentCategory

        checker = WorkerHeartbeatChecker(workers=[])
        assert checker.name == "background_workers"
        assert checker.category == ComponentCategory.workers
        assert checker.product_id is None

    async def test_empty_workers_is_healthy(self) -> None:
        checker = WorkerHeartbeatChecker(workers=[])
        result = await checker.check()
        assert result.status == ComponentStatus.healthy
        assert result.metadata == {}
