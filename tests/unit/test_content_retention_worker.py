"""Unit tests for ContentRetentionWorker."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.workers.content_retention import ContentRetentionWorker


def _make_worker(**kwargs: object) -> ContentRetentionWorker:
    defaults: dict[str, object] = {
        "service": AsyncMock(),
        "interval": 900,
    } | kwargs
    return ContentRetentionWorker(**defaults)  # type: ignore[arg-type]


async def test_worker_start_stop_idempotent() -> None:
    worker = _make_worker()

    await worker.start()
    await worker.start()  # idempotent — second call is a no-op
    assert worker.is_running is True

    await worker.stop()
    await worker.stop()  # idempotent — second call is a no-op
    assert worker.is_running is False


async def test_worker_run_once_calls_sweep() -> None:
    service = AsyncMock()
    worker = _make_worker(service=service)

    await worker.run_once()

    service.sweep.assert_awaited_once()


async def test_worker_run_once_propagates_service_exception() -> None:
    """run_once must not swallow failures — PeriodicWorker._run_loop only
    advances _last_tick_at when run_once returns without raising, so a
    persistently failing sweep must be visible to WorkerHeartbeatChecker."""
    service = AsyncMock()
    service.sweep = AsyncMock(side_effect=RuntimeError("boom"))
    worker = _make_worker(service=service)

    with pytest.raises(RuntimeError, match="boom"):
        await worker.run_once()


async def test_worker_uses_no_leader_lease() -> None:
    """D5: content retention deliberately opts out of the leader lease."""
    worker = _make_worker()
    # Lease is disabled regardless of Redis availability -> always acquires.
    assert await worker._lease.acquire_or_renew() is True
    assert worker.is_leader is True
