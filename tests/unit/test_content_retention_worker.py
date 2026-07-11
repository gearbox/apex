"""Unit tests for ContentRetentionWorker."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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


async def test_worker_loop_survives_service_exception() -> None:
    service = AsyncMock()
    service.sweep = AsyncMock(side_effect=RuntimeError("boom"))
    worker = _make_worker(service=service)

    with patch("src.workers.content_retention.logger") as logger_mock:
        await worker.run_once()  # must not raise

        logger_mock.exception.assert_called_once()
        args, kwargs = logger_mock.exception.call_args
        assert args[0] == "content_retention.error"
        assert kwargs["error"] == "boom"


async def test_worker_uses_no_leader_lease() -> None:
    """D5: content retention deliberately opts out of the leader lease."""
    worker = _make_worker()
    # Lease is disabled regardless of Redis availability -> always acquires.
    assert await worker._lease.acquire_or_renew() is True
    assert worker.is_leader is True
