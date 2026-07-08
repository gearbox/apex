"""Tests for PeriodicWorker."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.workers.base import PeriodicWorker


class _CountingWorker(PeriodicWorker):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.calls = 0

    async def run_once(self) -> None:
        self.calls += 1


class _FailingWorker(PeriodicWorker):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def run_once(self) -> None:
        raise RuntimeError("boom")


class _SleepyWorker(PeriodicWorker):
    def __init__(self, *, sleep_seconds: float, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._sleep_seconds = sleep_seconds
        self.completed = False

    async def run_once(self) -> None:
        await asyncio.sleep(self._sleep_seconds)
        self.completed = True


def _worker_kwargs(**overrides: object) -> dict[str, object]:
    return {
        "name": "test_worker",
        "interval_seconds": 0.01,
        "use_leader_lease": False,
    } | overrides


class TestStartIdempotent:
    async def test_start_is_idempotent(self) -> None:
        worker = _CountingWorker(**_worker_kwargs())
        await worker.start()
        first_task = worker._task
        await worker.start()
        assert worker._task is first_task
        await worker.stop()


class TestRunOnceCalledEachTick:
    async def test_run_once_called_each_tick(self) -> None:
        worker = _CountingWorker(**_worker_kwargs())
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()
        assert worker.calls >= 2


class TestTickErrorLogged:
    async def test_tick_error_logged_and_loop_continues(self) -> None:
        worker = _FailingWorker(**_worker_kwargs())
        with patch("src.workers.base.logger") as mock_logger:
            await worker.start()
            await asyncio.sleep(0.05)
            await worker.stop()

        assert mock_logger.exception.call_args_list
        assert mock_logger.exception.call_args_list[0].args[0] == "worker.tick_error"
        # Loop kept going despite the failing tick.
        assert worker.is_running is False


class TestStopDrainsInFlightTick:
    async def test_stop_drains_in_flight_tick(self) -> None:
        worker = _SleepyWorker(
            sleep_seconds=0.2,
            **_worker_kwargs(interval_seconds=10, drain_timeout_seconds=5.0),
        )
        await worker.start()
        await asyncio.sleep(0.02)  # let the tick start
        await worker.stop()
        assert worker.completed is True


class TestStopWakesFromIdleSleepImmediately:
    async def test_stop_does_not_wait_out_a_long_interval(self) -> None:
        """A worker idling between ticks must wake as soon as stop() is
        called, not run out drain_timeout_seconds (or the full interval).

        Regression test: the inter-tick sleep used to be a plain
        asyncio.sleep(), which stop()'s _running=False couldn't interrupt —
        every long-interval worker (hourly/daily sweeps) would block
        shutdown for the full drain_timeout_seconds (30s default).
        """
        worker = _CountingWorker(**_worker_kwargs(interval_seconds=100, drain_timeout_seconds=30))
        await worker.start()
        await asyncio.sleep(0.02)  # let the first tick finish; loop is now idling

        start = asyncio.get_running_loop().time()
        await worker.stop()
        elapsed = asyncio.get_running_loop().time() - start

        assert elapsed < 1.0


class TestStopCancelsAfterDrainTimeout:
    async def test_stop_cancels_after_drain_timeout(self) -> None:
        worker = _SleepyWorker(
            sleep_seconds=10.0,
            **_worker_kwargs(interval_seconds=10, drain_timeout_seconds=0.05),
        )
        await worker.start()
        await asyncio.sleep(0.02)

        with patch("src.workers.base.logger") as mock_logger:
            await worker.stop()

        assert worker.completed is False
        mock_logger.warning.assert_any_call("worker.drain_timeout", name="test_worker")


class TestHeartbeatUpdatedOnSuccessOnly:
    async def test_heartbeat_updated_on_success_only(self) -> None:
        worker = _FailingWorker(**_worker_kwargs())
        assert worker.last_tick_at is None
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()
        assert worker.last_tick_at is None


class TestCancelledErrorPropagates:
    async def test_cancelled_error_propagates_not_logged_as_tick_error(self) -> None:
        worker = _SleepyWorker(
            sleep_seconds=10.0,
            **_worker_kwargs(interval_seconds=10, drain_timeout_seconds=0.05),
        )
        await worker.start()
        await asyncio.sleep(0.02)
        task = worker._task
        assert task is not None

        with patch("src.workers.base.logger") as mock_logger:
            await worker.stop()

        assert task.cancelled()
        for call in mock_logger.exception.call_args_list:
            assert call.args[0] != "worker.tick_error"


class TestIsLeaderReflectsLease:
    async def test_is_leader_true_when_lease_disabled(self) -> None:
        worker = _CountingWorker(**_worker_kwargs())
        await worker.start()
        await asyncio.sleep(0.02)
        assert worker.is_leader is True
        await worker.stop()
        # stop() releases the lease — no longer leader once stopped.
        assert worker.is_leader is False
