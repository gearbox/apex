"""Tests for HealthSnapshotWorker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.api.services.health.worker import (
    _LEADER_LOCK_KEY,
    HealthSnapshotWorker,
)


def _make_worker(**kwargs: object) -> HealthSnapshotWorker:
    defaults: dict[str, object] = {
        "health_service": AsyncMock(),
        "db_manager": MagicMock(),
        "interval_seconds": 60,
        "retention_days": 30,
        "redis_url": None,
    } | kwargs
    return HealthSnapshotWorker(**defaults)  # type: ignore[arg-type]


class TestStartStop:
    async def test_start_creates_tasks(self) -> None:
        worker = _make_worker()
        await worker.start()
        assert worker._task is not None
        assert worker._cleanup_task is not None
        assert worker._running is True
        await worker.stop()

    async def test_stop_cancels_tasks(self) -> None:
        worker = _make_worker()
        await worker.start()
        await worker.stop()
        assert worker._task is None
        assert worker._running is False

    async def test_double_start_is_noop(self) -> None:
        worker = _make_worker()
        await worker.start()
        first_task = worker._task
        await worker.start()
        assert worker._task is first_task
        await worker.stop()


class TestRunOnce:
    async def test_calls_check_persist_publish(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "2026-01-01T00:00:00Z"}
        )

        worker = _make_worker(health_service=mock_service)
        worker._persist_snapshot = AsyncMock()
        worker._publish_to_redis = AsyncMock()

        await worker._run_once()

        mock_service.check_all_and_build.assert_awaited_once()
        worker._persist_snapshot.assert_awaited_once()
        worker._publish_to_redis.assert_awaited_once()

    async def test_persist_error_does_not_crash(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "x"}
        )

        worker = _make_worker(health_service=mock_service)
        worker._persist_snapshot = AsyncMock(side_effect=ConnectionError("db gone"))

        # Should not raise
        await worker._run_once()

    async def test_publishes_to_redis_on_success(self) -> None:
        """Verify snapshot is published to health:stream Redis channel."""
        import msgspec

        detailed = {"status": "healthy", "checked_at": "2026-01-01T00:00:00Z"}
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(return_value=detailed)
        worker = _make_worker(health_service=mock_service, redis_url="redis://localhost")
        worker._persist_snapshot = AsyncMock()

        with patch("src.core.redis.get_redis_client") as mock_get_redis:
            mock_client = AsyncMock()
            mock_client.publish = AsyncMock()
            mock_get_redis.return_value = mock_client

            await worker._run_once()

            mock_client.publish.assert_awaited_once()
            channel = mock_client.publish.call_args[0][0]
            payload = mock_client.publish.call_args[0][1]
            assert channel == "health:stream"
            # Verify payload is valid JSON encoding of the detailed dict
            decoded = msgspec.json.decode(payload)
            assert decoded == detailed

    async def test_publish_error_does_not_crash(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "x"}
        )

        worker = _make_worker(health_service=mock_service, redis_url="redis://localhost")
        worker._persist_snapshot = AsyncMock()
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_redis.return_value.publish = AsyncMock(side_effect=ConnectionError("no redis"))
            await worker._run_once()


class TestLeaderLock:
    async def test_no_redis_always_leader(self) -> None:
        worker = _make_worker(redis_url=None)
        assert await worker._try_acquire_leader_lock() is True

    async def test_acquires_lock_with_dynamic_ttl(self) -> None:
        worker = _make_worker(redis_url="redis://localhost", interval_seconds=300)
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_client = AsyncMock()
            mock_client.set = AsyncMock(return_value=True)
            mock_redis.return_value = mock_client

            assert await worker._try_acquire_leader_lock() is True
            mock_client.set.assert_awaited_once_with(
                _LEADER_LOCK_KEY,
                "1",
                nx=True,
                ex=600,  # max(300 * 2, 90) = 600
            )

    async def test_lock_ttl_minimum_90s(self) -> None:
        worker = _make_worker(redis_url="redis://localhost", interval_seconds=10)
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_client = AsyncMock()
            mock_client.set = AsyncMock(return_value=True)
            mock_redis.return_value = mock_client

            await worker._try_acquire_leader_lock()
            call_kwargs = mock_client.set.call_args
            assert call_kwargs.kwargs["ex"] == 90  # max(10 * 2, 90) = 90

    async def test_skips_when_lock_held(self) -> None:
        worker = _make_worker(redis_url="redis://localhost")
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_client = AsyncMock()
            mock_client.set = AsyncMock(return_value=None)  # NX failed
            mock_redis.return_value = mock_client

            assert await worker._try_acquire_leader_lock() is False

    async def test_redis_error_proceeds_anyway(self) -> None:
        worker = _make_worker(redis_url="redis://localhost")
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_redis.side_effect = ConnectionError("redis down")
            # Should proceed (degrade gracefully)
            assert await worker._try_acquire_leader_lock() is True


class TestCleanupOnce:
    async def test_cleanup_calls_repository(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_worker(db_manager=mock_db_manager, retention_days=7)

        with patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.cleanup = AsyncMock(return_value=10)
            mock_repo_cls.return_value = mock_repo

            await worker._cleanup_once()

            mock_repo.cleanup.assert_awaited_once_with(retention_days=7)
            mock_session.commit.assert_awaited_once()
