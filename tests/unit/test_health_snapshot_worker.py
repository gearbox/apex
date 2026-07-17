"""Tests for HealthSnapshotWorker and HealthSnapshotCleanupWorker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.api.schemas.ops_events import OpsEventType
from src.api.services.health.worker import HealthSnapshotCleanupWorker, HealthSnapshotWorker


def _make_worker(**kwargs: object) -> HealthSnapshotWorker:
    defaults: dict[str, object] = {
        "health_service": AsyncMock(),
        "db_manager": MagicMock(),
        "interval_seconds": 60,
        "redis_url": None,
    } | kwargs
    return HealthSnapshotWorker(**defaults)  # type: ignore[arg-type]


def _make_cleanup_worker(**kwargs: object) -> HealthSnapshotCleanupWorker:
    defaults: dict[str, object] = {
        "db_manager": MagicMock(),
        "retention_days": 30,
    } | kwargs
    return HealthSnapshotCleanupWorker(**defaults)  # type: ignore[arg-type]


# Generic start/stop/tick-error lifecycle is covered by test_periodic_worker.py.
# Only HealthSnapshotWorker/HealthSnapshotCleanupWorker-specific behavior lives here.


class TestRunOnce:
    async def test_calls_check_persist_publish(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "2026-01-01T00:00:00Z"}
        )

        worker = _make_worker(health_service=mock_service)
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock()  # type: ignore[method-assign]
        worker._publish_to_redis = AsyncMock()  # type: ignore[method-assign]

        await worker.run_once()

        mock_service.check_all_and_build.assert_awaited_once()
        worker._persist_snapshot.assert_awaited_once()
        worker._publish_to_redis.assert_awaited_once()

    async def test_persist_error_does_not_crash(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "x"}
        )

        worker = _make_worker(health_service=mock_service)
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock(side_effect=ConnectionError("db gone"))  # type: ignore[method-assign]

        # Should not raise
        await worker.run_once()

    async def test_publishes_to_redis_on_success(self) -> None:
        """Verify snapshot is published to health:stream Redis channel."""
        import msgspec

        detailed = {"status": "healthy", "checked_at": "2026-01-01T00:00:00Z"}
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(return_value=detailed)
        worker = _make_worker(health_service=mock_service, redis_url="redis://localhost")
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock()  # type: ignore[method-assign]

        with patch("src.core.redis.get_redis_client") as mock_get_redis:
            mock_client = AsyncMock()
            mock_client.publish = AsyncMock()
            mock_get_redis.return_value = mock_client

            await worker.run_once()

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
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock()  # type: ignore[method-assign]
        with patch("src.core.redis.get_redis_client") as mock_redis:
            mock_redis.return_value.publish = AsyncMock(side_effect=ConnectionError("no redis"))
            await worker.run_once()

    async def test_credit_guard_runs_after_checks(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "x"}
        )
        mock_guard = AsyncMock()

        worker = _make_worker(health_service=mock_service, session_credit_guard=mock_guard)
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock()  # type: ignore[method-assign]
        worker._publish_to_redis = AsyncMock()  # type: ignore[method-assign]

        await worker.run_once()

        mock_guard.run_cycle.assert_awaited_once()

    async def test_credit_guard_error_does_not_crash(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "x"}
        )
        mock_guard = AsyncMock()
        mock_guard.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))

        worker = _make_worker(health_service=mock_service, session_credit_guard=mock_guard)
        worker._load_previous_snapshot_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        worker._persist_snapshot = AsyncMock()  # type: ignore[method-assign]
        worker._publish_to_redis = AsyncMock()  # type: ignore[method-assign]

        # Should not raise
        await worker.run_once()


class TestPersistSnapshot:
    async def test_inserts_snapshot_to_db(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_worker(db_manager=mock_db_manager)
        detailed = {"status": "healthy", "checked_at": "2026-01-01T00:00:00"}

        with patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await worker._persist_snapshot(detailed)

        mock_repo.insert.assert_awaited_once()
        mock_session.commit.assert_awaited_once()


class TestHealthSnapshotCleanupWorker:
    async def test_run_once_calls_repository(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_cleanup_worker(db_manager=mock_db_manager, retention_days=7)

        with patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.cleanup = AsyncMock(return_value=10)
            mock_repo_cls.return_value = mock_repo

            await worker.run_once()

            mock_repo.cleanup.assert_awaited_once_with(retention_days=7)
            mock_session.commit.assert_awaited_once()

    async def test_run_once_zero_deleted_no_log(self) -> None:
        """run_once does not log when no rows are deleted."""
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_cleanup_worker(db_manager=mock_db_manager)

        with (
            patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls,
            patch("src.api.services.health.worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            mock_repo.cleanup = AsyncMock(return_value=0)
            mock_repo_cls.return_value = mock_repo

            await worker.run_once()

        mock_session.commit.assert_awaited_once()
        mock_logger.info.assert_not_called()

    async def test_deletes_old_snapshots_logs_count(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_cleanup_worker(db_manager=mock_db_manager, retention_days=14)

        with (
            patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls,
            patch("src.api.services.health.worker.logger") as mock_logger,
        ):
            mock_repo = AsyncMock()
            mock_repo.cleanup = AsyncMock(return_value=42)
            mock_repo_cls.return_value = mock_repo

            await worker.run_once()

        mock_logger.info.assert_called_once_with(
            "health.snapshot_cleanup_worker.cleanup",
            deleted=42,
            retention_days=14,
        )


class TestLoadPreviousSnapshotData:
    async def test_returns_none_when_no_previous_snapshot(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_worker(db_manager=mock_db_manager)

        with patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_latest = AsyncMock(return_value=None)
            mock_repo_cls.return_value = mock_repo

            result = await worker._load_previous_snapshot_data()

        assert result is None

    async def test_returns_snapshot_data_from_latest_row(self) -> None:
        mock_db_manager = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_db_manager.session.return_value = mock_session

        worker = _make_worker(db_manager=mock_db_manager)
        previous_row = MagicMock()
        previous_row.snapshot_data = {"status": "healthy"}

        with patch("src.api.services.health.worker.HealthSnapshotRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_latest = AsyncMock(return_value=previous_row)
            mock_repo_cls.return_value = mock_repo

            result = await worker._load_previous_snapshot_data()

        assert result == {"status": "healthy"}


class TestDetectAndPublishTransitions:
    async def test_none_ops_event_bus_is_a_no_op(self) -> None:
        worker = _make_worker(ops_event_bus=None)
        # Must not raise even with a real (non-baseline) diff.
        await worker._detect_and_publish_transitions(
            {
                "status": "healthy",
                "infrastructure": {"components": [{"name": "redis", "status": "healthy"}]},
            },
            {
                "status": "degraded",
                "infrastructure": {"components": [{"name": "redis", "status": "degraded"}]},
            },
        )

    async def test_degraded_transition_publishes_degraded_event_type(self) -> None:
        ops_bus = AsyncMock()
        worker = _make_worker(ops_event_bus=ops_bus)
        previous = {
            "status": "healthy",
            "infrastructure": {"components": [{"name": "redis", "status": "healthy"}]},
        }
        current = {
            "status": "degraded",
            "infrastructure": {"components": [{"name": "redis", "status": "degraded"}]},
        }

        await worker._detect_and_publish_transitions(previous, current)

        ops_bus.publish.assert_awaited_once()
        _, kwargs = ops_bus.publish.call_args
        assert kwargs["event_type"] == OpsEventType.HEALTH_SUBSYSTEM_DEGRADED
        assert kwargs["product_id"] == "platform"
        assert kwargs["payload"].subsystem == "redis"

    async def test_restored_transition_publishes_restored_event_type(self) -> None:
        ops_bus = AsyncMock()
        worker = _make_worker(ops_event_bus=ops_bus)
        previous = {
            "status": "unhealthy",
            "infrastructure": {"components": [{"name": "redis", "status": "unhealthy"}]},
        }
        current = {
            "status": "healthy",
            "infrastructure": {"components": [{"name": "redis", "status": "healthy"}]},
        }

        await worker._detect_and_publish_transitions(previous, current)

        ops_bus.publish.assert_awaited_once()
        _, kwargs = ops_bus.publish.call_args
        assert kwargs["event_type"] == OpsEventType.HEALTH_SUBSYSTEM_RESTORED

    async def test_baseline_none_previous_publishes_nothing(self) -> None:
        ops_bus = AsyncMock()
        worker = _make_worker(ops_event_bus=ops_bus)
        current = {
            "status": "degraded",
            "infrastructure": {"components": [{"name": "redis", "status": "degraded"}]},
        }

        await worker._detect_and_publish_transitions(None, current)

        ops_bus.publish.assert_not_awaited()
