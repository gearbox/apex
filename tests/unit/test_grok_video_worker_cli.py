"""Tests for the Grok video worker CLI runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers.grok_video import GrokVideoWorkerCLI


@pytest.fixture
def settings() -> MagicMock:
    """Settings with Grok and R2 configured."""
    s = MagicMock()
    s.grok_configured = True
    s.r2_configured = True
    s.database_url = "postgresql+asyncpg://apex:apex@localhost:5432/apex"
    s.db_pool_size = 5
    s.db_max_overflow = 10
    s.db_echo = False
    s.r2_account_id = "test-account"
    s.r2_access_key_id = "test-key"
    s.r2_secret_access_key = "test-secret"
    s.r2_bucket_name = "test-bucket"
    s.r2_public_url_base = "https://pub.example.com"
    s.retention_days = 7
    s.grok_video_poll_interval = 5
    s.grok_video_max_poll_time = 600
    s.grok_video_max_concurrent_polls = 8
    s.redis_url = None
    s.redis_socket_connect_timeout_seconds = 0.25
    s.redis_socket_timeout_seconds = 0.05
    s.redis_health_check_interval_seconds = 30.0
    s.redis_max_connections = 77
    return s


class TestGrokVideoWorkerCLIHandleSignal:
    """Tests for handle_signal."""

    def test_handle_signal_sets_shutdown_event(self, settings: MagicMock) -> None:
        """Calling handle_signal should set the internal shutdown event."""
        runner = GrokVideoWorkerCLI(settings)
        # The event is not set yet
        assert not runner._shutdown_event.is_set()
        runner.handle_signal()
        assert runner._shutdown_event.is_set()

    def test_handle_signal_idempotent(self, settings: MagicMock) -> None:
        """Calling handle_signal twice should not raise."""
        runner = GrokVideoWorkerCLI(settings)
        runner.handle_signal()
        runner.handle_signal()  # second call must not raise
        assert runner._shutdown_event.is_set()


class TestGrokVideoWorkerCLIRun:
    """Tests for the run() lifecycle method."""

    async def test_exits_when_grok_not_configured(self, settings: MagicMock) -> None:
        """run() should call sys.exit(1) when Grok API key is absent."""
        settings.grok_configured = False
        runner = GrokVideoWorkerCLI(settings)

        with pytest.raises(SystemExit) as exc_info:
            await runner.run()

        assert exc_info.value.code == 1

    async def test_exits_when_r2_not_configured(self, settings: MagicMock) -> None:
        """run() should call sys.exit(1) when R2 is not configured."""
        settings.r2_configured = False
        runner = GrokVideoWorkerCLI(settings)

        with pytest.raises(SystemExit) as exc_info:
            await runner.run()

        assert exc_info.value.code == 1

    async def test_successful_run_and_shutdown(self, settings: MagicMock) -> None:
        """run() should start the worker and clean up on shutdown."""
        mock_db_manager = AsyncMock()
        mock_r2_storage = AsyncMock()
        mock_grok_client = AsyncMock()
        mock_job_service = AsyncMock()
        mock_worker = AsyncMock()

        with (
            patch("src.workers.grok_video.init_db", return_value=mock_db_manager),
            patch(
                "src.workers.grok_video.R2StorageService",
                return_value=mock_r2_storage,
            ),
            patch(
                "src.workers.grok_video.GrokClient",
                return_value=mock_grok_client,
            ),
            patch(
                "src.workers.grok_video.GrokJobService",
                return_value=mock_job_service,
            ),
            patch(
                "src.workers.grok_video.GrokVideoWorker",
                return_value=mock_worker,
            ),
            patch("src.workers.grok_video.BillingService"),
        ):
            runner = GrokVideoWorkerCLI(settings)

            # Trigger shutdown immediately after the worker starts
            async def trigger_shutdown() -> None:
                await asyncio.sleep(0)
                runner.handle_signal()

            await asyncio.gather(runner.run(), trigger_shutdown())

        mock_grok_client.connect.assert_awaited_once()
        mock_job_service.connect.assert_awaited_once()
        mock_worker.start.assert_awaited_once()
        mock_worker.stop.assert_awaited_once()
        mock_job_service.close.assert_awaited_once()
        mock_grok_client.close.assert_awaited_once()
        mock_r2_storage.close.assert_awaited_once()
        mock_db_manager.close.assert_awaited_once()

    async def test_r2_settings_forwarded_correctly(self, settings: MagicMock) -> None:
        """run() should construct R2StorageSettings from the provided settings."""
        mock_db_manager = AsyncMock()
        mock_r2_storage = AsyncMock()
        mock_grok_client = AsyncMock()
        mock_job_service = AsyncMock()
        mock_worker = AsyncMock()

        with (
            patch("src.workers.grok_video.init_db", return_value=mock_db_manager),
            patch(
                "src.workers.grok_video.R2StorageService", return_value=mock_r2_storage
            ) as mock_r2_cls,
            patch("src.workers.grok_video.R2StorageSettings") as mock_r2_settings_cls,
            patch("src.workers.grok_video.GrokClient", return_value=mock_grok_client),
            patch("src.workers.grok_video.GrokJobService", return_value=mock_job_service),
            patch("src.workers.grok_video.GrokVideoWorker", return_value=mock_worker),
            patch("src.workers.grok_video.BillingService"),
        ):
            runner = GrokVideoWorkerCLI(settings)

            async def trigger_shutdown() -> None:
                await asyncio.sleep(0)
                runner.handle_signal()

            await asyncio.gather(runner.run(), trigger_shutdown())

        mock_r2_settings_cls.assert_called_once_with(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
            public_url_base=settings.r2_public_url_base,
            retention_days=settings.retention_days,
        )
        mock_r2_cls.assert_called_once_with(mock_r2_settings_cls.return_value)

    async def test_worker_receives_correct_settings(self, settings: MagicMock) -> None:
        """GrokVideoWorker should receive the db_manager, job_service, and settings."""
        mock_db_manager = AsyncMock()
        mock_r2_storage = AsyncMock()
        mock_grok_client = AsyncMock()
        mock_job_service = AsyncMock()
        mock_worker = AsyncMock()
        mock_billing_service = MagicMock()

        with (
            patch("src.workers.grok_video.init_db", return_value=mock_db_manager),
            patch("src.workers.grok_video.R2StorageService", return_value=mock_r2_storage),
            patch("src.workers.grok_video.R2StorageSettings"),
            patch("src.workers.grok_video.GrokClient", return_value=mock_grok_client),
            patch("src.workers.grok_video.GrokJobService", return_value=mock_job_service),
            patch(
                "src.workers.grok_video.GrokVideoWorker", return_value=mock_worker
            ) as mock_worker_cls,
            patch("src.workers.grok_video.BillingService", return_value=mock_billing_service),
        ):
            runner = GrokVideoWorkerCLI(settings)

            async def trigger_shutdown() -> None:
                await asyncio.sleep(0)
                runner.handle_signal()

            await asyncio.gather(runner.run(), trigger_shutdown())

        mock_worker_cls.assert_called_once_with(
            db_manager=mock_db_manager,
            job_service=mock_job_service,
            billing_service=mock_billing_service,
            settings=settings,
            redis_enabled=False,
        )

    async def test_redis_pool_initialized_with_max_connections_from_settings(
        self, settings: MagicMock
    ) -> None:
        """The standalone worker must pass max_connections through, or it
        silently falls back to init_redis_pool's default of 50 and ignores
        REDIS_MAX_CONNECTIONS entirely. No SSE pool is initialized here —
        this process has no long-lived subscribers."""
        settings.redis_url = "redis://localhost:6379"
        mock_db_manager = AsyncMock()
        mock_r2_storage = AsyncMock()
        mock_grok_client = AsyncMock()
        mock_job_service = AsyncMock()
        mock_worker = AsyncMock()

        with (
            patch("src.workers.grok_video.init_db", return_value=mock_db_manager),
            patch("src.workers.grok_video.R2StorageService", return_value=mock_r2_storage),
            patch("src.workers.grok_video.GrokClient", return_value=mock_grok_client),
            patch("src.workers.grok_video.GrokJobService", return_value=mock_job_service),
            patch("src.workers.grok_video.GrokVideoWorker", return_value=mock_worker),
            patch("src.workers.grok_video.BillingService"),
            patch("src.core.redis.init_redis_pool") as mock_init_redis_pool,
            patch("src.core.redis.init_sse_redis_pool") as mock_init_sse_redis_pool,
            patch("src.core.redis.close_redis_pool", new_callable=AsyncMock),
        ):
            runner = GrokVideoWorkerCLI(settings)

            async def trigger_shutdown() -> None:
                await asyncio.sleep(0)
                runner.handle_signal()

            await asyncio.gather(runner.run(), trigger_shutdown())

        mock_init_redis_pool.assert_called_once_with(
            settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_max_connections,
        )
        mock_init_sse_redis_pool.assert_not_called()
