"""Tests for WORKER_MODE gating in the app lifespan (init_services)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.api.dependencies import common as common_module
from src.core.config import Settings
from src.core.enums import WorkerMode

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


def _make_settings(*, worker_mode: WorkerMode) -> Settings:
    return Settings(
        jwt_secret_key=_JWT_SECRET,
        worker_mode=worker_mode,
        redis_url=None,
        # Keep optional stacks unconfigured so init_services takes the
        # short/no-op branches for R2, Grok, and the GPU session stack —
        # only the always-present workers (token cleanup, health snapshot,
        # aisha poller) need to be asserted on here.
        r2_account_id="",
        xai_api_key="",
        vastai_api_key="",
        aisha_poller_enabled=True,
    )


def _make_db_manager() -> MagicMock:
    db_manager = MagicMock()
    db_manager.session_factory = MagicMock()
    db_manager.close = AsyncMock()
    return db_manager


async def test_api_only_worker_classes_never_started() -> None:
    settings = _make_settings(worker_mode=WorkerMode.api_only)
    mock_db_manager = _make_db_manager()

    with (
        patch("src.api.dependencies.common.get_settings", return_value=settings),
        patch("src.api.dependencies.common.init_db", return_value=mock_db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch("src.api.dependencies.common.TokenCleanupWorker") as mock_token_cleanup,
        patch("src.api.dependencies.common.AishaJobPoller") as mock_aisha_poller,
        patch("src.api.dependencies.common.HealthSnapshotWorker") as mock_health_snapshot,
        patch("src.api.dependencies.common.HealthSnapshotCleanupWorker") as mock_health_cleanup,
    ):
        await common_module.init_services(settings)

        mock_token_cleanup.assert_not_called()
        mock_aisha_poller.assert_not_called()
        mock_health_snapshot.assert_not_called()
        mock_health_cleanup.assert_not_called()

        await common_module.shutdown_services()


async def test_workers_disabled_by_mode_logged() -> None:
    settings = _make_settings(worker_mode=WorkerMode.api_only)
    mock_db_manager = _make_db_manager()

    with (
        patch("src.api.dependencies.common.get_settings", return_value=settings),
        patch("src.api.dependencies.common.init_db", return_value=mock_db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch("src.api.dependencies.common.TokenCleanupWorker"),
        patch("src.api.dependencies.common.AishaJobPoller"),
        patch("src.api.dependencies.common.HealthSnapshotWorker"),
        patch("src.api.dependencies.common.HealthSnapshotCleanupWorker"),
        patch("src.api.dependencies.common.logger") as mock_logger,
    ):
        await common_module.init_services(settings)
        await common_module.shutdown_services()

    mock_logger.info.assert_any_call(
        "workers.disabled_by_mode", worker_mode=WorkerMode.api_only.value
    )


async def test_all_mode_starts_token_cleanup_and_health_workers() -> None:
    settings = _make_settings(worker_mode=WorkerMode.all)
    mock_db_manager = _make_db_manager()

    mock_token_cleanup_instance = AsyncMock()
    mock_aisha_poller_instance = AsyncMock()
    mock_health_snapshot_instance = AsyncMock()
    mock_health_cleanup_instance = AsyncMock()

    with (
        patch("src.api.dependencies.common.get_settings", return_value=settings),
        patch("src.api.dependencies.common.init_db", return_value=mock_db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch(
            "src.api.dependencies.common.TokenCleanupWorker",
            return_value=mock_token_cleanup_instance,
        ) as mock_token_cleanup_cls,
        patch(
            "src.api.dependencies.common.AishaJobPoller",
            return_value=mock_aisha_poller_instance,
        ) as mock_aisha_poller_cls,
        patch(
            "src.api.dependencies.common.HealthSnapshotWorker",
            return_value=mock_health_snapshot_instance,
        ),
        patch(
            "src.api.dependencies.common.HealthSnapshotCleanupWorker",
            return_value=mock_health_cleanup_instance,
        ),
    ):
        await common_module.init_services(settings)

        mock_token_cleanup_cls.assert_called_once()
        mock_token_cleanup_instance.start.assert_awaited_once()
        mock_health_snapshot_instance.start.assert_awaited_once()
        mock_health_cleanup_instance.start.assert_awaited_once()
        # aisha_poller_enabled=True and worker_mode=all -> poller constructed and started.
        mock_aisha_poller_cls.assert_called_once()
        mock_aisha_poller_instance.start.assert_awaited_once()

        await common_module.shutdown_services()
