"""Tests for the P2-1 best-effort R2 assets-bucket probe run during init_services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.api.dependencies import common as common_module
from src.api.services.storage.exceptions import StorageConnectionError
from src.core.config import Settings
from src.core.enums import WorkerMode

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"


def _make_settings() -> Settings:
    return Settings(
        jwt_secret_key=_JWT_SECRET,
        worker_mode=WorkerMode.api_only,
        redis_url=None,
        r2_account_id="acct",
        r2_access_key_id="key",
        r2_secret_access_key="secret",
        r2_bucket_name="apex-user-content",
        r2_public_assets_bucket="apex-public-assets",
        r2_public_assets_url_base="https://assets.example.com",
        xai_api_key="",
        vastai_api_key="",
        aisha_poller_enabled=False,
    )


def _make_db_manager() -> MagicMock:
    db_manager = MagicMock()
    db_manager.session_factory = MagicMock()
    db_manager.close = AsyncMock()
    return db_manager


async def test_probe_failure_logs_error_and_still_initializes_services() -> None:
    settings = _make_settings()
    mock_db_manager = _make_db_manager()

    fake_r2 = AsyncMock()
    fake_r2.exists = AsyncMock(side_effect=StorageConnectionError("403 Forbidden on HeadObject"))

    with (
        patch("src.api.dependencies.common.get_settings", return_value=settings),
        patch("src.api.dependencies.common.init_db", return_value=mock_db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch("src.api.dependencies.common.R2StorageService", return_value=fake_r2),
        patch("src.api.dependencies.common.logger") as mock_logger,
    ):
        await common_module.init_services(settings)
        sync_service_after_init = common_module._services.payment_currency_sync_service
        await common_module.shutdown_services()

    error_calls = [
        call
        for call in mock_logger.exception.call_args_list
        if call.args and call.args[0] == "payment_currency.logo_storage_probe_failed"
    ]
    assert len(error_calls) == 1
    assert sync_service_after_init is not None


async def test_probe_success_logs_enabled() -> None:
    settings = _make_settings()
    mock_db_manager = _make_db_manager()

    fake_r2 = AsyncMock()
    fake_r2.exists = AsyncMock(return_value=False)

    with (
        patch("src.api.dependencies.common.get_settings", return_value=settings),
        patch("src.api.dependencies.common.init_db", return_value=mock_db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch("src.api.dependencies.common.R2StorageService", return_value=fake_r2),
        patch("src.api.dependencies.common.logger") as mock_logger,
    ):
        await common_module.init_services(settings)
        await common_module.shutdown_services()

    mock_logger.info.assert_any_call(
        "payment_currency.logo_cache_enabled", bucket="apex-public-assets"
    )
