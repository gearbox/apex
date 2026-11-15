"""How ``init_services`` assembles the planner-run machinery (built by ``_start_studio_run_worker``)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.api.dependencies import common
from src.core.config import Settings
from src.studio.llm.registry import LlmModelRegistry
from src.workers.studio_run import StudioRunWorker
from tests.studio_runs_support import make_registry, make_settings

pytestmark = pytest.mark.unit


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> common.ServiceContainer:
    fresh = common.ServiceContainer()
    fresh.db_manager = MagicMock()
    fresh.event_bus = None
    monkeypatch.setattr(common, "_services", fresh)
    monkeypatch.setattr(common, "get_billing_service", MagicMock)
    return fresh


async def test_the_worker_is_built_from_the_settings_and_started(
    container: common.ServiceContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = AsyncMock()
    monkeypatch.setattr(StudioRunWorker, "start", start)
    settings = make_settings(studio_run_worker_interval_seconds=5, studio_run_max_concurrent=3)

    worker = await common._start_studio_run_worker(
        settings, make_registry(settings), redis_enabled=False
    )

    start.assert_awaited_once()
    assert worker.name == "studio_run"
    assert worker.interval_seconds == 5
    assert worker._max_concurrent == 3
    assert isinstance(container.studio_llm_http_client, httpx.AsyncClient)
    assert container.studio_llm_http_client.follow_redirects is False
    await container.studio_llm_http_client.aclose()


async def test_the_worker_needs_the_database(
    container: common.ServiceContainer,
) -> None:
    container.db_manager = None
    settings = make_settings()
    with pytest.raises(RuntimeError, match="database"):
        await common._start_studio_run_worker(
            settings, make_registry(settings), redis_enabled=False
        )


def test_the_run_service_gets_the_registry_the_settings_and_a_bus(
    container: common.ServiceContainer,
) -> None:
    settings = make_settings()
    container.studio_llm_registry = make_registry(settings)
    service = common.get_studio_run_service(MagicMock(), settings)
    assert service._registry is container.studio_llm_registry
    assert service._settings is settings


def test_a_registry_is_only_built_for_active_runs() -> None:
    off = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
    assert off.studio_runs_active is False
    assert isinstance(make_registry(), LlmModelRegistry)
