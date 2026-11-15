"""How ``init_services`` assembles the shared LLM access and the planner-run machinery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.api.dependencies import common
from src.core.config import Settings
from src.llm.factory import LlmClientFactory
from src.llm.registry import LlmModelRegistry
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


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API-key variables the test registry's entries name."""
    monkeypatch.setenv("KEY_A", "secret-a")
    monkeypatch.setenv("KEY_B", "secret-b")


class TestSharedLlmAccess:
    """The registry and the HTTP client exist when a feature needs them, once, and not otherwise."""

    def test_nothing_is_built_with_runs_off(
        self,
        container: common.ServiceContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from_settings = MagicMock()
        client = MagicMock()
        monkeypatch.setattr(LlmModelRegistry, "from_settings", from_settings)
        monkeypatch.setattr(common.httpx, "AsyncClient", client)
        # keys unset on purpose: a disabled feature must not need one
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.delenv("KEY_B", raising=False)

        for settings in (
            make_settings(studio_runs_enabled=False),
            make_settings(studio_enabled=False),
            Settings(_env_file=None, jwt_secret_key="x" * 40),  # type: ignore[call-arg]
        ):
            common._init_llm_access(settings)

        from_settings.assert_not_called()
        client.assert_not_called()
        assert container.llm_registry is None
        assert container.llm_http_client is None

    async def test_both_are_built_once_with_runs_on(
        self,
        container: common.ServiceContainer,
        keys: None,  # noqa: ARG002 - the fixture sets the API-key environment variables
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        built: list[LlmModelRegistry] = []
        real = LlmModelRegistry.from_settings

        def counting(settings: Settings) -> LlmModelRegistry:
            registry = real(settings)
            built.append(registry)
            return registry

        monkeypatch.setattr(LlmModelRegistry, "from_settings", counting)
        settings = make_settings()

        common._init_llm_access(settings)

        assert len(built) == 1
        assert container.llm_registry is built[0]
        assert isinstance(container.llm_http_client, httpx.AsyncClient)
        assert container.llm_http_client.follow_redirects is False
        await container.llm_http_client.aclose()

    def test_a_missing_api_key_fails_startup_naming_the_setting_only(
        self,
        container: common.ServiceContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KEY_A", "secret-a")
        monkeypatch.delenv("KEY_B", raising=False)
        with pytest.raises(Exception, match=r"llm_models\[qwen\]") as caught:
            common._init_llm_access(make_settings())
        assert "secret-a" not in str(caught.value)
        assert container.llm_registry is None
        assert container.llm_http_client is None


class TestStudioRunWorkerWiring:
    async def test_the_worker_is_built_from_the_settings_on_the_shared_client(
        self, container: common.ServiceContainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = AsyncMock()
        monkeypatch.setattr(StudioRunWorker, "start", start)
        settings = make_settings(studio_run_worker_interval_seconds=5, studio_run_max_concurrent=3)
        registry = make_registry(settings)
        shared = httpx.AsyncClient(follow_redirects=False)

        worker = await common._start_studio_run_worker(
            settings, registry, shared, redis_enabled=False
        )

        start.assert_awaited_once()
        assert worker.name == "studio_run"
        assert worker.interval_seconds == 5
        assert worker._max_concurrent == 3
        factory = worker._executor._llm_factory
        assert isinstance(factory, LlmClientFactory)
        assert factory._client is shared  # the worker makes no client of its own
        assert container.llm_http_client is None  # ... and does not register one
        await shared.aclose()

    async def test_the_worker_needs_the_database(self, container: common.ServiceContainer) -> None:
        container.db_manager = None
        settings = make_settings()
        async with httpx.AsyncClient() as shared:
            with pytest.raises(RuntimeError, match="database"):
                await common._start_studio_run_worker(
                    settings, make_registry(settings), shared, redis_enabled=False
                )


class TestRunServiceWiring:
    def test_the_run_service_gets_the_settings_and_a_bus_but_not_the_registry(
        self, container: common.ServiceContainer
    ) -> None:
        settings = make_settings()
        container.llm_registry = make_registry(settings)
        service = common.get_studio_run_service(MagicMock(), settings)
        assert service._settings is settings
        assert not hasattr(service, "_registry")  # only wiring and the executor see the registry

    def test_a_registry_is_only_needed_for_active_runs(self) -> None:
        off = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
        assert off.studio_runs_active is False
        assert off.llm_required is False
        assert isinstance(make_registry(), LlmModelRegistry)
