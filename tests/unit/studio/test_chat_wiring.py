"""How the chat is assembled at startup, and how the factory refuses a model that cannot stream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.api.dependencies import common
from src.core.llm_models import LlmRegistryError
from src.llm.factory import LlmClientFactory
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.llm.registry import LlmModelRegistry
from src.studio.chat.supervisor import ChatTaskSupervisor
from src.workers.studio_chat import StudioChatSweepWorker
from tests.studio_chat_support import make_chat_settings
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


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_A", "secret-a")
    monkeypatch.setenv("KEY_B", "secret-b")


class TestTheLlmRequirement:
    def test_the_chat_alone_builds_the_registry_and_the_client(
        self, container: common.ServiceContainer
    ) -> None:
        settings = make_chat_settings(studio_runs_enabled=False)
        assert settings.llm_required
        common._init_llm_access(settings)
        assert isinstance(container.llm_registry, LlmModelRegistry)
        assert isinstance(container.llm_http_client, httpx.AsyncClient)

    def test_a_disabled_chat_builds_nothing_and_needs_no_key(
        self, container: common.ServiceContainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.delenv("KEY_B", raising=False)
        for settings in (
            make_settings(studio_runs_enabled=False, studio_chat_enabled=False),
            make_chat_settings(studio_runs_enabled=False, studio_enabled=False),
        ):
            common._init_llm_access(settings)
        assert container.llm_registry is None
        assert container.llm_http_client is None


class TestTheSupervisor:
    async def test_it_is_built_from_port_builders_and_the_settings(
        self, container: common.ServiceContainer
    ) -> None:
        settings = make_chat_settings(studio_chat_shutdown_grace_seconds=7)
        registry = make_registry(settings)
        async with httpx.AsyncClient() as shared:
            supervisor = common._build_studio_chat_supervisor(settings, registry, shared)

        assert isinstance(supervisor, ChatTaskSupervisor)
        assert supervisor._grace == 7
        assert supervisor.in_flight == 0
        chat_port = supervisor._chat_port_for("grok")
        structured_port = supervisor._structured_port_for("qwen")
        assert isinstance(chat_port, OpenAICompatibleLlmAdapter)
        assert isinstance(structured_port, OpenAICompatibleLlmAdapter)
        assert container.llm_http_client is None  # it makes no client of its own

    async def test_it_needs_the_database(self, container: common.ServiceContainer) -> None:
        container.db_manager = None
        settings = make_chat_settings()
        async with httpx.AsyncClient() as shared:
            with pytest.raises(RuntimeError, match="database"):
                common._build_studio_chat_supervisor(settings, make_registry(settings), shared)

    def test_the_provider_refuses_when_the_chat_is_not_wired(
        self, container: common.ServiceContainer
    ) -> None:
        assert container.studio_chat_supervisor is None
        with pytest.raises(RuntimeError, match="not enabled"):
            common.get_studio_chat_supervisor()
        container.studio_chat_supervisor = MagicMock()
        assert common.get_studio_chat_supervisor() is container.studio_chat_supervisor


class TestTheSweepWorker:
    async def test_it_is_built_from_the_settings(
        self, container: common.ServiceContainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = AsyncMock()
        monkeypatch.setattr(StudioChatSweepWorker, "start", start)
        settings = make_chat_settings(studio_chat_sweep_interval_seconds=11)

        worker = await common._start_studio_chat_sweep_worker(settings, redis_enabled=False)

        start.assert_awaited_once()
        assert worker.name == "studio_chat_sweep"
        assert worker.interval_seconds == 11
        assert worker._sweeper._older_than == 120 + 30  # the deadline plus the grace
        assert container.studio_chat_sweep_worker is None  # the caller registers it

    async def test_it_needs_the_database(self, container: common.ServiceContainer) -> None:
        container.db_manager = None
        with pytest.raises(RuntimeError, match="database"):
            await common._start_studio_chat_sweep_worker(make_chat_settings(), redis_enabled=False)

    async def test_a_failing_sweep_raises_an_error_that_carries_only_its_type(self) -> None:
        sweeper = MagicMock()
        sweeper.sweep = AsyncMock(side_effect=ValueError("a message that quotes USER TEXT"))
        worker = StudioChatSweepWorker(
            sweeper=sweeper, interval_seconds=30, redis_client_factory=MagicMock
        )
        with pytest.raises(Exception, match="ValueError") as caught:
            await worker.run_once()
        assert "USER TEXT" not in str(caught.value)
        assert caught.value.__cause__ is None


class TestTheFactory:
    def _factory(self) -> LlmClientFactory:
        settings = make_chat_settings(
            llm_models={
                **make_chat_settings().llm_models,
                "plain": {
                    "base_url": "https://llm-c.example.test/v1",
                    "api_key_env": "KEY_A",
                    "model": "model-c",
                },
            }
        )
        return LlmClientFactory(make_registry(settings), httpx.AsyncClient())

    async def test_a_streaming_model_has_a_chat_port_and_a_structured_one(self) -> None:
        factory = self._factory()
        assert isinstance(factory.chat("grok"), OpenAICompatibleLlmAdapter)
        # one adapter per key, whichever port is asked for
        assert factory.chat("grok") is factory("grok")

    async def test_a_model_that_does_not_stream_has_no_chat_port(self) -> None:
        factory = self._factory()
        with pytest.raises(LlmRegistryError, match="streaming"):
            factory.chat("plain")
        assert isinstance(factory("plain"), OpenAICompatibleLlmAdapter)  # structured calls are fine
