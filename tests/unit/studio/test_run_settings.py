"""The studio's own run settings: the switches, prices, routing and the tracing guard."""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.enums import StudioRunKind
from src.core.llm_tracing import TracingEnabledError, force_tracing_off
from src.studio.services.run import model_key_for

pytestmark = pytest.mark.unit

SECRET = "sk-super-secret-value-123"

MODELS: dict[str, dict[str, Any]] = {
    "grok": {
        "base_url": "https://llm-a.example.test/v1",
        "api_key_env": "KEY_A",
        "model": "model-a",
    },
    "qwen": {
        "base_url": "https://llm-b.example.test/compatible-mode/v1",
        "api_key_env": "KEY_B",
        "model": "model-b",
        "structured_output": "json_object",
    },
}


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jwt_secret_key": "x" * 40,
        "studio_enabled": True,
        "studio_runs_enabled": True,
        "llm_models": MODELS,
        "llm_default_model": "grok",
    }
    return Settings(_env_file=None, **base | overrides)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestModelKeyFor:
    """The studio's routing: a per-kind value wins, otherwise the registry default."""

    def test_the_per_kind_value_wins(self) -> None:
        by_kind = {StudioRunKind.LYRICS: "qwen"}
        assert model_key_for(StudioRunKind.LYRICS, by_kind=by_kind, default_key="grok") == "qwen"

    def test_a_kind_without_an_entry_uses_the_default(self) -> None:
        by_kind = {StudioRunKind.LYRICS: "qwen"}
        assert model_key_for(StudioRunKind.PLAN, by_kind=by_kind, default_key="grok") == "grok"

    def test_no_routing_at_all_is_the_default(self) -> None:
        for kind in StudioRunKind:
            assert model_key_for(kind, by_kind={}, default_key="qwen") == "qwen"

    def test_it_reads_the_routing_from_the_settings(self) -> None:
        settings = _settings(studio_llm_model_by_kind={"plan": "qwen"})
        assert (
            model_key_for(
                StudioRunKind.PLAN,
                by_kind=settings.studio_llm_model_by_kind,
                default_key=settings.llm_default_model,
            )
            == "qwen"
        )


class TestRunSettings:
    """A feature that is switched on needs a usable registry (invariant 22)."""

    def test_a_disabled_feature_never_blocks_boot(self) -> None:
        settings = _settings(
            studio_runs_enabled=False,
            llm_models={},
            llm_default_model="zz",
            studio_llm_model_by_kind={"plan": "nope"},
        )
        assert settings.studio_runs_enabled is False

    def test_enabled_with_an_empty_registry_fails_naming_the_setting(self) -> None:
        with pytest.raises(ValidationError, match="llm_models must not be empty"):
            _settings(llm_models={})

    def test_enabled_with_a_bad_default_fails_naming_the_setting(self) -> None:
        with pytest.raises(ValidationError, match="llm_default_model"):
            _settings(llm_default_model="zz")

    def test_enabled_with_an_unknown_per_kind_value_fails_naming_the_setting(self) -> None:
        with pytest.raises(ValidationError, match=r"studio_llm_model_by_kind\[plan\]"):
            _settings(studio_llm_model_by_kind={"plan": "nope"})

    def test_enabled_with_known_per_kind_values_starts(self) -> None:
        settings = _settings(studio_llm_model_by_kind={"plan": "qwen", "lyrics": "grok"})
        assert settings.studio_llm_model_by_kind == {
            StudioRunKind.PLAN: "qwen",
            StudioRunKind.LYRICS: "grok",
        }

    def test_enabled_with_an_http_url_fails(self) -> None:
        models = {**MODELS, "grok": {**MODELS["grok"], "base_url": "http://x.test/v1"}}
        with pytest.raises(ValidationError, match="base_url"):
            _settings(llm_models=models)

    def test_every_kind_must_be_priced(self) -> None:
        with pytest.raises(ValidationError, match="price every run kind"):
            _settings(studio_run_prices={"plan": 1})

    def test_a_negative_price_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _settings(studio_run_prices={"plan": -1, "lyrics": 1, "revise_lyrics": 1})

    def test_defaults(self) -> None:
        settings = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
        assert settings.studio_runs_enabled is False
        assert settings.studio_llm_model_by_kind == {}
        assert settings.studio_run_prices == {
            StudioRunKind.PLAN: 1,
            StudioRunKind.LYRICS: 1,
            StudioRunKind.REVISE_LYRICS: 1,
        }
        assert settings.studio_run_deadline_seconds == 180
        assert settings.studio_run_max_concurrent == 4
        assert settings.studio_run_max_active_per_user == 3
        assert settings.studio_runs_per_user_per_hour == 30


class TestTracingGuard:
    @pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
    @pytest.mark.parametrize("value", ["true", "1", "True", "yes"])
    def test_a_truthy_switch_is_refused(self, name: str, value: str) -> None:
        with pytest.raises(TracingEnabledError, match=name):
            force_tracing_off({name: value})

    @pytest.mark.parametrize("name", ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"])
    def test_a_tracing_api_key_is_refused_without_echoing_it(self, name: str) -> None:
        with pytest.raises(TracingEnabledError) as caught:
            force_tracing_off({name: SECRET})
        assert name in str(caught.value)
        assert SECRET not in str(caught.value)

    def test_both_switches_are_pinned_to_false(self) -> None:
        env: dict[str, str] = {"LANGSMITH_TRACING": "0"}
        force_tracing_off(env)
        assert env == {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}

    def test_enabled_runs_refuse_to_start_with_tracing_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        with pytest.raises(ValidationError, match="LANGSMITH_TRACING"):
            _settings()

    def test_importing_the_graphs_pins_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        import src.studio.planner.graphs as graphs

        importlib.reload(graphs)
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_importing_the_graphs_refuses_a_truthy_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.studio.planner.graphs as graphs

        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        with pytest.raises(TracingEnabledError):
            importlib.reload(graphs)


class TestRunsActive:
    @pytest.mark.parametrize(
        ("studio", "runs", "active"),
        [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
    )
    def test_both_switches_are_needed(self, studio: bool, runs: bool, active: bool) -> None:
        settings = _settings(studio_enabled=studio, studio_runs_enabled=runs)
        assert settings.studio_runs_active is active
        assert settings.llm_required is active
