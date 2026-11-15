"""The model registry and the startup guards around it (invariants 13 and 16)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from src.core.config import Settings
from src.core.enums import StudioRunKind
from src.core.llm_models import LlmModelEntry, LlmRegistryError, parse_model_entries
from src.core.llm_tracing import TracingEnabledError, force_tracing_off
from src.studio.llm.registry import LlmModelRegistry

pytestmark = pytest.mark.unit

SECRET = "sk-super-secret-value-123"


def _models(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
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
            "extra_body": {"enable_thinking": False},
        },
    }
    models.update(overrides)
    return models


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jwt_secret_key": "x" * 40,
        "studio_enabled": True,
        "studio_runs_enabled": True,
        "studio_llm_models": _models(),
        "studio_llm_default_model": "grok",
    }
    return Settings(_env_file=None, **base | overrides)  # type: ignore[call-arg]


ENV = {"KEY_A": SECRET, "KEY_B": "sk-other"}


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestRegistry:
    def test_resolves_the_default_and_a_per_kind_override(self) -> None:
        registry = LlmModelRegistry.from_settings(
            _settings(studio_llm_model_by_kind={"lyrics": "qwen"}), ENV
        )
        key, entry = registry.resolve(StudioRunKind.PLAN)
        assert (key, entry.model) == ("grok", "model-a")
        key, entry = registry.resolve(StudioRunKind.LYRICS)
        assert (key, entry.model) == ("qwen", "model-b")
        assert entry.structured_output == "json_object"
        assert entry.extra_body == {"enable_thinking": False}

    def test_keys_are_read_into_secrets(self) -> None:
        registry = LlmModelRegistry.from_settings(_settings(), ENV)
        assert registry.api_key("grok").get_secret_value() == SECRET
        assert SECRET not in repr(registry.api_key("grok"))
        assert registry.keys == ("grok", "qwen")

    def test_a_missing_key_environment_variable_names_the_setting_not_a_value(self) -> None:
        with pytest.raises(LlmRegistryError) as caught:
            LlmModelRegistry.from_settings(_settings(), {"KEY_A": SECRET})
        assert "studio_llm_models[qwen]" in str(caught.value)
        assert "api_key_env" in str(caught.value)
        assert SECRET not in str(caught.value)

    def test_an_empty_key_environment_variable_is_missing(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"studio_llm_models\[grok\]"):
            LlmModelRegistry.from_settings(_settings(), {"KEY_A": "  ", "KEY_B": "x"})

    def test_a_default_that_is_not_an_entry(self) -> None:
        with pytest.raises(LlmRegistryError, match="studio_llm_default_model"):
            LlmModelRegistry(
                parse_model_entries(_models()),
                "missing",
                {},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    def test_a_by_kind_that_is_not_an_entry(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"studio_llm_model_by_kind\[plan\]"):
            LlmModelRegistry(
                parse_model_entries(_models()),
                "grok",
                {StudioRunKind.PLAN: "nope"},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    @pytest.mark.parametrize(
        "url", ["http://llm.example.test/v1", "ftp://x.test", "llm.example.test"]
    )
    def test_a_non_https_base_url_is_refused(self, url: str) -> None:
        models = _models(grok={**_models()["grok"], "base_url": url})
        with pytest.raises(LlmRegistryError, match=r"studio_llm_models\[grok\]\.base_url"):
            LlmModelRegistry(
                parse_model_entries(models),
                "grok",
                {},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    def test_the_url_must_stop_before_the_endpoint(self) -> None:
        models = _models(
            grok={**_models()["grok"], "base_url": "https://x.test/v1/chat/completions"}
        )
        with pytest.raises(LlmRegistryError, match="chat/completions"):
            LlmModelRegistry(
                parse_model_entries(models),
                "grok",
                {},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    def test_extra_body_may_not_replace_protected_keys(self) -> None:
        models = _models(grok={**_models()["grok"], "extra_body": {"messages": []}})
        with pytest.raises(LlmRegistryError, match="messages"):
            LlmModelRegistry(
                parse_model_entries(models),
                "grok",
                {},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    def test_an_entry_with_unknown_fields_is_refused(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"studio_llm_models\[grok\]"):
            parse_model_entries({"grok": {**_models()["grok"], "api_key": "inline-secret"}})

    def test_limits_are_bounded(self) -> None:
        with pytest.raises(LlmRegistryError):
            parse_model_entries({"grok": {**_models()["grok"], "timeout_s": 1}})
        assert isinstance(parse_model_entries(_models())["grok"], LlmModelEntry)


class TestSettings:
    def test_a_disabled_feature_never_blocks_boot(self) -> None:
        settings = _settings(
            studio_runs_enabled=False, studio_llm_models={}, studio_llm_default_model="zz"
        )
        assert settings.studio_runs_enabled is False

    def test_enabled_with_a_bad_default_fails_naming_the_setting(self) -> None:
        with pytest.raises(ValidationError, match="studio_llm_default_model"):
            _settings(studio_llm_default_model="zz")

    def test_enabled_with_an_http_url_fails(self) -> None:
        models = _models(grok={**_models()["grok"], "base_url": "http://x.test/v1"})
        with pytest.raises(ValidationError, match="base_url"):
            _settings(studio_llm_models=models)

    def test_every_kind_must_be_priced(self) -> None:
        with pytest.raises(ValidationError, match="price every run kind"):
            _settings(studio_run_prices={"plan": 1})

    def test_a_negative_price_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _settings(studio_run_prices={"plan": -1, "lyrics": 1, "revise_lyrics": 1})

    def test_defaults(self) -> None:
        settings = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
        assert settings.studio_runs_enabled is False
        assert settings.studio_run_prices == {
            StudioRunKind.PLAN: 1,
            StudioRunKind.LYRICS: 1,
            StudioRunKind.REVISE_LYRICS: 1,
        }
        assert settings.studio_run_deadline_seconds == 180
        assert settings.studio_run_max_concurrent == 4
        assert settings.studio_run_max_active_per_user == 3
        assert settings.studio_runs_per_user_per_hour == 30

    def test_the_example_env_registry_parses(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[3] / ".env.example").read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("STUDIO_LLM_MODELS="))
        entries = parse_model_entries(json.loads(line.split("=", 1)[1]))
        assert set(entries) == {"grok", "qwen"}
        assert entries["grok"].structured_output == "json_schema"
        assert entries["qwen"].structured_output == "json_object"
        assert entries["qwen"].extra_body == {"enable_thinking": False}


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
        import importlib
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        import src.studio.planner.graphs as graphs

        importlib.reload(graphs)
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_importing_the_graphs_refuses_a_truthy_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

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
