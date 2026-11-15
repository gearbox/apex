"""The shared model registry and the settings that feed it (invariants 13, 16 and 22)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from src.core.config import Settings
from src.core.llm_models import (
    LlmModelEntry,
    LlmRegistryError,
    parse_model_entries,
    validate_model_entries,
    validate_model_keys,
)
from src.llm.registry import LlmModelRegistry

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
        "llm_models": _models(),
        "llm_default_model": "grok",
    }
    return Settings(_env_file=None, **base | overrides)  # type: ignore[call-arg]


def _registry(models: dict[str, dict[str, Any]], default: str = "grok") -> LlmModelRegistry:
    return LlmModelRegistry(
        parse_model_entries(models),
        default,
        {"grok": SecretStr("a"), "qwen": SecretStr("b")},
    )


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
    def test_it_knows_its_default_and_its_entries_but_not_who_asks(self) -> None:
        registry = LlmModelRegistry.from_settings(_settings(), ENV)
        assert registry.default_key == "grok"
        assert registry.entry("grok").model == "model-a"
        entry = registry.entry("qwen")
        assert (entry.model, entry.structured_output) == ("model-b", "json_object")
        assert entry.extra_body == {"enable_thinking": False}
        assert not hasattr(registry, "resolve")  # routing is the consumer's own

    def test_the_constructor_takes_no_per_kind_routing(self) -> None:
        constructor: Any = LlmModelRegistry  # called with the old four-argument shape on purpose
        with pytest.raises(TypeError):
            constructor(
                parse_model_entries(_models()),
                "grok",
                {},
                {"grok": SecretStr("a"), "qwen": SecretStr("b")},
            )

    def test_keys_are_read_into_secrets(self) -> None:
        registry = LlmModelRegistry.from_settings(_settings(), ENV)
        assert registry.api_key("grok").get_secret_value() == SECRET
        assert SECRET not in repr(registry.api_key("grok"))
        assert registry.keys == ("grok", "qwen")

    def test_a_missing_key_environment_variable_names_the_setting_not_a_value(self) -> None:
        with pytest.raises(LlmRegistryError) as caught:
            LlmModelRegistry.from_settings(_settings(), {"KEY_A": SECRET})
        assert "llm_models[qwen]" in str(caught.value)
        assert "api_key_env" in str(caught.value)
        assert SECRET not in str(caught.value)

    def test_an_empty_key_environment_variable_is_missing(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"llm_models\[grok\]"):
            LlmModelRegistry.from_settings(_settings(), {"KEY_A": "  ", "KEY_B": "x"})

    def test_a_default_that_is_not_an_entry_names_the_setting(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"llm_default_model is not a key of llm_models"):
            _registry(_models(), "missing")

    @pytest.mark.parametrize(
        "url", ["http://llm.example.test/v1", "ftp://x.test", "llm.example.test"]
    )
    def test_a_non_https_base_url_is_refused(self, url: str) -> None:
        models = _models(grok={**_models()["grok"], "base_url": url})
        with pytest.raises(LlmRegistryError, match=r"llm_models\[grok\]\.base_url"):
            _registry(models)

    def test_the_url_must_stop_before_the_endpoint(self) -> None:
        models = _models(
            grok={**_models()["grok"], "base_url": "https://x.test/v1/chat/completions"}
        )
        with pytest.raises(LlmRegistryError, match="chat/completions"):
            _registry(models)

    def test_extra_body_may_not_replace_protected_keys(self) -> None:
        models = _models(grok={**_models()["grok"], "extra_body": {"messages": []}})
        with pytest.raises(LlmRegistryError, match="messages"):
            _registry(models)

    def test_an_entry_with_unknown_fields_is_refused(self) -> None:
        with pytest.raises(LlmRegistryError, match=r"llm_models\[grok\]"):
            parse_model_entries({"grok": {**_models()["grok"], "api_key": "inline-secret"}})

    def test_limits_are_bounded(self) -> None:
        with pytest.raises(LlmRegistryError):
            parse_model_entries({"grok": {**_models()["grok"], "timeout_s": 1}})
        assert isinstance(parse_model_entries(_models())["grok"], LlmModelEntry)

    def test_no_message_names_a_consumer(self) -> None:
        """Whoever calls the registry, its errors speak of ``llm_*`` settings only."""
        with pytest.raises(LlmRegistryError) as caught:
            _registry(_models(), "missing")
        assert "studio" not in str(caught.value).lower()


class TestValidateModelKeys:
    def test_every_known_key_passes(self) -> None:
        validate_model_keys(
            "some_setting", ["grok", "qwen", "grok"], parse_model_entries(_models())
        )
        validate_model_keys("some_setting", [], parse_model_entries(_models()))

    def test_an_unknown_key_names_the_setting_and_never_the_key(self) -> None:
        with pytest.raises(LlmRegistryError) as caught:
            validate_model_keys("my_routing[x]", ["grok", "nope"], parse_model_entries(_models()))
        assert str(caught.value) == "my_routing[x] is not a key of llm_models"

    def test_the_default_key_is_optional_when_only_the_shape_is_checked(self) -> None:
        validate_model_entries(parse_model_entries(_models()))
        with pytest.raises(LlmRegistryError, match="llm_default_model"):
            validate_model_entries(parse_model_entries(_models()), default_key="zz")


class TestSettings:
    """What the shared registry settings guarantee at startup (invariant 22)."""

    def test_the_shape_is_checked_with_every_feature_off(self) -> None:
        bad = _models(grok={**_models()["grok"], "base_url": "http://x.test/v1"})
        with pytest.raises(ValidationError, match=r"llm_models\[grok\]\.base_url"):
            _settings(studio_enabled=False, studio_runs_enabled=False, llm_models=bad)

    def test_an_entry_that_does_not_decode_is_refused_with_every_feature_off(self) -> None:
        bad = _models(grok={**_models()["grok"], "timeout_s": 1})
        with pytest.raises(ValidationError, match=r"llm_models\[grok\]"):
            _settings(studio_enabled=False, studio_runs_enabled=False, llm_models=bad)

    def test_a_blank_model_id_is_refused_with_every_feature_off(self) -> None:
        bad = _models(qwen={**_models()["qwen"], "model": "  "})
        with pytest.raises(ValidationError, match=r"llm_models\[qwen\]\.model"):
            _settings(studio_enabled=False, studio_runs_enabled=False, llm_models=bad)

    def test_an_empty_registry_is_fine_with_every_feature_off(self) -> None:
        settings = _settings(
            studio_enabled=False,
            studio_runs_enabled=False,
            llm_models={},
            llm_default_model="not-a-key",
        )
        assert settings.llm_model_entries() == {}
        assert settings.llm_required is False

    def test_an_unknown_default_is_not_an_error_while_nothing_uses_it(self) -> None:
        settings = _settings(studio_runs_enabled=False, llm_default_model="zz")
        assert settings.llm_required is False

    def test_keys_are_not_needed_to_start_with_every_feature_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.delenv("KEY_B", raising=False)
        settings = _settings(studio_runs_enabled=False)
        assert settings.llm_required is False
        assert set(settings.llm_model_entries()) == {"grok", "qwen"}  # configured, not built

    def test_llm_required_follows_the_features_that_call_a_model(self) -> None:
        assert _settings().llm_required is True
        assert _settings(studio_enabled=False).llm_required is False
        assert _settings(studio_runs_enabled=False).llm_required is False

    def test_the_registry_settings_have_their_own_names(self) -> None:
        settings = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
        assert settings.llm_models == {}
        assert settings.llm_default_model == "grok"
        # no compatibility alias for the names these two settings had before they were shared
        assert not hasattr(settings, "studio_llm_" + "models")
        assert not hasattr(settings, "studio_llm_" + "default_model")

    def test_the_example_env_registry_parses(self) -> None:
        text = (Path(__file__).resolve().parents[3] / ".env.example").read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("LLM_MODELS="))
        entries = parse_model_entries(json.loads(line.split("=", 1)[1]))
        assert set(entries) == {"grok", "qwen"}
        assert entries["grok"].structured_output == "json_schema"
        assert entries["qwen"].structured_output == "json_object"
        assert entries["qwen"].extra_body == {"enable_thinking": False}
        validate_model_entries(entries, default_key="grok")
