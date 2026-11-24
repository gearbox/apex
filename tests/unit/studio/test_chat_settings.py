"""The chat's own settings: the switch, the models, the prices and the startup checks (C1)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.llm_models import LlmModelEntry, LlmRegistryError, validate_streaming_keys

pytestmark = pytest.mark.unit

MODELS: dict[str, dict[str, Any]] = {
    "grok": {
        "base_url": "https://llm-a.example.test/v1",
        "api_key_env": "KEY_A",
        "model": "model-a",
        "supports_streaming": True,
    },
    "qwen": {
        "base_url": "https://llm-b.example.test/compatible-mode/v1",
        "api_key_env": "KEY_B",
        "model": "model-b",
        "structured_output": "json_object",
        "supports_streaming": True,
    },
    "plain": {
        "base_url": "https://llm-c.example.test/v1",
        "api_key_env": "KEY_C",
        "model": "model-c",
    },
}


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jwt_secret_key": "x" * 40,
        "studio_enabled": True,
        "studio_chat_enabled": True,
        "llm_models": MODELS,
        "llm_default_model": "grok",
    }
    return Settings(_env_file=None, **base | overrides)  # type: ignore[call-arg]


class TestDefaults:
    def test_the_chat_is_off_by_default(self) -> None:
        settings = Settings(_env_file=None, jwt_secret_key="x" * 40)  # type: ignore[call-arg]
        assert settings.studio_chat_enabled is False
        assert settings.studio_chat_active is False

    def test_the_documented_defaults(self) -> None:
        settings = _settings()
        assert settings.studio_chat_message_price == 1
        assert settings.studio_memory_draft_price == 1
        assert settings.studio_chat_max_output_tokens == 700
        assert settings.studio_chat_reply_deadline_seconds == 120
        assert settings.studio_chat_max_pending_per_user == 3
        assert settings.studio_chat_jobs_per_user_per_hour == 120
        assert settings.studio_chat_sweep_interval_seconds == 30
        assert settings.studio_chat_shutdown_grace_seconds == 10

    def test_both_switches_are_needed_to_be_active(self) -> None:
        assert _settings().studio_chat_active
        assert not _settings(studio_enabled=False).studio_chat_active
        assert not _settings(studio_chat_enabled=False).studio_chat_active

    def test_the_chat_alone_needs_the_llm_registry(self) -> None:
        assert _settings().llm_required
        assert not _settings(studio_chat_enabled=False).llm_required

    def test_a_disabled_chat_never_blocks_boot(self) -> None:
        # a chat model that does not stream is only an error once the chat is switched on
        _settings(studio_chat_enabled=False, studio_chat_model="plain")


class TestModelRouting:
    def test_unset_means_the_default_model(self) -> None:
        settings = _settings()
        assert settings.studio_chat_model_key("vex") == "grok"
        assert settings.studio_memory_draft_model_key == "grok"

    def test_the_chat_model_setting_wins_over_the_default(self) -> None:
        assert _settings(studio_chat_model="qwen").studio_chat_model_key("vex") == "qwen"

    def test_a_product_override_wins_over_the_chat_model(self) -> None:
        settings = _settings(
            studio_chat_model="qwen", studio_chat_model_by_product={"synthara": "grok"}
        )
        assert settings.studio_chat_model_key("synthara") == "grok"
        assert settings.studio_chat_model_key("vex") == "qwen"

    def test_the_draft_model_is_its_own_setting(self) -> None:
        assert _settings(studio_memory_draft_model="qwen").studio_memory_draft_model_key == "qwen"


class TestStartupChecks:
    def test_a_chat_model_without_streaming_is_a_startup_error(self) -> None:
        with pytest.raises(ValidationError, match="studio_chat_model names a model without"):
            _settings(studio_chat_model="plain")

    def test_a_default_model_without_streaming_is_a_startup_error_when_it_is_the_chat_model(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="supports_streaming"):
            _settings(llm_default_model="plain")

    def test_an_unknown_chat_model_names_the_setting(self) -> None:
        with pytest.raises(ValidationError, match="studio_chat_model is not a key"):
            _settings(studio_chat_model="nope")

    def test_a_product_override_must_stream_too(self) -> None:
        with pytest.raises(ValidationError, match=r"studio_chat_model_by_product\[synthara\]"):
            _settings(studio_chat_model_by_product={"synthara": "plain"})

    def test_a_product_override_must_name_a_known_product(self) -> None:
        with pytest.raises(ValidationError, match="unknown product"):
            _settings(studio_chat_model_by_product={"nope": "grok"})

    def test_the_draft_model_need_not_stream_but_must_exist(self) -> None:
        _settings(studio_memory_draft_model="plain")
        with pytest.raises(ValidationError, match="studio_memory_draft_model is not a key"):
            _settings(studio_memory_draft_model="nope")

    def test_an_empty_registry_is_a_startup_error(self) -> None:
        with pytest.raises(ValidationError, match="llm_models must not be empty"):
            _settings(llm_models={})

    def test_a_message_never_names_a_value(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _settings(studio_chat_model="plain")
        assert "model-c" not in str(caught.value)
        assert "KEY_C" not in str(caught.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("studio_chat_message_price", -1),
            ("studio_memory_draft_price", -1),
            ("studio_chat_max_output_tokens", 63),
            ("studio_chat_max_output_tokens", 4_001),
            ("studio_chat_max_pending_per_user", 0),
            ("studio_chat_jobs_per_user_per_hour", 0),
        ],
    )
    def test_out_of_range_values_are_refused(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            _settings(**{field: value})

    def test_a_free_price_is_allowed(self) -> None:
        settings = _settings(studio_chat_message_price=0, studio_memory_draft_price=0)
        assert settings.studio_chat_message_price == 0


class TestTheStreamingFlag:
    def test_it_defaults_to_false(self) -> None:
        entry = LlmModelEntry(base_url="https://x.example/v1", api_key_env="K", model="m")
        assert entry.supports_streaming is False

    def test_validate_streaming_keys_names_the_setting(self) -> None:
        entries = {
            "yes": LlmModelEntry(
                base_url="https://x.example/v1", api_key_env="K", model="m", supports_streaming=True
            ),
            "no": LlmModelEntry(base_url="https://x.example/v1", api_key_env="K", model="m"),
        }
        validate_streaming_keys("some_setting", ["yes"], entries)
        with pytest.raises(LlmRegistryError, match="some_setting names a model without"):
            validate_streaming_keys("some_setting", ["no"], entries)
        with pytest.raises(LlmRegistryError, match="some_setting is not a key"):
            validate_streaming_keys("some_setting", ["missing"], entries)
