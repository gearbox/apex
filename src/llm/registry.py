"""The model registry: the configured models, their entries and their API keys.

Built once at startup from settings. API keys are looked up **by environment-variable name** (the
entry's ``api_key_env``), read once into ``SecretStr`` and never logged; a missing or empty one
fails startup naming the setting, not the value. Which model serves which piece of work is the
consumer's own routing (it names a registry key); the registry only knows a default.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import SecretStr

from src.core.llm_models import LlmModelEntry, LlmRegistryError, validate_model_entries

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.core.config import Settings

__all__ = ["LlmModelEntry", "LlmModelRegistry", "LlmRegistryError"]


class LlmModelRegistry:
    """The configured models, by registry key."""

    def __init__(
        self,
        entries: Mapping[str, LlmModelEntry],
        default_key: str,
        keys: Mapping[str, SecretStr],
    ) -> None:
        """Validate and freeze the registry.

        Args:
            entries: Registry key -> entry.
            default_key: Key for any consumer without a routing setting of its own.
            keys: Registry key -> API key, one per entry.

        Raises:
            LlmRegistryError: The configuration is unusable (messages name the setting and never
                a secret): an unknown default key, non-https ``base_url``, a protected
                ``extra_body`` key, or an entry without a non-empty API key.
        """
        validate_model_entries(entries, default_key=default_key)
        for key in entries:
            secret = keys.get(key)
            if secret is None or not secret.get_secret_value().strip():
                raise LlmRegistryError(
                    f"llm_models[{key}]: the environment variable named by api_key_env "
                    "is not set or is empty"
                )
        self._entries = dict(entries)
        self._default_key = default_key
        self._keys = dict(keys)

    @classmethod
    def from_settings(
        cls, settings: Settings, environ: Mapping[str, str] | None = None
    ) -> LlmModelRegistry:
        """Build the registry from ``LLM_*`` settings, reading API keys from ``environ``.

        Raises:
            LlmRegistryError: See :meth:`__init__`.
        """
        env = os.environ if environ is None else environ
        entries = settings.llm_model_entries()
        keys = {key: SecretStr(env.get(entry.api_key_env, "")) for key, entry in entries.items()}
        return cls(entries, settings.llm_default_model, keys)

    def entry(self, key: str) -> LlmModelEntry:
        """The entry of a registry key (``KeyError`` for an unknown one)."""
        return self._entries[key]

    def api_key(self, key: str) -> SecretStr:
        """The API key of a registry key (``KeyError`` for an unknown one)."""
        return self._keys[key]

    @property
    def default_key(self) -> str:
        """The key used by a consumer that has no routing setting of its own."""
        return self._default_key

    @property
    def keys(self) -> tuple[str, ...]:
        """Every configured registry key."""
        return tuple(self._entries)
