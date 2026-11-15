"""The model registry: which configured model answers which kind of run.

Built once at startup from settings. API keys are looked up **by environment-variable name** (the
entry's ``api_key_env``), read once into ``SecretStr`` and never logged; a missing or empty one
fails startup naming the setting, not the value.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import SecretStr

from src.core.llm_models import (
    LlmModelEntry,
    LlmRegistryError,
    validate_model_entries,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.core.config import Settings
    from src.core.enums import StudioRunKind

__all__ = ["LlmModelEntry", "LlmModelRegistry", "LlmRegistryError"]


class LlmModelRegistry:
    """Resolves a run kind to a registry key and its configured model entry."""

    def __init__(
        self,
        entries: Mapping[str, LlmModelEntry],
        default_key: str,
        by_kind: Mapping[StudioRunKind, str],
        keys: Mapping[str, SecretStr],
    ) -> None:
        """Validate and freeze the registry.

        Args:
            entries: Registry key -> entry.
            default_key: Key used by kinds without an entry in ``by_kind``.
            by_kind: Optional run kind -> registry key.
            keys: Registry key -> API key, one per entry.

        Raises:
            LlmRegistryError: The configuration is unusable (messages name the setting and never
                a secret): unknown default / per-kind key, non-https ``base_url``, a protected
                ``extra_body`` key, or an entry without a non-empty API key.
        """
        validate_model_entries(entries, default_key=default_key, by_kind=by_kind)
        for key in entries:
            secret = keys.get(key)
            if secret is None or not secret.get_secret_value().strip():
                raise LlmRegistryError(
                    f"studio_llm_models[{key}]: the environment variable named by api_key_env "
                    "is not set or is empty"
                )
        self._entries = dict(entries)
        self._default_key = default_key
        self._by_kind = dict(by_kind)
        self._keys = dict(keys)

    @classmethod
    def from_settings(
        cls, settings: Settings, environ: Mapping[str, str] | None = None
    ) -> LlmModelRegistry:
        """Build the registry from ``STUDIO_LLM_*`` settings, reading API keys from ``environ``.

        Raises:
            LlmRegistryError: See :meth:`__init__`.
        """
        env = os.environ if environ is None else environ
        entries = settings.studio_llm_model_entries()
        keys = {key: SecretStr(env.get(entry.api_key_env, "")) for key, entry in entries.items()}
        return cls(
            entries,
            settings.studio_llm_default_model,
            settings.studio_llm_model_by_kind,
            keys,
        )

    def resolve(self, kind: StudioRunKind) -> tuple[str, LlmModelEntry]:
        """The registry key and entry that serve ``kind``."""
        key = self._by_kind.get(kind, self._default_key)
        return key, self._entries[key]

    def entry(self, key: str) -> LlmModelEntry:
        """The entry of a registry key (``KeyError`` for an unknown one)."""
        return self._entries[key]

    def api_key(self, key: str) -> SecretStr:
        """The API key of a registry key (``KeyError`` for an unknown one)."""
        return self._keys[key]

    @property
    def keys(self) -> tuple[str, ...]:
        """Every configured registry key."""
        return tuple(self._entries)
