"""Model registry entries for structured-output LLM calls (``LLM_MODELS``).

Pure and dependency-light on purpose: ``src.core.config`` validates the setting with it, and the
shared registry (``src/llm/registry.py``) is built from it. An entry names the **env var** that
holds the API key (``api_key_env``), never the key; model ids come from configuration only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Final, Literal
from urllib.parse import urlparse

import msgspec
from msgspec import Meta, Struct

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "PROTECTED_BODY_KEYS",
    "LlmModelEntry",
    "LlmRegistryError",
    "parse_model_entries",
    "validate_model_entries",
    "validate_model_keys",
    "validate_streaming_keys",
]

PROTECTED_BODY_KEYS: Final = frozenset({"model", "messages", "response_format"})
"""Request-body keys ``extra_body`` may not replace: they carry the contract with the caller."""


class LlmRegistryError(ValueError):
    """The model registry configuration is unusable. Messages name the setting, never a secret."""


class LlmModelEntry(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One model that can be called, over an OpenAI-compatible ``/chat/completions`` endpoint."""

    base_url: str
    """https origin plus path up to, not including, ``/chat/completions``."""
    api_key_env: str
    """NAME of the environment variable that holds the API key; never the key itself."""
    model: str
    structured_output: Literal["json_schema", "json_object"] = "json_schema"
    max_output_tokens: Annotated[int, Meta(ge=256, le=32_000)] = 8_000
    timeout_s: Annotated[int, Meta(ge=5, le=300)] = 90
    temperature: float | None = None
    extra_body: dict[str, Any] = {}
    """Merged into the request body last; may not replace ``model`` / ``messages`` /
    ``response_format``."""
    supports_streaming: bool = False
    """Whether the endpoint streams a chat completion (``stream: true``, server-sent events). A
    consumer that streams its replies needs a model with this set; its startup check refuses one
    without it (``validate_streaming_keys``)."""


def parse_model_entries(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, LlmModelEntry]:
    """Decode the ``LLM_MODELS`` mapping into entries.

    Raises:
        LlmRegistryError: An entry does not decode (the message names the key, and the
            ``msgspec`` reason, which can mention field names and types but never a secret: an
            entry holds none).
    """
    entries: dict[str, LlmModelEntry] = {}
    for key, value in raw.items():
        try:
            entries[key] = msgspec.convert(dict(value), LlmModelEntry)
        except msgspec.ValidationError as exc:
            raise LlmRegistryError(f"llm_models[{key}] is invalid: {exc}") from None
    return entries


def validate_model_entries(
    entries: Mapping[str, LlmModelEntry], *, default_key: str | None = None
) -> None:
    """Check everything about the registry that needs no environment.

    Args:
        entries: Registry key -> entry.
        default_key: When given, it must be one of ``entries``. ``None`` checks the entries' shape
            only, which is what a process that does not use the registry still wants.

    Raises:
        LlmRegistryError: ``default_key`` is not an entry, an entry's ``base_url`` is not
            ``https`` with a host, its ``api_key_env`` or ``model`` is blank, or its ``extra_body``
            replaces a protected key. Each message names the setting.
    """
    if default_key is not None:
        validate_model_keys("llm_default_model", [default_key], entries)
    for key, entry in entries.items():
        parsed = urlparse(entry.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise LlmRegistryError(f"llm_models[{key}].base_url must be an https URL")
        if entry.base_url.endswith("/chat/completions"):
            raise LlmRegistryError(f"llm_models[{key}].base_url must stop before /chat/completions")
        if not entry.api_key_env.strip():
            raise LlmRegistryError(f"llm_models[{key}].api_key_env must not be blank")
        if not entry.model.strip():
            raise LlmRegistryError(f"llm_models[{key}].model must not be blank")
        if clash := PROTECTED_BODY_KEYS & entry.extra_body.keys():
            raise LlmRegistryError(
                f"llm_models[{key}].extra_body may not set {', '.join(sorted(clash))}"
            )


def validate_model_keys(
    setting: str, keys: Iterable[str], entries: Mapping[str, LlmModelEntry]
) -> None:
    """Every key must be a registry entry; the message names ``setting``.

    A consumer that routes work to models by its own setting (a per-kind map, say) checks that
    setting's values with this, so a typo fails startup naming the setting instead of failing a
    request later.

    Raises:
        LlmRegistryError: A key is not one of ``entries``.
    """
    for key in keys:
        if key not in entries:
            raise LlmRegistryError(f"{setting} is not a key of llm_models")


def validate_streaming_keys(
    setting: str, keys: Iterable[str], entries: Mapping[str, LlmModelEntry]
) -> None:
    """Every key must be a registry entry **that streams** (``supports_streaming``).

    A consumer that streams its replies checks its routing setting with this, so a model without
    streaming is a startup error naming the setting, never a failure on the first reply.

    Raises:
        LlmRegistryError: A key is not one of ``entries``, or its entry does not stream.
    """
    keys = list(keys)
    validate_model_keys(setting, keys, entries)
    for key in keys:
        if not entries[key].supports_streaming:
            raise LlmRegistryError(f"{setting} names a model without supports_streaming")
