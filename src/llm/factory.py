"""One adapter per registry key, sharing the process-wide HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.llm_models import LlmRegistryError
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter

if TYPE_CHECKING:
    import httpx

    from src.llm.port import ChatLlmPort, StructuredLlmPort
    from src.llm.registry import LlmModelRegistry

__all__ = ["LlmClientFactory"]


class LlmClientFactory:
    """``factory(model_key)`` -> the structured port that serves that registry key (built once,
    cached); ``factory.chat(model_key)`` -> the streaming port of a key that streams."""

    def __init__(self, registry: LlmModelRegistry, client: httpx.AsyncClient) -> None:
        self._registry = registry
        self._client = client
        self._adapters: dict[str, OpenAICompatibleLlmAdapter] = {}

    def _adapter(self, model_key: str) -> OpenAICompatibleLlmAdapter:
        adapter = self._adapters.get(model_key)
        if adapter is None:
            adapter = OpenAICompatibleLlmAdapter(
                self._registry.entry(model_key),
                self._registry.api_key(model_key),
                self._client,
                model_key=model_key,
            )
            self._adapters[model_key] = adapter
        return adapter

    def __call__(self, model_key: str) -> StructuredLlmPort:
        return self._adapter(model_key)

    def chat(self, model_key: str) -> ChatLlmPort:
        """The streaming port of ``model_key``.

        Raises:
            LlmRegistryError: The entry does not stream (``supports_streaming`` is false). Startup
                has already refused such a key for every configured consumer; this is the
                backstop for a key that reached here some other way.
        """
        if not self._registry.entry(model_key).supports_streaming:
            raise LlmRegistryError("the chat model does not support streaming")
        return self._adapter(model_key)
