"""One adapter per registry key, sharing the process-wide HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.llm.openai_compatible import OpenAICompatibleLlmAdapter

if TYPE_CHECKING:
    import httpx

    from src.llm.port import StructuredLlmPort
    from src.llm.registry import LlmModelRegistry

__all__ = ["LlmClientFactory"]


class LlmClientFactory:
    """``factory(model_key)`` -> the port that serves that registry key (built once, cached)."""

    def __init__(self, registry: LlmModelRegistry, client: httpx.AsyncClient) -> None:
        self._registry = registry
        self._client = client
        self._ports: dict[str, StructuredLlmPort] = {}

    def __call__(self, model_key: str) -> StructuredLlmPort:
        port = self._ports.get(model_key)
        if port is None:
            port = OpenAICompatibleLlmAdapter(
                self._registry.entry(model_key),
                self._registry.api_key(model_key),
                self._client,
                model_key=model_key,
            )
            self._ports[model_key] = port
        return port
