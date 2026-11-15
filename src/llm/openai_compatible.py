"""Adapter for any ``/chat/completions`` endpoint that speaks the common wire format, on ``httpx``.

One adapter serves every configured model: the endpoint, key, model id and structured-output mode
come from a registry entry, so adding a provider is configuration. Nothing here is specific to a
vendor.

Privacy: request and response **bodies are never logged** and never put in an exception; a call logs
the registry key, status code, duration, token counts and the attempt number only.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Final

import httpx
import msgspec
import structlog

from src.core.llm_models import PROTECTED_BODY_KEYS
from src.llm.port import (
    LlmMessage,
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmResult,
    LlmUnavailable,
    LlmUsage,
)
from src.llm.strict_schema import to_strict_schema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from pydantic import SecretStr

    from src.core.llm_models import LlmModelEntry

__all__ = ["OpenAICompatibleLlmAdapter"]

logger = structlog.get_logger(__name__)

_MAX_ATTEMPTS: Final = 2
"""One call plus one transport retry (timeout, connection error, 429, 5xx)."""
_DEFAULT_BACKOFF_SECONDS: Final = 0.5
_MAX_RETRY_AFTER_SECONDS: Final = 10.0
_JSON_OBJECT_INSTRUCTION: Final = "Return only a JSON object that matches this JSON Schema:"
"""Provider JSON mode needs the word "JSON" in the prompt, so the schema travels as text."""


class OpenAICompatibleLlmAdapter:
    """``StructuredLlmPort`` over a chat-completions endpoint."""

    def __init__(
        self,
        entry: LlmModelEntry,
        api_key: SecretStr,
        client: httpx.AsyncClient,
        *,
        model_key: str,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create an adapter.

        Args:
            entry: The registry entry (endpoint, model id, mode, limits).
            api_key: The bearer token; never logged.
            client: The process-wide shared client (closed on shutdown by its owner).
            model_key: The registry key, for logs only.
            sleep: Awaited between the two attempts (injectable for tests).
        """
        self._entry = entry
        self._api_key = api_key
        self._client = client
        self._model_key = model_key
        self._sleep = sleep
        self._url = entry.base_url.rstrip("/") + "/chat/completions"

    async def complete[T: msgspec.Struct](
        self, messages: Sequence[LlmMessage], *, schema: type[T], schema_name: str
    ) -> LlmResult[T]:
        """See :meth:`StructuredLlmPort.complete`."""
        body = self._body(messages, schema, schema_name)
        payload = await self._post(body)
        return self._decode(payload, schema)

    # -- request -----------------------------------------------------------------------------

    def _body(
        self, messages: Sequence[LlmMessage], schema: type[msgspec.Struct], schema_name: str
    ) -> dict[str, Any]:
        entry = self._entry
        strict = to_strict_schema(schema)
        wire_messages = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {"max_tokens": entry.max_output_tokens}
        if entry.temperature is not None:
            body["temperature"] = entry.temperature
        if entry.structured_output == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": strict},
            }
        else:
            response_format = {"type": "json_object"}
            _put_in_leading_system_message(
                wire_messages,
                f"{_JSON_OBJECT_INSTRUCTION} {json.dumps(strict, separators=(',', ':'))}",
            )
        # ``extra_body`` goes in last, but the request's contract keys cannot be replaced.
        extra = {k: v for k, v in entry.extra_body.items() if k not in PROTECTED_BODY_KEYS}
        return {
            **body,
            **extra,
            "model": entry.model,
            "messages": wire_messages,
            "response_format": response_format,
        }

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST with one retry; returns the decoded JSON object of a ``200`` reply."""
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        failure = "transport"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            status: int | None = None
            retry_after: float | None = None
            try:
                response = await self._client.post(
                    self._url, json=body, headers=headers, timeout=self._entry.timeout_s
                )
            except (httpx.TimeoutException, httpx.TransportError):
                failure = "transport"
            else:
                status = response.status_code
                if status == httpx.codes.OK:
                    return self._payload(response, attempt, started)
                if status != httpx.codes.TOO_MANY_REQUESTS and status < 500:
                    self._log(status, started, attempt, usage=None)
                    raise LlmRejected(status)
                failure = f"http_{status}"
                retry_after = _retry_after(response)
            self._log(status, started, attempt, usage=None)
            if attempt < _MAX_ATTEMPTS:
                await self._sleep(
                    retry_after if retry_after is not None else _DEFAULT_BACKOFF_SECONDS
                )
        raise LlmUnavailable(f"LLM provider unavailable ({failure})")

    def _payload(self, response: httpx.Response, attempt: int, started: float) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            self._log(response.status_code, started, attempt, usage=None)
            raise LlmOutputInvalid("LLM reply is not JSON") from None
        if not isinstance(payload, dict):
            self._log(response.status_code, started, attempt, usage=None)
            raise LlmOutputInvalid("LLM reply is not a JSON object")
        self._log(response.status_code, started, attempt, usage=_reported_usage(payload))
        return payload

    # -- response ----------------------------------------------------------------------------

    @staticmethod
    def _decode[T: msgspec.Struct](payload: dict[str, Any], schema: type[T]) -> LlmResult[T]:
        # A reply that arrives but is unusable has been paid for: its token counts travel on the
        # error (numbers only) so a caller that meters tokens still counts it.
        usage = _reported_usage(payload)
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            raise LlmOutputInvalid("LLM reply has no choice", usage=usage)
        if choice.get("finish_reason") == "length":
            raise LlmOutputTruncated("LLM reply was cut off at the output-token limit", usage=usage)
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LlmOutputInvalid("LLM reply is empty", usage=usage)
        try:
            value = msgspec.json.decode(content, type=schema)
        except (msgspec.DecodeError, msgspec.ValidationError):
            # No chaining: msgspec's message can quote a value from the reply.
            raise LlmOutputInvalid("LLM reply does not match the schema", usage=usage) from None
        model = payload.get("model")
        return LlmResult(
            value=value,
            usage=usage or LlmUsage(prompt_tokens=0, completion_tokens=0),
            model_id=model if isinstance(model, str) else "",
        )

    # -- logging -----------------------------------------------------------------------------

    def _log(
        self, status: int | None, started: float, attempt: int, usage: LlmUsage | None
    ) -> None:
        logger.info(
            "llm.call",
            model_key=self._model_key,
            status_code=status,
            duration_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            attempt=attempt,
        )


def _reported_usage(payload: dict[str, Any]) -> LlmUsage | None:
    """The token counts of a reply, or ``None`` when the provider sent no ``usage`` object."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    return LlmUsage(
        prompt_tokens=prompt if isinstance(prompt, int) else 0,
        completion_tokens=completion if isinstance(completion, int) else 0,
    )


def _put_in_leading_system_message(wire_messages: list[dict[str, Any]], text: str) -> None:
    """Add ``text`` to the first system message (after a blank line), or open with a new one.

    Several chat endpoints accept a system message only at the very start of the conversation, so
    the instruction never travels as a later message. ``wire_messages`` is the adapter's own copy
    of the request; the caller's messages are never touched.
    """
    for message in wire_messages:
        if message["role"] == "system":
            message["content"] = f"{message['content']}\n\n{text}"
            return
    wire_messages.insert(0, {"role": "system", "content": text})


def _retry_after(response: httpx.Response) -> float | None:
    """The ``Retry-After`` delay in seconds, capped; ``None`` when absent or not a number."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_SECONDS))
