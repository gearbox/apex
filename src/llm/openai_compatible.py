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
from dataclasses import dataclass
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
    StreamEnd,
    TextDelta,
)
from src.llm.strict_schema import to_strict_schema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from pydantic import SecretStr

    from src.core.llm_models import LlmModelEntry
    from src.llm.port import ChatChunk

__all__ = ["OpenAICompatibleLlmAdapter"]

logger = structlog.get_logger(__name__)

_MAX_ATTEMPTS: Final = 2
"""One call plus one transport retry (timeout, connection error, 429, 5xx)."""
_DEFAULT_BACKOFF_SECONDS: Final = 0.5
_MAX_RETRY_AFTER_SECONDS: Final = 10.0
_JSON_OBJECT_INSTRUCTION: Final = "Return only a JSON object that matches this JSON Schema:"
"""Provider JSON mode needs the word "JSON" in the prompt, so the schema travels as text."""


@dataclass(frozen=True, slots=True)
class _Retry:
    """An attempt that may be repeated: why, and how long to wait first."""

    failure: str
    """``"transport"`` or ``"http_<status>"``; only ever part of the final error's message."""
    delay: float


class _StreamRetry(Exception):
    """A stream attempt failed **before any text was delivered**: it may be repeated once.

    Internal control flow between ``_stream_attempt`` and ``stream``; it never leaves this module.
    """

    def __init__(self, retry: _Retry) -> None:
        super().__init__(retry.failure)
        self.retry = retry


@dataclass(slots=True)
class _StreamState:
    """What one stream attempt has seen so far (reset for every attempt)."""

    delivered: bool = False
    """Whether any text has been handed to the caller: from then on nothing may be retried."""
    finish_reason: str | None = None
    usage: LlmUsage | None = None
    model_id: str | None = None
    chunks: int = 0


class OpenAICompatibleLlmAdapter:
    """``StructuredLlmPort`` and ``ChatLlmPort`` over a chat-completions endpoint."""

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

    async def stream(
        self, messages: Sequence[LlmMessage], *, max_output_tokens: int
    ) -> AsyncIterator[ChatChunk]:
        """See :meth:`ChatLlmPort.stream`.

        The single transport retry applies **only before the first text delta**: once the caller
        has seen text, a reply cannot be spliced, so any later failure raises.
        """
        body = self._stream_body(messages, max_output_tokens)
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        failure = "transport"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            state = _StreamState()
            try:
                async for chunk in self._stream_attempt(body, headers, attempt, state):
                    yield chunk
            except _StreamRetry as retry:
                failure = retry.retry.failure
                if attempt < _MAX_ATTEMPTS:
                    await self._sleep(retry.retry.delay)
                continue
            return
        raise LlmUnavailable(f"LLM provider unavailable ({failure})")

    def _stream_body(
        self, messages: Sequence[LlmMessage], max_output_tokens: int
    ) -> dict[str, Any]:
        entry = self._entry
        body: dict[str, Any] = {}
        if entry.temperature is not None:
            body["temperature"] = entry.temperature
        extra = {k: v for k, v in entry.extra_body.items() if k not in PROTECTED_BODY_KEYS}
        # The contract keys go in last: ``extra_body`` cannot turn streaming or usage off.
        return {
            **body,
            **extra,
            "max_tokens": max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "model": entry.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

    async def _stream_attempt(
        self, body: dict[str, Any], headers: dict[str, str], attempt: int, state: _StreamState
    ) -> AsyncIterator[ChatChunk]:
        """One streaming request: text deltas, then the end.

        Raises:
            _StreamRetry: Failed before any text was delivered, in a way worth one more attempt.
            LlmRejected: A final ``4xx``.
            LlmUnavailable: The stream broke after text was delivered, or ended without an end.
            LlmOutputInvalid: A line of the stream is not a completion chunk.
        """
        started = time.perf_counter()
        done = False
        try:
            async with self._client.stream(
                "POST",
                self._url,
                json=body,
                headers=headers,
                timeout=httpx.Timeout(self._entry.timeout_s),
            ) as response:
                status = response.status_code
                if status != httpx.codes.OK:
                    self._log_stream("failed", started, attempt, state, status=status)
                    if not _is_retryable(status):
                        raise LlmRejected(status)
                    retry_after = _retry_after(response)
                    raise _StreamRetry(
                        _Retry(
                            f"http_{status}",
                            retry_after if retry_after is not None else _DEFAULT_BACKOFF_SECONDS,
                        )
                    )
                async for line in response.aiter_lines():
                    payload = _data_of(line)
                    if payload is None:
                        continue
                    if payload == "[DONE]":
                        done = True
                        break
                    if text := self._read_chunk(payload, state):
                        state.delivered = True
                        yield TextDelta(text)
        except (httpx.TimeoutException, httpx.TransportError):
            self._log_stream("failed", started, attempt, state, status=None)
            if state.delivered:
                raise LlmUnavailable("LLM stream was interrupted") from None
            raise _StreamRetry(_Retry("transport", _DEFAULT_BACKOFF_SECONDS)) from None
        if not done and state.finish_reason is None:
            # The body closed without ``[DONE]`` and without a finish chunk: a cut-off stream,
            # not a reply that ended.
            self._log_stream("failed", started, attempt, state, status=httpx.codes.OK)
            if state.delivered:
                raise LlmUnavailable("LLM stream ended unexpectedly")
            raise _StreamRetry(_Retry("transport", _DEFAULT_BACKOFF_SECONDS))
        self._log_stream("finished", started, attempt, state, status=httpx.codes.OK)
        yield StreamEnd(
            finish_reason=state.finish_reason,
            usage=state.usage or LlmUsage(prompt_tokens=0, completion_tokens=0),
            model_id=state.model_id,
        )

    def _read_chunk(self, payload: str, state: _StreamState) -> str:
        """Fold one ``data:`` payload into ``state``; returns the text delta it carried (or ``""``).

        Raises:
            LlmOutputInvalid: The payload is not a JSON object. No text, no chaining: the payload
                is reply content.
            LlmUnavailable: The provider reported an error inside the stream after text was
                delivered (before that it is retried).
        """
        try:
            chunk = json.loads(payload)
        except ValueError:
            raise LlmOutputInvalid("LLM stream chunk is not JSON") from None
        if not isinstance(chunk, dict):
            raise LlmOutputInvalid("LLM stream chunk is not a JSON object")
        state.chunks += 1
        if "error" in chunk and not chunk.get("choices"):
            if state.delivered:
                raise LlmUnavailable("LLM stream reported an error")
            raise _StreamRetry(_Retry("stream_error", _DEFAULT_BACKOFF_SECONDS))
        usage = _reported_usage(chunk)
        if usage is not None:
            state.usage = usage
        model = chunk.get("model")
        if isinstance(model, str) and model:
            state.model_id = model
        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            return ""  # the usage chunk (``choices: []``) and keep-alives carry no text
        finish = choice.get("finish_reason")
        if isinstance(finish, str) and finish:
            state.finish_reason = finish
        delta = choice.get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        return content if isinstance(content, str) else ""

    def _log_stream(
        self,
        outcome: str,
        started: float,
        attempt: int,
        state: _StreamState,
        *,
        status: int | None,
    ) -> None:
        """Metadata only: never a delta, never a prompt."""
        prompt_tokens = state.usage.prompt_tokens if state.usage else None
        completion_tokens = state.usage.completion_tokens if state.usage else None
        duration_ms = int((time.perf_counter() - started) * 1000)
        if outcome == "finished":
            logger.info(
                "llm.stream_finished",
                model_key=self._model_key,
                status_code=status,
                finish_reason=state.finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                attempt=attempt,
            )
        else:
            logger.info(
                "llm.stream_failed",
                model_key=self._model_key,
                status_code=status,
                delivered=state.delivered,
                duration_ms=duration_ms,
                attempt=attempt,
            )

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
            outcome = await self._attempt(body, headers, attempt)
            if not isinstance(outcome, _Retry):
                return outcome
            failure = outcome.failure
            if attempt < _MAX_ATTEMPTS:
                await self._sleep(outcome.delay)
        raise LlmUnavailable(f"LLM provider unavailable ({failure})")

    async def _attempt(
        self, body: dict[str, Any], headers: dict[str, str], attempt: int
    ) -> dict[str, Any] | _Retry:
        """One request: the payload of a ``200``, a :class:`_Retry`, or ``LlmRejected`` (final 4xx)."""
        started = time.perf_counter()
        try:
            response = await self._client.post(
                self._url, json=body, headers=headers, timeout=self._entry.timeout_s
            )
        except (httpx.TimeoutException, httpx.TransportError):
            self._log(None, started, attempt, usage=None)
            return _Retry("transport", _DEFAULT_BACKOFF_SECONDS)
        status = response.status_code
        if status == httpx.codes.OK:
            return self._payload(response, attempt, started)
        self._log(status, started, attempt, usage=None)
        if not _is_retryable(status):
            raise LlmRejected(status)
        retry_after = _retry_after(response)
        return _Retry(
            f"http_{status}", retry_after if retry_after is not None else _DEFAULT_BACKOFF_SECONDS
        )

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


def _data_of(line: str) -> str | None:
    """The payload of a server-sent ``data:`` line; ``None`` for blank lines, comments (``:``) and
    every other field (``event:``, ``id:``, ``retry:``)."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    return payload or None


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


def _is_retryable(status: int) -> bool:
    """Whether a non-``200`` status is worth one more attempt: rate limited or a server error."""
    return status == httpx.codes.TOO_MANY_REQUESTS or status >= 500


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
