"""``OpenAICompatibleLlmAdapter.stream`` against ``httpx.MockTransport`` (C3). No network."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from src.core.llm_models import LlmModelEntry
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.llm.port import (
    ChatChunk,
    LlmMessage,
    LlmOutputInvalid,
    LlmRejected,
    LlmUnavailable,
    LlmUsage,
    StreamEnd,
    TextDelta,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

pytestmark = pytest.mark.unit

API_KEY = "sk-test-secret-key-value"
LEAK = "THE-REPLY-TEXT-OF-THE-ARTIST"
MESSAGES = [LlmMessage("system", "rules"), LlmMessage("user", "tell me about the night")]


def _entry(**overrides: Any) -> LlmModelEntry:
    base: dict[str, Any] = {
        "base_url": "https://llm.example.test/v1",
        "api_key_env": "SOME_ENV_VAR",
        "model": "test-model-1",
        "supports_streaming": True,
    }
    return LlmModelEntry(**{**base, **overrides})


def _chunk(text: str | None = None, *, finish: str | None = None, **extra: Any) -> str:
    delta: dict[str, Any] = {} if text is None else {"content": text}
    payload = {
        "model": "served-model-id",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }
    return "data: " + json.dumps(payload)


def _usage_chunk(prompt: int = 11, completion: int = 7) -> str:
    return "data: " + json.dumps(
        {
            "model": "served-model-id",
            "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }
    )


def _sse(*lines: str, status: int = 200) -> httpx.Response:
    """A streamed reply: the lines, each followed by the blank line that ends an SSE event."""
    body = "".join(f"{line}\n\n" for line in lines)
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class Recorder:
    def __init__(self, *script: httpx.Response | Exception) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)  # type: ignore[no-any-return]


def _adapter(
    handler: Callable[[httpx.Request], httpx.Response],
    entry: LlmModelEntry | None = None,
    sleeps: list[float] | None = None,
) -> OpenAICompatibleLlmAdapter:
    async def sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return OpenAICompatibleLlmAdapter(
        entry or _entry(),
        SecretStr(API_KEY),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        model_key="test",
        sleep=sleep,
    )


async def _collect(
    adapter: OpenAICompatibleLlmAdapter, max_output_tokens: int = 700
) -> list[ChatChunk]:
    return [c async for c in adapter.stream(MESSAGES, max_output_tokens=max_output_tokens)]


class TestNulCharacters:
    """N4: a streamed reply is stored after it is written, and PostgreSQL cannot store U+0000."""

    async def test_it_is_removed_from_every_delta(self) -> None:
        recorder = Recorder(
            _sse(
                _chunk("He\x00llo, "),
                _chunk("\x00\x00wor\x00ld\x00"),
                _chunk(finish="stop"),
                "data: [DONE]",
            )
        )

        chunks = await _collect(_adapter(recorder))

        assert chunks[:-1] == [TextDelta("Hello, "), TextDelta("world")]
        assert isinstance(chunks[-1], StreamEnd)

    async def test_a_chunk_of_nothing_else_yields_no_delta(self) -> None:
        recorder = Recorder(
            _sse(
                _chunk("\x00"),
                _chunk("text"),
                _chunk("\x00\x00"),
                _chunk(finish="stop"),
                "data: [DONE]",
            )
        )

        chunks = await _collect(_adapter(recorder))

        assert chunks[:-1] == [TextDelta("text")]
        assert len(chunks) == 2

    async def test_only_real_text_counts_as_delivered_so_the_attempt_may_be_repeated(
        self,
    ) -> None:
        """The first stream breaks off after a chunk of nothing but U+0000: the caller has seen
        no text, so the one transport retry still applies."""
        recorder = Recorder(
            _sse(_chunk("\x00")),  # no finish, no [DONE]: cut off
            _sse(_chunk("second try"), _chunk(finish="stop"), "data: [DONE]"),
        )

        chunks = await _collect(_adapter(recorder))

        assert chunks[0] == TextDelta("second try")
        assert len(recorder.requests) == 2

    async def test_the_six_characters_that_spell_the_escape_are_text(self) -> None:
        spelled = "type \\u0000 here"
        recorder = Recorder(_sse(_chunk(spelled), _chunk(finish="stop"), "data: [DONE]"))

        chunks = await _collect(_adapter(recorder))

        assert chunks[0] == TextDelta(spelled)
        assert len(spelled) == len("type  here") + 6


class TestAMultiChunkStream:
    async def test_it_yields_the_deltas_then_exactly_one_end(self) -> None:
        recorder = Recorder(
            _sse(
                _chunk("Hel"),
                _chunk("lo, "),
                _chunk("world"),
                _chunk(finish="stop"),
                _usage_chunk(),
                "data: [DONE]",
            )
        )
        chunks = await _collect(_adapter(recorder))

        assert chunks[:3] == [TextDelta("Hel"), TextDelta("lo, "), TextDelta("world")]
        assert chunks[3] == StreamEnd(
            finish_reason="stop",
            usage=LlmUsage(prompt_tokens=11, completion_tokens=7),
            model_id="served-model-id",
        )
        assert len(chunks) == 4
        assert sum(isinstance(c, StreamEnd) for c in chunks) == 1

    async def test_the_request_asks_for_a_stream_with_usage_and_no_response_format(self) -> None:
        recorder = Recorder(_sse(_chunk("x"), _chunk(finish="stop"), "data: [DONE]"))
        await _collect(_adapter(recorder, _entry(temperature=0.7)), max_output_tokens=321)

        body = recorder.body
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["max_tokens"] == 321
        assert body["model"] == "test-model-1"
        assert body["temperature"] == 0.7
        assert "response_format" not in body
        assert body["messages"] == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "tell me about the night"},
        ]
        assert recorder.requests[-1].headers["authorization"] == f"Bearer {API_KEY}"

    async def test_extra_body_applies_but_cannot_turn_streaming_off(self) -> None:
        recorder = Recorder(_sse(_chunk("x"), _chunk(finish="stop"), "data: [DONE]"))
        entry = _entry(extra_body={"enable_thinking": False, "stream": False, "max_tokens": 5})
        await _collect(_adapter(recorder, entry), max_output_tokens=100)

        assert recorder.body["enable_thinking"] is False
        assert recorder.body["stream"] is True
        assert recorder.body["max_tokens"] == 100

    async def test_comments_blank_lines_and_other_fields_are_skipped(self) -> None:
        recorder = Recorder(
            _sse(
                ": keep-alive",
                "event: message",
                "id: 7",
                _chunk("a"),
                "",
                _chunk(finish="stop"),
                "data: [DONE]",
            )
        )
        chunks = await _collect(_adapter(recorder))
        assert [c.text for c in chunks if isinstance(c, TextDelta)] == ["a"]

    async def test_a_chunk_without_choices_other_than_usage_is_ignored(self) -> None:
        recorder = Recorder(
            _sse(
                'data: {"id": "x"}',
                _chunk("a"),
                _chunk(finish="stop"),
                "data: [DONE]",
            )
        )
        chunks = await _collect(_adapter(recorder))
        assert chunks[0] == TextDelta("a")

    async def test_an_empty_delta_is_not_a_chunk(self) -> None:
        recorder = Recorder(
            _sse(_chunk(""), _chunk(None), _chunk("a"), _chunk(finish="stop"), "data: [DONE]")
        )
        chunks = await _collect(_adapter(recorder))
        assert [type(c) for c in chunks] == [TextDelta, StreamEnd]


class TestHowTheStreamEnds:
    async def test_usage_missing_is_zero(self) -> None:
        recorder = Recorder(_sse(_chunk("a"), _chunk(finish="stop"), "data: [DONE]"))
        end = (await _collect(_adapter(recorder)))[-1]
        assert end == StreamEnd(
            finish_reason="stop",
            usage=LlmUsage(prompt_tokens=0, completion_tokens=0),
            model_id="served-model-id",
        )

    async def test_usage_can_ride_on_the_finish_chunk(self) -> None:
        recorder = Recorder(
            _sse(
                _chunk("a"),
                _chunk(finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 2}),
                "data: [DONE]",
            )
        )
        end = (await _collect(_adapter(recorder)))[-1]
        assert isinstance(end, StreamEnd)
        assert end.usage == LlmUsage(prompt_tokens=5, completion_tokens=2)

    @pytest.mark.parametrize("finish", ["length", "content_filter", "something_new"])
    async def test_the_finish_reason_is_passed_through(self, finish: str) -> None:
        recorder = Recorder(
            _sse(_chunk("a"), _chunk(finish=finish), _usage_chunk(), "data: [DONE]")
        )
        end = (await _collect(_adapter(recorder)))[-1]
        assert isinstance(end, StreamEnd)
        assert end.finish_reason == finish

    async def test_done_without_a_finish_chunk_ends_with_no_reason(self) -> None:
        recorder = Recorder(_sse(_chunk("a"), "data: [DONE]"))
        end = (await _collect(_adapter(recorder)))[-1]
        assert isinstance(end, StreamEnd)
        assert end.finish_reason is None

    async def test_a_body_that_just_stops_after_a_finish_chunk_is_a_complete_stream(self) -> None:
        recorder = Recorder(_sse(_chunk("a"), _chunk(finish="stop")))
        end = (await _collect(_adapter(recorder)))[-1]
        assert isinstance(end, StreamEnd)
        assert end.finish_reason == "stop"

    async def test_a_body_cut_off_with_no_finish_and_no_done_is_not_a_reply(self) -> None:
        recorder = Recorder(_sse(_chunk("a")))
        with pytest.raises(LlmUnavailable):
            await _collect(_adapter(recorder))


class TestRetriesHappenOnlyBeforeTheFirstDelta:
    async def test_a_5xx_before_any_text_is_retried_once(self) -> None:
        sleeps: list[float] = []
        recorder = Recorder(
            httpx.Response(503),
            _sse(_chunk("ok"), _chunk(finish="stop"), "data: [DONE]"),
        )
        chunks = await _collect(_adapter(recorder, sleeps=sleeps))

        assert chunks[0] == TextDelta("ok")
        assert len(recorder.requests) == 2
        assert len(sleeps) == 1

    async def test_a_429_with_retry_after_waits_that_long(self) -> None:
        sleeps: list[float] = []
        recorder = Recorder(
            httpx.Response(429, headers={"retry-after": "2"}),
            _sse(_chunk("ok"), _chunk(finish="stop"), "data: [DONE]"),
        )
        await _collect(_adapter(recorder, sleeps=sleeps))
        assert sleeps == [2.0]

    async def test_a_transport_error_before_any_text_is_retried(self) -> None:
        recorder = Recorder(
            httpx.ConnectError("boom"),
            _sse(_chunk("ok"), _chunk(finish="stop"), "data: [DONE]"),
        )
        chunks = await _collect(_adapter(recorder))
        assert chunks[0] == TextDelta("ok")

    async def test_two_failures_before_any_text_raise_unavailable(self) -> None:
        recorder = Recorder(httpx.Response(500), httpx.Response(502))
        with pytest.raises(LlmUnavailable):
            await _collect(_adapter(recorder))
        assert len(recorder.requests) == 2

    async def test_an_error_event_before_any_text_is_retried(self) -> None:
        recorder = Recorder(
            _sse('data: {"error": {"message": "overloaded"}}'),
            _sse(_chunk("ok"), _chunk(finish="stop"), "data: [DONE]"),
        )
        chunks = await _collect(_adapter(recorder))
        assert chunks[0] == TextDelta("ok")

    async def test_a_4xx_is_final_and_not_retried(self) -> None:
        recorder = Recorder(httpx.Response(400))
        with pytest.raises(LlmRejected) as caught:
            await _collect(_adapter(recorder))
        assert caught.value.status_code == 400
        assert len(recorder.requests) == 1

    async def test_after_the_first_delta_an_error_event_raises_and_is_not_retried(self) -> None:
        recorder = Recorder(
            _sse(_chunk("a"), 'data: {"error": {"message": "boom"}}'),
            _sse(_chunk("never"), _chunk(finish="stop"), "data: [DONE]"),
        )
        seen: list[ChatChunk] = []
        with pytest.raises(LlmUnavailable):
            async for chunk in _adapter(recorder).stream(MESSAGES, max_output_tokens=10):
                seen.append(chunk)
        assert seen == [TextDelta("a")]
        assert len(recorder.requests) == 1  # no retry: the reply cannot be spliced

    async def test_after_the_first_delta_a_dropped_connection_raises_and_is_not_retried(
        self,
    ) -> None:
        class _Dropping(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield (_chunk("a") + "\n\n").encode()
                raise httpx.ReadError("connection reset")

        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, stream=_Dropping())

        seen: list[ChatChunk] = []
        with pytest.raises(LlmUnavailable):
            async for chunk in _adapter(handler).stream(MESSAGES, max_output_tokens=10):
                seen.append(chunk)
        assert seen == [TextDelta("a")]
        assert len(calls) == 1


class TestAMalformedStream:
    async def test_a_line_that_is_not_json_is_an_invalid_output(self) -> None:
        recorder = Recorder(_sse("data: {not json"))
        with pytest.raises(LlmOutputInvalid):
            await _collect(_adapter(recorder))

    async def test_a_json_value_that_is_not_an_object_is_an_invalid_output(self) -> None:
        recorder = Recorder(_sse("data: [1, 2]"))
        with pytest.raises(LlmOutputInvalid):
            await _collect(_adapter(recorder))


class TestPrivacy:
    async def test_a_stream_is_logged_by_metadata_only(self) -> None:
        recorder = Recorder(
            _sse(_chunk(LEAK), _chunk(finish="stop"), _usage_chunk(), "data: [DONE]")
        )
        with capture_logs() as logs:
            await _collect(_adapter(recorder))

        finished = [log for log in logs if log["event"] == "llm.stream_finished"]
        assert len(finished) == 1
        assert finished[0]["model_key"] == "test"
        assert finished[0]["finish_reason"] == "stop"
        assert finished[0]["prompt_tokens"] == 11
        assert finished[0]["completion_tokens"] == 7
        assert "duration_ms" in finished[0]
        rendered = json.dumps(logs, default=str)
        assert LEAK not in rendered
        assert API_KEY not in rendered
        assert "tell me about the night" not in rendered

    async def test_errors_never_carry_reply_text(self) -> None:
        recorder = Recorder(_sse(_chunk(LEAK), "data: {broken " + LEAK))
        with pytest.raises(LlmOutputInvalid) as caught:
            await _collect(_adapter(recorder))
        assert LEAK not in str(caught.value)
        assert caught.value.__cause__ is None
