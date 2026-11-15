"""``OpenAICompatibleLlmAdapter`` against ``httpx.MockTransport`` (invariant 15). No network."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import msgspec
import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from src.core.llm_models import LlmModelEntry
from src.llm.openai_compatible import OpenAICompatibleLlmAdapter
from src.llm.port import (
    LlmMessage,
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
    LlmUsage,
)
from src.studio.planner.drafts import PlanDraft
from tests.studio_llm import make_draft

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

API_KEY = "sk-test-secret-key-value"
LEAK = "THE-BRIEF-AND-THE-PERSONA"
MESSAGES = [LlmMessage("system", "rules"), LlmMessage("user", "make a song")]


def _entry(**overrides: Any) -> LlmModelEntry:
    base: dict[str, Any] = {
        "base_url": "https://llm.example.test/v1",
        "api_key_env": "SOME_ENV_VAR",
        "model": "test-model-1",
    }
    return LlmModelEntry(**{**base, **overrides})


def _reply(draft: PlanDraft | None = None, *, finish: str = "stop", **extra: Any) -> dict[str, Any]:
    content = msgspec.json.encode(draft or make_draft()).decode()
    return {
        "model": "served-model-id",
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        **extra,
    }


class Recorder:
    """A transport handler that replays a script of responses and records every request."""

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


def ok(payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload or _reply())


class TestRequestShape:
    async def test_json_schema_mode(self) -> None:
        rec = Recorder(ok())
        result = await _adapter(rec, _entry(temperature=0.3, max_output_tokens=2000)).complete(
            MESSAGES, schema=PlanDraft, schema_name="plan_draft"
        )

        request = rec.requests[0]
        assert str(request.url) == "https://llm.example.test/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        body = rec.body
        assert body["model"] == "test-model-1"
        assert body["max_tokens"] == 2000
        assert body["temperature"] == 0.3
        assert body["messages"] == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "make a song"},
        ]
        fmt = body["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "plan_draft"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"]["additionalProperties"] is False
        assert result.value == make_draft()

    async def test_temperature_is_omitted_when_unset(self) -> None:
        rec = Recorder(ok())
        await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert "temperature" not in rec.body

    async def test_json_object_mode_puts_the_schema_in_the_one_leading_system_message(
        self,
    ) -> None:
        """Invariant 25: exactly one system message, first, with the schema and the word JSON."""
        rec = Recorder(ok())
        await _adapter(rec, _entry(structured_output="json_object")).complete(
            MESSAGES, schema=PlanDraft, schema_name="plan_draft"
        )
        body = rec.body
        assert body["response_format"] == {"type": "json_object"}
        messages = body["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]  # nothing is appended
        assert sum(m["role"] == "system" for m in messages) == 1
        system = messages[0]["content"]
        assert system.startswith("rules\n\n")  # the caller's text first, then a blank line
        assert "Return only a JSON object that matches this JSON Schema:" in system
        assert "JSON" in system
        assert '"refusal"' in system
        assert messages[1] == {"role": "user", "content": "make a song"}
        assert "json_schema" not in json.dumps(body["response_format"])

    async def test_json_object_mode_opens_a_system_message_when_there_is_none(self) -> None:
        rec = Recorder(ok())
        conversation = [LlmMessage("user", "make a song"), LlmMessage("assistant", "ok")]
        await _adapter(rec, _entry(structured_output="json_object")).complete(
            conversation, schema=PlanDraft, schema_name="x"
        )
        messages = rec.body["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert messages[0]["content"].startswith("Return only a JSON object")
        assert "JSON" in messages[0]["content"]
        assert not messages[0]["content"].startswith("\n")

    async def test_json_object_mode_extends_the_first_system_message_only(self) -> None:
        rec = Recorder(ok())
        conversation = [
            LlmMessage("system", "first"),
            LlmMessage("user", "u"),
            LlmMessage("system", "second"),
        ]
        await _adapter(rec, _entry(structured_output="json_object")).complete(
            conversation, schema=PlanDraft, schema_name="x"
        )
        messages = rec.body["messages"]
        assert messages[0]["content"].startswith("first\n\nReturn only a JSON object")
        assert messages[2] == {"role": "system", "content": "second"}

    async def test_json_object_mode_never_mutates_the_callers_messages(self) -> None:
        conversation = [LlmMessage("system", "rules"), LlmMessage("user", "make a song")]
        before = list(conversation)
        adapter = _adapter(Recorder(ok(), ok()), _entry(structured_output="json_object"))
        await adapter.complete(conversation, schema=PlanDraft, schema_name="x")
        await adapter.complete(conversation, schema=PlanDraft, schema_name="x")  # twice: no growth
        assert conversation == before
        assert conversation[0].content == "rules"

    async def test_json_schema_mode_leaves_the_messages_alone(self) -> None:
        rec = Recorder(ok())
        await _adapter(rec).complete(
            [LlmMessage("user", "only a user message")], schema=PlanDraft, schema_name="x"
        )
        assert rec.body["messages"] == [{"role": "user", "content": "only a user message"}]

    async def test_extra_body_is_merged_last(self) -> None:
        rec = Recorder(ok())
        entry = _entry(extra_body={"enable_thinking": False, "max_tokens": 123})
        await _adapter(rec, entry).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert rec.body["enable_thinking"] is False
        assert rec.body["max_tokens"] == 123  # not a protected key: the operator may override it

    async def test_extra_body_cannot_replace_protected_keys(self) -> None:
        rec = Recorder(ok())
        entry = _entry(
            extra_body={"model": "evil", "messages": [], "response_format": {"type": "text"}}
        )
        await _adapter(rec, entry).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        body = rec.body
        assert body["model"] == "test-model-1"
        assert len(body["messages"]) == 2
        assert body["response_format"]["type"] == "json_schema"

    async def test_trailing_slash_in_base_url_is_harmless(self) -> None:
        rec = Recorder(ok())
        await _adapter(rec, _entry(base_url="https://llm.example.test/v1/")).complete(
            MESSAGES, schema=PlanDraft, schema_name="x"
        )
        assert str(rec.requests[0].url) == "https://llm.example.test/v1/chat/completions"


class TestResponse:
    async def test_usage_and_model_are_read(self) -> None:
        result = await _adapter(Recorder(ok())).complete(
            MESSAGES, schema=PlanDraft, schema_name="x"
        )
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 7)
        assert result.model_id == "served-model-id"

    async def test_missing_usage_is_zero(self) -> None:
        payload = _reply()
        del payload["usage"]
        result = await _adapter(Recorder(ok(payload))).complete(
            MESSAGES, schema=PlanDraft, schema_name="x"
        )
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (0, 0)

    async def test_finish_reason_length_is_truncation(self) -> None:
        with pytest.raises(LlmOutputTruncated):
            await _adapter(Recorder(ok(_reply(finish="length")))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": []},
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": '{"title": 1}'}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": '["x"]'}}]},
        ],
        ids=["no-choice", "empty", "null", "not-json", "wrong-shape", "not-an-object"],
    )
    async def test_an_unusable_reply_is_invalid(self, payload: dict[str, Any]) -> None:
        with pytest.raises(LlmOutputInvalid):
            await _adapter(Recorder(ok(payload))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )

    async def test_a_non_json_body_is_invalid(self) -> None:
        with pytest.raises(LlmOutputInvalid):
            await _adapter(Recorder(httpx.Response(200, text="<html>nope</html>"))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )

    async def test_errors_never_carry_response_text(self) -> None:
        leaky = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"kind": LEAK, "title": LEAK})},
                }
            ]
        }
        with pytest.raises(LlmOutputInvalid) as invalid:
            await _adapter(Recorder(ok(leaky))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        with pytest.raises(LlmRejected) as rejected:
            await _adapter(Recorder(httpx.Response(400, text=LEAK))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        for exc in (invalid.value, rejected.value):
            assert LEAK not in str(exc)
            assert LEAK not in repr(exc)
            assert exc.__cause__ is None
            assert API_KEY not in str(exc)


class TestUsageOfUnusableReplies:
    """Invariant 26: a reply that was paid for but cannot be used still reports its tokens."""

    USAGE: ClassVar[dict[str, int]] = {"prompt_tokens": 11, "completion_tokens": 7}

    async def test_a_truncated_reply_carries_its_usage(self) -> None:
        with pytest.raises(LlmOutputTruncated) as caught:
            await _adapter(Recorder(ok(_reply(finish="length")))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        assert caught.value.usage == LlmUsage(prompt_tokens=11, completion_tokens=7)

    @pytest.mark.parametrize(
        "choices",
        [
            [],
            [{"finish_reason": "stop", "message": {"content": ""}}],
            [{"finish_reason": "stop", "message": {"content": "not json"}}],
            [{"finish_reason": "stop", "message": {"content": '{"title": 1}'}}],
        ],
        ids=["no-choice", "empty", "not-json", "wrong-shape"],
    )
    async def test_an_undecodable_reply_carries_its_usage(self, choices: list[Any]) -> None:
        payload = {"choices": choices, "usage": self.USAGE}
        with pytest.raises(LlmOutputInvalid) as caught:
            await _adapter(Recorder(ok(payload))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        assert caught.value.usage == LlmUsage(prompt_tokens=11, completion_tokens=7)

    async def test_a_reply_without_a_usage_block_has_none_to_carry(self) -> None:
        payload = {"choices": [{"finish_reason": "length", "message": {"content": "x"}}]}
        with pytest.raises(LlmOutputTruncated) as caught:
            await _adapter(Recorder(ok(payload))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        assert caught.value.usage is None

    async def test_a_body_that_is_not_json_has_no_usage(self) -> None:
        with pytest.raises(LlmOutputInvalid) as caught:
            await _adapter(Recorder(httpx.Response(200, text="<html>nope</html>"))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        assert caught.value.usage is None

    async def test_usage_is_numbers_only(self) -> None:
        """The usage block is read field by field: text in it never reaches the error."""
        payload = {
            "choices": [{"finish_reason": "length", "message": {"content": LEAK}}],
            "usage": {"prompt_tokens": LEAK, "completion_tokens": 5},
        }
        with pytest.raises(LlmOutputTruncated) as caught:
            await _adapter(Recorder(ok(payload))).complete(
                MESSAGES, schema=PlanDraft, schema_name="x"
            )
        assert caught.value.usage == LlmUsage(prompt_tokens=0, completion_tokens=5)
        assert LEAK not in str(caught.value)
        assert LEAK not in repr(caught.value.usage)

    async def test_a_usable_reply_still_reports_zero_without_a_usage_block(self) -> None:
        payload = _reply()
        del payload["usage"]
        result = await _adapter(Recorder(ok(payload))).complete(
            MESSAGES, schema=PlanDraft, schema_name="x"
        )
        assert result.usage == LlmUsage(prompt_tokens=0, completion_tokens=0)

    def test_the_errors_take_usage_as_an_optional_keyword(self) -> None:
        assert LlmOutputInvalid("x").usage is None
        assert LlmOutputTruncated("x").usage is None
        usage = LlmUsage(prompt_tokens=1, completion_tokens=2)
        assert LlmOutputInvalid("x", usage=usage).usage is usage
        assert LlmOutputTruncated("x", usage=usage).usage is usage


class TestRetry:
    @pytest.mark.parametrize(
        "first",
        [
            httpx.Response(429),
            httpx.Response(500),
            httpx.Response(503),
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("down"),
        ],
        ids=["429", "500", "503", "timeout", "connect"],
    )
    async def test_one_retry_then_success(self, first: httpx.Response | Exception) -> None:
        rec = Recorder(first, ok())
        sleeps: list[float] = []
        result = await _adapter(rec, sleeps=sleeps).complete(
            MESSAGES, schema=PlanDraft, schema_name="x"
        )
        assert len(rec.requests) == 2
        assert len(sleeps) == 1
        assert result.value == make_draft()

    async def test_two_failures_are_unavailable(self) -> None:
        rec = Recorder(httpx.Response(503, text=LEAK), httpx.Response(503, text=LEAK))
        with pytest.raises(LlmUnavailable) as caught:
            await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert len(rec.requests) == 2  # never a third
        assert LEAK not in str(caught.value)

    async def test_the_failure_after_the_last_attempt_names_the_last_cause(self) -> None:
        rec = Recorder(httpx.Response(429), httpx.Response(502))
        with pytest.raises(LlmUnavailable, match=r"http_502"):
            await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")

        rec = Recorder(httpx.Response(502), httpx.ReadTimeout("slow"))
        with pytest.raises(LlmUnavailable, match=r"transport"):
            await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")

    async def test_there_is_no_sleep_after_the_last_attempt(self) -> None:
        sleeps: list[float] = []
        rec = Recorder(httpx.Response(503), httpx.Response(503))
        with pytest.raises(LlmUnavailable):
            await _adapter(rec, sleeps=sleeps).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert len(sleeps) == 1  # between the two attempts only

    async def test_a_timeout_twice_is_unavailable(self) -> None:
        rec = Recorder(httpx.ReadTimeout("a"), httpx.ReadTimeout("b"))
        with pytest.raises(LlmUnavailable):
            await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")

    async def test_retry_after_is_honoured_and_capped(self) -> None:
        sleeps: list[float] = []
        rec = Recorder(httpx.Response(429, headers={"retry-after": "3"}), ok())
        await _adapter(rec, sleeps=sleeps).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert sleeps == [3.0]

        sleeps.clear()
        rec = Recorder(httpx.Response(429, headers={"retry-after": "9999"}), ok())
        await _adapter(rec, sleeps=sleeps).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert sleeps == [10.0]

    async def test_a_client_error_is_not_retried(self) -> None:
        rec = Recorder(httpx.Response(401, text="bad key"))
        with pytest.raises(LlmRejected) as caught:
            await _adapter(rec).complete(MESSAGES, schema=PlanDraft, schema_name="x")
        assert caught.value.status_code == 401
        assert len(rec.requests) == 1


class TestLogging:
    async def test_a_call_logs_only_metadata(self) -> None:
        rec = Recorder(ok())
        with capture_logs() as logs:
            await _adapter(rec).complete(
                [LlmMessage("user", LEAK)], schema=PlanDraft, schema_name="x"
            )
        calls = [log for log in logs if log["event"] == "llm.call"]
        assert len(calls) == 1
        entry = calls[0]
        assert entry["model_key"] == "test"
        assert entry["status_code"] == 200
        assert entry["prompt_tokens"] == 11
        assert entry["attempt"] == 1
        assert "duration_ms" in entry
        rendered = json.dumps(logs, default=str)
        assert LEAK not in rendered
        assert API_KEY not in rendered
