"""Model Studio client wire behaviour, failure acceptance rules (C14) and log hygiene (C17)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.model_studio import (
    ModelStudioAPIError,
    ModelStudioClient,
    ModelStudioTaskStatus,
    ModelStudioTransientError,
)

pytestmark = pytest.mark.unit

_KEY = "sk-super-secret-key"
_PROMPT = "a very private prompt about the neighbours"
_VIDEO_URL = "https://oss.example.com/result.mp4?Signature=abc"
_INPUT_URL = "https://r2.example.com/in.png?X-Amz-Signature=def"


def _client(handler: Any) -> ModelStudioClient:
    return ModelStudioClient(
        base_url="https://ws.ap-southeast-1.maas.aliyuncs.com",
        api_key=SecretStr(_KEY),
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )


def _json(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status, json=body)


_PAYLOAD = {
    "model": "wan3.0-video",
    "input": {"prompt": _PROMPT, "media": [{"type": "first_frame", "url": _INPUT_URL}]},
    "parameters": {"resolution": "720P"},
}


async def test_create_sends_the_async_header_bearer_and_json_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return _json(
            200, {"output": {"task_status": "PENDING", "task_id": "t-1"}, "request_id": "r-1"}
        )

    client = _client(handler)
    await client.connect()
    ref = await client.create_video_task(_PAYLOAD)
    await client.close()

    assert seen["method"] == "POST"
    assert seen["url"] == (
        "https://ws.ap-southeast-1.maas.aliyuncs.com"
        "/api/v1/services/aigc/video-generation/video-synthesis"
    )
    assert seen["headers"]["x-dashscope-async"] == "enable"
    assert seen["headers"]["authorization"] == f"Bearer {_KEY}"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"] == _PAYLOAD
    assert (ref.task_id, ref.request_id) == ("t-1", "r-1")


async def test_get_task_parses_a_successful_task_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/tasks/t-1"
        assert "x-dashscope-async" not in request.headers
        return _json(
            200,
            {
                "request_id": "r-2",
                "output": {
                    "task_id": "t-1",
                    "task_status": "SUCCEEDED",
                    "video_url": _VIDEO_URL,
                    "submit_time": "2026-10-09 10:00:00.000",
                    "extra_unknown_field": 1,
                },
                "usage": {"duration": 7.0, "input_video_duration": 2.0, "SR": 720, "ratio": "16:9"},
            },
        )

    client = _client(handler)
    await client.connect()
    task = await client.get_task("t-1")
    await client.close()

    assert task.task_status is ModelStudioTaskStatus.SUCCEEDED
    assert task.video_url == _VIDEO_URL
    assert task.usage is not None
    assert (task.usage.duration, task.usage.sr) == (7.0, 720)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PENDING", ModelStudioTaskStatus.PENDING),
        ("RUNNING", ModelStudioTaskStatus.RUNNING),
        ("FAILED", ModelStudioTaskStatus.FAILED),
        ("CANCELED", ModelStudioTaskStatus.CANCELED),
        ("UNKNOWN", ModelStudioTaskStatus.UNKNOWN),
        ("SOMETHING_NEW", ModelStudioTaskStatus.UNKNOWN),
    ],
)
async def test_get_task_maps_every_status_with_a_safe_fallback(
    raw: str, expected: ModelStudioTaskStatus
) -> None:
    client = _client(
        lambda _: _json(200, {"output": {"task_id": "t", "task_status": raw, "code": "X"}})
    )
    await client.connect()

    task = await client.get_task("t")

    assert task.task_status is expected
    await client.close()


# ---- create-time acceptance rules ------------------------------------------------------


async def test_create_4xx_is_definitely_not_accepted() -> None:
    client = _client(
        lambda _: _json(400, {"code": "InvalidParameter", "message": "bad", "request_id": "r"})
    )
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    failure = raised.value.failure
    assert failure.kind is ProviderFailureKind.INVALID_REQUEST
    assert failure.provider_request_accepted is False
    assert failure.provider_request_id == "r"
    await client.close()


async def test_create_moderation_rejection_is_classified_from_the_code() -> None:
    client = _client(lambda _: _json(400, {"code": "DataInspectionFailed", "message": "x"}))
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    assert raised.value.failure.kind is ProviderFailureKind.MODERATION_REJECTED
    await client.close()


async def test_create_timeout_is_ambiguous_and_logs_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler)
    await client.connect()

    with capture_logs() as logs, pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    failure = raised.value.failure
    assert failure.kind is ProviderFailureKind.TIMEOUT
    assert failure.provider_request_accepted is None
    assert any(entry["event"] == "alibaba.submit_ambiguous" for entry in logs)
    await client.close()


async def test_create_connection_error_is_ambiguous() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    assert raised.value.failure.kind is ProviderFailureKind.PROVIDER_UNAVAILABLE
    assert raised.value.failure.provider_request_accepted is None
    await client.close()


async def test_create_5xx_is_ambiguous() -> None:
    client = _client(lambda _: _json(502, {}))
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    assert raised.value.failure.provider_request_accepted is None
    await client.close()


@pytest.mark.parametrize("response", [httpx.Response(200, content=b"not json"), _json(200, {})])
async def test_create_unreadable_success_is_malformed_and_ambiguous(
    response: httpx.Response,
) -> None:
    client = _client(lambda _: response)
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    assert raised.value.failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert raised.value.failure.provider_request_accepted is None
    await client.close()


# ---- query-time rules ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (429, {"code": "Throttling.RateQuota"}, ProviderFailureKind.RATE_LIMITED),
        (500, {"code": "InternalError"}, ProviderFailureKind.PROVIDER_UNAVAILABLE),
        (429, {"code": "Arrearage"}, ProviderFailureKind.PROVIDER_UNAVAILABLE),
    ],
)
async def test_poll_transient_failures_do_not_fail_the_job(
    status: int, body: dict[str, Any], kind: ProviderFailureKind
) -> None:
    client = _client(lambda _: _json(status, body))
    await client.connect()

    with pytest.raises(ModelStudioTransientError) as raised:
        await client.get_task("t-1")

    assert raised.value.failure.kind is kind
    assert raised.value.failure.provider_request_accepted is True
    await client.close()


async def test_poll_timeout_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler)
    await client.connect()

    with pytest.raises(ModelStudioTransientError) as raised:
        await client.get_task("t-1")

    assert raised.value.failure.kind is ProviderFailureKind.TIMEOUT
    await client.close()


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "InvalidApiKey"), (403, "AccessDenied"), (403, "Workspace.AccessDenied")],
)
async def test_poll_auth_failure_is_transient_and_logged_at_error(status: int, code: str) -> None:
    client = _client(lambda _: _json(status, {"code": code, "message": f"secret {_KEY}"}))
    await client.connect()

    with capture_logs() as logs, pytest.raises(ModelStudioTransientError) as raised:
        await client.get_task("t-1")

    assert raised.value.failure.kind is ProviderFailureKind.AUTHENTICATION_FAILED
    assert raised.value.failure.provider_request_accepted is True
    (event,) = [e for e in logs if e["event"] == "alibaba.poll_auth_failed"]
    assert event["log_level"] == "error"
    assert event["provider_status"] == status
    assert event["provider_error_code"] == code
    assert event["provider_job_id"] == "t-1"
    _assert_clean(repr(logs))
    await client.close()


async def test_create_auth_failure_stays_terminal_and_non_billable() -> None:
    """R4 is query-only: nothing was accepted on create, so it must still fail the request."""
    client = _client(lambda _: _json(401, {"code": "InvalidApiKey"}))
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task({"model": "wan3.0-video", "input": {"prompt": "x"}})

    assert not isinstance(raised.value, ModelStudioTransientError)
    assert raised.value.failure.kind is ProviderFailureKind.AUTHENTICATION_FAILED
    assert raised.value.failure.billable is False
    await client.close()


@pytest.mark.parametrize("response", [httpx.Response(200, content=b"<html>"), _json(200, {})])
async def test_poll_unreadable_body_is_malformed_but_accepted(response: httpx.Response) -> None:
    client = _client(lambda _: response)
    await client.connect()

    with pytest.raises(ModelStudioAPIError) as raised:
        await client.get_task("t-1")

    assert raised.value.failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert raised.value.failure.provider_request_accepted is True
    await client.close()


async def test_client_requires_connect() -> None:
    client = _client(lambda _: _json(200, {}))

    with pytest.raises(RuntimeError, match="connect"):
        await client.get_task("t-1")


# ---- C17: no secrets in logs or exceptions ---------------------------------------------


def _assert_clean(text: str) -> None:
    for secret in (_KEY, _PROMPT, _VIDEO_URL, _INPUT_URL, "Signature"):
        assert secret not in text


async def test_c17_a_failing_create_leaks_neither_logs_nor_exceptions() -> None:
    body = {
        "code": "InvalidParameter",
        "message": f"bad {_PROMPT} {_INPUT_URL} {_KEY}",
        "request_id": "r-9",
    }
    client = _client(lambda _: _json(400, body))
    await client.connect()

    with capture_logs() as logs, pytest.raises(ModelStudioAPIError) as raised:
        await client.create_video_task(_PAYLOAD)

    _assert_clean(repr(logs))
    _assert_clean(str(raised.value))
    _assert_clean(repr(raised.value.failure))
    await client.close()


async def test_c17_a_failing_poll_leaks_neither_logs_nor_exceptions() -> None:
    body = {
        "request_id": "r-9",
        "output": {
            "task_id": "t-1",
            "task_status": "FAILED",
            "code": "DataInspectionFailed",
            "message": f"{_PROMPT} {_VIDEO_URL}",
            "video_url": _VIDEO_URL,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/t-1"):
            return httpx.Response(500, json={"message": f"{_VIDEO_URL} {_KEY} {_PROMPT}"})
        return _json(200, body)

    client = _client(handler)
    await client.connect()

    with capture_logs() as logs, pytest.raises(ModelStudioTransientError) as raised:
        await client.get_task("t-1")

    _assert_clean(repr(logs))
    _assert_clean(str(raised.value))
    _assert_clean(repr(raised.value.failure))
    await client.close()
