"""Tests for the Grok API client."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.api.services.grok import (
    GrokAPIError,
    GrokClient,
    GrokModerationError,
    GrokVideoResult,
)


class _DeferredStatus:
    DONE = 1
    PENDING = 2
    FAILED = 3
    EXPIRED = 4


@dataclass(frozen=True, slots=True)
class _Video:
    url: str = ""
    respect_moderation: bool = True


@dataclass(frozen=True, slots=True)
class _Error:
    code: str
    message: str


class _DeferredPayload:
    def __init__(self, *, video: _Video | None = None, error: _Error | None = None) -> None:
        self.video = video
        self.error = error

    def HasField(self, name: str) -> bool:
        return getattr(self, name) is not None


class _DeferredResponse:
    def __init__(self, status: int, payload: _DeferredPayload | None = None) -> None:
        self.status = status
        self.response = payload

    def HasField(self, name: str) -> bool:
        return name == "response" and self.response is not None


class TestParseDeferredVideoResult:
    def test_done_returns_video_result(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.DONE,
            _DeferredPayload(video=_Video(url="https://example.test/video.mp4")),
        )

        result = GrokClient._parse_deferred_video_result(
            response,
            _DeferredStatus,
            request_id="req-1",
        )

        assert result == GrokVideoResult(url="https://example.test/video.mp4")

    def test_pending_returns_none(self) -> None:
        result = GrokClient._parse_deferred_video_result(
            _DeferredResponse(_DeferredStatus.PENDING),
            _DeferredStatus,
            request_id="req-1",
        )

        assert result is None

    def test_done_without_response_raises_api_error(self) -> None:
        with pytest.raises(GrokAPIError, match="no response returned"):
            GrokClient._parse_deferred_video_result(
                _DeferredResponse(_DeferredStatus.DONE),
                _DeferredStatus,
                request_id="req-1",
            )

    def test_done_without_url_and_failed_moderation_raises_moderation_error(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.DONE,
            _DeferredPayload(video=_Video(respect_moderation=False)),
        )

        with pytest.raises(GrokModerationError, match="flagged by moderation"):
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-1",
            )

    def test_failed_with_error_payload_raises_api_error_with_code_and_message(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.FAILED,
            _DeferredPayload(error=_Error(code="E42", message="upstream failed")),
        )

        with pytest.raises(GrokAPIError, match=r"\[E42\] upstream failed"):
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-1",
            )

    def test_expired_raises_api_error(self) -> None:
        with pytest.raises(GrokAPIError, match="expired before completion"):
            GrokClient._parse_deferred_video_result(
                _DeferredResponse(_DeferredStatus.EXPIRED),
                _DeferredStatus,
                request_id="req-1",
            )

    def test_unknown_status_is_treated_as_pending(self) -> None:
        result = GrokClient._parse_deferred_video_result(
            _DeferredResponse(999),
            _DeferredStatus,
            request_id="req-1",
        )

        assert result is None
