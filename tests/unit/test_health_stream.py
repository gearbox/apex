"""Tests for SSE stream generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.api.services.health.stream import health_sse_generator


def test_health_stream_channel_constant_exists() -> None:
    from src.api.services.health import HEALTH_STREAM_CHANNEL

    assert HEALTH_STREAM_CHANNEL == "health:stream"


class TestPollingFallback:
    async def test_yields_snapshot_event(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            return_value={"status": "healthy", "checked_at": "2026-01-01T00:00:00Z"},
        )

        mock_settings = MagicMock()
        mock_settings.redis_url = None
        mock_settings.health_snapshot_interval_seconds = 0.01  # fast for test

        events = []
        async for event in health_sse_generator(
            health_service=mock_service,
            settings=mock_settings,
        ):
            events.append(event)
            if len(events) >= 2:
                break

        assert events[0]["event"] == "health.snapshot"
        assert "healthy" in events[0]["data"]

    async def test_yields_error_comment_on_failure(self) -> None:
        mock_service = AsyncMock()
        mock_service.check_all_and_build = AsyncMock(
            side_effect=RuntimeError("check failed"),
        )

        mock_settings = MagicMock()
        mock_settings.redis_url = None
        mock_settings.health_snapshot_interval_seconds = 0.01

        events = []
        async for event in health_sse_generator(
            health_service=mock_service,
            settings=mock_settings,
        ):
            events.append(event)
            if events:
                break

        assert events[0] == {"comment": "error"}
