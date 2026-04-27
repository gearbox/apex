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


class TestRedisStream:
    """Tests for _redis_stream via health_sse_generator with redis_url set."""

    async def test_yields_snapshot_event_from_redis_message(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        message = {"type": "message", "data": b'{"status":"healthy"}'}
        mock_pubsub.get_message = AsyncMock(return_value=message)

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        events: list[dict[str, str]] = []
        with patch("src.api.services.health.stream.get_redis_client", return_value=mock_client):
            async for event in _redis_stream():
                events.append(event)
                if events:
                    break

        assert events[0]["event"] == "health.snapshot"
        assert "healthy" in events[0]["data"]

    async def test_yields_keepalive_when_no_message(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        # Returns None — no message
        mock_pubsub.get_message = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        events: list[dict[str, str]] = []
        with patch("src.api.services.health.stream.get_redis_client", return_value=mock_client):
            async for event in _redis_stream():
                events.append(event)
                if events:
                    break

        assert events[0] == {"comment": "keepalive"}

    async def test_yields_keepalive_on_timeout(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_pubsub.get_message = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        events: list[dict[str, str]] = []
        with patch("src.api.services.health.stream.get_redis_client", return_value=mock_client):
            async for event in _redis_stream():
                events.append(event)
                if events:
                    break

        assert events[0] == {"comment": "keepalive"}

    async def test_health_sse_generator_routes_to_redis_when_url_set(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_service = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost"

        async def _fake_redis_stream() -> object:
            yield {"event": "health.snapshot", "data": "{}"}

        events: list[dict[str, str]] = []
        with patch(
            "src.api.services.health.stream._redis_stream",
            return_value=_fake_redis_stream(),
        ):
            async for event in health_sse_generator(
                health_service=mock_service, settings=mock_settings
            ):
                events.append(event)
                break

        assert events[0]["event"] == "health.snapshot"
