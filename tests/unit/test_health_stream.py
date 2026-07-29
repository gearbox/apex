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
        with patch("src.api.services.health.stream.get_sse_redis_client", return_value=mock_client):
            async for event in _redis_stream():
                events.append(event)
                if events:
                    break

        assert events[0]["event"] == "health.snapshot"
        assert "healthy" in events[0]["data"]

    async def test_uses_sse_pool_not_shared_pool(self) -> None:
        """_redis_stream must check out its connection from the dedicated SSE
        pool, never the shared pool used by TokenRevocationService/LeaderLease —
        a long-lived per-admin subscriber on the shared pool can exhaust it."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        with (
            patch(
                "src.api.services.health.stream.get_sse_redis_client",
                return_value=mock_client,
            ) as mock_get_sse_client,
            patch("src.core.redis.get_redis_client") as mock_get_client,
        ):
            async for _event in _redis_stream():
                break

        mock_get_sse_client.assert_called_once()
        mock_get_client.assert_not_called()

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
        with patch("src.api.services.health.stream.get_sse_redis_client", return_value=mock_client):
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
        with patch("src.api.services.health.stream.get_sse_redis_client", return_value=mock_client):
            async for event in _redis_stream():
                events.append(event)
                if events:
                    break

        assert events[0] == {"comment": "keepalive"}

    async def test_subscribe_failure_closes_pubsub_and_logs(self) -> None:
        """A failed initial subscribe() (SSE pool exhaustion, or Redis down)
        must still release the PubSub via aclose(), log
        health.sse.subscribe_failed with cause detail, and re-raise —
        this is the E2 regression."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytest
        from redis.exceptions import ConnectionError as RedisConnectionError

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock(side_effect=RedisConnectionError("Too many connections"))
        mock_pubsub.aclose = AsyncMock()
        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        mock_pool = MagicMock(max_connections=500)

        with (
            patch(
                "src.api.services.health.stream.get_sse_redis_client",
                return_value=mock_client,
            ),
            patch(
                "src.api.services.health.stream.get_sse_redis_pool",
                return_value=mock_pool,
            ),
            patch("src.api.services.health.stream.logger") as mock_logger,
            pytest.raises(RedisConnectionError),
        ):
            async for _event in _redis_stream():
                pass

        mock_pubsub.aclose.assert_awaited_once()
        mock_logger.exception.assert_called_once()
        assert mock_logger.exception.call_args[0][0] == "health.sse.subscribe_failed"
        kwargs = mock_logger.exception.call_args[1]
        assert kwargs["error"] == "Too many connections"
        assert kwargs["error_type"] == "ConnectionError"
        assert kwargs["sse_pool_max"] == 500

    async def test_cancelled_error_on_disconnect_logs_and_closes_cleanly(self) -> None:
        """A client disconnect (CancelledError) is still handled by the
        dedicated except arm, not swallowed by the new subscribe-failure
        handler, and cleanup still runs."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.services.health.stream import _redis_stream

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = AsyncMock(side_effect=asyncio.CancelledError)
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub

        with (
            patch(
                "src.api.services.health.stream.get_sse_redis_client",
                return_value=mock_client,
            ),
            patch("src.api.services.health.stream.logger") as mock_logger,
        ):
            async for _event in _redis_stream():
                pass

        mock_logger.debug.assert_called_once_with("health.sse.client_disconnected")
        mock_pubsub.unsubscribe.assert_awaited_once()
        mock_pubsub.aclose.assert_awaited_once()

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
