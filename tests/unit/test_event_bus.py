"""Unit tests for EventBus — mocked Redis client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.events import (
    EventEnvelope,
    EventType,
    JobStatusPayload,
    SystemNotificationPayload,
)
from src.api.services.event_bus import EventBus
from src.core.enums import NotificationLevel

_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(EventEnvelope)

_HEARTBEAT_INTERVAL = 15.0


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def user_id():  # type: ignore[no-untyped-def]
    return uuid4()


@pytest.fixture
def job_status_payload():  # type: ignore[no-untyped-def]
    return JobStatusPayload(
        job_id=uuid4(),
        status="completed",
        previous_status="running",
        generation_type="t2v",
        provider="grok",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wire(payload: object, event_type: EventType = EventType.JOB_STATUS_CHANGED) -> bytes:
    import datetime

    envelope = EventEnvelope(
        event_type=event_type,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.datetime(2026, 3, 17, 12, tzinfo=datetime.UTC),
        event_id="evt-1",
    )
    return _encoder.encode(envelope)


def _mock_pubsub(get_message_side_effect: object) -> AsyncMock:
    """Return a mock pubsub whose get_message is pre-configured."""
    mock_pubsub = AsyncMock()
    mock_pubsub.get_message = AsyncMock(side_effect=get_message_side_effect)
    return mock_pubsub


def _mock_client(pubsub: AsyncMock) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.pubsub = MagicMock(return_value=pubsub)
    return mock_client


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


class TestEventBusPublish:
    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_calls_redis_publish(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await event_bus.publish(
            user_id=user_id,
            event_type=EventType.JOB_STATUS_CHANGED,
            payload=job_status_payload,
        )

        mock_client.publish.assert_awaited_once()
        mock_client.aclose.assert_not_awaited()

        channel_arg = mock_client.publish.call_args[0][0]
        assert channel_arg == f"user:{user_id}"

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_data_is_valid_envelope(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await event_bus.publish(
            user_id=user_id,
            event_type=EventType.JOB_STATUS_CHANGED,
            payload=job_status_payload,
        )

        raw_data = mock_client.publish.call_args[0][1]
        envelope = _decoder.decode(raw_data)
        assert envelope.event_type == EventType.JOB_STATUS_CHANGED
        inner = msgspec.json.decode(bytes(envelope.payload), type=JobStatusPayload)
        assert inner.status == "completed"

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_does_not_close_shared_pool_on_error(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        mock_client = AsyncMock()
        mock_client.publish.side_effect = RuntimeError("connection lost")
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError):
            await event_bus.publish(
                user_id=user_id,
                event_type=EventType.JOB_STATUS_CHANGED,
                payload=job_status_payload,
            )

        mock_client.aclose.assert_not_awaited()


# ---------------------------------------------------------------------------
# Publish system
# ---------------------------------------------------------------------------


class TestEventBusPublishSystem:
    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_system_uses_broadcast_channel(
        self, mock_get_client: MagicMock, event_bus: EventBus
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        payload = SystemNotificationPayload(level=NotificationLevel.INFO, title="T", message="M")
        await event_bus.publish_system(
            event_type=EventType.SYSTEM_NOTIFICATION,
            payload=payload,
        )

        mock_client.publish.assert_awaited_once()
        channel_arg = mock_client.publish.call_args[0][0]
        assert channel_arg == "system:broadcast"
        mock_client.aclose.assert_not_awaited()

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_system_data_is_valid_envelope(
        self, mock_get_client: MagicMock, event_bus: EventBus
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        payload = SystemNotificationPayload(
            level=NotificationLevel.CRITICAL, title="Down", message="Outage"
        )
        await event_bus.publish_system(event_type=EventType.SYSTEM_NOTIFICATION, payload=payload)

        raw_data = mock_client.publish.call_args[0][1]
        envelope = _decoder.decode(raw_data)
        assert envelope.event_type == EventType.SYSTEM_NOTIFICATION
        inner = msgspec.json.decode(bytes(envelope.payload), type=SystemNotificationPayload)
        assert inner.level == "critical"


# ---------------------------------------------------------------------------
# Subscribe — get_message-based implementation
# ---------------------------------------------------------------------------


class TestEventBusSubscribe:
    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_yields_decoded_envelope(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        wire = _make_wire(job_status_payload)
        pubsub = _mock_pubsub([{"type": "message", "channel": f"user:{user_id}", "data": wire}])
        mock_get_client.return_value = _mock_client(pubsub)

        results = []
        async for item in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
            if item is not None:
                results.append(item)
                break

        assert len(results) == 1
        assert results[0].event_type == EventType.JOB_STATUS_CHANGED

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_yields_none_on_get_message_timeout(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        """get_message returning None (no message in interval) → subscribe yields None."""
        pubsub = _mock_pubsub([None])
        mock_get_client.return_value = _mock_client(pubsub)

        results = []
        async for item in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
            results.append(item)
            break  # stop after first heartbeat tick

        assert results == [None]

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_get_message_called_with_heartbeat_interval(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        """get_message must be called with timeout=heartbeat_interval."""
        pubsub = _mock_pubsub([None])
        mock_get_client.return_value = _mock_client(pubsub)

        async for _ in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
            break

        pubsub.get_message.assert_awaited_with(
            ignore_subscribe_messages=True, timeout=_HEARTBEAT_INTERVAL
        )

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_skips_decode_errors_and_continues(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        """A bad message is skipped and the next good message is still yielded."""
        wire = _make_wire(job_status_payload)
        pubsub = _mock_pubsub(
            [
                {"type": "message", "channel": f"user:{user_id}", "data": b"not-valid-json"},
                {"type": "message", "channel": f"user:{user_id}", "data": wire},
            ]
        )
        mock_get_client.return_value = _mock_client(pubsub)

        results = []
        async for item in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
            if item is not None:
                results.append(item)
                break

        assert len(results) == 1
        assert results[0].event_type == EventType.JOB_STATUS_CHANGED

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_skips_non_message_type(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        """Messages with type != 'message' are skipped."""
        wire = _make_wire(job_status_payload)
        pubsub = _mock_pubsub(
            [
                {"type": "psubscribe", "data": 1},  # skipped
                {"type": "message", "channel": f"user:{user_id}", "data": wire},
            ]
        )
        mock_get_client.return_value = _mock_client(pubsub)

        results = []
        async for item in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
            if item is not None:
                results.append(item)
                break

        assert len(results) == 1

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_unsubscribes_and_closes_on_exit(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        pubsub = _mock_pubsub([None])
        mock_get_client.return_value = _mock_client(pubsub)

        gen = event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL)
        async for _ in gen:
            break
        # Python defers async-generator finalization after break; drive it now
        await gen.aclose()

        pubsub.unsubscribe.assert_awaited_once()
        pubsub.aclose.assert_awaited_once()
        # shared pool client must NOT be closed
        mock_get_client.return_value.aclose.assert_not_awaited()

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_unsubscribes_on_cancellation(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        """Cleanup (unsubscribe + aclose) runs even when the generator is cancelled."""

        async def cancel_on_second(*_args: object, **_kwargs: object) -> None:
            cancel_on_second.calls = getattr(cancel_on_second, "calls", 0) + 1  # type: ignore[attr-defined]
            if cancel_on_second.calls >= 2:  # type: ignore[attr-defined]
                raise asyncio.CancelledError
            return None  # type: ignore[return-value]

        pubsub = AsyncMock()
        pubsub.get_message = AsyncMock(side_effect=cancel_on_second)
        mock_get_client.return_value = _mock_client(pubsub)

        with pytest.raises(asyncio.CancelledError):
            async for _ in event_bus.subscribe(user_id, heartbeat_interval=_HEARTBEAT_INTERVAL):
                pass

        pubsub.unsubscribe.assert_awaited_once()
        pubsub.aclose.assert_awaited_once()
