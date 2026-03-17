"""Unit tests for EventBus — mocked Redis client."""

from __future__ import annotations

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

_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(EventEnvelope)


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
        mock_client.aclose.assert_awaited_once()

        # Check channel name
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
    async def test_publish_closes_client_on_error(
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

        # aclose must still be called even after error
        mock_client.aclose.assert_awaited_once()


class TestEventBusPublishSystem:
    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_system_uses_broadcast_channel(
        self, mock_get_client: MagicMock, event_bus: EventBus
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        payload = SystemNotificationPayload(level="info", title="T", message="M")
        await event_bus.publish_system(
            event_type=EventType.SYSTEM_NOTIFICATION,
            payload=payload,
        )

        mock_client.publish.assert_awaited_once()
        channel_arg = mock_client.publish.call_args[0][0]
        assert channel_arg == "system:broadcast"
        mock_client.aclose.assert_awaited_once()

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_publish_system_data_is_valid_envelope(
        self, mock_get_client: MagicMock, event_bus: EventBus
    ) -> None:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        payload = SystemNotificationPayload(level="critical", title="Down", message="Outage")
        await event_bus.publish_system(event_type=EventType.SYSTEM_NOTIFICATION, payload=payload)

        raw_data = mock_client.publish.call_args[0][1]
        envelope = _decoder.decode(raw_data)
        assert envelope.event_type == EventType.SYSTEM_NOTIFICATION
        inner = msgspec.json.decode(bytes(envelope.payload), type=SystemNotificationPayload)
        assert inner.level == "critical"


class TestEventBusSubscribe:
    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_yields_decoded_envelopes(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id, job_status_payload
    ) -> None:
        import datetime

        # Build a wire-format message as Redis would deliver it
        envelope = EventEnvelope(
            event_type=EventType.JOB_STATUS_CHANGED,
            payload=msgspec.Raw(_encoder.encode(job_status_payload)),
            timestamp=datetime.datetime(2026, 3, 17, 12, tzinfo=datetime.UTC),
            event_id="evt-1",
        )
        wire = _encoder.encode(envelope)

        # Mock Redis pubsub — listen() returns an async iterable directly (not a coroutine)
        mock_pubsub = AsyncMock()
        mock_pubsub.listen = MagicMock(
            return_value=aiter_from_list(
                [
                    {"type": "subscribe", "data": 1},
                    {"type": "message", "channel": f"user:{user_id}", "data": wire},
                ]
            )
        )

        # pubsub() is a sync call on the client (not awaited)
        mock_client = AsyncMock()
        mock_client.pubsub = MagicMock(return_value=mock_pubsub)
        mock_get_client.return_value = mock_client

        results = []
        async for evt in event_bus.subscribe(user_id):
            results.append(evt)
            break  # Stop after first message

        assert len(results) == 1
        assert results[0].event_type == EventType.JOB_STATUS_CHANGED

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_skips_decode_errors(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        mock_pubsub = AsyncMock()
        mock_pubsub.listen = MagicMock(
            return_value=aiter_from_list(
                [
                    {"type": "message", "channel": f"user:{user_id}", "data": b"not-valid-json"},
                ]
            )
        )

        mock_client = AsyncMock()
        mock_client.pubsub = MagicMock(return_value=mock_pubsub)
        mock_get_client.return_value = mock_client

        results = []
        # Should not raise — decode errors are logged and skipped
        async for evt in event_bus.subscribe(user_id):
            results.append(evt)

        assert not results

    @patch("src.api.services.event_bus.get_redis_client")
    async def test_subscribe_unsubscribes_on_exit(
        self, mock_get_client: MagicMock, event_bus: EventBus, user_id
    ) -> None:
        mock_pubsub = AsyncMock()
        mock_pubsub.listen = MagicMock(return_value=aiter_from_list([]))

        mock_client = AsyncMock()
        mock_client.pubsub = MagicMock(return_value=mock_pubsub)
        mock_get_client.return_value = mock_client

        async for _ in event_bus.subscribe(user_id):
            pass

        mock_pubsub.unsubscribe.assert_awaited_once()
        mock_pubsub.aclose.assert_awaited_once()
        mock_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Helper: turn a plain list into an async iterator
# ---------------------------------------------------------------------------


async def aiter_from_list(items):  # type: ignore[no-untyped-def]
    for item in items:
        yield item
