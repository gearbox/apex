"""Unit tests for the Web Push dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import msgspec
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.events import (
    EventEnvelope,
    EventType,
    JobProgressPayload,
    SystemNotificationPayload,
)
from src.core.enums import NotificationLevel
from src.workers.push_dispatcher import PushDispatcher

_encoder = msgspec.json.Encoder()


class FakeBegin:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeBegin:
        self._session.begin_entered = True
        return self

    async def __aexit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        self._session.begin_committed = exc_type is None


class FakeSession:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.begin_entered = False
        self.begin_committed = False

    async def __aenter__(self) -> FakeSession:
        self.entered = True
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.exited = True

    def begin(self) -> FakeBegin:
        return FakeBegin(self)


class FakeLease:
    def __init__(self, acquire_results: list[bool] | None = None) -> None:
        self._acquire_results = acquire_results or [True]
        self.released = False

    async def acquire_or_renew(self) -> bool:
        if len(self._acquire_results) == 1:
            return self._acquire_results[0]
        return self._acquire_results.pop(0)

    async def release(self) -> None:
        self.released = True


class FakePubSub:
    def __init__(self, messages: list[dict[str, Any] | None]) -> None:
        self._messages = messages
        self.psubscribed: list[str] = []
        self.subscribed: list[str] = []
        self.punsubscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False

    async def psubscribe(self, pattern: str) -> None:
        self.psubscribed.append(pattern)

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    async def get_message(self, **kwargs: object) -> object:
        assert kwargs["ignore_subscribe_messages"] is True
        assert kwargs["timeout"] == 5.0
        if not self._messages:
            return None
        return self._messages.pop(0)

    async def punsubscribe(self, pattern: str) -> None:
        self.punsubscribed.append(pattern)

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed.append(channel)

    async def aclose(self) -> None:
        self.closed = True


class FakeRedisClient:
    def __init__(self, pubsub: FakePubSub) -> None:
        self._pubsub = pubsub

    def pubsub(self) -> FakePubSub:
        return self._pubsub


def _session_factory(session: FakeSession) -> Callable[[], AsyncSession]:
    return cast("Callable[[], AsyncSession]", lambda: session)


def _event_data(event_type: EventType, payload: object) -> str:
    envelope = EventEnvelope(
        event_type=event_type,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )
    return _encoder.encode(envelope).decode()


def _system_notification_data() -> str:
    return _event_data(
        EventType.SYSTEM_NOTIFICATION,
        SystemNotificationPayload(
            level=NotificationLevel.WARNING,
            title="Scheduled maintenance",
            message="The service will be down for 10 minutes.",
        ),
    )


def _ignored_job_progress_data() -> str:
    return _event_data(
        EventType.JOB_PROGRESS,
        JobProgressPayload(job_id=uuid4(), progress_pct=50, generation_type="t2i"),
    )


async def test_handle_raw_message_commits_dispatch_session() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )

    await dispatcher._handle_raw_message(
        channel="system:broadcast",
        data=_system_notification_data(),
    )

    assert session.entered is True
    assert session.exited is True
    assert session.begin_entered is True
    assert session.begin_committed is True
    push_service.send_broadcast.assert_awaited_once()


async def test_handle_raw_message_dispatches_user_channel() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )
    user_id = uuid4()

    await dispatcher._handle_raw_message(
        channel=f"user:{user_id}",
        data=_system_notification_data(),
    )

    push_service.send_to_user.assert_awaited_once()
    call_args = push_service.send_to_user.call_args
    assert call_args.args[0] == user_id
    assert call_args.kwargs["session"] is session


async def test_handle_raw_message_ignores_unmapped_event() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )

    await dispatcher._handle_raw_message(
        channel="system:broadcast",
        data=_ignored_job_progress_data(),
    )

    assert session.entered is False
    push_service.send_broadcast.assert_not_awaited()
    push_service.send_to_user.assert_not_awaited()


async def test_handle_raw_message_ignores_decode_error() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )

    await dispatcher._handle_raw_message(channel="system:broadcast", data="not-json")

    assert session.entered is False
    push_service.send_broadcast.assert_not_awaited()


async def test_handle_raw_message_logs_send_error_without_raising() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    push_service.send_broadcast.side_effect = RuntimeError("send failed")
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )

    await dispatcher._handle_raw_message(
        channel="system:broadcast",
        data=_system_notification_data(),
    )

    assert session.begin_entered is True
    assert session.begin_committed is False
    push_service.send_broadcast.assert_awaited_once()


async def test_listen_while_leader_handles_messages_and_cleans_up() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=_session_factory(session),
        redis_enabled=False,
    )
    cast("Any", dispatcher)._running = True
    cast("Any", dispatcher)._lease = FakeLease([False])
    pubsub = FakePubSub(
        [
            {"type": "subscribe", "channel": "ignored", "data": "ignored"},
            {
                "type": "message",
                "channel": "system:broadcast",
                "data": _system_notification_data(),
            },
            None,
        ]
    )

    with patch(
        "src.workers.push_dispatcher.get_redis_client", return_value=FakeRedisClient(pubsub)
    ):
        await dispatcher._listen_while_leader()

    assert pubsub.psubscribed == ["user:*"]
    assert pubsub.subscribed == ["system:broadcast"]
    assert pubsub.punsubscribed == ["user:*"]
    assert pubsub.unsubscribed == ["system:broadcast"]
    assert pubsub.closed is True
    push_service.send_broadcast.assert_awaited_once()


async def test_start_is_idempotent_and_stop_releases_lease() -> None:
    session = FakeSession()
    dispatcher = PushDispatcher(
        push_service=AsyncMock(),
        session_factory=_session_factory(session),
        redis_enabled=False,
    )
    fake_lease = FakeLease()
    cast("Any", dispatcher)._lease = fake_lease
    cast("Any", dispatcher)._run_loop = AsyncMock()

    await dispatcher.start()
    first_task = cast("Any", dispatcher)._task
    await dispatcher.start()
    assert dispatcher.is_running is True
    assert cast("Any", dispatcher)._task is first_task

    await dispatcher.stop()

    assert dispatcher.is_running is False
    assert cast("Any", dispatcher)._task is None
    assert fake_lease.released is True


async def test_stop_is_noop_when_not_running() -> None:
    dispatcher = PushDispatcher(
        push_service=AsyncMock(),
        session_factory=_session_factory(FakeSession()),
        redis_enabled=False,
    )
    fake_lease = FakeLease()
    cast("Any", dispatcher)._lease = fake_lease

    await dispatcher.stop()

    assert fake_lease.released is False


async def test_interruptible_sleep_times_out_without_raising() -> None:
    dispatcher = PushDispatcher(
        push_service=AsyncMock(),
        session_factory=_session_factory(FakeSession()),
        redis_enabled=False,
    )

    await dispatcher._interruptible_sleep(0)
