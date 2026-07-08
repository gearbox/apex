"""Unit tests for the Web Push dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock

import msgspec
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.events import EventEnvelope, EventType, SystemNotificationPayload
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


def _system_notification_data() -> str:
    payload = SystemNotificationPayload(
        level=NotificationLevel.WARNING,
        title="Scheduled maintenance",
        message="The service will be down for 10 minutes.",
    )
    envelope = EventEnvelope(
        event_type=EventType.SYSTEM_NOTIFICATION,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )
    return _encoder.encode(envelope).decode()


async def test_handle_raw_message_commits_dispatch_session() -> None:
    session = FakeSession()
    push_service = AsyncMock()
    dispatcher = PushDispatcher(
        push_service=push_service,
        session_factory=cast("Callable[[], AsyncSession]", lambda: session),
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
