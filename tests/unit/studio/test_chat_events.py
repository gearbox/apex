"""``StudioChatEvents``: the owner's channel, ids and a status, and never an exception."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from structlog.testing import capture_logs

from src.api.schemas.events import EventType
from src.studio.services.chat_events import StudioChatEvents

pytestmark = pytest.mark.unit

USER, CHAT, MESSAGE, DRAFT, SOURCE = (UUID(int=i) for i in range(1, 6))
SECRET = "THE-ARTISTS-WORDS-IN-A-DELTA"


class _Bus:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.published: list[tuple[UUID, EventType, Any]] = []
        self.balances: list[Any] = []

    async def publish(self, *, user_id: UUID, event_type: EventType, payload: object) -> None:
        if self.fail:
            raise ConnectionError("redis is down")
        self.published.append((user_id, event_type, payload))

    async def publish_balance(self, event: object) -> None:
        self.balances.append(event)


def _message() -> Any:
    return SimpleNamespace(id=MESSAGE, user_id=USER, chat_id=CHAT, status="complete", revision=2)


def _draft() -> Any:
    return SimpleNamespace(
        id=DRAFT, user_id=USER, chat_id=CHAT, source_message_id=SOURCE, status="pending", revision=1
    )


async def test_each_event_goes_to_the_owner_with_its_documented_payload() -> None:
    bus = _Bus()
    events = StudioChatEvents(bus)  # type: ignore[arg-type]
    await events.delta(user_id=USER, chat_id=CHAT, message_id=MESSAGE, seq=3, text="hi")
    await events.message_updated(_message())
    await events.draft_updated(_draft())

    assert [(user, kind) for user, kind, _ in bus.published] == [
        (USER, EventType.STUDIO_CHAT_DELTA),
        (USER, EventType.STUDIO_CHAT_MESSAGE_UPDATED),
        (USER, EventType.STUDIO_MEMORY_DRAFT_UPDATED),
    ]
    delta, updated, drafted = (payload for _, _, payload in bus.published)
    assert (delta.chat_id, delta.message_id, delta.seq, delta.text) == (CHAT, MESSAGE, 3, "hi")
    assert (updated.message_id, updated.status, updated.revision) == (MESSAGE, "complete", 2)
    assert (drafted.draft_id, drafted.source_message_id, drafted.status) == (
        DRAFT,
        SOURCE,
        "pending",
    )


async def test_a_bus_that_fails_never_raises_and_the_log_has_ids_only() -> None:
    events = StudioChatEvents(_Bus(fail=True))  # type: ignore[arg-type]
    with capture_logs() as logs:
        await events.delta(user_id=USER, chat_id=CHAT, message_id=MESSAGE, seq=1, text=SECRET)
        await events.message_updated(_message())
        await events.draft_updated(_draft())

    assert [log["event"] for log in logs] == ["studio.chat.event_publish_failed"] * 3
    assert SECRET not in str(logs)
    assert {log.get("message_id") or log.get("draft_id") for log in logs} == {
        str(MESSAGE),
        str(DRAFT),
    }


async def test_without_a_bus_every_method_is_a_no_op() -> None:
    events = StudioChatEvents(None)
    await events.delta(user_id=USER, chat_id=CHAT, message_id=MESSAGE, seq=1, text="x")
    await events.message_updated(_message())
    await events.draft_updated(_draft())
    await events.balance(object())  # type: ignore[arg-type]


async def test_a_balance_event_is_handed_to_the_bus() -> None:
    bus = _Bus()
    marker = object()
    await StudioChatEvents(bus).balance(marker)  # type: ignore[arg-type]
    assert bus.balances == [marker]
