"""Unit tests for TelegramDispatcher: recipient resolution, filtering, throttling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import msgspec

from src.api.schemas.ops_events import (
    GenerationCreatedOpsPayload,
    HealthTransitionOpsPayload,
    OpsEventEnvelope,
    OpsEventType,
    UserRegisteredOpsPayload,
)
from src.db.repositories.admin_notifications import RecipientRow
from src.workers.telegram_dispatcher import TelegramDispatcher

_encoder = msgspec.json.Encoder()


def _raw_envelope(event_type: OpsEventType, product_id: str, payload: object) -> str:
    envelope = OpsEventEnvelope(
        event_type=event_type,
        product_id=product_id,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )
    return _encoder.encode(envelope).decode()


class _FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.fail_for_chat_id: int | None = None

    async def send_message(self, *, chat_id: int, text: str) -> None:
        if chat_id == self.fail_for_chat_id:
            raise RuntimeError("simulated send failure")
        self.sent.append((chat_id, text))

    async def get_me(self) -> str:
        return "test_bot"

    async def get_updates(self, *, offset: int | None, timeout_seconds: int) -> list[object]:  # noqa: ARG002
        return []


def _make_dispatcher(sender: _FakeSender) -> TelegramDispatcher:
    @asynccontextmanager
    async def _session_factory():  # type: ignore[no-untyped-def]
        yield AsyncMock()

    return TelegramDispatcher(
        sender=sender,  # type: ignore[arg-type]
        session_factory=_session_factory,  # type: ignore[arg-type]
        redis_enabled=False,
    )


async def test_decode_failure_is_skipped() -> None:
    dispatcher = _make_dispatcher(_FakeSender())
    await dispatcher._handle_raw_message("not valid json")  # should not raise


async def test_unknown_event_type_is_skipped_without_recipient_lookup() -> None:
    dispatcher = _make_dispatcher(_FakeSender())
    data = _raw_envelope(
        OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=uuid4())
    )

    # Corrupt the encoded event_type post-hoc to something unrecognized.
    data = data.replace("ops.user.registered", "ops.future.unknown")

    with patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls:
        await dispatcher._handle_raw_message(data)
        mock_repo_cls.assert_not_called()


async def test_product_scoped_event_not_delivered_cross_product() -> None:
    sender = _FakeSender()
    dispatcher = _make_dispatcher(sender)
    data = _raw_envelope(
        OpsEventType.GENERATION_CREATED,
        "vex",
        GenerationCreatedOpsPayload(
            job_id=uuid4(), user_id=uuid4(), provider="grok", generation_type="t2i"
        ),
    )
    synthara_admin = RecipientRow(
        user_id=uuid4(), product_id="synthara", chat_id=555, min_interval_seconds=0
    )

    with patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.list_recipients_for_class = AsyncMock(return_value=[synthara_admin])
        mock_repo_cls.return_value = mock_repo

        await dispatcher._handle_raw_message(data)

    assert sender.sent == []


async def test_health_event_delivered_cross_product() -> None:
    sender = _FakeSender()
    dispatcher = _make_dispatcher(sender)
    data = _raw_envelope(
        OpsEventType.HEALTH_SUBSYSTEM_DEGRADED,
        "platform",
        HealthTransitionOpsPayload(
            subsystem="redis",
            previous_status="healthy",
            current_status="degraded",
            overall_status="degraded",
        ),
    )
    vex_admin = RecipientRow(user_id=uuid4(), product_id="vex", chat_id=777, min_interval_seconds=0)

    with patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.list_recipients_for_class = AsyncMock(return_value=[vex_admin])
        mock_repo_cls.return_value = mock_repo

        await dispatcher._handle_raw_message(data)

    assert len(sender.sent) == 1
    assert sender.sent[0][0] == 777


async def test_send_failure_for_one_recipient_does_not_block_next() -> None:
    sender = _FakeSender()
    sender.fail_for_chat_id = 1
    dispatcher = _make_dispatcher(sender)
    data = _raw_envelope(
        OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=uuid4())
    )
    recipients = [
        RecipientRow(user_id=uuid4(), product_id="vex", chat_id=1, min_interval_seconds=0),
        RecipientRow(user_id=uuid4(), product_id="vex", chat_id=2, min_interval_seconds=0),
    ]

    with patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.list_recipients_for_class = AsyncMock(return_value=recipients)
        mock_repo_cls.return_value = mock_repo

        await dispatcher._handle_raw_message(data)

    assert [chat_id for chat_id, _ in sender.sent] == [2]


class TestThrottle:
    async def test_second_send_within_window_is_suppressed(self) -> None:
        sender = _FakeSender()
        dispatcher = _make_dispatcher(sender)
        user_id = uuid4()

        with patch("time.monotonic", return_value=1000.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=60,
                text="first",
                event_id="e1",
            )
        with patch("time.monotonic", return_value=1010.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=60,
                text="second",
                event_id="e2",
            )

        assert len(sender.sent) == 1
        assert sender.sent[0][1] == "first"

    async def test_third_after_window_carries_suppressed_count(self) -> None:
        sender = _FakeSender()
        dispatcher = _make_dispatcher(sender)
        user_id = uuid4()

        with patch("time.monotonic", return_value=1000.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=60,
                text="first",
                event_id="e1",
            )
        with patch("time.monotonic", return_value=1010.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=60,
                text="second (suppressed)",
                event_id="e2",
            )
        with patch("time.monotonic", return_value=1070.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=60,
                text="third",
                event_id="e3",
            )

        assert len(sender.sent) == 2
        assert sender.sent[1][1] == "third\n<i>(+1 suppressed)</i>"

    async def test_zero_interval_never_throttles(self) -> None:
        sender = _FakeSender()
        dispatcher = _make_dispatcher(sender)
        user_id = uuid4()

        with patch("time.monotonic", return_value=1000.0):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=0,
                text="first",
                event_id="e1",
            )
        with patch("time.monotonic", return_value=1000.1):
            await dispatcher._send_throttled(
                user_id=user_id,
                chat_id=42,
                notification_class="user.registered",
                min_interval_seconds=0,
                text="second",
                event_id="e2",
            )

        assert len(sender.sent) == 2
