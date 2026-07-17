"""Unit tests for TelegramDispatcher: recipient resolution, filtering, throttling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
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


async def test_unmapped_event_is_skipped_without_recipient_lookup() -> None:
    """A validly-decoded envelope whose event_type has no notification mapping is skipped.

    Distinct from the malformed-payload decode-failure case: here decode
    succeeds and map_ops_event legitimately returns None (module docstring:
    "e.g. an unknown event_type during a rolling deploy").
    """
    dispatcher = _make_dispatcher(_FakeSender())
    data = _raw_envelope(
        OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=uuid4())
    )

    with (
        patch("src.workers.telegram_dispatcher.map_ops_event", return_value=None),
        patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls,
    ):
        await dispatcher._handle_raw_message(data)
        mock_repo_cls.assert_not_called()


async def test_recipient_lookup_failure_is_logged_and_does_not_raise() -> None:
    sender = _FakeSender()
    dispatcher = _make_dispatcher(sender)
    data = _raw_envelope(
        OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=uuid4())
    )

    with patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls:
        mock_repo = AsyncMock()
        mock_repo.list_recipients_for_class = AsyncMock(side_effect=RuntimeError("db down"))
        mock_repo_cls.return_value = mock_repo

        await dispatcher._handle_raw_message(data)  # must not raise

    assert sender.sent == []


class TestLifecycle:
    async def test_start_is_idempotent_and_stop_releases_lease(self) -> None:
        dispatcher = _make_dispatcher(_FakeSender())
        dispatcher._run_loop = AsyncMock()  # type: ignore[method-assign]
        dispatcher._lease.release = AsyncMock()  # type: ignore[method-assign]

        await dispatcher.start()
        first_task = dispatcher._task
        await dispatcher.start()  # idempotent — no second task created

        assert dispatcher.is_running is True
        assert dispatcher._task is first_task

        await dispatcher.stop()

        assert dispatcher.is_running is False
        assert dispatcher._task is None
        dispatcher._lease.release.assert_awaited_once()

    async def test_stop_is_noop_when_not_running(self) -> None:
        dispatcher = _make_dispatcher(_FakeSender())
        dispatcher._lease.release = AsyncMock()  # type: ignore[method-assign]

        await dispatcher.stop()

        dispatcher._lease.release.assert_not_awaited()

    async def test_interruptible_sleep_times_out_without_raising(self) -> None:
        dispatcher = _make_dispatcher(_FakeSender())
        await dispatcher._interruptible_sleep(0)

    async def test_run_loop_backs_off_when_not_leader(self) -> None:
        dispatcher = _make_dispatcher(_FakeSender())
        dispatcher._running = True
        dispatcher._lease.acquire_or_renew = AsyncMock(return_value=False)  # type: ignore[method-assign]
        dispatcher._interruptible_sleep = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda *_: setattr(dispatcher, "_running", False)
        )

        await dispatcher._run_loop()

        dispatcher._interruptible_sleep.assert_awaited_once()

    async def test_run_loop_backs_off_on_listen_error(self) -> None:
        dispatcher = _make_dispatcher(_FakeSender())
        dispatcher._running = True
        dispatcher._lease.acquire_or_renew = AsyncMock(return_value=True)  # type: ignore[method-assign]
        dispatcher._listen_while_leader = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("boom")
        )
        dispatcher._interruptible_sleep = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda *_: setattr(dispatcher, "_running", False)
        )

        await dispatcher._run_loop()

        dispatcher._interruptible_sleep.assert_awaited_once()


class TestLeaseRenewal:
    async def test_renewed_every_iteration_and_stops_on_lost_leadership(self) -> None:
        """F2: renewal must happen every loop iteration, not only when idle.

        A continuous message stream (no idle window) with acquire_or_renew
        called only in the `message is None` branch would never renew and
        the lease would expire under sustained traffic. This asserts
        renewal happens once per delivered message, and that losing
        leadership mid-stream stops processing immediately.
        """
        sender = _FakeSender()
        dispatcher = _make_dispatcher(sender)

        data = _raw_envelope(
            OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=uuid4())
        )
        recipient = RecipientRow(
            user_id=uuid4(), product_id="vex", chat_id=99, min_interval_seconds=0
        )

        fake_pubsub = AsyncMock()
        fake_pubsub.get_message = AsyncMock(return_value={"type": "message", "data": data})

        fake_client = MagicMock()
        fake_client.pubsub = MagicMock(return_value=fake_pubsub)

        dispatcher._lease.acquire_or_renew = AsyncMock(  # type: ignore[method-assign]
            side_effect=[True, True, True, False]
        )
        dispatcher._running = True

        with (
            patch("src.workers.telegram_dispatcher.get_redis_client", return_value=fake_client),
            patch("src.workers.telegram_dispatcher.AdminNotificationRepository") as mock_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo.list_recipients_for_class = AsyncMock(return_value=[recipient])
            mock_repo_cls.return_value = mock_repo

            await dispatcher._listen_while_leader()

        # One renewal per delivered message, plus the renewal that reports
        # lost leadership and ends the loop.
        assert dispatcher._lease.acquire_or_renew.await_count == 4
        # No message was processed after leadership was lost.
        assert len(sender.sent) == 3
        assert fake_pubsub.get_message.await_count == 3


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
