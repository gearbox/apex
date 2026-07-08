"""Unit tests for PushService using a fake WebPushSender (no network, no DB)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pywebpush import WebPushException

from src.api.schemas.push import PushNotificationPayload
from src.api.services.push import (
    PushSendError,
    PushService,
    PushSubscriptionExpiredError,
    PywebpushSender,
)

_REPO_PATH = "src.api.services.push.PushSubscriptionRepository"

_PAYLOAD = PushNotificationPayload(
    title="Generation complete",
    body="Your t2i generation has completed.",
    url="/gallery/abc",
    tag="job-abc",
    category="job",
    level="info",
)


def _response(status_code: int) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    return response


def _make_subscription(
    *, subscription_id: UUID | None = None, endpoint: str | None = None
) -> MagicMock:
    sub = MagicMock()
    sub.id = subscription_id or uuid4()
    sub.endpoint = endpoint or f"https://push.example/{sub.id}"
    sub.p256dh = "p256dh-key"
    sub.auth = "auth-key"
    return sub


class FakeSender:
    """Records every send() call; can be configured to fail per-endpoint."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.expired_endpoints: set[str] = set()
        self.failing_endpoints: set[str] = set()
        self._concurrent = 0
        self.max_concurrent = 0
        self._delay = 0.0

    async def send(self, *, endpoint: str, p256dh: str, auth: str, payload: dict[str, Any]) -> None:
        self._concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self._concurrent)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            self.calls.append(
                {"endpoint": endpoint, "p256dh": p256dh, "auth": auth, "payload": payload}
            )
            if endpoint in self.expired_endpoints:
                raise PushSubscriptionExpiredError("gone")
            if endpoint in self.failing_endpoints:
                raise PushSendError("boom")
        finally:
            self._concurrent -= 1


# ---------------------------------------------------------------------------
# PywebpushSender
# ---------------------------------------------------------------------------


class TestPywebpushSender:
    def test_send_sync_calls_pywebpush_with_subscription_and_vapid_claims(self) -> None:
        sender = PywebpushSender(private_key="private", subject="mailto:ops@example.com")

        with patch("src.api.services.push.webpush") as mock_webpush:
            sender._send_sync(
                endpoint="https://push.example/endpoint",
                p256dh="p256dh",
                auth="auth",
                payload={"title": "Hello"},
            )

        mock_webpush.assert_called_once()
        kwargs = mock_webpush.call_args.kwargs
        assert kwargs["subscription_info"] == {
            "endpoint": "https://push.example/endpoint",
            "keys": {"p256dh": "p256dh", "auth": "auth"},
        }
        assert kwargs["data"] == '{"title": "Hello"}'
        assert kwargs["vapid_private_key"] == "private"
        assert kwargs["vapid_claims"] == {"sub": "mailto:ops@example.com"}

    def test_send_sync_maps_404_410_to_expired_subscription(self) -> None:
        sender = PywebpushSender(private_key="private", subject="mailto:ops@example.com")

        with (
            patch(
                "src.api.services.push.webpush",
                side_effect=WebPushException("gone", response=_response(410)),
            ),
            pytest.raises(PushSubscriptionExpiredError),
        ):
            sender._send_sync(
                endpoint="https://push.example/endpoint",
                p256dh="p256dh",
                auth="auth",
                payload={"title": "Hello"},
            )

    def test_send_sync_maps_other_webpush_errors_to_send_error(self) -> None:
        sender = PywebpushSender(private_key="private", subject="mailto:ops@example.com")

        with (
            patch(
                "src.api.services.push.webpush",
                side_effect=WebPushException("server error", response=_response(500)),
            ),
            pytest.raises(PushSendError),
        ):
            sender._send_sync(
                endpoint="https://push.example/endpoint",
                p256dh="p256dh",
                auth="auth",
                payload={"title": "Hello"},
            )


class TrackingRepo:
    """Fake PushSubscriptionRepository that records concurrent prune calls."""

    def __init__(self, subscriptions: list[MagicMock]) -> None:
        self._subscriptions = subscriptions
        self.deleted_ids: list[UUID] = []
        self._concurrent_deletes = 0
        self.max_concurrent_deletes = 0

    async def list_by_user(self, _user_id: UUID) -> list[MagicMock]:
        return self._subscriptions

    async def delete_by_id(self, subscription_id: UUID) -> None:
        self._concurrent_deletes += 1
        self.max_concurrent_deletes = max(
            self.max_concurrent_deletes,
            self._concurrent_deletes,
        )
        try:
            await asyncio.sleep(0.01)
            self.deleted_ids.append(subscription_id)
        finally:
            self._concurrent_deletes -= 1


# ---------------------------------------------------------------------------
# send_to_user
# ---------------------------------------------------------------------------


class TestSendToUser:
    async def test_sends_to_every_subscription(self) -> None:
        subs = [_make_subscription() for _ in range(3)]
        sender = FakeSender()
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_by_user.return_value = subs
            mock_repo_cls.return_value = mock_repo

            await service.send_to_user(uuid4(), _PAYLOAD, session=MagicMock())

        assert len(sender.calls) == 3
        assert {c["endpoint"] for c in sender.calls} == {s.endpoint for s in subs}

    async def test_prunes_subscription_on_expired_error(self) -> None:
        expired = _make_subscription()
        healthy = _make_subscription()
        sender = FakeSender()
        sender.expired_endpoints.add(expired.endpoint)
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_by_user.return_value = [expired, healthy]
            mock_repo_cls.return_value = mock_repo

            await service.send_to_user(uuid4(), _PAYLOAD, session=MagicMock())

            mock_repo.delete_by_id.assert_awaited_once_with(expired.id)

        # Both sends were attempted despite the pruning.
        assert len(sender.calls) == 2

    async def test_prunes_expired_subscriptions_sequentially_after_concurrent_sends(self) -> None:
        expired = [_make_subscription() for _ in range(4)]
        sender = FakeSender()
        sender._delay = 0.01
        sender.expired_endpoints = {sub.endpoint for sub in expired}
        service = PushService(sender=sender, broadcast_concurrency=4)
        repo = TrackingRepo(expired)

        with patch(_REPO_PATH, return_value=repo):
            await service.send_to_user(uuid4(), _PAYLOAD, session=MagicMock())

        assert sender.max_concurrent > 1
        assert set(repo.deleted_ids) == {sub.id for sub in expired}
        assert repo.max_concurrent_deletes == 1

    async def test_generic_send_error_does_not_prune_and_does_not_raise(self) -> None:
        failing = _make_subscription()
        healthy = _make_subscription()
        sender = FakeSender()
        sender.failing_endpoints.add(failing.endpoint)
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_by_user.return_value = [failing, healthy]
            mock_repo_cls.return_value = mock_repo

            await service.send_to_user(uuid4(), _PAYLOAD, session=MagicMock())

            mock_repo.delete_by_id.assert_not_awaited()

        assert len(sender.calls) == 2

    async def test_no_subscriptions_sends_nothing(self) -> None:
        sender = FakeSender()
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_by_user.return_value = []
            mock_repo_cls.return_value = mock_repo

            await service.send_to_user(uuid4(), _PAYLOAD, session=MagicMock())

        assert sender.calls == []


# ---------------------------------------------------------------------------
# send_broadcast — keyset pagination + concurrency limit
# ---------------------------------------------------------------------------


class TestSendBroadcast:
    async def test_paginates_through_all_batches(self) -> None:
        batch1 = [_make_subscription() for _ in range(3)]
        batch2 = [_make_subscription() for _ in range(2)]
        sender = FakeSender()
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_batch.side_effect = [batch1, batch2]
            mock_repo_cls.return_value = mock_repo

            await service.send_broadcast(_PAYLOAD, session=MagicMock(), batch_size=3)

        assert len(sender.calls) == 5
        # First page fetched with no cursor, second page cursored on last id of page 1.
        first_call_kwargs = mock_repo.list_batch.call_args_list[0].kwargs
        second_call_kwargs = mock_repo.list_batch.call_args_list[1].kwargs
        assert first_call_kwargs["cursor_id"] is None
        assert second_call_kwargs["cursor_id"] == batch1[-1].id

    async def test_stops_when_batch_is_empty(self) -> None:
        sender = FakeSender()
        service = PushService(sender=sender)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_batch.return_value = []
            mock_repo_cls.return_value = mock_repo

            await service.send_broadcast(_PAYLOAD, session=MagicMock())

        assert sender.calls == []
        mock_repo.list_batch.assert_awaited_once()

    async def test_concurrency_is_bounded_by_broadcast_concurrency(self) -> None:
        subs = [_make_subscription() for _ in range(10)]
        sender = FakeSender()
        sender._delay = 0.01
        service = PushService(sender=sender, broadcast_concurrency=2)

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_batch.side_effect = [subs, []]
            mock_repo_cls.return_value = mock_repo

            await service.send_broadcast(_PAYLOAD, session=MagicMock(), batch_size=100)

        assert len(sender.calls) == 10
        assert sender.max_concurrent <= 2


# ---------------------------------------------------------------------------
# upsert / delete delegate to the repository
# ---------------------------------------------------------------------------


class TestSubscriptionCrud:
    async def test_upsert_subscription_delegates_to_repo(self) -> None:
        sender = FakeSender()
        service = PushService(sender=sender)
        user_id = uuid4()

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            expected = _make_subscription()
            mock_repo.upsert.return_value = expected
            mock_repo_cls.return_value = mock_repo

            result = await service.upsert_subscription(
                user_id=user_id,
                product_id="vex",
                endpoint="https://push.example/x",
                p256dh="p",
                auth="a",
                user_agent="ua",
                session=MagicMock(),
            )

            mock_repo.upsert.assert_awaited_once_with(
                user_id=user_id,
                product_id="vex",
                endpoint="https://push.example/x",
                p256dh="p",
                auth="a",
                user_agent="ua",
            )
        assert result is expected

    async def test_delete_subscription_delegates_to_repo(self) -> None:
        sender = FakeSender()
        service = PushService(sender=sender)
        user_id = uuid4()

        with patch(_REPO_PATH) as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await service.delete_subscription(
                user_id=user_id,
                endpoint="https://push.example/x",
                session=MagicMock(),
            )

            mock_repo.delete_by_endpoint.assert_awaited_once_with(
                "https://push.example/x", user_id=user_id
            )
