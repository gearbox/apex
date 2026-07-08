"""Web Push subscription management and delivery.

Layering: PushSubscriptionRepository (DB only) -> PushService (this module —
upsert/delete/send, stateless like BillingService) -> PushDispatcher (event
mapping + fan-out trigger, src/workers/push_dispatcher.py).

The actual HTTP delivery to a push service (FCM, Mozilla autopush, ...) is
behind the ``WebPushSender`` protocol so tests can inject a fake sender
without touching the network or pywebpush.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Protocol

import msgspec
import structlog
from pywebpush import WebPushException, webpush

from src.db.repositories.push_subscription import PushSubscriptionRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.push import PushNotificationPayload
    from src.db.models.push_subscription import PushSubscription

logger = structlog.get_logger(__name__)

_DEFAULT_BATCH_SIZE = 100


class PushSendError(Exception):
    """Generic push delivery failure (network error, non-2xx other than 404/410)."""


class PushSubscriptionExpiredError(PushSendError):
    """The push service reported the subscription no longer exists (HTTP 404/410)."""


class WebPushSender(Protocol):
    """Delivers one Web Push message. Implementations own their own transport."""

    async def send(
        self,
        *,
        endpoint: str,
        p256dh: str,
        auth: str,
        payload: dict[str, Any],
    ) -> None:
        """Send payload to one subscription.

        Raises:
            PushSubscriptionExpiredError: The push service reports the
                subscription is gone (HTTP 404/410) — caller should prune it.
            PushSendError: Any other delivery failure.
        """
        ...


class PywebpushSender:
    """``WebPushSender`` backed by the synchronous ``pywebpush`` library.

    pywebpush is requests-based (synchronous), so each send is dispatched via
    ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, *, private_key: str, subject: str) -> None:
        self._private_key = private_key
        self._subject = subject

    async def send(
        self,
        *,
        endpoint: str,
        p256dh: str,
        auth: str,
        payload: dict[str, Any],
    ) -> None:
        await asyncio.to_thread(self._send_sync, endpoint, p256dh, auth, payload)

    def _send_sync(
        self,
        endpoint: str,
        p256dh: str,
        auth: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            webpush(
                subscription_info={
                    "endpoint": endpoint,
                    "keys": {"p256dh": p256dh, "auth": auth},
                },
                data=json.dumps(payload),
                vapid_private_key=self._private_key,
                # webpush() mutates this dict (adds 'aud'/'exp') — pass a
                # fresh copy every call so calls never cross-pollute.
                vapid_claims={"sub": self._subject},
            )
        except WebPushException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 410):
                raise PushSubscriptionExpiredError(str(exc)) from exc
            raise PushSendError(str(exc)) from exc


class PushService:
    """Upsert/delete subscriptions and fan out notifications. Stateless — takes ``session`` per call."""

    def __init__(self, *, sender: WebPushSender, broadcast_concurrency: int = 10) -> None:
        self._sender = sender
        self._broadcast_concurrency = broadcast_concurrency

    async def upsert_subscription(
        self,
        *,
        user_id: UUID,
        product_id: str,
        endpoint: str,
        p256dh: str,
        auth: str,
        user_agent: str | None,
        session: AsyncSession,
    ) -> PushSubscription:
        repo = PushSubscriptionRepository(session)
        return await repo.upsert(
            user_id=user_id,
            product_id=product_id,
            endpoint=endpoint,
            p256dh=p256dh,
            auth=auth,
            user_agent=user_agent,
        )

    async def delete_subscription(
        self,
        *,
        user_id: UUID,
        endpoint: str,
        session: AsyncSession,
    ) -> None:
        """Idempotent delete — no error if the endpoint doesn't exist or isn't owned by user_id."""
        repo = PushSubscriptionRepository(session)
        await repo.delete_by_endpoint(endpoint, user_id=user_id)

    async def send_to_user(
        self,
        user_id: UUID,
        payload: PushNotificationPayload,
        *,
        session: AsyncSession,
    ) -> None:
        repo = PushSubscriptionRepository(session)
        subscriptions = await repo.list_by_user(user_id)
        await self._send_to_many(subscriptions, payload, session=session)

    async def send_broadcast(
        self,
        payload: PushNotificationPayload,
        *,
        session: AsyncSession,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Fan out to every subscription, in keyset-paginated batches."""
        repo = PushSubscriptionRepository(session)
        cursor_id: UUID | None = None
        while True:
            batch = await repo.list_batch(limit=batch_size, cursor_id=cursor_id)
            if not batch:
                return
            await self._send_to_many(batch, payload, session=session)
            cursor_id = batch[-1].id
            if len(batch) < batch_size:
                return

    async def _send_to_many(
        self,
        subscriptions: list[PushSubscription] | Any,
        payload: PushNotificationPayload,
        *,
        session: AsyncSession,
    ) -> None:
        if not subscriptions:
            return
        semaphore = asyncio.Semaphore(self._broadcast_concurrency)
        await asyncio.gather(
            *(self._send_one_guarded(sub, payload, session, semaphore) for sub in subscriptions)
        )

    async def _send_one_guarded(
        self,
        subscription: PushSubscription,
        payload: PushNotificationPayload,
        session: AsyncSession,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            await self._send_one(subscription, payload, session=session)

    async def _send_one(
        self,
        subscription: PushSubscription,
        payload: PushNotificationPayload,
        *,
        session: AsyncSession,
    ) -> None:
        try:
            await self._sender.send(
                endpoint=subscription.endpoint,
                p256dh=subscription.p256dh,
                auth=subscription.auth,
                payload=msgspec.to_builtins(payload),
            )
        except PushSubscriptionExpiredError:
            repo = PushSubscriptionRepository(session)
            await repo.delete_by_id(subscription.id)
            logger.info(
                "push.subscription_expired_pruned",
                subscription_id=str(subscription.id),
            )
        except PushSendError:
            logger.info(
                "push.send_failed",
                subscription_id=str(subscription.id),
            )
