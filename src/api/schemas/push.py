"""Web Push API schemas — subscription CRUD requests/responses and the wire payload."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec


class VapidPublicKeyResponse(msgspec.Struct, kw_only=True):
    public_key: str


class PushSubscriptionKeys(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Mirrors the browser PushSubscription.toJSON().keys shape."""

    p256dh: str
    auth: str


class PushSubscriptionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    endpoint: str
    keys: PushSubscriptionKeys
    user_agent: str | None = None


class PushSubscriptionResponse(msgspec.Struct, kw_only=True):
    id: UUID
    endpoint: str
    created_at: datetime


class PushSubscriptionDeleteRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    endpoint: str


class PushNotificationPayload(msgspec.Struct, kw_only=True):
    """Wire payload delivered as the push message body.

    The frontend service worker consumes exactly this JSON shape — do not
    add, rename, or remove fields without a coordinated frontend change.
    """

    title: str
    body: str
    url: str
    tag: str
    category: str  # "job" | "gpu_credit" | "system" | "balance"
    level: str  # "info" | "warning" | "critical"
