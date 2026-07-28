"""Shared push-subscription cleanup for bulk-revocation call sites.

Extracted from AuthService/UserService/EmailVerificationService, which each
called this with identical semantics from their own bulk-revocation methods
(logout_all, change_password, deactivate_account, reset_password,
token_reuse_detected) — see push-cleanup-on-revocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.schemas.ops_events import (
    PLATFORM_PRODUCT_ID,
    OpsEventType,
    PushSubscriptionsCleanupFailedOpsPayload,
)
from src.db.repositories.push_subscription import PushSubscriptionRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.ops_event_bus import OpsEventBus

logger = structlog.get_logger(__name__)


async def delete_user_push_subscriptions(
    session: AsyncSession,
    ops_event_bus: OpsEventBus,
    *,
    user_id: UUID,
    op: str,
    source: str,
) -> None:
    """Delete every push subscription for the user (push-cleanup-on-revocation).

    Runs alongside the bulk access-token revocation, not inside
    TokenRevocationService (D3) — push deletion is a plain DB operation
    with its own failure semantics, independent of Redis availability.

    Wrapped in a SAVEPOINT so a failure here rolls back only the delete,
    never poisoning the caller's outer transaction: logout-all/password
    change/reset/deactivation must still commit even if this fails (D4).
    Never raises. Logs the outcome truthfully and publishes an ops event
    only on failure — mirrors _report_revocation_outcome's pattern, whose
    unconditional-success-logging mistake this must not repeat.

    Args:
        session: DB session to run the delete against (caller-owned).
        ops_event_bus: Publishes the failure ops event.
        user_id: User whose subscriptions are being deleted.
        op: Triggering action, e.g. "logout_all", "change_password",
            "deactivate_account", "reset_password", "token_reuse_detected".
        source: Log-event-name prefix of the calling service (e.g. "auth",
            "user", "email") — preserves each call site's existing log keys.
    """
    deleted_event = f"{source}.push_subscriptions_deleted"
    failed_event = f"{source}.push_subscriptions_cleanup_failed"
    try:
        async with session.begin_nested():
            deleted = await PushSubscriptionRepository(session).delete_all_for_user(user_id)
    except Exception:
        logger.exception(failed_event, user_id=str(user_id), op=op)
        await ops_event_bus.publish(
            event_type=OpsEventType.PUSH_SUBSCRIPTIONS_CLEANUP_FAILED,
            product_id=PLATFORM_PRODUCT_ID,
            payload=PushSubscriptionsCleanupFailedOpsPayload(user_id=user_id, op=op),
        )
        return
    logger.info(deleted_event, user_id=str(user_id), op=op, deleted=deleted)
