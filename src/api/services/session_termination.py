"""Terminate every session of one user — the shared "revoke all" sequence.

Refresh tokens, live access/content tokens (Redis epoch) and Web Push
subscriptions are all credentials a user holds on some device. Every path that
must end all of them — logout-all, password change / reset, account
deactivation, refresh-token reuse detection, claiming an unverified account on
OAuth sign-in — uses this instead of repeating the revoke / report /
push-cleanup sequence.

Never commits — the caller owns the transaction. Callers terminate *before*
their commit, so if the commit later fails the only effect is that the user
was signed out (fails safe).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.ops_events import (
    PLATFORM_PRODUCT_ID,
    OpsEventType,
    TokenRevocationFailedOpsPayload,
)
from src.api.services.push_cleanup import delete_user_push_subscriptions

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.token_revocation import TokenRevocationService
    from src.db.repositories.user import UserRepository

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionTerminationResult:
    """Outcome of :meth:`SessionTerminationService.terminate_all`."""

    revoked_refresh_tokens: int
    epoch: int | None
    """Redis-clock second written as the revocation epoch; ``None`` when the write
    failed *or* Redis is not configured."""

    @property
    def bulk_access_revoked(self) -> bool:
        """False when the Redis epoch write failed *or* Redis is not configured."""
        return self.epoch is not None


class SessionTerminationService:
    """Terminate every session of one user: refresh tokens, live access/content tokens, push.

    The single implementation of the "revoke everything" sequence, shared by all
    revoke-all paths (``AuthService``, ``UserService``, the password-reset and
    OAuth-claim flows).
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        user_repo: UserRepository,
        token_revocation: TokenRevocationService,
        ops_event_bus: OpsEventBus,
    ) -> None:
        """Initialise the service.

        Args:
            session: Caller-owned DB session (used for the push-subscription delete).
            user_repo: Repository bound to the same session.
            token_revocation: Writes the bulk access-token revocation epoch.
            ops_event_bus: Publishes an alert when the epoch write fails.
        """
        self._session = session
        self._users = user_repo
        self._token_revocation = token_revocation
        self._ops_event_bus = ops_event_bus

    async def terminate_all(
        self, user_id: UUID, *, op: str, source: str
    ) -> SessionTerminationResult:
        """Revoke every refresh token, access/content token and push subscription of a user.

        A failed Redis epoch write never raises — blocking the caller's primary
        action on a cache outage is worse than the bounded exposure of a live
        access token (F5). The outcome is reported truthfully instead.

        Args:
            user_id: User whose sessions to terminate.
            op: Triggering action, e.g. ``"reset_password"`` (ops events, logs).
            source: Log-event-name prefix of the caller, e.g. ``"email"``, ``"oauth"``.

        Returns:
            How many refresh tokens were revoked and the epoch written for the
            bulk access-token revocation (``None`` if it did not land).
        """
        # Issue #142 G1: revoke_all_refresh_tokens takes the user-row lock itself
        # (user row first, then refresh-token rows), so it cannot be forgotten.
        revoked = await self._users.revoke_all_refresh_tokens(user_id)
        epoch = await self._token_revocation.revoke_user_sessions(user_id)
        await self._report_revocation_outcome(
            bulk_access_revoked=epoch is not None, user_id=user_id, op=op, source=source
        )
        await delete_user_push_subscriptions(
            self._session, self._ops_event_bus, user_id=user_id, op=op, source=source
        )
        return SessionTerminationResult(revoked_refresh_tokens=revoked, epoch=epoch)

    async def _report_revocation_outcome(
        self, *, bulk_access_revoked: bool, user_id: UUID, op: str, source: str
    ) -> None:
        """F5 — surface a failed bulk access-token revocation to operators.

        Only alert-worthy when Redis is actually configured
        (``token_revocation.enabled``) — a failed outcome with Redis unset is
        the documented no-op, already logged once at startup, not a fresh
        degradation.
        """
        if bulk_access_revoked or not self._token_revocation.enabled:
            return
        failed_event = f"{source}.bulk_revocation_failed"
        logger.error(failed_event, user_id=str(user_id), op=op)
        await self._ops_event_bus.publish(
            event_type=OpsEventType.TOKEN_REVOCATION_FAILED,
            product_id=PLATFORM_PRODUCT_ID,
            payload=TokenRevocationFailedOpsPayload(user_id=user_id, op=op),
        )


SessionTerminationFactory = Callable[["AsyncSession", "UserRepository"], SessionTerminationService]
"""Builds a session-bound :class:`SessionTerminationService` for a caller-owned session.

Process-wide singletons (which cannot hold a request-scoped instance) take this
instead of constructing the service inline, so the dependency is visible in their
constructor and substitutable in tests.
"""


def make_session_termination_factory(
    *, token_revocation: TokenRevocationService, ops_event_bus: OpsEventBus
) -> SessionTerminationFactory:
    """Build the factory closed over the shared revocation service and ops bus.

    The single construction rule for :class:`SessionTerminationService`.
    """

    def factory(session: AsyncSession, user_repo: UserRepository) -> SessionTerminationService:
        return SessionTerminationService(
            session=session,
            user_repo=user_repo,
            token_revocation=token_revocation,
            ops_event_bus=ops_event_bus,
        )

    return factory
