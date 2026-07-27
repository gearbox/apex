"""Email verification and password reset service.

Extracted from AuthService to keep that class focused on JWT token operations.
This service owns the full lifecycle of time-limited auth tokens:
  - Generating tokens and sending emails
  - Consuming tokens and applying side effects (mark verified, update password)

Depends on:
  - ``UserRepository`` — read/write user records
  - ``AuthTokenRepository`` — manage token hashes
  - ``EmailService`` — send the actual emails
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.schemas.ops_events import (
    PLATFORM_PRODUCT_ID,
    OpsEventType,
    PushSubscriptionsCleanupFailedOpsPayload,
    TokenRevocationFailedOpsPayload,
)
from src.api.services.ops_event_bus import OpsEventBus
from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.push_subscription import PushSubscriptionRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.email import EmailService
    from src.api.services.token_revocation import TokenRevocationService
    from src.db.models.user import User

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidTokenError(Exception):
    """Token is missing, expired, or already used."""


class UserNotFoundError(Exception):
    """Referenced user does not exist."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EmailVerificationService:
    """Handles email verification and password reset token lifecycles.

    Stateless — all state lives in the DB session passed per method.
    """

    def __init__(
        self,
        *,
        email_service: EmailService,
        app_url: str,
        token_revocation_service: TokenRevocationService,
        app_name: str = "Apex",
        ops_event_bus: OpsEventBus | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            email_service: Provider that actually sends emails.
            app_url: Base URL of the frontend app, e.g. ``https://app.apex.ai``.
                     Used to build verification/reset links.
            token_revocation_service: Bulk-revokes access tokens/content
                cookies on password reset (see
                src.api.services.token_revocation) — the account-recovery
                path a user reaches because they believe their account is
                compromised, so it must invalidate live credentials, not
                just refresh tokens. Required — callers that intentionally
                want revocation to no-op (tests, older call sites) must pass
                an explicit
                ``TokenRevocationService(None, max_token_ttl_seconds=0)`` so
                the choice is visible rather than a silent default (issue
                #142 A1).
            app_name: Public-facing product name for email branding.
            ops_event_bus: Publishes an alert when a bulk access-token
                revocation write fails against a configured Redis (issue
                #142 F5). Defaults to a disabled bus so callers that don't
                wire one (tests, older call sites) simply skip publishing.
        """
        self._email = email_service
        self._app_url = app_url.rstrip("/")
        self._app_name = app_name
        self._token_revocation = token_revocation_service
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )

    # -------------------------------------------------------------------------
    # Email verification
    # -------------------------------------------------------------------------

    async def send_verification_email(
        self,
        user_id: UUID,
        *,
        session: AsyncSession,
    ) -> None:
        """Generate a verification token and email it to the user.

        Invalidates any previous active verification tokens first.
        Safe to call multiple times (resend flow).

        Args:
            user_id: User to verify.
            session: DB session (caller commits).

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        user_repo = UserRepository(session)
        user = await user_repo.get_active_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        token_repo = AuthTokenRepository(session)
        raw_token = await token_repo.create_verification_token(user_id)

        verification_url = f"{self._app_url}/verify-email?token={raw_token}"

        await self._email.send_verification_email(
            to=user.email,
            display_name=user.display_name,
            verification_url=verification_url,
            locale=user.locale,
            app_name=self._app_name,
        )

        logger.info("email.verification_sent", user_id=str(user_id))

    async def verify_email(
        self,
        raw_token: str,
        *,
        session: AsyncSession,
    ) -> User:
        """Consume a verification token and mark the user's email as verified.

        Args:
            raw_token: Token from the verification URL query parameter.
            session: DB session (caller commits).

        Returns:
            The updated User object.

        Raises:
            InvalidTokenError: If the token is invalid, expired, or already used.
        """
        token_repo = AuthTokenRepository(session)
        user_id = await token_repo.consume_verification_token(raw_token)

        if user_id is None:
            raise InvalidTokenError("Invalid or expired verification token")

        user_repo = UserRepository(session)
        user = await user_repo.mark_email_verified(user_id)

        if user is None:
            raise UserNotFoundError(f"User {user_id} not found after token verification")

        logger.info("email.verified", user_id=str(user_id))
        return user

    # -------------------------------------------------------------------------
    # Password reset
    # -------------------------------------------------------------------------

    async def send_password_reset_email(
        self,
        email: str,
        *,
        session: AsyncSession,
        ip_address: str | None = None,
    ) -> None:
        """Send a password reset email if the address is registered.

        **Always returns successfully**, even if the email is not found.
        This prevents email enumeration attacks — callers should return 200
        regardless of whether the address exists.

        Args:
            email: Recipient email address.
            session: DB session (caller commits).
            ip_address: Originating IP for audit trail.
        """
        user_repo = UserRepository(session)
        user = await user_repo.get_active_user_by_email(email)

        if user is None:
            # Silent — do not reveal that the address is unknown
            logger.info("email.password_reset_requested")
            return

        token_repo = AuthTokenRepository(session)
        raw_token = await token_repo.create_reset_token(user.id, ip_address=ip_address)

        reset_url = f"{self._app_url}/reset-password?token={raw_token}"

        await self._email.send_password_reset_email(
            to=user.email,
            display_name=user.display_name,
            reset_url=reset_url,
            locale=user.locale,
            app_name=self._app_name,
        )

        logger.info(
            "email.password_reset_sent",
            user_id=str(user.id),
            email=user.email,
            ip=ip_address,
        )

    async def reset_password(
        self,
        raw_token: str,
        new_password: str,
        *,
        session: AsyncSession,
    ) -> User:
        """Consume a reset token and update the user's password.

        Also revokes all active refresh tokens and all live access
        tokens/content cookies (issue #142) — forces re-authentication
        everywhere. This is the account-recovery path a user reaches
        *because* they believe their account is compromised, so an
        attacker holding a stolen access token or content cookie must not
        keep it for its remaining lifetime after the reset.

        Args:
            raw_token: Token from the reset URL query parameter.
            new_password: New plain-text password (hashed here).
            session: DB session (caller commits).

        Returns:
            The updated User object.

        Raises:
            InvalidTokenError: If the token is invalid, expired, or already used.
            UserNotFoundError: If the referenced user no longer exists.
        """
        from src.api.security import PasswordService

        token_repo = AuthTokenRepository(session)
        user_id = await token_repo.consume_reset_token(raw_token)

        if user_id is None:
            raise InvalidTokenError("Invalid or expired password reset token")

        password_service = PasswordService()
        new_hash = await password_service.ahash(new_password)

        user_repo = UserRepository(session)
        user = await user_repo.update_user(user_id, password_hash=new_hash)

        if user is None:
            raise UserNotFoundError(f"User {user_id} not found after token consumption")

        # Revoke all refresh tokens — forces re-authentication on all devices
        revoked = await user_repo.revoke_all_refresh_tokens(user_id)
        # Bulk-revoke live access tokens/content cookies too (issue #142) —
        # otherwise a stolen access token or content cookie survives a
        # password reset for its full remaining lifetime. This must never
        # block the password reset itself completing (F5) — blocking
        # account recovery on a cache outage is a worse failure than the
        # bounded exposure of a live access token.
        epoch = await self._token_revocation.revoke_user_sessions(user_id)
        bulk_access_revoked = epoch is not None
        await self._report_revocation_outcome(
            bulk_access_revoked=bulk_access_revoked, user_id=user_id, op="reset_password"
        )
        await self._delete_push_subscriptions(user_id, session=session, op="reset_password")
        logger.info(
            "email.password_reset_done",
            user_id=str(user_id),
            revoked_tokens=revoked,
            bulk_access_revoked=bulk_access_revoked,
        )

        return user

    async def _report_revocation_outcome(
        self, *, bulk_access_revoked: bool, user_id: UUID, op: str
    ) -> None:
        """F5 — surface a failed bulk access-token revocation to operators.

        Only alert-worthy when Redis is actually configured
        (`token_revocation.enabled`) — a failed outcome with Redis unset is
        the documented no-op, already logged once at startup, not a fresh
        degradation.
        """
        if bulk_access_revoked or not self._token_revocation.enabled:
            return
        logger.error("email.bulk_revocation_failed", user_id=str(user_id), op=op)
        await self._ops_event_bus.publish(
            event_type=OpsEventType.TOKEN_REVOCATION_FAILED,
            product_id=PLATFORM_PRODUCT_ID,
            payload=TokenRevocationFailedOpsPayload(user_id=user_id, op=op),
        )

    async def _delete_push_subscriptions(
        self, user_id: UUID, *, session: AsyncSession, op: str
    ) -> None:
        """Delete every push subscription for the user (push-cleanup-on-revocation).

        Runs alongside the bulk access-token revocation above, not inside
        TokenRevocationService (D3) — push deletion is a plain DB operation
        with its own failure semantics, independent of Redis availability.

        Wrapped in a SAVEPOINT so a failure here rolls back only the delete,
        never poisoning the caller's outer transaction: the password reset
        must still commit even if this fails (D4). Never raises. Logs the
        outcome truthfully and publishes an ops event only on failure —
        mirrors _report_revocation_outcome above, whose
        unconditional-success-logging mistake this must not repeat.
        """
        try:
            async with session.begin_nested():
                deleted = await PushSubscriptionRepository(session).delete_all_for_user(user_id)
        except Exception:
            logger.exception("email.push_subscriptions_cleanup_failed", user_id=str(user_id), op=op)
            await self._ops_event_bus.publish(
                event_type=OpsEventType.PUSH_SUBSCRIPTIONS_CLEANUP_FAILED,
                product_id=PLATFORM_PRODUCT_ID,
                payload=PushSubscriptionsCleanupFailedOpsPayload(user_id=user_id, op=op),
            )
            return
        logger.info(
            "email.push_subscriptions_deleted", user_id=str(user_id), op=op, deleted=deleted
        )
