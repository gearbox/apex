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

from src.api.services.email import EmailService
from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

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
        app_name: str = "Apex",
    ) -> None:
        """Initialise the service.

        Args:
            email_service: Provider that actually sends emails.
            app_url: Base URL of the frontend app, e.g. ``https://app.apex.ai``.
                     Used to build verification/reset links.
            app_name: Public-facing product name for email branding.
        """
        self._email = email_service
        self._app_url = app_url.rstrip("/")
        self._app_name = app_name

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

        Also revokes all active refresh tokens (forces re-login everywhere).

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
        logger.info(
            "email.password_reset_done",
            user_id=str(user_id),
            revoked_tokens=revoked,
        )

        return user
