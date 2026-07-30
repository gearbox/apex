"""Authentication service for user registration and login."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.ops_events import (
    PLATFORM_PRODUCT_ID,
    OpsEventType,
    TokenRevocationFailedOpsPayload,
    UserRegisteredOpsPayload,
)
from src.api.security import (
    JWTService,
    PasswordService,
    generate_token,
    hash_token,
)
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.push_cleanup import delete_user_push_subscriptions
from src.core.enums import RefreshTokenRevocationReason
from src.core.uid import new_id
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.security.jwt import TokenPayload
    from src.api.services.email_verification import EmailVerificationService
    from src.api.services.token_revocation import TokenRevocationService
    from src.db.models import User
    from src.db.repositories import UserRepository

logger = structlog.get_logger(__name__)


class AuthError(Exception):
    """Base authentication error."""


class InvalidCredentialsError(AuthError):
    """Invalid email or password."""


class EmailAlreadyExistsError(AuthError):
    """Email is already registered."""


class UserNotFoundError(AuthError):
    """User not found."""


class UserInactiveError(AuthError):
    """User account is deactivated."""


class InvalidRefreshTokenError(AuthError):
    """Refresh token is invalid, expired, or revoked."""


class TokenReuseDetectedError(AuthError):
    """Potential token theft detected - revoked token was reused."""


@dataclass
class TokenPair:
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    expires_at: datetime
    expires_in: int


def _reuse_detected_message(*, bulk_access_revoked: bool) -> str:
    """TokenReuseDetectedError's user-facing message (issue #142 F5).

    Branches on whether the Redis-backed bulk access-token/content-cookie
    revocation actually landed — the refresh-token family revocation
    (DB-side) always succeeds by the time this is called, so only the
    access-token side can fail (Redis down) or be skipped (Redis not
    configured). Claiming "all sessions invalidated" when that write failed
    would over-promise a security guarantee that wasn't actually met, so the
    message is branched rather than a single fixed string.
    """
    if bulk_access_revoked:
        return (
            "Security alert: This refresh token was already used. "
            "All sessions have been invalidated."
        )
    return (
        "Security alert: This refresh token was already used. Your refresh-token "
        "session has been invalidated, but we could not confirm that other active "
        "access tokens were revoked — change your password immediately if you "
        "suspect compromise."
    )


class AuthService:
    """Authentication service.

    Handles user registration, login, token refresh, and logout.
    Implements secure token rotation with family tracking for
    detecting potential token theft.
    """

    def __init__(
        self,
        repository: UserRepository,
        jwt_service: JWTService,
        password_service: PasswordService,
        *,
        token_revocation_service: TokenRevocationService,
        session: AsyncSession | None = None,
        email_verification_service: EmailVerificationService | None = None,
        ops_event_bus: OpsEventBus | None = None,
    ) -> None:
        """Initialize auth service.

        Args:
            repository: User repository.
            jwt_service: JWT token service.
            password_service: Password hashing service.
            token_revocation_service: Bulk-revokes access tokens issued
                before logout_all and on refresh-token reuse detection (see
                src.api.services.token_revocation). Required — callers that
                intentionally want revocation to no-op (tests, older call
                sites) must pass an explicit
                ``TokenRevocationService(None, max_token_ttl_seconds=0)`` so
                the choice is visible rather than a silent default (issue
                #142 A1).
            session: Database session (for billing account creation).
            email_verification_service: Optional — when provided, sends a
                verification email immediately after successful registration.
                Pass ``None`` to skip email sending (e.g. in tests).
            ops_event_bus: Publishes admin ops notifications (Telegram).
                Defaults to a disabled bus so callers that don't wire one
                (tests, older call sites) simply skip publishing.
        """
        self._repo = repository
        self._jwt = jwt_service
        self._password = password_service
        self._session = session
        self._email_verification = email_verification_service
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )
        self._token_revocation = token_revocation_service

    async def register(
        self,
        *,
        email: str,
        password: str,
        product_id: str,
        display_name: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Register a new user.

        Args:
            email: User email.
            password: Plain text password.
            product_id: Product the user is registering on.
            display_name: Optional display name.

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            EmailAlreadyExistsError: If email is taken on this product.
        """
        # Check for existing email within the same product
        if await self._repo.email_exists(email, product_id=product_id):
            raise EmailAlreadyExistsError(f"Email {email} is already registered")

        # Create user
        user_id = new_id()
        password_hash = await self._password.ahash(password)

        user = await self._repo.create_user(
            id=user_id,
            email=email,
            password_hash=password_hash,
            product_id=product_id,
            display_name=display_name,
        )

        # Create personal token account in the same transaction
        if self._session is not None:
            billing_repo = BillingRepository(self._session)
            await billing_repo.create_personal_account(
                id=new_id(), user_id=user_id, product_id=product_id
            )
            logger.info("billing.account_created", user_id=str(user_id))

        logger.info("user.registered", user_id=str(user_id))
        await self._ops_event_bus.publish(
            event_type=OpsEventType.USER_REGISTERED,
            product_id=product_id,
            payload=UserRegisteredOpsPayload(user_id=user_id),
        )

        # Generate tokens
        tokens, _refresh_token_id = await self._create_token_pair(user_id, product_id=product_id)

        # Send verification email — non-blocking failure: if the email provider
        # is down we still complete registration and let the user resend manually.
        if self._email_verification is not None and self._session is not None:
            try:
                await self._email_verification.send_verification_email(
                    user_id, session=self._session
                )
            except Exception:
                logger.exception(
                    "auth.email_verification_failed",
                    user_id=str(user_id),
                    email=email,
                )

        return user, tokens

    async def login(
        self,
        *,
        email: str,
        password: str,
        product_id: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[User, TokenPair]:
        """Authenticate user and return tokens.

        Args:
            email: User email.
            password: Plain text password.
            product_id: Product the user is logging into.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            Tuple of (User, TokenPair).

        Raises:
            InvalidCredentialsError: If email or password is wrong, or user
                does not exist on this product.
            UserInactiveError: If user account is deactivated.
        """
        user = await self._repo.get_active_user_by_email(email, product_id=product_id)

        if user is None:
            # Prevent timing attacks — covers both not-found and inactive accounts
            await self._password.ahash("dummy_password")
            raise InvalidCredentialsError("Invalid email or password")

        if not await self._password.averify(user.password_hash, password):
            raise InvalidCredentialsError("Invalid email or password")

        # Check if password needs rehash (parameter upgrade)
        if self._password.needs_rehash(user.password_hash):
            new_hash = await self._password.ahash(password)
            await self._repo.update_user(user.id, password_hash=new_hash)
            logger.info("user.password_rehashed", user_id=str(user.id))

        logger.info("user.logged_in", user_id=str(user.id))

        tokens, _refresh_token_id = await self._create_token_pair(
            user.id,
            product_id=product_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return user, tokens

    async def refresh_tokens(
        self,
        refresh_token: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[TokenPair, UUID]:
        """Refresh access token using refresh token.

        Implements token rotation: the old refresh token is revoked
        and a new one is issued in the same family.

        Args:
            refresh_token: Current refresh token.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            New TokenPair.

        Raises:
            InvalidRefreshTokenError: If token is invalid/expired, or was
                revoked by a bulk revocation that won the race against this
                refresh (issue #142 B2 — reported as an ended session, not
                theft).
            TokenReuseDetectedError: If revoked token was reused for any
                other reason (including legacy rows with no recorded
                reason).
        """
        token_hash = hash_token(refresh_token)

        # G1 — look up the owning user_id first via a scalar (non-ORM-entity)
        # read so the user-row lock below can be acquired *before* the row
        # lock on the refresh-token row itself, without loading a
        # RefreshToken object into the session's identity map ahead of the
        # locked read further down. See
        # UserRepository.get_refresh_token_owner's docstring: doing this
        # via a full-entity read (e.g. get_refresh_token_by_hash) would
        # cause the later get_refresh_token_by_hash_for_update call to
        # return that same, already-loaded (and now stale) Python object
        # instead of one reflecting the freshly locked row — silently
        # reintroducing exactly the stale read G1 exists to close.
        # token_hash is immutable once a token is created, so this
        # preliminary read cannot race with anything that would change
        # which user owns this hash; the actual decision below is made
        # from the locked re-read, not this one.
        owner_user_id = await self._repo.get_refresh_token_owner(token_hash)
        if owner_user_id is None:
            raise InvalidRefreshTokenError("Invalid refresh token")

        # G1 — lock ordering is user row -> refresh-token row in every path
        # that acquires both: this method and every bulk-revocation method
        # (UserRepository.revoke_all_user_tokens/revoke_all_refresh_tokens)
        # acquire the user row first. Reversing the order in either path
        # deadlocks against the other — see
        # UserRepository.lock_user_for_session_change's docstring. With
        # this lock held, a concurrent bulk revocation either already
        # committed (so is_revoked/revoked_reason below reflect it) or
        # blocks until this transaction commits/rolls back — the new
        # refresh-token row minted further down is never invisible to it.
        await self._repo.lock_user_for_session_change(owner_user_id)

        # Re-read row-locked (issue #142 F2, G1): is_revoked/revoked_reason
        # below are read from this same locked row, already a re-read under
        # both locks, not a stale pre-lock value.
        stored_token = await self._repo.get_refresh_token_by_hash_for_update(token_hash)

        if stored_token is None:
            raise InvalidRefreshTokenError("Invalid refresh token")

        # Check if token was already revoked (potential theft, or a benign
        # lost race against a bulk revocation — issue #142 B2)
        if stored_token.is_revoked:
            if stored_token.revoked_reason == RefreshTokenRevocationReason.BULK_REVOCATION.value:
                # B2 — this token lost the user-row lock race to a bulk
                # revocation (logout-all/password-change/deactivation/
                # password-reset). That is an ordinary, benign outcome, not
                # theft: no ops alert, no warning log. Every other value —
                # including NULL, for rows revoked before this column
                # existed — falls through to the theft path below.
                logger.info(
                    "auth.refresh_lost_bulk_revocation_race",
                    user_id=str(stored_token.user_id),
                )
                raise InvalidRefreshTokenError("Your session has ended. Please sign in again.")

            # Revoke entire token family as precaution
            revoked_count = await self._repo.revoke_token_family(stored_token.family_id)
            # Bulk-revoke live access tokens/content cookies too (issue #142)
            # — otherwise the "all sessions have been invalidated" message
            # below is false: an access token or content cookie the
            # attacker already holds would keep working for its full
            # remaining lifetime.
            epoch = await self._token_revocation.revoke_user_sessions(stored_token.user_id)
            bulk_access_revoked = epoch is not None
            await self._report_revocation_outcome(
                bulk_access_revoked=bulk_access_revoked,
                user_id=stored_token.user_id,
                op="token_reuse_detected",
            )
            await self._delete_push_subscriptions(stored_token.user_id, op="token_reuse_detected")
            logger.warning(
                "auth.token_reuse_detected",
                user_id=str(stored_token.user_id),
                revoked=revoked_count,
                family=str(stored_token.family_id),
                bulk_access_revoked=bulk_access_revoked,
            )
            raise TokenReuseDetectedError(
                _reuse_detected_message(bulk_access_revoked=bulk_access_revoked)
            )

        # Check expiration
        if stored_token.expires_at < datetime.now(UTC):
            raise InvalidRefreshTokenError("Refresh token has expired")

        # Verify user is still active
        user = await self._repo.get_active_user(stored_token.user_id)
        if user is None:
            raise UserInactiveError("User account is deactivated")

        # F2 — now a backstop, not the primary guarantee (G1's
        # lock_user_for_session_change above is): it covers a revocation
        # arriving via a path that somehow bypasses the lock. Read the
        # epoch before minting so such a revocation can be detected below —
        # this is Redis-backed and entirely independent of the Postgres
        # locking above, so it isn't made redundant by it.
        epoch_before = await self._token_revocation.get_current_epoch(stored_token.user_id)

        # Revoke current token (rotation)
        await self._repo.revoke_refresh_token(
            stored_token.id, reason=RefreshTokenRevocationReason.ROTATED
        )

        # Issue new token pair in same family
        tokens, new_refresh_token_id = await self._create_token_pair(
            stored_token.user_id,
            product_id=stored_token.product_id,
            family_id=stored_token.family_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        epoch_after = await self._token_revocation.get_current_epoch(stored_token.user_id)
        if epoch_after is not None and epoch_after != epoch_before:
            # F2 backstop tripped — a bulk revocation landed between our
            # epoch read and the mint above via a path that bypassed the
            # G1 lock. Refuse deterministically rather than trusting the
            # race: revoke what we just minted (both the new refresh-token
            # row and the new access token's jti) and raise the same error
            # the reuse-detection path raises above.
            await self._repo.revoke_refresh_token(
                new_refresh_token_id, reason=RefreshTokenRevocationReason.BULK_REVOCATION
            )
            access_payload = self._jwt.decode_access_token(tokens.access_token)
            if access_payload is not None:
                await self._token_revocation.revoke_token(access_payload.jti, access_payload.exp)
            logger.warning(
                "auth.refresh_race_detected",
                user_id=str(stored_token.user_id),
            )
            raise TokenReuseDetectedError(_reuse_detected_message(bulk_access_revoked=True))

        logger.debug("auth.token_refreshed", user_id=str(stored_token.user_id))

        return tokens, stored_token.user_id

    async def logout(
        self,
        refresh_token: str,
        *,
        token_payload: TokenPayload | None = None,
    ) -> bool:
        """Revoke the refresh token and, when an access token was presented,
        deny-list its jti.

        The user-row lock is taken **first**, before either refresh-token
        operation below. G1 lock ordering is user row -> refresh-token row
        in every path that acquires both: ``revoke_refresh_token`` below
        issues an ``UPDATE`` that locks the refresh-token row, and
        ``AuthService.refresh_tokens``/``revoke_all_user_tokens``/
        ``revoke_all_refresh_tokens`` all take the user row first.
        Acquiring the user-row lock after the token row (as this method
        used to) inverts that order and deadlocks against every one of
        them — see ``UserRepository.lock_user_for_session_change``.

        With the lock held first, the jti write is also serialized the
        same way every bulk revocation is: a mutating request that passed
        ``recheck_revocation_or_raise`` holds that row lock until it
        commits, so this blocks behind it and the two outcomes are
        deterministic. Without the lock the Redis write cannot be observed
        by an in-flight mutation and the grant lands after logout returns —
        see src/api/security/revocation_recheck.py.

        Deliberately unchanged: logout still does **not** call
        ``revoke_user_sessions``. The lock serializes single-device logout;
        it must not widen it into a bulk revocation.

        Args:
            refresh_token: Refresh token to revoke.
            token_payload: The decoded access token presented alongside the
                refresh token, if any. Only present when the caller had a
                valid, unexpired bearer token — logout is reachable with an
                expired or absent one, and only then is there a jti (and a
                ``sub`` to lock) to act on. A payload whose ``sub`` isn't a
                valid UUID (only reachable with the JWT signing key) skips
                both the lock and the jti denylist — there is no user row
                to serialize against.

        Returns:
            True if the refresh token was revoked, False if not found.
        """
        # G1 lock ordering: user row -> refresh-token row, in every path
        # that acquires both. revoke_refresh_token below UPDATEs (and so
        # locks) the refresh-token row, and AuthService.refresh_tokens /
        # revoke_all_user_tokens / revoke_all_refresh_tokens all take the
        # user row first — acquiring it after the token row here would
        # deadlock against every one of them. See
        # UserRepository.lock_user_for_session_change.
        actor_id = token_payload.user_id if token_payload is not None else None
        if actor_id is not None:
            await self._repo.lock_user_for_session_change(actor_id)

        token_hash = hash_token(refresh_token)
        stored_token = await self._repo.get_refresh_token_by_hash(token_hash)

        revoked = False
        if stored_token is not None:
            await self._repo.revoke_refresh_token(
                stored_token.id, reason=RefreshTokenRevocationReason.SINGLE_LOGOUT
            )
            logger.debug("auth.logged_out", user_id=str(stored_token.user_id))
            revoked = True

        if (
            token_payload is not None
            and actor_id is not None
            and not await self._token_revocation.revoke_token(token_payload.jti, token_payload.exp)
        ):
            # Redis is configured but the write failed — a genuine
            # degradation, not the documented Redis-unset no-op (F5).
            # Logout still succeeds: the refresh token is already
            # revoked above, which is the primary guarantee.
            logger.error("auth.jti_denylist_failed", jti=token_payload.jti)

        return revoked

    async def logout_all(self, user_id: UUID) -> int:
        """Logout from all devices by revoking all tokens.

        Args:
            user_id: User ID.

        Returns:
            Number of tokens revoked.
        """
        count = await self._repo.revoke_all_user_tokens(user_id)
        epoch = await self._token_revocation.revoke_user_sessions(user_id)
        bulk_access_revoked = epoch is not None
        await self._report_revocation_outcome(
            bulk_access_revoked=bulk_access_revoked, user_id=user_id, op="logout_all"
        )
        await self._delete_push_subscriptions(user_id, op="logout_all")
        logger.info(
            "auth.tokens_revoked",
            user_id=str(user_id),
            count=count,
            bulk_access_revoked=bulk_access_revoked,
        )
        return count

    async def _report_revocation_outcome(
        self, *, bulk_access_revoked: bool, user_id: UUID, op: str
    ) -> None:
        """F5 — surface a failed bulk access-token revocation to operators.

        Only alert-worthy when Redis is actually configured
        (`token_revocation.enabled`) — a `False` outcome with Redis unset is
        the documented no-op, already logged once at startup, not a fresh
        degradation. Never raises and never blocks the caller's primary
        action (password reset/logout-all/etc. must still complete even if
        this publish fails — OpsEventBus.publish already guarantees that).
        """
        if bulk_access_revoked or not self._token_revocation.enabled:
            return
        logger.error("auth.bulk_revocation_failed", user_id=str(user_id), op=op)
        await self._ops_event_bus.publish(
            event_type=OpsEventType.TOKEN_REVOCATION_FAILED,
            product_id=PLATFORM_PRODUCT_ID,
            payload=TokenRevocationFailedOpsPayload(user_id=user_id, op=op),
        )

    async def _delete_push_subscriptions(self, user_id: UUID, *, op: str) -> None:
        """Delete every push subscription for the user, if a session is wired.

        A missing session (M3) means a mis-wired construction would
        otherwise skip this security-relevant cleanup silently — production
        always wires one (get_auth_service), so this only fires for direct
        construction (mostly tests).
        """
        if self._session is None:
            logger.warning("auth.push_subscriptions_cleanup_skipped_no_session", op=op)
            return
        await delete_user_push_subscriptions(
            self._session, self._ops_event_bus, user_id=user_id, op=op, source="auth"
        )

    async def _create_token_pair(
        self,
        user_id: UUID,
        *,
        product_id: str | None = None,
        family_id: UUID | None = None,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[TokenPair, UUID]:
        """Create new access and refresh token pair.

        Args:
            user_id: User ID.
            product_id: Product scope to embed in the JWT.
            family_id: Token family ID (for rotation).
            user_agent: Client user agent.
            ip_address: Client IP address.

        Returns:
            Tuple of (TokenPair, the new refresh token's DB row id) — the id
            lets refresh_tokens' post-mint race check (F2) revoke this exact
            row if a bulk revocation is detected to have landed mid-mint.
        """
        # Create access token
        access_token, expires_at = self._jwt.create_access_token(user_id, product_id=product_id)
        expires_in = int(self._jwt.access_token_lifetime.total_seconds())

        # Create refresh token
        refresh_token = generate_token(32)
        refresh_token_hash = hash_token(refresh_token)
        refresh_expires_at = datetime.now(UTC) + self._jwt.refresh_token_lifetime
        refresh_token_id = new_id()

        await self._repo.create_refresh_token(
            id=refresh_token_id,
            user_id=user_id,
            token_hash=refresh_token_hash,
            family_id=family_id or new_id(),
            expires_at=refresh_expires_at,
            product_id=product_id or "vex",
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return (
            TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                expires_in=expires_in,
            ),
            refresh_token_id,
        )
