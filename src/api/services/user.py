"""User profile service for account management."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.ops_events import (
    PLATFORM_PRODUCT_ID,
    OpsEventType,
    TokenRevocationFailedOpsPayload,
)
from src.api.schemas.user import (
    UserProfileResponse,
    UserStatsResponse,
)
from src.api.services.age_verification import AgeVerificationError, AgeVerificationService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.push_cleanup import delete_user_push_subscriptions

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.security import PasswordService
    from src.api.services.storage import R2StorageService
    from src.api.services.token_revocation import TokenRevocationService
    from src.core.product import ProductConfig
    from src.db.models import User
    from src.db.repositories import UserRepository

logger = structlog.get_logger(__name__)


class UserServiceError(Exception):
    """Base user service error."""


class UserNotFoundError(UserServiceError):
    """User not found."""


class EmailAlreadyExistsError(UserServiceError):
    """Email is already taken."""


class InvalidPasswordError(UserServiceError):
    """Current password is incorrect."""


class UserService:
    """User profile management service.

    Handles profile viewing, updating, and soft deletion.
    """

    def __init__(
        self,
        repository: UserRepository,
        password_service: PasswordService,
        age_verification_service: AgeVerificationService,
        *,
        token_revocation_service: TokenRevocationService,
        r2_storage: R2StorageService | None = None,
        ops_event_bus: OpsEventBus | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Initialize user service.

        Args:
            repository: User repository.
            password_service: Password hashing service.
            age_verification_service: Age gate claim validator.
            token_revocation_service: Bulk-revokes access tokens issued
                before a password change or account deactivation (see
                src.api.services.token_revocation). Required — callers that
                intentionally want revocation to no-op (tests, older call
                sites) must pass an explicit
                ``TokenRevocationService(None, max_token_ttl_seconds=0)`` so
                the choice is visible rather than a silent default (issue
                #142 A1).
            r2_storage: R2 storage service for presigned URL generation (optional).
            ops_event_bus: Publishes an alert when a bulk access-token
                revocation write fails against a configured Redis (issue
                #142 F5). Defaults to a disabled bus so callers that don't
                wire one (tests, older call sites) simply skip publishing.
            session: Database session, used to delete the user's push
                subscriptions alongside a bulk revocation
                (push-cleanup-on-revocation). Optional — callers that don't
                wire one (tests, older call sites) simply skip that cleanup.
        """
        self._repo = repository
        self._password = password_service
        self._age_verification = age_verification_service
        self._r2 = r2_storage
        self._token_revocation = token_revocation_service
        self._ops_event_bus = (
            ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False)
        )
        self._session = session

    async def get_profile(self, user_id: UUID) -> UserProfileResponse:
        """Get user profile.

        Args:
            user_id: User ID.

        Returns:
            UserProfileResponse.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        return self._to_profile_response(user)

    async def update_profile(
        self,
        user_id: UUID,
        *,
        product_config: ProductConfig,
        display_name: str | None = None,
        email: str | None = None,
        locale: str | None = None,
        age_confirmed: bool | None = None,
        date_of_birth: date | None = None,
    ) -> UserProfileResponse:
        """Update user profile.

        Args:
            user_id: User ID.
            product_config: Active product config (used for age gate policy).
            display_name: New display name (optional).
            email: New email (optional).
            locale: New locale (optional).
            age_confirmed: Age checkbox value (optional, for CHECKBOX products).
            date_of_birth: Date of birth (optional, for DATE_OF_BIRTH products).

        Returns:
            Updated UserProfileResponse.

        Raises:
            UserNotFoundError: If user not found.
            EmailAlreadyExistsError: If new email is taken.
            AgeVerificationError: If age claim is invalid or DOB conflicts.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # Check email uniqueness if changing
        if (
            email is not None
            and email.lower() != user.email
            and await self._repo.email_exists(email, exclude_user_id=user_id)
        ):
            raise EmailAlreadyExistsError(f"Email {email} is already taken")

        # Validate age claim and compute desired values
        desired_verified_at, desired_dob = self._age_verification.verify(
            product_config,
            age_confirmed=age_confirmed,
            date_of_birth=date_of_birth,
        )

        # Monotonic: only set age_verified_at on first verification
        new_age_verified_at: datetime | None = None
        if desired_verified_at is not None and user.age_verified_at is None:
            new_age_verified_at = desired_verified_at

        # Write-once: reject a different DOB if one is already stored
        new_dob: date | None = None
        if desired_dob is not None:
            if user.date_of_birth is not None and desired_dob != user.date_of_birth:
                raise AgeVerificationError("date_of_birth cannot be changed once set")
            if user.date_of_birth is None:
                new_dob = desired_dob

        updated_user = await self._repo.update_user(
            user_id,
            email=email,
            display_name=display_name,
            locale=locale,
            age_verified_at=new_age_verified_at,
            date_of_birth=new_dob,
        )

        if updated_user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        logger.info("user.profile_updated", user_id=str(user_id))

        return self._to_profile_response(updated_user)

    async def change_password(
        self,
        user_id: UUID,
        *,
        current_password: str,
        new_password: str,
    ) -> None:
        """Change user password.

        Revokes all refresh tokens and all live access tokens/content
        cookies after password change — the most security-sensitive of the
        three bulk-revocation sites (issue #142), since a password change is
        often a reaction to suspected compromise.

        Args:
            user_id: User ID.
            current_password: Current password for verification.
            new_password: New password.

        Raises:
            UserNotFoundError: If user not found.
            InvalidPasswordError: If current password is wrong.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # Verify current password
        if not await self._password.averify(user.password_hash, current_password):
            raise InvalidPasswordError("Current password is incorrect")

        # Hash and update new password
        new_hash = await self._password.ahash(new_password)
        await self._repo.update_user(user_id, password_hash=new_hash)

        # Revoke all refresh tokens (force re-login on all devices)
        revoked = await self._repo.revoke_all_user_tokens(user_id)
        # Bulk-revoke live access tokens/content cookies too (issue #142) —
        # otherwise a stolen access token survives a password change for its
        # full remaining lifetime.
        epoch = await self._token_revocation.revoke_user_sessions(user_id)
        bulk_access_revoked = epoch is not None
        await self._report_revocation_outcome(
            bulk_access_revoked=bulk_access_revoked, user_id=user_id, op="change_password"
        )
        await self._delete_push_subscriptions(user_id, op="change_password")
        logger.info(
            "user.password_changed",
            user_id=str(user_id),
            revoked_tokens=revoked,
            bulk_access_revoked=bulk_access_revoked,
        )

    async def deactivate_account(self, user_id: UUID) -> datetime:
        """Soft delete user account.

        Sets is_active to False and revokes all refresh tokens plus any
        live access tokens/content cookies (issue #142).

        Args:
            user_id: User ID.

        Returns:
            Deactivation timestamp.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._repo.soft_delete_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # Revoke all tokens
        await self._repo.revoke_all_user_tokens(user_id)
        # Bulk-revoke live access tokens/content cookies too (issue #142).
        epoch = await self._token_revocation.revoke_user_sessions(user_id)
        bulk_access_revoked = epoch is not None
        await self._report_revocation_outcome(
            bulk_access_revoked=bulk_access_revoked, user_id=user_id, op="deactivate_account"
        )
        await self._delete_push_subscriptions(user_id, op="deactivate_account")

        deactivated_at = datetime.now(UTC)
        logger.info(
            "user.deactivated", user_id=str(user_id), bulk_access_revoked=bulk_access_revoked
        )

        return deactivated_at

    async def _report_revocation_outcome(
        self, *, bulk_access_revoked: bool, user_id: UUID, op: str
    ) -> None:
        """F5 — surface a failed bulk access-token revocation to operators.

        Only alert-worthy when Redis is actually configured
        (`token_revocation.enabled`) — a failed outcome with Redis unset is
        the documented no-op, already logged once at startup, not a fresh
        degradation. Never raises and never blocks the caller's primary
        action — password change/deactivation must still complete even if
        this publish fails (OpsEventBus.publish already guarantees that).
        """
        if bulk_access_revoked or not self._token_revocation.enabled:
            return
        logger.error("user.bulk_revocation_failed", user_id=str(user_id), op=op)
        await self._ops_event_bus.publish(
            event_type=OpsEventType.TOKEN_REVOCATION_FAILED,
            product_id=PLATFORM_PRODUCT_ID,
            payload=TokenRevocationFailedOpsPayload(user_id=user_id, op=op),
        )

    async def _delete_push_subscriptions(self, user_id: UUID, *, op: str) -> None:
        """Delete every push subscription for the user, if a session is wired.

        A missing session (M3) means a mis-wired construction would
        otherwise skip this security-relevant cleanup silently — production
        always wires one (get_user_service), so this only fires for direct
        construction (mostly tests).
        """
        if self._session is None:
            logger.warning("user.push_subscriptions_cleanup_skipped_no_session", op=op)
            return
        await delete_user_push_subscriptions(
            self._session, self._ops_event_bus, user_id=user_id, op=op, source="user"
        )

    async def get_stats(self, user_id: UUID) -> UserStatsResponse:
        """Get user statistics.

        Args:
            user_id: User ID.

        Returns:
            UserStatsResponse.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        job_counts = await self._repo.get_user_job_count(user_id)
        output_count = await self._repo.get_user_output_count(user_id)
        upload_count = await self._repo.get_user_upload_count(user_id)
        storage_bytes = await self._repo.get_user_storage_bytes(user_id)

        return UserStatsResponse(
            total_jobs=job_counts["total"],
            completed_jobs=job_counts["completed"],
            failed_jobs=job_counts["failed"],
            total_outputs=output_count,
            total_uploads=upload_count,
            storage_used_bytes=storage_bytes,
        )

    def _to_profile_response(self, user: User) -> UserProfileResponse:
        """Convert User model to profile response.

        Args:
            user: User model.

        Returns:
            UserProfileResponse.
        """
        return UserProfileResponse(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            subscription_tier=str(user.subscription_tier),
            locale=user.locale,
            role=str(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            age_verified=user.age_verified_at is not None,
            age_verified_at=user.age_verified_at,
            date_of_birth=user.date_of_birth,
        )
