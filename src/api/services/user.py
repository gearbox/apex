"""User profile service for account management."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.user import (
    UserProfileResponse,
    UserStatsResponse,
)
from src.api.services.age_verification import AgeVerificationError, AgeVerificationService

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.security import PasswordService
    from src.api.services.legal.acceptance import LegalAcceptanceService, RequestContext
    from src.api.services.session_termination import SessionTerminationService
    from src.core.product import ProductConfig
    from src.db.models import User
    from src.db.repositories import UserRepository
    from src.db.repositories.user_identity import UserIdentityRepository

logger = structlog.get_logger(__name__)


class UserServiceError(Exception):
    """Base user service error."""


class UserNotFoundError(UserServiceError):
    """User not found."""


class EmailAlreadyExistsError(UserServiceError):
    """Email is already taken."""


class InvalidPasswordError(UserServiceError):
    """Current password is incorrect."""


class PasswordNotSetError(UserServiceError):
    """The account has no password (OAuth-only) — set one via forgot-password instead."""


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
        legal_acceptance_service: LegalAcceptanceService,
        identity_repository: UserIdentityRepository,
        session_termination: SessionTerminationService,
    ) -> None:
        """Initialize user service.

        Args:
            repository: User repository.
            password_service: Password hashing service.
            age_verification_service: Age gate claim validator.
            legal_acceptance_service: Records the sensitive-data consent
                withdrawal on account closure.
            identity_repository: Deletes the user's OAuth identity links on
                self-closure so the subject can sign up again. Required —
                callers that intentionally do not use OAuth must pass an
                explicit repository so this security-relevant cleanup is
                never silently skipped (issue #142 A1).
            session_termination: Owns the "revoke everything" sequence (refresh
                tokens, access-token epoch, push cleanup) run after a password
                change or account deactivation. Required — there is no
                fallback that skips any of the steps (issue #142 A1).
        """
        self._repo = repository
        self._password = password_service
        self._age_verification = age_verification_service
        self._legal = legal_acceptance_service
        self._identities = identity_repository
        self._sessions = session_termination

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
            and await self._repo.email_exists(
                email, product_id=user.product_id, exclude_user_id=user_id
            )
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
            PasswordNotSetError: If the account has no password (OAuth-only).
            InvalidPasswordError: If current password is wrong.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        if user.password_hash is None:
            raise PasswordNotSetError("This account has no password set")

        # Verify current password
        if not await self._password.averify(user.password_hash, current_password):
            raise InvalidPasswordError("Current password is incorrect")

        # Hash and update new password
        new_hash = await self._password.ahash(new_password)
        await self._repo.update_user(user_id, password_hash=new_hash)

        # Force re-login on all devices: refresh tokens, live access tokens/
        # content cookies (issue #142 — otherwise a stolen access token
        # survives a password change for its full remaining lifetime), push.
        result = await self._sessions.terminate_all(user_id, op="change_password", source="user")
        logger.info(
            "user.password_changed",
            user_id=str(user_id),
            revoked_tokens=result.revoked_refresh_tokens,
            bulk_access_revoked=result.bulk_access_revoked,
        )

    async def deactivate_account(
        self,
        user_id: UUID,
        *,
        product: ProductConfig,
        context: RequestContext,
    ) -> datetime:
        """Soft delete user account.

        Sets is_active to False, records withdrawal of sensitive-data consent
        (closure *is* withdrawal — see legal/{product}/sensitive_data_consent),
        and revokes all refresh tokens plus any live access tokens/content
        cookies (issue #142).

        Args:
            user_id: User ID.
            product: The user's product (decides whether consent is tracked).
            context: Client IP / user agent, stored with the withdrawal event.

        Returns:
            Deactivation timestamp.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._repo.soft_delete_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # Same transaction as the soft delete.
        today = datetime.now(UTC).date()
        await self._legal.record_consent_withdrawal(
            user_id=user_id, product=product, context=context, today=today
        )
        # Unlink OAuth identities in the same transaction, so the provider
        # subject can sign up afresh. (Admin deactivation keeps them — it's
        # reversible; self-closure is not.)
        unlinked = await self._identities.delete_for_user(user_id)
        if unlinked:
            logger.info("user.identities_unlinked", user_id=str(user_id), count=unlinked)

        # Revoke every session (issue #142).
        result = await self._sessions.terminate_all(user_id, op="deactivate_account", source="user")

        deactivated_at = datetime.now(UTC)
        logger.info(
            "user.deactivated",
            user_id=str(user_id),
            bulk_access_revoked=result.bulk_access_revoked,
        )

        return deactivated_at

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
            email_verified=user.email_verified_at is not None,
            has_password=user.password_hash is not None,
            age_verified_at=user.age_verified_at,
            date_of_birth=user.date_of_birth,
        )
