"""User profile service for account management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.schemas.user import (
    JobSummaryResponse,
    UserJobsResponse,
    UserProfileResponse,
    UserStatsResponse,
)
from src.api.security import PasswordService
from src.db.repositories import UserRepository

if TYPE_CHECKING:
    from src.api.services.storage import R2StorageService
    from src.db.models import User

logger = structlog.get_logger(__name__)


class UserServiceError(Exception):
    """Base user service error."""

    pass


class UserNotFoundError(UserServiceError):
    """User not found."""

    pass


class EmailAlreadyExistsError(UserServiceError):
    """Email is already taken."""

    pass


class InvalidPasswordError(UserServiceError):
    """Current password is incorrect."""

    pass


class UserService:
    """User profile management service.

    Handles profile viewing, updating, and soft deletion.
    """

    def __init__(
        self,
        repository: UserRepository,
        password_service: PasswordService,
        r2_storage: R2StorageService | None = None,
    ) -> None:
        """Initialize user service.

        Args:
            repository: User repository.
            password_service: Password hashing service.
            r2_storage: R2 storage service for presigned URL generation (optional).
        """
        self._repo = repository
        self._password = password_service
        self._r2 = r2_storage

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
        display_name: str | None = None,
        email: str | None = None,
    ) -> UserProfileResponse:
        """Update user profile.

        Args:
            user_id: User ID.
            display_name: New display name (optional).
            email: New email (optional).

        Returns:
            Updated UserProfileResponse.

        Raises:
            UserNotFoundError: If user not found.
            EmailAlreadyExistsError: If new email is taken.
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

        # Update fields
        updated_user = await self._repo.update_user(
            user_id,
            email=email,
            display_name=display_name,
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

        Revokes all refresh tokens after password change.

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
        if not self._password.verify(user.password_hash, current_password):
            raise InvalidPasswordError("Current password is incorrect")

        # Hash and update new password
        new_hash = self._password.hash(new_password)
        await self._repo.update_user(user_id, password_hash=new_hash)

        # Revoke all refresh tokens (force re-login on all devices)
        revoked = await self._repo.revoke_all_user_tokens(user_id)
        logger.info("user.password_changed", user_id=str(user_id), revoked_tokens=revoked)

    async def deactivate_account(self, user_id: UUID) -> datetime:
        """Soft delete user account.

        Sets is_active to False and revokes all tokens.

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

        deactivated_at = datetime.now(UTC)
        logger.info("user.deactivated", user_id=str(user_id))

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

    async def get_jobs(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> UserJobsResponse:
        """Get user's generation jobs.

        Args:
            user_id: User ID.
            limit: Max results.
            offset: Results to skip.

        Returns:
            UserJobsResponse.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._repo.get_user(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        jobs = await self._repo.list_user_jobs(user_id, limit=limit, offset=offset)
        total = await self._repo.count_user_jobs(user_id)

        job_ids = [job.id for job in jobs]

        # Both batch queries in a single round-trip each — no N+1
        output_counts, first_outputs = await asyncio.gather(
            self._repo.count_outputs_for_jobs(job_ids),
            self._repo.get_first_outputs_for_jobs(job_ids),
        )

        # Sign thumbnail URLs in parallel (pure HMAC, no R2 round-trips)
        thumbnail_urls: dict[UUID, str] = {}
        if self._r2 and first_outputs:
            sign_results = await asyncio.gather(
                *(self._r2.sign_key(out.storage_key) for out in first_outputs.values()),
                return_exceptions=True,
            )
            for job_id, result in zip(first_outputs.keys(), sign_results, strict=True):
                if isinstance(result, str):
                    thumbnail_urls[job_id] = result

        items = [
            JobSummaryResponse(
                id=str(job.id),
                name=job.name,
                status=str(job.status),
                generation_type=str(job.generation_type),
                prompt=(f"{job.prompt[:200]}..." if len(job.prompt) > 200 else job.prompt),
                output_count=output_counts.get(job.id, 0),
                thumbnail_url=thumbnail_urls.get(job.id),
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
            for job in jobs
        ]

        return UserJobsResponse(items=items, total=total)

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
            role=str(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
