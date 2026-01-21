"""Repository for user-related database operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import GenerationJob, GenerationOutput, RefreshToken, User, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence


class UserRepository:
    """Repository for user database operations.

    Provides data access methods for User and RefreshToken models.
    All methods are async and use the provided session for
    transaction management.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    # -------------------------------------------------------------------------
    # User operations
    # -------------------------------------------------------------------------

    async def create_user(
        self,
        *,
        id: UUID,
        email: str,
        password_hash: str,
        display_name: str | None = None,
    ) -> User:
        """Create a new user.

        Args:
            id: User ID.
            email: User email (must be unique).
            password_hash: Hashed password.
            display_name: Optional display name.

        Returns:
            Created User instance.
        """
        user = User(
            id=id,
            email=email.lower(),
            password_hash=password_hash,
            display_name=display_name,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_user(self, user_id: UUID) -> User | None:
        """Get user by ID.

        Args:
            user_id: User ID.

        Returns:
            User if found, None otherwise.
        """
        return await self._session.get(User, user_id)

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: User email.

        Returns:
            User if found, None otherwise.
        """
        result = await self._session.execute(select(User).where(User.email == email.lower()))
        return result.scalar_one_or_none()

    async def get_active_user(self, user_id: UUID) -> User | None:
        """Get active (non-deleted) user by ID.

        Args:
            user_id: User ID.

        Returns:
            User if found and active, None otherwise.
        """
        result = await self._session.execute(
            select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def get_active_user_by_email(self, email: str) -> User | None:
        """Get active user by email.

        Args:
            email: User email.

        Returns:
            User if found and active, None otherwise.
        """
        result = await self._session.execute(
            select(User).where(
                User.email == email.lower(),
                User.is_active == True,  # noqa: E712
            )
        )
        return result.scalar_one_or_none()

    async def email_exists(self, email: str, exclude_user_id: UUID | None = None) -> bool:
        """Check if email is already registered.

        Args:
            email: Email to check.
            exclude_user_id: Optional user ID to exclude from check.

        Returns:
            True if email exists, False otherwise.
        """
        query = select(func.count()).select_from(User).where(User.email == email.lower())
        if exclude_user_id:
            query = query.where(User.id != exclude_user_id)
        result = await self._session.execute(query)
        return (result.scalar() or 0) > 0

    async def update_user(
        self,
        user_id: UUID,
        *,
        email: str | None = None,
        password_hash: str | None = None,
        display_name: str | None = None,
        subscription_tier: str | None = None,
        is_active: bool | None = None,
    ) -> User | None:
        """Update user fields.

        Args:
            user_id: User ID to update.
            email: New email (optional).
            password_hash: New password hash (optional).
            display_name: New display name (optional).
            subscription_tier: New subscription tier (optional).
            is_active: New active status (optional).

        Returns:
            Updated User if found, None otherwise.
        """
        user = await self.get_user(user_id)
        if user is None:
            return None

        if email is not None:
            user.email = email.lower()
        if password_hash is not None:
            user.password_hash = password_hash
        if display_name is not None:
            user.display_name = display_name
        if subscription_tier is not None:
            user.subscription_tier = subscription_tier
        if is_active is not None:
            user.is_active = is_active

        user.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return user

    async def soft_delete_user(self, user_id: UUID) -> User | None:
        """Soft delete a user by setting is_active to False.

        Args:
            user_id: User ID to deactivate.

        Returns:
            Updated User if found, None otherwise.
        """
        return await self.update_user(user_id, is_active=False)

    # -------------------------------------------------------------------------
    # Refresh token operations
    # -------------------------------------------------------------------------

    async def create_refresh_token(
        self,
        *,
        id: UUID,
        user_id: UUID,
        token_hash: str,
        family_id: UUID,
        expires_at: datetime,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> RefreshToken:
        """Create a new refresh token.

        Args:
            id: Token ID.
            user_id: Owner user ID.
            token_hash: Hashed token value.
            family_id: Token family ID for rotation tracking.
            expires_at: Token expiration time.
            user_agent: Client user agent (optional).
            ip_address: Client IP address (optional).

        Returns:
            Created RefreshToken instance.
        """
        token = RefreshToken(
            id=id,
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        self._session.add(token)
        await self._session.flush()
        return token

    async def get_refresh_token_by_hash(self, token_hash: str) -> RefreshToken | None:
        """Get refresh token by its hash.

        Args:
            token_hash: SHA-256 hash of token.

        Returns:
            RefreshToken if found, None otherwise.
        """
        result = await self._session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def get_valid_refresh_token(self, token_hash: str) -> RefreshToken | None:
        """Get a valid (non-revoked, non-expired) refresh token.

        Args:
            token_hash: SHA-256 hash of token.

        Returns:
            RefreshToken if valid, None otherwise.
        """
        now = datetime.now(timezone.utc)
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.is_revoked == False,  # noqa: E712
                RefreshToken.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def revoke_refresh_token(self, token_id: UUID) -> bool:
        """Revoke a refresh token.

        Args:
            token_id: Token ID to revoke.

        Returns:
            True if revoked, False if not found.
        """
        result = await self._session.execute(
            update(RefreshToken)
            .where(RefreshToken.id == token_id)
            .values(is_revoked=True, revoked_at=datetime.now(timezone.utc))
        )
        return result.rowcount > 0

    async def revoke_token_family(self, family_id: UUID) -> int:
        """Revoke all tokens in a family.

        Used when detecting potential token theft (reuse of revoked token).

        Args:
            family_id: Token family ID.

        Returns:
            Number of tokens revoked.
        """
        result = await self._session.execute(
            update(RefreshToken)
            .where(
                RefreshToken.family_id == family_id,
                RefreshToken.is_revoked == False,  # noqa: E712
            )
            .values(is_revoked=True, revoked_at=datetime.now(timezone.utc))
        )
        return result.rowcount

    async def revoke_all_user_tokens(self, user_id: UUID) -> int:
        """Revoke all refresh tokens for a user.

        Used on password change or logout-all.

        Args:
            user_id: User ID.

        Returns:
            Number of tokens revoked.
        """
        result = await self._session.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,  # noqa: E712
            )
            .values(is_revoked=True, revoked_at=datetime.now(timezone.utc))
        )
        return result.rowcount

    async def cleanup_expired_tokens(self) -> int:
        """Delete expired tokens.

        Called by background cleanup task.

        Returns:
            Number of tokens deleted.
        """
        now = datetime.now(timezone.utc)
        result = await self._session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < now)
        )
        return result.rowcount

    # -------------------------------------------------------------------------
    # User statistics
    # -------------------------------------------------------------------------

    async def get_user_job_count(self, user_id: UUID) -> dict[str, int]:
        """Get job counts by status for a user.

        Args:
            user_id: User ID.

        Returns:
            Dict with total, completed, failed counts.
        """
        total_result = await self._session.execute(
            select(func.count()).select_from(GenerationJob).where(GenerationJob.user_id == user_id)
        )
        total = total_result.scalar() or 0

        completed_result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.user_id == user_id,
                GenerationJob.status == "completed",
            )
        )
        completed = completed_result.scalar() or 0

        failed_result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.user_id == user_id,
                GenerationJob.status == "failed",
            )
        )
        failed = failed_result.scalar() or 0

        return {"total": total, "completed": completed, "failed": failed}

    async def get_user_output_count(self, user_id: UUID) -> int:
        """Get total output count for a user.

        Args:
            user_id: User ID.

        Returns:
            Total output count.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(GenerationOutput)
            .where(GenerationOutput.user_id == user_id)
        )
        return result.scalar() or 0

    async def get_user_upload_count(self, user_id: UUID) -> int:
        """Get total upload count for a user.

        Args:
            user_id: User ID.

        Returns:
            Total upload count.
        """
        result = await self._session.execute(
            select(func.count()).select_from(UserImage).where(UserImage.user_id == user_id)
        )
        return result.scalar() or 0

    async def get_user_storage_bytes(self, user_id: UUID) -> int:
        """Get total storage used by a user.

        Args:
            user_id: User ID.

        Returns:
            Total bytes used.
        """
        uploads_result = await self._session.execute(
            select(func.coalesce(func.sum(UserImage.size_bytes), 0)).where(
                UserImage.user_id == user_id
            )
        )
        uploads_bytes = uploads_result.scalar() or 0

        outputs_result = await self._session.execute(
            select(func.coalesce(func.sum(GenerationOutput.size_bytes), 0)).where(
                GenerationOutput.user_id == user_id
            )
        )
        outputs_bytes = outputs_result.scalar() or 0

        return uploads_bytes + outputs_bytes

    async def list_user_jobs(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[GenerationJob]:
        """List generation jobs for a user.

        Args:
            user_id: User ID.
            limit: Max results.
            offset: Results to skip.

        Returns:
            List of GenerationJob.
        """
        result = await self._session.execute(
            select(GenerationJob)
            .where(GenerationJob.user_id == user_id)
            .order_by(GenerationJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_user_jobs(self, user_id: UUID) -> int:
        """Count total jobs for a user.

        Args:
            user_id: User ID.

        Returns:
            Total job count.
        """
        result = await self._session.execute(
            select(func.count()).select_from(GenerationJob).where(GenerationJob.user_id == user_id)
        )
        return result.scalar() or 0

    async def count_job_outputs(self, job_id: UUID) -> int:
        """Count outputs for a job.

        Args:
            job_id: Job ID.

        Returns:
            Output count.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(GenerationOutput)
            .where(GenerationOutput.job_id == job_id)
        )
        return result.scalar() or 0
