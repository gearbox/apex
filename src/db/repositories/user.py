"""Repository for user-related database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import JobStatus, SubscriptionTier, UserRole
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
            select(User).where(User.email == email.lower(), User.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def email_exists(self, email: str, exclude_user_id: UUID | None = None) -> bool:
        """Check if email is already registered by an active user.

        Args:
            email: Email to check.
            exclude_user_id: Optional user ID to exclude from check.

        Returns:
            True if an active user with this email exists, False otherwise.
        """
        query = (
            select(func.count())
            .select_from(User)
            .where(User.email == email.lower(), User.is_active == True)  # noqa: E712
        )
        if exclude_user_id:
            query = query.where(User.id != exclude_user_id)
        result = await self._session.execute(query)
        return int(result.scalar() or 0) > 0

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
            user.subscription_tier = SubscriptionTier(subscription_tier)
        if is_active is not None:
            user.is_active = is_active

        user.updated_at = datetime.now(UTC)
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

    async def mark_email_verified(self, user_id: UUID) -> User | None:
        """Set email_verified_at to now() for a user.

        Args:
            user_id: User to verify.

        Returns:
            Updated User, or None if not found.
        """
        await self._session.execute(
            update(User).where(User.id == user_id).values(email_verified_at=datetime.now(UTC))
        )
        await self._session.flush()
        return await self.get_active_user(user_id)

    async def revoke_all_refresh_tokens(self, user_id: UUID) -> int:
        """Revoke all active refresh tokens for a user.

        Used after a password reset to force re-authentication on all devices.

        Args:
            user_id: User whose tokens to revoke.

        Returns:
            Number of tokens revoked.
        """
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == user_id)
                .where(RefreshToken.is_revoked == False)  # noqa: E712
                .values(is_revoked=True)
            ),
        )
        await self._session.flush()
        return result.rowcount or 0

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
        now = datetime.now(UTC)
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
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                update(RefreshToken)
                .where(RefreshToken.id == token_id)
                .values(is_revoked=True, revoked_at=datetime.now(UTC))
            ),
        )
        return (result.rowcount or 0) > 0

    async def revoke_token_family(self, family_id: UUID) -> int:
        """Revoke all tokens in a family.

        Used when detecting potential token theft (reuse of revoked token).

        Args:
            family_id: Token family ID.

        Returns:
            Number of tokens revoked.
        """
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.family_id == family_id,
                    RefreshToken.is_revoked == False,  # noqa: E712
                )
                .values(is_revoked=True, revoked_at=datetime.now(UTC))
            ),
        )
        return result.rowcount or 0

    async def revoke_all_user_tokens(self, user_id: UUID) -> int:
        """Revoke all refresh tokens for a user.

        Used on password change or logout-all.

        Args:
            user_id: User ID.

        Returns:
            Number of tokens revoked.
        """
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.user_id == user_id,
                    RefreshToken.is_revoked == False,  # noqa: E712
                )
                .values(is_revoked=True, revoked_at=datetime.now(UTC))
            ),
        )
        return result.rowcount or 0

    async def cleanup_expired_tokens(self) -> int:
        """Delete expired tokens.

        Called by background cleanup task.

        Returns:
            Number of tokens deleted.
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[tuple[()]],
            await self._session.execute(delete(RefreshToken).where(RefreshToken.expires_at < now)),
        )
        return result.rowcount or 0

    # -------------------------------------------------------------------------
    # Admin operations
    # -------------------------------------------------------------------------

    async def list_users(
        self,
        *,
        is_active: bool | None = None,
        role: str | None = None,
        email_contains: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Sequence[User], int]:
        """List users with optional filtering, excluding SYSTEM role users.

        Args:
            is_active: Filter by active status when not None.
            role: Filter by exact role value when not None.
            email_contains: Case-insensitive partial email match when not None.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (users, total_count) matching the filters.
        """
        base = select(User).where(User.role != UserRole.SYSTEM.value)
        count_base = (
            select(func.count(User.id))
            .select_from(User)
            .where(User.role != UserRole.SYSTEM.value)
        )

        if is_active is not None:
            base = base.where(User.is_active == is_active)
            count_base = count_base.where(User.is_active == is_active)
        if role is not None:
            base = base.where(User.role == role)
            count_base = count_base.where(User.role == role)
        if email_contains is not None:
            pattern = f"%{email_contains}%"
            base = base.where(User.email.ilike(pattern))
            count_base = count_base.where(User.email.ilike(pattern))

        count_result = await self._session.execute(count_base)
        total = int(count_result.scalar_one())

        result = await self._session.execute(
            base.order_by(User.created_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all(), total

    async def update_user_admin(
        self,
        user_id: UUID,
        *,
        role: str | None = None,
        subscription_tier: str | None = None,
        is_active: bool | None = None,
    ) -> User | None:
        """Update user fields as an admin operation.

        Uses a single UPDATE … RETURNING round-trip when there are changes.
        Returns the existing user unchanged when no fields are provided.

        Args:
            user_id: ID of the user to update.
            role: New role value (must not be UserRole.SYSTEM).
            subscription_tier: New subscription tier value.
            is_active: New active status.

        Returns:
            Updated (or unchanged) User, or None if user does not exist.

        Raises:
            ValueError: If role is set to UserRole.SYSTEM.
        """
        if role is not None and role == UserRole.SYSTEM.value:
            raise ValueError("Cannot set user role to SYSTEM")

        values: dict[str, object] = {}
        if role is not None:
            values["role"] = role
        if subscription_tier is not None:
            values["subscription_tier"] = subscription_tier
        if is_active is not None:
            values["is_active"] = is_active

        if not values:
            return await self.get_user(user_id)

        values["updated_at"] = datetime.now(UTC)

        result = await self._session.execute(
            update(User).where(User.id == user_id).values(**values).returning(User)
        )
        return result.scalar_one_or_none()

    # -------------------------------------------------------------------------
    # User statistics
    # -------------------------------------------------------------------------

    async def set_preferred_billing_account(
        self,
        user_id: UUID,
        account_type: str | None,
    ) -> User | None:
        """Set or clear the preferred billing account for a user.

        Returns the updated User, or None if not found.
        """
        result = await self._session.execute(
            update(User)
            .where(User.id == user_id)
            .values(preferred_billing_account=account_type, updated_at=datetime.now(UTC))
            .returning(User)
        )
        return result.scalar_one_or_none()

    async def get_preferred_billing_account(self, user_id: UUID) -> str | None:
        """Fetch only the preferred_billing_account field for a user.

        Returns the preference string, or None if not set or user not found.
        """
        result = await self._session.execute(
            select(User.preferred_billing_account).where(User.id == user_id)
        )
        row = result.one_or_none()
        return None if row is None else row[0]

    async def get_user_job_count(self, user_id: UUID) -> dict[str, int]:
        """Get job counts by status for a user.

        Args:
            user_id: User ID.

        Returns:
            Dict with total, completed, failed counts.
        """
        query = (
            select(
                GenerationJob.status,
                func.count().label("count"),
            )
            .where(GenerationJob.user_id == user_id)
            .group_by(GenerationJob.status)
        )
        result = await self._session.execute(query)
        rows = result.all()

        total = 0
        completed = 0
        failed = 0

        for row in rows:
            status, count = row
            total += count
            if status == JobStatus.COMPLETED.value:
                completed = count
            elif status == JobStatus.FAILED.value:
                failed = count

        return {"total": total, "completed": completed, "failed": failed}

    async def _get_user_job_count_(self, user_id: UUID) -> dict[str, int]:
        """Get job counts by status for a user.

        Args:
            user_id: User ID.

        Returns:
            Dict with total, completed, failed counts.
        """
        # TODO: doublecheck if this method is needed
        total_result = await self._session.execute(
            select(func.count()).select_from(GenerationJob).where(GenerationJob.user_id == user_id)
        )
        total = int(total_result.scalar() or 0)

        completed_result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.user_id == user_id,
                GenerationJob.status == JobStatus.COMPLETED.value,
            )
        )
        completed = int(completed_result.scalar() or 0)

        failed_result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.user_id == user_id,
                GenerationJob.status == JobStatus.FAILED.value,
            )
        )
        failed = int(failed_result.scalar() or 0)

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
        return int(result.scalar() or 0)

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
        return int(result.scalar() or 0)

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
        uploads_bytes = int(uploads_result.scalar() or 0)

        outputs_result = await self._session.execute(
            select(func.coalesce(func.sum(GenerationOutput.size_bytes), 0)).where(
                GenerationOutput.user_id == user_id
            )
        )
        outputs_bytes = int(outputs_result.scalar() or 0)

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
        return int(result.scalar() or 0)

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
        return int(result.scalar() or 0)

    async def count_outputs_for_jobs(self, job_ids: list[UUID]) -> dict[UUID, int]:
        """Count outputs for multiple jobs in a single query.

        Args:
            job_ids: List of job IDs.

        Returns:
            Dict mapping job_id to output count.
        """
        if not job_ids:
            return {}

        result = await self._session.execute(
            select(GenerationOutput.job_id, func.count().label("count"))
            .where(GenerationOutput.job_id.in_(job_ids))
            .group_by(GenerationOutput.job_id)
        )
        rows = result.all()

        # Initialize all job_ids with 0, then update with actual counts
        counts: dict[UUID, int] = dict.fromkeys(job_ids, 0)
        for job_id, count in rows:
            counts[job_id] = count

        return counts
