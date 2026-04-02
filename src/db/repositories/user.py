"""Repository for user-related database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from sqlalchemy import CursorResult, case, delete, func, literal, select, tuple_, update
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
        product_id: str,
        display_name: str | None = None,
    ) -> User:
        """Create a new user.

        Args:
            id: User ID.
            email: User email (must be unique within product).
            password_hash: Hashed password.
            product_id: Product the user is registering on.
            display_name: Optional display name.

        Returns:
            Created User instance.
        """
        user = User(
            id=id,
            email=email.lower(),
            password_hash=password_hash,
            product_id=product_id,
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

    async def get_active_user_by_email(
        self,
        email: str,
        *,
        product_id: str | None = None,
    ) -> User | None:
        """Get active user by email, optionally scoped to a product.

        Args:
            email: User email.
            product_id: When provided, only return user if product_id matches.

        Returns:
            User if found and active, None otherwise.
        """
        conditions = [User.email == email.lower(), User.is_active == True]  # noqa: E712
        if product_id is not None:
            conditions.append(User.product_id == product_id)
        result = await self._session.execute(select(User).where(*conditions))
        return result.scalar_one_or_none()

    async def email_exists(
        self,
        email: str,
        *,
        product_id: str | None = None,
        exclude_user_id: UUID | None = None,
    ) -> bool:
        """Check if email is already registered by an active user.

        Args:
            email: Email to check.
            product_id: When provided, scope check to this product.
            exclude_user_id: Optional user ID to exclude from check.

        Returns:
            True if an active user with this email exists, False otherwise.
        """
        query = (
            select(func.count())
            .select_from(User)
            .where(User.email == email.lower(), User.is_active == True)  # noqa: E712
        )
        if product_id is not None:
            query = query.where(User.product_id == product_id)
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
        locale: str | None = None,
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
        if locale is not None:
            user.locale = locale

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
        product_id: str,
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
            product_id: Product this token is scoped to.
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
            product_id=product_id,
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
        product_id: str | None = None,
        is_active: bool | None = None,
        role: str | None = None,
        email_contains: str | None = None,
        limit: int = 50,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[User]:
        """List users with optional filtering, excluding SYSTEM role users.

        Uses cursor-based (keyset) pagination with limit+1 fetch pattern.
        Caller checks ``len(result) > limit`` to determine ``has_more``.

        Args:
            product_id: When provided, filter by product.
            is_active: Filter by active status when not None.
            role: Filter by exact role value when not None.
            email_contains: Case-insensitive partial email match when not None.
            limit: Maximum number of results to return (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            Sequence of User instances matching the filters.
        """
        base = select(User).where(User.role != UserRole.SYSTEM.value)

        if product_id is not None:
            base = base.where(User.product_id == product_id)
        if is_active is not None:
            base = base.where(User.is_active == is_active)
        if role is not None:
            base = base.where(User.role == role)
        if email_contains is not None:
            pattern = f"%{email_contains}%"
            base = base.where(User.email.ilike(pattern))

        if cursor_ts is not None and cursor_id is not None:
            base = base.where(
                tuple_(User.created_at, User.id) < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            base.order_by(User.created_at.desc(), User.id.desc()).limit(limit + 1)
        )
        return result.scalars().all()

    async def list_users_by_roles(
        self,
        product_id: str,
        roles: Sequence[str],
        limit: int = 500,
    ) -> Sequence[User]:
        """List active users matching any of the given roles in a product.

        Excludes SYSTEM role. Ordered: superadmin first, then admin, by created_at desc.
        """
        result = await self._session.execute(
            select(User)
            .where(
                User.product_id == product_id,
                User.role.in_(roles),
                User.role != UserRole.SYSTEM.value,
                User.is_active == True,  # noqa: E712
            )
            .order_by(
                case(
                    (User.role == UserRole.SUPERADMIN.value, 0),
                    (User.role == UserRole.ADMIN.value, 1),
                    else_=2,
                ),
                User.created_at.desc(),
            )
            .limit(limit)
        )
        return result.scalars().all()

    async def count_by_roles(
        self,
        product_id: str,
        roles: Sequence[str],
    ) -> int:
        """Count active users with any of the given roles in a product."""
        result = await self._session.execute(
            select(func.count(User.id)).where(
                User.product_id == product_id,
                User.role.in_(roles),
                User.is_active == True,  # noqa: E712
            )
        )
        return result.scalar_one()

    async def update_user_admin(
        self,
        user_id: UUID,
        *,
        role: str | None = None,
        subscription_tier: str | None = None,
        is_active: bool | None = None,
        locale: str | None = None,
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
        if locale is not None:
            values["locale"] = locale

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
