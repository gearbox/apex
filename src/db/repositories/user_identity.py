"""Repository for ``user_identities`` (OAuth/OIDC subject links)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select, update

from src.db.models.user_identity import UserIdentity

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.product import OAuthProvider


class UserIdentityRepository:
    """Data access for identity-provider links.

    Never flushes or commits — the owning service decides that (the OAuth
    service flushes inside a savepoint to detect unique violations).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_subject(
        self, *, product_id: str, provider: OAuthProvider, subject: str
    ) -> UserIdentity | None:
        """The identity for a provider subject on one product, if linked."""
        stmt = select(UserIdentity).where(
            UserIdentity.product_id == product_id,
            UserIdentity.provider == provider,
            UserIdentity.subject == subject,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_for_user(self, *, user_id: UUID, provider: OAuthProvider) -> UserIdentity | None:
        """The user's link for one provider, if any."""
        stmt = select(UserIdentity).where(
            UserIdentity.user_id == user_id, UserIdentity.provider == provider
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    def add(
        self,
        *,
        id: UUID,
        user_id: UUID,
        product_id: str,
        provider: OAuthProvider,
        subject: str,
    ) -> UserIdentity:
        """Stage a new identity row in the session (no flush)."""
        identity = UserIdentity(
            id=id,
            user_id=user_id,
            product_id=product_id,
            provider=provider,
            subject=subject,
        )
        self._session.add(identity)
        return identity

    async def touch_last_login(self, identity_id: UUID) -> None:
        """Record a successful sign-in through this identity."""
        await self._session.execute(
            update(UserIdentity)
            .where(UserIdentity.id == identity_id)
            .values(last_login_at=func.now())
        )

    async def delete_for_user(self, user_id: UUID) -> int:
        """Delete every identity of a user (self-closure). Returns the row count."""
        result = cast(
            "CursorResult[tuple[()]]",
            await self._session.execute(
                delete(UserIdentity).where(UserIdentity.user_id == user_id)
            ),
        )
        return result.rowcount or 0
