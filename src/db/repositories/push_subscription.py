"""Repository for Web Push subscription database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.uid import new_id
from src.db.models.push_subscription import PushSubscription
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class PushSubscriptionRepository(BaseRepository[PushSubscription]):
    """Data access layer for PushSubscription records."""

    _model = PushSubscription

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_endpoint(self, endpoint: str) -> PushSubscription | None:
        """Look up a subscription by its unique browser endpoint."""
        result = await self._session.execute(
            select(PushSubscription).where(PushSubscription.endpoint == endpoint)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        endpoint: str,
        p256dh: str,
        auth: str,
        user_agent: str | None,
    ) -> PushSubscription:
        """Atomically insert or update a subscription row keyed by endpoint.

        Reassigns ownership to ``user_id`` if the endpoint was previously
        registered under a different user (shared device / account switch).
        Updates ``last_seen_at`` on every call.
        """
        now = datetime.now(UTC)
        stmt = (
            pg_insert(PushSubscription)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                endpoint=endpoint,
                p256dh=p256dh,
                auth=auth,
                user_agent=user_agent,
            )
            .on_conflict_do_update(
                constraint="uq_push_subscriptions_endpoint",
                set_={
                    "user_id": user_id,
                    "product_id": product_id,
                    "p256dh": p256dh,
                    "auth": auth,
                    "user_agent": user_agent,
                    "last_seen_at": now,
                },
            )
            .returning(PushSubscription)
            .execution_options(populate_existing=True)
        )
        result = await self._session.execute(stmt)
        subscription = result.scalar_one()
        await self._session.flush()
        return subscription

    async def delete_by_endpoint(self, endpoint: str, *, user_id: UUID) -> bool:
        """Delete a subscription owned by ``user_id``. Returns True if a row was deleted."""
        result = await self._session.execute(
            delete(PushSubscription).where(
                PushSubscription.endpoint == endpoint,
                PushSubscription.user_id == user_id,
            )
        )
        await self._session.flush()
        return bool(result.rowcount)  # type: ignore[attr-defined]

    async def delete_by_id(self, subscription_id: UUID) -> None:
        """Delete a subscription by primary key (used to prune expired subscriptions)."""
        await self._session.execute(
            delete(PushSubscription).where(PushSubscription.id == subscription_id)
        )
        await self._session.flush()

    async def list_by_user(self, user_id: UUID) -> Sequence[PushSubscription]:
        """List all subscriptions for a user (no pagination — per-user fan-out is small)."""
        result = await self._session.execute(
            select(PushSubscription).where(PushSubscription.user_id == user_id)
        )
        return result.scalars().all()

    async def list_batch(
        self,
        *,
        limit: int,
        cursor_id: UUID | None,
    ) -> Sequence[PushSubscription]:
        """Keyset-paginated batch ordered by id, for broadcast fan-out."""
        query = select(PushSubscription)
        if cursor_id is not None:
            query = query.where(PushSubscription.id > cursor_id)
        result = await self._session.execute(query.order_by(PushSubscription.id).limit(limit))
        return result.scalars().all()
