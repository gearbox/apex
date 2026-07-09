"""Repository for idempotency key operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.idempotency import IdempotencyKey

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class IdempotencyRepository:
    """Data access for idempotency key deduplication."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def try_acquire(
        self,
        *,
        id: UUID,
        user_id: UUID,
        product_id: str,
        idempotency_key: str,
        operation: str,
        request_hash: str,
        expires_at: datetime,
    ) -> IdempotencyKey | None:
        """Attempt to insert a new idempotency record.

        Uses INSERT ... ON CONFLICT DO NOTHING. Returns the inserted row
        if successful, or None if the key already exists (conflict).
        """
        stmt = (
            pg_insert(IdempotencyKey)
            .values(
                id=id,
                user_id=user_id,
                product_id=product_id,
                idempotency_key=idempotency_key,
                operation=operation,
                status="processing",
                request_hash=request_hash,
                expires_at=expires_at,
            )
            .on_conflict_do_nothing(constraint="uq_idempotency_user_product_key")
            .returning(IdempotencyKey)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            await self._session.flush()
        return row

    async def get_existing(
        self,
        user_id: UUID,
        product_id: str,
        idempotency_key: str,
    ) -> IdempotencyKey | None:
        """Look up an existing idempotency record (non-expired)."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(IdempotencyKey).where(
                IdempotencyKey.user_id == user_id,
                IdempotencyKey.product_id == product_id,
                IdempotencyKey.idempotency_key == idempotency_key,
                IdempotencyKey.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def mark_completed(
        self,
        record_id: UUID,
        *,
        resource_id: UUID | None = None,
        response_status_code: int,
        response_body: dict[str, Any],
    ) -> None:
        """Mark an idempotency record as completed with cached response."""
        record = await self._session.get(IdempotencyKey, record_id)
        if record is None:
            return
        record.status = "completed"
        record.resource_id = resource_id
        record.response_status_code = response_status_code
        record.response_body = response_body
        record.completed_at = datetime.now(UTC)
        await self._session.flush()

    async def reclaim_stale(
        self,
        record_id: UUID,
        *,
        request_hash: str,
        expires_at: datetime,
        stale_before: datetime,
    ) -> bool:
        """Atomically take over a stale 'processing' record.

        Guarded UPDATE: only matches rows still 'processing' with
        ``created_at`` older than ``stale_before``. Resets ``created_at`` to
        now as part of the same statement — this is what makes a second,
        near-simultaneous reclaim attempt lose: it re-evaluates the WHERE
        clause against the just-refreshed ``created_at`` (no longer stale)
        once it gets past this UPDATE's row lock, so at most one caller ever
        sees rowcount > 0. Also refreshes request_hash (matches the caller's
        already-verified request) and expires_at (fresh TTL window instead of
        inheriting whatever was left on the original attempt).

        Returns:
            True if this call won the reclaim, False if the row was no
            longer stale/processing by the time the lock was granted.
        """
        now = datetime.now(UTC)
        result = await self._session.execute(
            update(IdempotencyKey)
            .where(
                IdempotencyKey.id == record_id,
                IdempotencyKey.status == "processing",
                IdempotencyKey.created_at < stale_before,
            )
            .values(created_at=now, request_hash=request_hash, expires_at=expires_at)
        )
        await self._session.flush()
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount > 0

    async def mark_failed(self, record_id: UUID) -> None:
        """Mark an idempotency record as failed (allows retry with same key)."""
        record = await self._session.get(IdempotencyKey, record_id)
        if record is None:
            return
        record.status = "failed"
        record.completed_at = datetime.now(UTC)
        await self._session.flush()

    async def cleanup_expired(self) -> int:
        """Delete expired idempotency records. Returns count deleted."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            delete(IdempotencyKey).where(IdempotencyKey.expires_at < now)
        )
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount
