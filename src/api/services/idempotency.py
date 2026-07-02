"""Idempotency service — orchestrates deduplication for mutation endpoints."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.uid import new_id
from src.db.repositories.idempotency import IdempotencyRepository

logger = structlog.get_logger(__name__)


class IdempotencyConflictError(Exception):
    """Raised when a request with the same key is already in-flight. → HTTP 409"""


class IdempotencyReplayResult:
    """Cached response from a previous identical request."""

    __slots__ = ("status_code", "body")

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body


class IdempotencyService:
    """Manages idempotency key lifecycle.

    Usage pattern in a route handler:

        result = await idempotency_service.check(...)
        if isinstance(result, IdempotencyReplayResult):
            return Response(content=result.body, status_code=result.status_code)

        record_id = result  # UUID of the new idempotency record
        try:
            ... do the actual work ...
            await idempotency_service.complete(record_id, ...)
        except Exception:
            await idempotency_service.fail(record_id, session=session)
            raise
    """

    def __init__(self, ttl_hours: int = 24, processing_stale_after_seconds: int = 120) -> None:
        self._ttl_hours = ttl_hours
        self._processing_stale_after_seconds = processing_stale_after_seconds

    @staticmethod
    def hash_request(body: bytes) -> str:
        """SHA-256 hex digest of the raw request body for mismatch detection."""
        return hashlib.sha256(body).hexdigest()

    async def check(
        self,
        *,
        user_id: UUID,
        product_id: str,
        idempotency_key: str,
        operation: str,
        request_hash: str,
        session: AsyncSession,
    ) -> UUID | IdempotencyReplayResult:
        """Check idempotency key and either acquire or replay.

        Returns:
            UUID: The idempotency record ID — caller should proceed with
                  the operation and then call complete() or fail().
            IdempotencyReplayResult: Cached response from previous execution.

        Raises:
            IdempotencyConflictError: Another request with the same key is
                currently in-flight (status='processing').
        """
        repo = IdempotencyRepository(session)
        expires_at = datetime.now(UTC) + timedelta(hours=self._ttl_hours)

        # Attempt atomic insert
        record = await repo.try_acquire(
            id=new_id(),
            user_id=user_id,
            product_id=product_id,
            idempotency_key=idempotency_key,
            operation=operation,
            request_hash=request_hash,
            expires_at=expires_at,
        )
        if record is not None:
            # New key — caller should proceed
            logger.debug(
                "idempotency.acquired",
                key=idempotency_key,
                operation=operation,
            )
            return record.id

        # Key exists — fetch the existing record
        existing = await repo.get_existing(user_id, product_id, idempotency_key)
        if existing is None:
            # Key expired between insert attempt and lookup — treat as new
            record = await repo.try_acquire(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=request_hash,
                expires_at=expires_at,
            )
            if record is not None:
                return record.id
            # Extremely unlikely race — fallback to conflict
            raise IdempotencyConflictError("Concurrent request with the same idempotency key")

        # Check request body mismatch
        if existing.request_hash != request_hash:
            logger.warning(
                "idempotency.request_mismatch",
                key=idempotency_key,
                operation=operation,
            )
            raise IdempotencyConflictError("Idempotency key reused with different request body")

        # Still processing — either a concurrent in-flight request, or the
        # owning request died (crash / poisoned session) and left the key
        # stranded. Distinguish by age: only a record older than the stale
        # threshold is eligible for reclaim.
        if existing.status == "processing":
            age = datetime.now(UTC) - existing.created_at
            if age.total_seconds() < self._processing_stale_after_seconds:
                logger.info(
                    "idempotency.in_flight",
                    key=idempotency_key,
                    operation=operation,
                )
                raise IdempotencyConflictError("Concurrent request with the same idempotency key")

            stale_before = datetime.now(UTC) - timedelta(
                seconds=self._processing_stale_after_seconds
            )
            reclaimed = await repo.reclaim_stale(
                existing.id,
                request_hash=request_hash,
                expires_at=expires_at,
                stale_before=stale_before,
            )
            if reclaimed:
                logger.warning(
                    "idempotency.stale_reclaimed",
                    key=idempotency_key,
                    operation=operation,
                    age_s=int(age.total_seconds()),
                )
                return existing.id
            # Lost the reclaim race to a concurrent retry — that retry now
            # owns the record.
            raise IdempotencyConflictError("Concurrent request with the same idempotency key")

        # Failed — allow retry (delete old record and re-acquire)
        if existing.status == "failed":
            await session.delete(existing)
            await session.flush()
            record = await repo.try_acquire(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=request_hash,
                expires_at=expires_at,
            )
            if record is not None:
                return record.id
            raise IdempotencyConflictError("Concurrent request with the same idempotency key")

        # Completed — replay cached response
        logger.info(
            "idempotency.replay",
            key=idempotency_key,
            operation=operation,
            resource_id=str(existing.resource_id),
        )
        return IdempotencyReplayResult(
            status_code=existing.response_status_code or 200,
            body=existing.response_body or {},
        )

    async def complete(
        self,
        record_id: UUID,
        *,
        resource_id: UUID | None = None,
        response_status_code: int,
        response_body: dict[str, Any],
        session: AsyncSession,
    ) -> None:
        """Mark idempotency record as completed with cached response."""
        repo = IdempotencyRepository(session)
        await repo.mark_completed(
            record_id,
            resource_id=resource_id,
            response_status_code=response_status_code,
            response_body=response_body,
        )

    async def fail(
        self,
        record_id: UUID,
        *,
        session: AsyncSession,
    ) -> None:
        """Mark idempotency record as failed (allows retry)."""
        repo = IdempotencyRepository(session)
        await repo.mark_failed(record_id)

    async def cleanup_expired(self, session: AsyncSession) -> int:
        """Remove expired idempotency records. Returns count deleted."""
        repo = IdempotencyRepository(session)
        count = await repo.cleanup_expired()
        logger.info("idempotency.cleanup", deleted=count)
        return count
