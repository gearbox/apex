"""Integration tests for IdempotencyRepository against real PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.idempotency import IdempotencyKey
from src.db.repositories.idempotency import IdempotencyRepository


@pytest.fixture
def idempotency_repo(db_session: AsyncSession) -> IdempotencyRepository:
    return IdempotencyRepository(db_session)


async def test_try_acquire_success(idempotency_repo: IdempotencyRepository, make_user) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="test-key-1",
        operation="generation",
        request_hash="abc123",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None
    assert record.status == "processing"
    assert record.idempotency_key == "test-key-1"


async def test_try_acquire_conflict_returns_none(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    kwargs = {
        "user_id": user.id,
        "product_id": "vex",
        "idempotency_key": "dup-key",
        "operation": "generation",
        "request_hash": "abc123",
        "expires_at": datetime.now(UTC) + timedelta(hours=24),
    }
    first = await idempotency_repo.try_acquire(id=uuid4(), **kwargs)
    assert first is not None

    second = await idempotency_repo.try_acquire(id=uuid4(), **kwargs)
    assert second is None


async def test_different_users_same_key_no_conflict(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user1 = await make_user(email=f"idem1-{uuid4().hex[:6]}@example.com")
    user2 = await make_user(email=f"idem2-{uuid4().hex[:6]}@example.com")
    base = {
        "product_id": "vex",
        "idempotency_key": "shared-key",
        "operation": "generation",
        "request_hash": "abc",
        "expires_at": datetime.now(UTC) + timedelta(hours=24),
    }
    r1 = await idempotency_repo.try_acquire(id=uuid4(), user_id=user1.id, **base)
    r2 = await idempotency_repo.try_acquire(id=uuid4(), user_id=user2.id, **base)
    assert r1 is not None
    assert r2 is not None


async def test_different_products_same_key_no_conflict(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    base = {
        "user_id": user.id,
        "idempotency_key": "cross-product-key",
        "operation": "generation",
        "request_hash": "abc",
        "expires_at": datetime.now(UTC) + timedelta(hours=24),
    }
    r1 = await idempotency_repo.try_acquire(id=uuid4(), product_id="vex", **base)
    r2 = await idempotency_repo.try_acquire(id=uuid4(), product_id="synthara", **base)
    assert r1 is not None
    assert r2 is not None


async def test_mark_completed_and_get_existing(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="complete-key",
        operation="payment",
        request_hash="xyz",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None

    resource_id = uuid4()
    await idempotency_repo.mark_completed(
        record.id,
        resource_id=resource_id,
        response_status_code=201,
        response_body={"payment_id": str(resource_id)},
    )

    existing = await idempotency_repo.get_existing(user.id, "vex", "complete-key")
    assert existing is not None
    assert existing.status == "completed"
    assert existing.response_status_code == 201
    assert existing.resource_id == resource_id


async def test_mark_failed_and_get_existing(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="fail-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None

    await idempotency_repo.mark_failed(record.id)

    existing = await idempotency_repo.get_existing(user.id, "vex", "fail-key")
    assert existing is not None
    assert existing.status == "failed"
    assert existing.completed_at is not None


async def test_get_existing_returns_none_for_expired(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    # Insert with past expires_at (already expired)
    await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="expired-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )

    result = await idempotency_repo.get_existing(user.id, "vex", "expired-key")
    assert result is None


async def test_reclaim_stale_succeeds_when_old_enough(
    idempotency_repo: IdempotencyRepository, make_user, db_session: AsyncSession
) -> None:
    """A 'processing' record older than stale_before is atomically reclaimed,
    with created_at refreshed to now (so a second reclaim pass can't also
    succeed against the same stale window)."""
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="stale-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None

    # Backdate created_at to simulate a stranded 'processing' record (the
    # owning request crashed before calling complete()/fail()).
    old_created_at = datetime.now(UTC) - timedelta(seconds=200)
    await db_session.execute(
        update(IdempotencyKey)
        .where(IdempotencyKey.id == record.id)
        .values(created_at=old_created_at)
    )
    await db_session.flush()

    new_expires_at = datetime.now(UTC) + timedelta(hours=24)
    reclaimed = await idempotency_repo.reclaim_stale(
        record.id,
        request_hash="abc",
        expires_at=new_expires_at,
        stale_before=datetime.now(UTC) - timedelta(seconds=120),
    )
    assert reclaimed is True

    existing = await idempotency_repo.get_existing(user.id, "vex", "stale-key")
    assert existing is not None
    assert existing.status == "processing"  # reclaimed, not completed/failed
    assert existing.created_at > old_created_at  # refreshed to "now"


async def test_reclaim_stale_fails_when_too_young(
    idempotency_repo: IdempotencyRepository, make_user
) -> None:
    """A 'processing' record younger than stale_before is not reclaimable —
    it's a genuinely concurrent in-flight request, not a dead owner."""
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="fresh-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None

    reclaimed = await idempotency_repo.reclaim_stale(
        record.id,
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
        stale_before=datetime.now(UTC) - timedelta(seconds=120),
    )
    assert reclaimed is False


async def test_reclaim_stale_fails_when_already_completed(
    idempotency_repo: IdempotencyRepository, make_user, db_session: AsyncSession
) -> None:
    """A record that finished (status != 'processing') by the time reclaim
    runs must not be touched — the WHERE guard excludes it."""
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    record = await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="done-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    assert record is not None
    await idempotency_repo.mark_completed(
        record.id, resource_id=uuid4(), response_status_code=201, response_body={}
    )

    old_created_at = datetime.now(UTC) - timedelta(seconds=200)
    await db_session.execute(
        update(IdempotencyKey)
        .where(IdempotencyKey.id == record.id)
        .values(created_at=old_created_at)
    )
    await db_session.flush()

    reclaimed = await idempotency_repo.reclaim_stale(
        record.id,
        request_hash="abc",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
        stale_before=datetime.now(UTC) - timedelta(seconds=120),
    )
    assert reclaimed is False


async def test_cleanup_expired(idempotency_repo: IdempotencyRepository, make_user) -> None:
    user = await make_user(email=f"idem-{uuid4().hex[:6]}@example.com")
    # Create expired record
    await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="expired-key",
        operation="generation",
        request_hash="abc",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    # Create non-expired record
    await idempotency_repo.try_acquire(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        idempotency_key="valid-key",
        operation="generation",
        request_hash="def",
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )

    deleted = await idempotency_repo.cleanup_expired()
    assert deleted == 1

    # Valid key still exists
    valid = await idempotency_repo.get_existing(user.id, "vex", "valid-key")
    assert valid is not None
