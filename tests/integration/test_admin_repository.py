"""Integration tests for AdminRepository.get_audit_log keyset pagination."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.admin import AdminAuditLog
from src.db.repositories.admin import AdminRepository
from tests.integration.conftest import UserFactory


async def _insert_audit_log(
    session: AsyncSession,
    *,
    product_id: str,
    actor_id: UUID,
    target_user_id: UUID,
    created_at: datetime,
    action: str = "role.grant",
) -> AdminAuditLog:
    entry = AdminAuditLog(
        id=uuid4(),
        actor_id=actor_id,
        target_user_id=target_user_id,
        product_id=product_id,
        action=action,
        detail="test detail",
        source="api",
        created_at=created_at,
    )
    session.add(entry)
    await session.flush()
    return entry


async def test_get_audit_log_cursor_paginates(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    """Two-page cursor fetch returns non-overlapping, gap-free slices in created_at DESC order."""
    product_id = "vex"
    actor = await make_user(email=f"actor-{uuid4().hex[:6]}@example.com", product_id=product_id)
    target = await make_user(email=f"target-{uuid4().hex[:6]}@example.com", product_id=product_id)

    base_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    n = 5
    inserted = []
    for i in range(n):
        entry = await _insert_audit_log(
            db_session,
            product_id=product_id,
            actor_id=actor.id,
            target_user_id=target.id,
            created_at=base_ts + timedelta(minutes=i),
        )
        inserted.append(entry)

    # inserted[0] is oldest, inserted[4] is newest
    # ORDER BY created_at DESC → newest first: [4, 3, 2, 1, 0]
    repo = AdminRepository(db_session)
    limit = 3

    # Page 1: limit+1 rows returned
    page1_raw = await repo.get_audit_log(product_id, limit=limit)
    assert len(page1_raw) == limit + 1  # 4 rows → has_more
    page1 = list(page1_raw)[:limit]

    # Verify descending order
    assert page1[0].created_at > page1[1].created_at
    assert page1[1].created_at > page1[2].created_at

    # Page 2: use cursor from last item of page 1
    last = page1[-1]
    page2_raw = await repo.get_audit_log(
        product_id,
        limit=limit,
        cursor_ts=last.created_at,
        cursor_id=last.id,
    )
    page2 = list(page2_raw)[:limit]

    # Page 1 IDs and page 2 IDs must not overlap
    page1_ids = {e.id for e in page1}
    page2_ids = {e.id for e in page2}
    assert page1_ids.isdisjoint(page2_ids)

    # Together they cover all n inserted rows with no gap
    assert page1_ids | page2_ids == {e.id for e in inserted}

    # Page 2 items are older than page 1 items
    for e2 in page2:
        for e1 in page1:
            assert e2.created_at <= e1.created_at


async def test_get_audit_log_product_scoping(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    """Rows from a different product_id are excluded."""
    actor = await make_user(email=f"actor2-{uuid4().hex[:6]}@example.com", product_id="vex")
    target = await make_user(email=f"target2-{uuid4().hex[:6]}@example.com", product_id="vex")

    ts = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)
    vex_entry = await _insert_audit_log(
        db_session,
        product_id="vex",
        actor_id=actor.id,
        target_user_id=target.id,
        created_at=ts,
    )
    # Synthara user for the FK — same actor/target since FKs reference users table not product-scoped
    await _insert_audit_log(
        db_session,
        product_id="synthara",
        actor_id=actor.id,
        target_user_id=target.id,
        created_at=ts + timedelta(seconds=1),
    )

    repo = AdminRepository(db_session)
    results = await repo.get_audit_log("vex", limit=50)
    result_ids = {e.id for e in results}

    assert vex_entry.id in result_ids
    # All returned rows must be for vex
    for e in results:
        assert e.product_id == "vex"


async def test_get_audit_log_target_user_id_filter(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    """target_user_id filter narrows results to the given user only."""
    product_id = "vex"
    actor = await make_user(email=f"actor3-{uuid4().hex[:6]}@example.com", product_id=product_id)
    target_a = await make_user(email=f"ta-{uuid4().hex[:6]}@example.com", product_id=product_id)
    target_b = await make_user(email=f"tb-{uuid4().hex[:6]}@example.com", product_id=product_id)

    ts = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
    entry_a = await _insert_audit_log(
        db_session,
        product_id=product_id,
        actor_id=actor.id,
        target_user_id=target_a.id,
        created_at=ts,
    )
    await _insert_audit_log(
        db_session,
        product_id=product_id,
        actor_id=actor.id,
        target_user_id=target_b.id,
        created_at=ts + timedelta(seconds=1),
    )

    repo = AdminRepository(db_session)
    results = await repo.get_audit_log(product_id, target_user_id=target_a.id, limit=50)

    assert len(results) >= 1
    result_ids = {e.id for e in results}
    assert entry_a.id in result_ids
    for e in results:
        assert e.target_user_id == target_a.id


async def test_get_audit_log_empty_first_page_when_no_rows(db_session: AsyncSession) -> None:
    """Returns empty sequence when no audit rows exist for the product."""
    repo = AdminRepository(db_session)
    results = await repo.get_audit_log(f"no-such-product-{uuid4().hex}", limit=10)
    assert not list(results)
