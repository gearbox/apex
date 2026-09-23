"""PostgreSQL contracts for durable PDQ ledger rows and asyncpg BIT values."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from sqlalchemy import bindparam, func, select
from sqlalchemy.exc import IntegrityError

from src.core.uid import new_id
from src.db.models.media_hash import MediaHash
from src.db.types import PdqBit256

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.integration.conftest import JobFactory, UserFactory

pytestmark = pytest.mark.asyncio


def _row(*, user_id: UUID, source_id: UUID, job_id: UUID | None = None) -> MediaHash:
    return MediaHash(
        id=new_id(),
        product_id="vex",
        user_id=user_id,
        job_id=job_id,
        source_kind="output",
        source_id=source_id,
        source_media_type="image",
        hash_profile="pdq-image-rgb-white-v1",
        sampling_profile="still-v1",
        sample_index=0,
        frame_timestamp_ms=None,
        pdq=bytes(range(32)),
        pdq_quality=73,
    )


async def test_asyncpg_bit_round_trip_typed_query_and_fk_lifecycle(
    db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
) -> None:
    """Raw 32-byte PDQ values round-trip and survive only the intended deletes."""
    user = await make_user(email=f"hash-{uuid4().hex}@example.com")
    job = await make_job(user=user)
    row = _row(user_id=user.id, job_id=job.id, source_id=uuid4())
    db_session.add(row)
    await db_session.flush()

    stored = (
        await db_session.execute(select(MediaHash.pdq).where(MediaHash.id == row.id))
    ).scalar_one()
    assert stored == bytes(range(32))

    query_hash = bindparam("query_hash", bytes(range(32)), type_=PdqBit256())
    distance = (
        await db_session.execute(
            select(func.bit_count(MediaHash.pdq.bitwise_xor(query_hash))).where(
                MediaHash.id == row.id
            )
        )
    ).scalar_one()
    assert distance == 0

    await db_session.delete(job)
    await db_session.flush()
    assert (
        await db_session.execute(select(MediaHash.job_id).where(MediaHash.id == row.id))
    ).scalar_one() is None

    user_id = user.id
    await db_session.delete(user)
    await db_session.flush()
    remaining = select(func.count()).select_from(MediaHash).where(MediaHash.user_id == user_id)
    assert (await db_session.execute(remaining)).scalar_one() == 0


async def test_ledger_sample_identity_is_unique(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email=f"hash-unique-{uuid4().hex}@example.com")
    source_id = uuid4()
    db_session.add(_row(user_id=user.id, source_id=source_id))
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(_row(user_id=user.id, source_id=source_id))
            await db_session.flush()

    stored = select(func.count()).select_from(MediaHash).where(MediaHash.source_id == source_id)
    assert (await db_session.execute(stored)).scalar_one() == 1
