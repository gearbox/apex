"""W1-a — ``revoke_all_refresh_tokens`` really blocks on the user-row lock (G1).

A second transaction holds ``SELECT ... FOR UPDATE`` on the user row; the bulk
revocation must queue behind it in Postgres and only run once it is released.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import RefreshTokenRevocationReason
from src.core.uid import new_id
from src.db.models.user import RefreshToken, User
from src.db.repositories.user import UserRepository
from tests.integration.pg_locks import wait_for_row_lock_waiter

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine


async def _seed(engine: AsyncEngine, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(
            User(
                id=user_id,
                email=f"w1a-lock-{uuid4().hex[:8]}@example.com",
                password_hash="x" * 64,
                product_id="vex",
                is_active=True,
            )
        )
        await session.flush()
        for _ in range(2):
            session.add(
                RefreshToken(
                    id=new_id(),
                    user_id=user_id,
                    token_hash=uuid4().hex,
                    family_id=new_id(),
                    expires_at=datetime.now(UTC) + timedelta(days=7),
                    product_id="vex",
                )
            )
        await session.commit()


async def _cleanup(engine: AsyncEngine, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def test_w1_a_revoke_all_refresh_tokens_waits_for_the_user_row_lock(
    db_engine: AsyncEngine,
) -> None:
    user_id = uuid4()
    await _seed(db_engine, user_id)
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as holder:
            await UserRepository(holder).lock_user_for_session_change(user_id)

            async def revoke() -> int:
                async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                    count = await UserRepository(session).revoke_all_refresh_tokens(user_id)
                    await session.commit()
                    return count

            task = asyncio.create_task(revoke())
            await wait_for_row_lock_waiter(db_engine, relation="users")
            assert not task.done(), "bulk revocation ran without waiting for the user-row lock"

            await holder.commit()
            assert await asyncio.wait_for(task, timeout=5) == 2

        async with AsyncSession(bind=db_engine) as session:
            reasons = (
                (
                    await session.execute(
                        select(RefreshToken.revoked_reason).where(RefreshToken.user_id == user_id)
                    )
                )
                .scalars()
                .all()
            )
        assert reasons == [RefreshTokenRevocationReason.BULK_REVOCATION.value] * 2
    finally:
        await _cleanup(db_engine, user_id)
