"""The retention sweep honours holds (S1-P3, W3, invariant 1 and P3-10.1).

The sweep opens its own sessions and commits, so everything here is committed through
``hold_world`` (see ``tests/library_holds_support.py``) and the sweep runs for real.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from structlog.testing import capture_logs

from src.api.services.content_retention import ContentRetentionService
from src.api.services.library_holds import (
    AssetHolder,
    AssetHoldService,
    AssetNotFoundError,
)
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository
from tests.library_holds_support import (
    AUDIO,
    IMAGES,
    HoldWorld,
    make_output,
    output_ref,
    past,
    upload_ref,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.core.library_ref import AssetRef

RETENTION_DAYS = 7


class FakeR2:
    """Records the keys the sweep asks to delete (never touches R2)."""

    def __init__(self) -> None:
        self.deleted_keys: list[str] = []

    async def delete_many(self, storage_keys: list[str]) -> int:
        self.deleted_keys.extend(storage_keys)
        return len(storage_keys)


def sweeper(factory: async_sessionmaker[AsyncSession], r2: FakeR2) -> ContentRetentionService:
    return ContentRetentionService(
        session_factory=factory,
        storage=r2,  # type: ignore[arg-type]
        batch_size=100,
        max_batches_per_run=5,
    )


async def hold(
    world: HoldWorld, ref: AssetRef, user_id: UUID, *, kind: str = "test.holder"
) -> UUID:
    async with world.factory() as session:
        hold_id = await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
            ref,
            holder=AssetHolder(kind=kind, id=uuid4()),
            user_id=user_id,
            product_id="vex",
            accept=IMAGES | AUDIO,  # eligibility is not what these tests are about
        )
        await session.commit()
        return hold_id


async def release(world: HoldWorld, hold_id: UUID) -> None:
    async with world.factory() as session:
        await AssetHoldService(session, retention_days=RETENTION_DAYS).release([hold_id])
        await session.commit()


class TestSweepSkipsHeldAssets:
    async def test_a_held_expired_upload_survives_and_an_unheld_one_does_not(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        held = await hold_world.upload(user, expires_at=past())
        free = await hold_world.upload(user, expires_at=past())
        await hold(hold_world, upload_ref(held), user.id)
        r2 = FakeR2()

        result = await sweeper(hold_world.factory, r2).sweep()

        assert result.uploads_deleted == 1
        assert await hold_world.upload_exists(held)
        assert not await hold_world.upload_exists(free)
        assert r2.deleted_keys == [free.storage_key]  # no R2 object of the held upload

    async def test_a_held_expired_output_survives_with_its_job_and_thumbnail(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        held = await hold_world.output(user, job, expires_at=past())
        free = await hold_world.output(user, job, expires_at=past(), output_index=1)
        async with hold_world.factory() as session:
            await make_output(session, user, job, is_thumbnail=True, parent=held, expires_at=past())
            await session.commit()
        await hold(hold_world, output_ref(held), user.id)

        result = await sweeper(hold_world.factory, FakeR2()).sweep()

        assert result.outputs_deleted == 1
        assert await hold_world.output_exists(held)
        assert not await hold_world.output_exists(free)
        async with hold_world.factory() as session:
            job_row = (
                await session.execute(
                    select(GenerationJob.is_deleted).where(GenerationJob.id == job.id)
                )
            ).scalar_one()
            thumbnails = (
                await session.execute(
                    select(GenerationOutput.id).where(GenerationOutput.parent_output_id == held.id)
                )
            ).all()
        assert job_row is False  # the held output keeps the job visible
        assert len(thumbnails) == 1

    async def test_an_asset_is_swept_again_only_after_its_clock_restarted_and_ran_out(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user, expires_at=past())
        hold_id = await hold(hold_world, upload_ref(image), user.id)

        await sweeper(hold_world.factory, FakeR2()).sweep()
        assert await hold_world.upload_exists(image)  # held: skipped

        await release(hold_world, hold_id)  # the last release re-arms it
        await sweeper(hold_world.factory, FakeR2()).sweep()
        assert await hold_world.upload_exists(image)  # fresh again: not expired

        # expire it again, now with no hold: the next sweep deletes it
        async with hold_world.factory() as session:
            await session.execute(
                update(UserImage).where(UserImage.id == image.id).values(expires_at=past())
            )
            await session.commit()
        await sweeper(hold_world.factory, FakeR2()).sweep()
        assert not await hold_world.upload_exists(image)

    async def test_get_expired_returns_only_unheld_rows_and_locks_them(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        held = await hold_world.upload(user, expires_at=past())
        free = await hold_world.upload(user, expires_at=past())
        await hold(hold_world, upload_ref(held), user.id)

        async with hold_world.factory() as session:
            rows = await UserImageRepository(session).get_expired()
            ids = {row.id for row in rows}
            assert free.id in ids
            assert held.id not in ids
            # SKIP LOCKED: a second session does not wait for, nor see, the locked rows.
            async with hold_world.factory() as other:
                other_ids = {row.id for row in await UserImageRepository(other).get_expired()}
            assert free.id not in other_ids


class TestHoldConflictBackstop:
    """A hold that commits after ``get_expired``'s snapshot is refused by the foreign key."""

    async def test_an_upload_batch_the_foreign_key_refuses_is_rolled_back(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user, expires_at=past())
        await hold(hold_world, upload_ref(image), user.id)
        r2 = FakeR2()

        async def stale_snapshot(self: UserImageRepository, **_: object):
            # The sweep's view just before the hold committed: the held asset is "expired".
            return list(
                (
                    await self._session.execute(
                        select(UserImage).where(UserImage.id == image.id).with_for_update()
                    )
                )
                .scalars()
                .all()
            )

        with (
            patch.object(UserImageRepository, "get_expired", stale_snapshot),
            capture_logs() as logs,
        ):
            result = await sweeper(hold_world.factory, r2).sweep()

        assert result.uploads_deleted == 0
        assert r2.deleted_keys == []  # never delete R2 objects of rows that were not deleted
        assert await hold_world.upload_exists(image)
        conflicts = [log for log in logs if log["event"] == "content_retention.hold_conflict"]
        assert conflicts == [
            {
                "event": "content_retention.hold_conflict",
                "content_type": "upload",
                "rows": 1,
                "log_level": "warning",
            }
        ]
        # The next tick works as usual (the sweep did not crash or wedge).
        result = await sweeper(hold_world.factory, r2).sweep()
        assert result.uploads_deleted == 0
        assert await hold_world.upload_exists(image)

    async def test_an_output_batch_the_foreign_key_refuses_is_rolled_back(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job, expires_at=past())
        await hold(hold_world, output_ref(output), user.id)
        r2 = FakeR2()

        async def stale_snapshot(self: OutputRepository, **_: object):
            return list(
                (
                    await self._session.execute(
                        select(GenerationOutput)
                        .where(GenerationOutput.id == output.id)
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )

        with (
            patch.object(OutputRepository, "get_expired", stale_snapshot),
            capture_logs() as logs,
        ):
            result = await sweeper(hold_world.factory, r2).sweep()

        assert result.outputs_deleted == 0
        assert result.jobs_soft_deleted == 0
        assert r2.deleted_keys == []
        assert await hold_world.output_exists(output)
        assert [
            log["content_type"] for log in logs if log["event"] == "content_retention.hold_conflict"
        ] == ["output"]
        async with hold_world.factory() as session:
            assert (
                await session.execute(
                    select(GenerationJob.is_deleted).where(GenerationJob.id == job.id)
                )
            ).scalar_one() is False

    async def test_an_unrelated_integrity_error_is_not_swallowed(
        self, hold_world: HoldWorld
    ) -> None:
        # Only the hold foreign keys are the backstop's business.
        from sqlalchemy.exc import IntegrityError

        user = await hold_world.user()
        await hold_world.upload(user, expires_at=past())

        async def boom(*_: object, **__: object) -> int:
            raise IntegrityError("stmt", {}, Exception("some other constraint"))

        with (
            patch.object(ContentRetentionService, "_purge_metadata", boom),
            pytest.raises(IntegrityError),
        ):
            await sweeper(hold_world.factory, FakeR2()).sweep()


# ---------------------------------------------------------------------------------------------
# The property: whatever happens, no held asset is swept and every unheld expired one is.
# ---------------------------------------------------------------------------------------------

_ASSETS = (
    6  # 2 image uploads + 2 image outputs of one completed job + an audio upload + an audio output
)
_HOLDERS = 3

_operation = st.one_of(
    st.tuples(st.just("hold"), st.integers(0, _ASSETS - 1), st.integers(0, _HOLDERS - 1)),
    st.tuples(st.just("release"), st.integers(0, _ASSETS - 1), st.integers(0, _HOLDERS - 1)),
    st.tuples(st.just("expire"), st.integers(0, _ASSETS - 1), st.just(0)),
    st.tuples(st.just("sweep"), st.just(0), st.just(0)),
)


@dataclass
class _Model:
    """What the database must look like: the oracle the property compares against."""

    exists: list[bool] = field(default_factory=lambda: [True] * _ASSETS)
    expired: list[bool] = field(default_factory=lambda: [False] * _ASSETS)
    holders: list[dict[int, UUID]] = field(default_factory=lambda: [{} for _ in range(_ASSETS)])

    def sweep(self) -> None:
        for index in range(_ASSETS):
            if self.exists[index] and self.expired[index] and not self.holders[index]:
                self.exists[index] = False


async def _expire(world: HoldWorld, asset: UserImage | GenerationOutput) -> None:
    async with world.factory() as session:
        table = UserImage if isinstance(asset, UserImage) else GenerationOutput
        await session.execute(update(table).where(table.id == asset.id).values(expires_at=past()))
        await session.commit()


async def _present(world: HoldWorld, asset: UserImage | GenerationOutput) -> bool:
    if isinstance(asset, UserImage):
        return await world.upload_exists(asset)
    return await world.output_exists(asset)


async def _run_property(url: str, operations: list[tuple[str, int, int]]) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    world = HoldWorld(factory=async_sessionmaker(bind=engine, expire_on_commit=False))
    try:
        user = await world.user()
        job = await world.job(user)
        assets: list[UserImage | GenerationOutput] = [
            await world.upload(user),
            await world.upload(user),
            await world.output(user, job, output_index=0),
            await world.output(user, job, output_index=1),
            # audio is in the random pool too: a held one is never swept, an unheld expired one is
            # (and its audio profile goes with it)
            await world.audio(user),
            await world.output(user, job, output_index=2, content_type="audio/mpeg"),
        ]
        refs = [
            upload_ref(asset) if isinstance(asset, UserImage) else output_ref(asset)
            for asset in assets
        ]
        model = _Model()
        r2 = FakeR2()

        for kind, asset, holder_index in operations:
            if kind == "hold":
                if holder_index in model.holders[asset]:
                    continue  # one holder holds an asset once (a programming error otherwise)
                if not model.exists[asset]:
                    with pytest.raises(AssetNotFoundError):
                        await hold(world, refs[asset], user.id, kind=f"test.holder{holder_index}")
                    continue
                model.holders[asset][holder_index] = await hold(
                    world, refs[asset], user.id, kind=f"test.holder{holder_index}"
                )
            elif kind == "release":
                hold_id = model.holders[asset].pop(holder_index, None)
                if hold_id is None:
                    continue
                await release(world, hold_id)
                if not model.holders[asset]:
                    model.expired[asset] = False  # the last release restarts the clock
            elif kind == "expire":
                if model.exists[asset]:
                    await _expire(world, assets[asset])
                    model.expired[asset] = True
            else:
                await sweeper(world.factory, r2).sweep()
                model.sweep()
                for index, row in enumerate(assets):
                    assert await _present(world, row) is model.exists[index], (index, operations)

        # Whatever the sequence was, a held asset still exists.
        for index, row in enumerate(assets):
            if model.holders[index]:
                assert await _present(world, row), (index, operations)
    finally:
        await world.cleanup()
        await engine.dispose()


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(operations=st.lists(_operation, min_size=1, max_size=25))
def test_no_held_asset_is_ever_swept_and_every_unheld_expired_one_is(
    test_database_url: str,
    db_engine: AsyncEngine,  # noqa: ARG001 - makes sure the migrations ran
    operations: list[tuple[str, int, int]],
) -> None:
    """Random hold / release / expire / sweep sequences, checked against a model of the state."""
    asyncio.run(_run_property(test_database_url, operations))
