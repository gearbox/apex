"""Two-session races around holds (S1-P3, W10.4 (a)-(e) and invariant 6: never a 500).

Every test proves, in PostgreSQL itself, that the second request is **blocked behind a specific
backend** (``tests.db_locks``), lets the first one finish, and then checks that the loser got a
typed outcome and the state is consistent. The racing task reports its backend pid first and works
in that same session; every task is awaited with ``gather(..., return_exceptions=True)`` in a
``finally`` *after* the block that holds the lock has exited, so a failure is one clean ``FAILED``
and never a transaction left behind.

Rows are committed (``hold_world``): a second connection must see them.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, update

from src.api.services.content_proxy import ContentProxyService
from src.api.services.content_retention import ContentRetentionService
from src.api.services.library import LibraryService
from src.api.services.library_holds import (
    AssetHolder,
    AssetHoldService,
    AssetIneligibleReason,
    AssetNotEligibleError,
    AssetNotFoundError,
)
from src.core.enums import AssetDeleteOutcome, SoftDeleteOutcome
from src.core.library_ref import format_asset_ref
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.repositories.job import JobRepository
from src.db.repositories.library import LibraryRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository
from tests.db_locks import backend_pid, wait_until_blocked
from tests.library_holds_support import (
    IMAGES,
    RETENTION_DAYS,
    HoldWorld,
    output_ref,
    past,
    upload_ref,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.core.library_ref import AssetRef
    from src.db.models.user import User

KIND = "studio.cast_reference"


class FakeR2:
    def __init__(self) -> None:
        self.deleted_keys: list[str] = []

    async def delete_many(self, storage_keys: list[str]) -> int:
        self.deleted_keys.extend(storage_keys)
        return len(storage_keys)


def _sweeper(world: HoldWorld) -> ContentRetentionService:
    return ContentRetentionService(
        session_factory=world.factory,
        storage=FakeR2(),  # type: ignore[arg-type]
        batch_size=100,
        max_batches_per_run=5,
    )


class Racer:
    """A request that reports its backend pid, then runs ``work(session)`` in that session."""

    def __init__(self, world: HoldWorld, work: Callable[[AsyncSession], Awaitable[Any]]) -> None:
        self._world = world
        self._work = work
        self.pid = 0
        self.reported = asyncio.Event()
        self.task: asyncio.Task[Any] | None = None

    async def _run(self) -> Any:
        async with self._world.factory() as session:
            self.pid = await backend_pid(session)
            self.reported.set()
            try:
                result = await self._work(session)
                await session.commit()
            except BaseException:
                await session.rollback()
                raise
            return result

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())
        await asyncio.wait_for(self.reported.wait(), 5)

    async def outcome(self) -> Any:
        """The result, or the exception the request raised (never both)."""
        assert self.task is not None
        (result,) = await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 15)
        return result


async def _hold(session: AsyncSession, ref: AssetRef, user: User, *, kind: str = KIND) -> UUID:
    return await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
        ref,
        holder=AssetHolder(kind=kind, id=uuid4()),
        user_id=user.id,
        product_id="vex",
        accept=IMAGES,
    )


async def _committed_hold(world: HoldWorld, ref: AssetRef, user: User, *, kind: str = KIND) -> UUID:
    async with world.factory() as session:
        hold_id = await _hold(session, ref, user, kind=kind)
        await session.commit()
        return hold_id


async def _await_blocked(engine: AsyncEngine, racer: Racer) -> None:
    """Prove the racer's backend waits on a lock - and that it is not simply done."""
    assert racer.task is not None
    await wait_until_blocked(engine, racer.pid)
    assert not racer.task.done()


# ---------------------------------------------------------------------------------------------
# (a) the sweep holds the asset FOR UPDATE: hold() waits, then the asset is gone
# ---------------------------------------------------------------------------------------------


class TestSweepThenHold:
    async def test_hold_waits_for_the_sweep_and_then_answers_asset_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user, expires_at=past())
        racer = Racer(hold_world, lambda s: _hold(s, upload_ref(image), user))
        try:
            async with hold_world.factory() as sweep:
                # What the sweep does: select the expired rows (locked FOR UPDATE) ...
                locked = await UserImageRepository(sweep).get_expired()
                assert [row.id for row in locked] == [image.id]
                await racer.start()
                await _await_blocked(db_engine, racer)
                # ... delete them, and commit.
                await sweep.execute(delete(UserImage).where(UserImage.id == image.id))
                await sweep.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AssetNotFoundError)
        assert await hold_world.hold_count() == 0

    async def test_an_output_hold_waits_for_the_sweep_too_and_nothing_deadlocks(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        # The sweep locks an output FOR UPDATE and then UPDATEs its job (it soft-deletes a job
        # left without outputs). hold() takes the output first and the job second - the sweep's own
        # order - so the two can never wait for each other. (Job first would: hold waits for the
        # output while holding the job, the sweep waits for the job while holding the output.)
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job, expires_at=past())  # the job's only output
        racer = Racer(hold_world, lambda s: _hold(s, output_ref(output), user))
        try:
            async with hold_world.factory() as sweep:
                locked = await OutputRepository(sweep).get_expired()
                assert [row.id for row in locked] == [output.id]
                await racer.start()
                await _await_blocked(db_engine, racer)
                await sweep.execute(
                    delete(GenerationOutput).where(GenerationOutput.id == output.id)
                )
                # the sweep's next statement: it needs the job row, which hold() must not hold
                await sweep.execute(
                    update(GenerationJob).where(GenerationJob.id == job.id).values(is_deleted=True)
                )
                await sweep.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AssetNotFoundError)  # not a DeadlockDetectedError


# ---------------------------------------------------------------------------------------------
# (b) a hold in flight: the sweep skips the asset, now and after the commit
# ---------------------------------------------------------------------------------------------


class TestHoldThenSweep:
    @pytest.mark.parametrize("source", ["upload", "output"])
    async def test_the_sweep_skips_an_asset_whose_hold_is_in_flight_and_after_the_commit(
        self, source: str, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        asset: UserImage | GenerationOutput
        if source == "upload":
            asset = await hold_world.upload(user, expires_at=past())
            ref = upload_ref(asset)
            exists = hold_world.upload_exists
        else:
            asset = await hold_world.output(user, job, expires_at=past())
            ref = output_ref(asset)
            exists = hold_world.output_exists

        async with hold_world.factory() as holder_session:
            await _hold(holder_session, ref, user)  # in flight: uncommitted, KEY SHARE taken

            await _sweeper(hold_world).sweep()  # skips the locked row (SKIP LOCKED)
            assert await exists(asset)  # type: ignore[arg-type]

            await holder_session.commit()

        await _sweeper(hold_world).sweep()  # the committed hold excludes it (NOT EXISTS)
        assert await exists(asset)  # type: ignore[arg-type]
        assert await hold_world.hold_count() == 1
        async with hold_world.factory() as session:
            if source == "output":
                # the held output keeps its job visible
                assert (
                    await session.execute(
                        select(GenerationJob.is_deleted).where(GenerationJob.id == job.id)
                    )
                ).scalar_one() is False


# ---------------------------------------------------------------------------------------------
# (c) library delete vs hold(), both orders
# ---------------------------------------------------------------------------------------------


def _content_proxy() -> ContentProxyService:
    storage = MagicMock()
    storage.delete = AsyncMock()
    settings = MagicMock()
    settings.content_url_ttl = 60
    return ContentProxyService(storage=storage, settings=settings)


class TestLibraryDeleteVsHold:
    async def test_the_delete_wins_the_hold_gets_asset_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        racer = Racer(hold_world, lambda s: _hold(s, upload_ref(image), user))
        try:
            async with hold_world.factory() as deleter:
                # the DELETE has run, the transaction is still open
                await deleter.execute(delete(UserImage).where(UserImage.id == image.id))
                await racer.start()
                await _await_blocked(db_engine, racer)
                await deleter.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AssetNotFoundError)
        assert await hold_world.hold_count() == 0

    async def test_the_hold_wins_the_delete_is_refused_and_keeps_the_metadata(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = upload_ref(image)
        async with hold_world.factory() as session:
            await LibraryRepository(session).upsert_metadata(
                user.id, "vex", ref.source, ref.asset_id, is_favorite=True
            )
            await session.commit()

        async def delete_through_the_service(session: AsyncSession) -> AssetDeleteOutcome:
            return await LibraryService(session=session).delete_asset(
                format_asset_ref(ref.source, ref.asset_id),
                user.id,
                "vex",
                session=session,
                content_proxy=_content_proxy(),
            )

        racer = Racer(hold_world, delete_through_the_service)
        try:
            async with hold_world.factory() as holder:
                await _hold(holder, ref, user)  # in flight: it holds KEY SHARE on the upload
                await racer.start()
                # The delete sees no *committed* hold, purges the metadata and then blocks on the
                # row (the hold's KEY SHARE conflicts with DELETE's FOR UPDATE).
                await _await_blocked(db_engine, racer)
                await holder.commit()
        finally:
            outcome = await racer.outcome()

        assert outcome is AssetDeleteOutcome.HELD  # the FK refused; rolled back; typed
        assert await hold_world.upload_exists(image)
        async with hold_world.factory() as session:
            metadata = await LibraryRepository(session).get_metadata(
                user.id, "vex", ref.source, ref.asset_id
            )
        assert metadata is not None
        assert metadata.is_favorite is True  # the flushed purge went away with the rollback


# ---------------------------------------------------------------------------------------------
# (d) job soft-delete vs hold() of its output, both orders
# ---------------------------------------------------------------------------------------------


class TestJobSoftDeleteVsHold:
    async def test_the_soft_delete_wins_the_hold_gets_job_not_available(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job)
        racer = Racer(hold_world, lambda s: _hold(s, output_ref(output), user))
        try:
            async with hold_world.factory() as deleter:
                assert (
                    await JobRepository(deleter).soft_delete(job.id, user_id=user.id)
                    is SoftDeleteOutcome.DELETED
                )
                await racer.start()
                await _await_blocked(db_engine, racer)  # waiting for the job's share lock
                await deleter.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AssetNotEligibleError)
        assert outcome.reason is AssetIneligibleReason.JOB_NOT_AVAILABLE
        assert await hold_world.hold_count() == 0

    async def test_the_hold_wins_the_soft_delete_is_held(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job)
        racer = Racer(hold_world, lambda s: JobRepository(s).soft_delete(job.id, user_id=user.id))
        try:
            async with hold_world.factory() as holder:
                await _hold(holder, output_ref(output), user)  # job FOR SHARE, uncommitted
                await racer.start()
                await _await_blocked(db_engine, racer)  # soft-delete waits on FOR UPDATE
                await holder.commit()
        finally:
            outcome = await racer.outcome()

        # The soft-delete re-ran its hold check in a NEW statement after the lock: it sees the hold.
        assert outcome is SoftDeleteOutcome.HELD
        async with hold_world.factory() as session:
            assert (
                await session.execute(
                    select(GenerationJob.is_deleted).where(GenerationJob.id == job.id)
                )
            ).scalar_one() is False
        assert await hold_world.hold_count() == 1


# ---------------------------------------------------------------------------------------------
# (e) two releases of the last two holds of one asset: re-armed (not skipped by both)
# ---------------------------------------------------------------------------------------------


class TestConcurrentReleases:
    async def test_the_second_release_sees_the_first_ones_delete_and_rearms(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user, expires_at=past(hours=5))
        first = await _committed_hold(hold_world, upload_ref(image), user, kind="a.one")
        second = await _committed_hold(hold_world, upload_ref(image), user, kind="b.two")
        expired_at = await hold_world.upload_expires_at(image)

        racer = Racer(
            hold_world,
            lambda s: AssetHoldService(s, retention_days=RETENTION_DAYS).release([second]),
        )
        try:
            async with hold_world.factory() as releasing:
                await AssetHoldService(releasing, retention_days=RETENTION_DAYS).release([first])
                # the first release did not re-arm: the other hold is still there
                assert (
                    await releasing.execute(
                        select(UserImage.expires_at).where(UserImage.id == image.id)
                    )
                ).scalar_one() == expired_at
                await racer.start()
                await _await_blocked(db_engine, racer)  # serialized on the asset row
                await releasing.commit()
        finally:
            outcome = await racer.outcome()

        assert outcome is None
        # Without the asset lock both releases would have seen the other's hold: nothing re-armed.
        async with hold_world.factory() as session:
            db_now = (await session.execute(select(func.now()))).scalar_one()
        assert await hold_world.upload_expires_at(image) >= db_now - timedelta(
            minutes=1
        ) + timedelta(days=RETENTION_DAYS)
        assert await hold_world.hold_count() == 0


async def test_a_release_does_not_block_a_concurrent_hold_of_the_same_asset(
    hold_world: HoldWorld,
) -> None:
    """FOR NO KEY UPDATE (release) is compatible with FOR KEY SHARE (hold): holds never queue."""
    user = await hold_world.user()
    image = await hold_world.upload(user)
    existing = await _committed_hold(hold_world, upload_ref(image), user, kind="a.one")

    async with hold_world.factory() as releasing:
        await AssetHoldService(releasing, retention_days=RETENTION_DAYS).release([existing])
        async with hold_world.factory() as holding:
            hold_id = await asyncio.wait_for(_hold(holding, upload_ref(image), user), 5)
            await holding.commit()
        await releasing.commit()

    assert hold_id is not None
    # The release saw no hold left (the new one was uncommitted) and re-armed - harmless: the
    # hold is what keeps the asset from being swept, and ``greatest()`` never shortens a life.
    assert await hold_world.hold_count() == 1
