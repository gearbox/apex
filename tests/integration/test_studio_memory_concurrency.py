"""Overlapping memory requests: the 200-active cap is a hard rule, and PATCHes serialize (RD8).

``create`` and an un-archive lock the **artist** row first and count under that lock, so two
overlapping requests at 199 cannot both succeed; a PATCH takes artist, then memory, so overlapping
PATCHes of different list fields never lose each other's field. Lock order is artist, then memory.

Committed rows and separate sessions (the integration ``db_session`` is one rolled-back
transaction, which a second connection cannot see). Each test pauses request A while it holds the
artist lock, proves in ``pg_stat_activity`` that request B is waiting on it, then releases A.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.studio import ArtistCreate, ArtistPatch, MemoryCreate, MemoryPatch
from src.core.enums import StudioMemoryStatus
from src.db.models.studio import StudioArtistMemory
from src.db.models.user import User
from src.studio.constraints import MAX_ACTIVE_MEMORIES_PER_ARTIST
from src.studio.errors import MemoryLimitReached
from src.studio.services.artist import ArtistService
from src.studio.services.memory import MemoryService
from tests.integration.test_studio_artist_concurrency import _wait_until_blocked_on_a_lock

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncEngine

PRODUCT = "vex"
Factory = async_sessionmaker[AsyncSession]


class World:
    def __init__(self, factory: Factory, user_id: UUID, artist_id: UUID) -> None:
        self.factory = factory
        self.user_id = user_id
        self.artist_id = artist_id

    async def seed(self, *, active: int, archived: int = 0) -> list[UUID]:
        """Insert rows directly (fast); returns the ids of the archived ones."""
        rows = [
            {
                "id": uuid4(),
                "artist_id": self.artist_id,
                "title": f"m{n}",
                "body": "b",
                "status": "active" if n < active else "archived",
            }
            for n in range(active + archived)
        ]
        async with self.factory() as session, session.begin():
            await session.execute(insert(StudioArtistMemory), rows)
        return [row["id"] for row in rows if row["status"] == "archived"]

    async def count(self, status: StudioMemoryStatus = StudioMemoryStatus.ACTIVE) -> int:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(func.count())
                    .select_from(StudioArtistMemory)
                    .where(
                        StudioArtistMemory.artist_id == self.artist_id,
                        StudioArtistMemory.status == status.value,
                    )
                )
            ).scalar_one()

    async def row(self, memory_id: UUID) -> StudioArtistMemory:
        async with self.factory() as session:
            return (
                await session.execute(
                    select(StudioArtistMemory)
                    .where(StudioArtistMemory.id == memory_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one()


@pytest.fixture
async def world(db_engine: AsyncEngine) -> AsyncGenerator[World]:
    factory: Factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"memory-conc-{uuid4().hex}@example.com",
        password_hash="x",
        product_id=PRODUCT,
    )
    try:
        async with factory() as session, session.begin():
            session.add(user)
            await session.flush()
            artist = await ArtistService(session).create(
                user.id, PRODUCT, ArtistCreate(name="Concurrent")
            )
        yield World(factory, user.id, artist.id)
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(User).where(User.id == user.id))


class _Paused:
    """Request A: runs a service call, pausing at ``method`` of the memory repository - which is
    after the artist lock was taken and (for the cap) after the count."""

    def __init__(
        self,
        world: World,
        method: str,
        call: Callable[[MemoryService], Awaitable[object]],
    ) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.result: object = None
        self._world = world
        self._method = method
        self._call = call

    async def run(self) -> None:
        async with self._world.factory() as session:
            service = MemoryService(session)
            real = getattr(service._memories, self._method)

            async def paused(*args: Any, **kwargs: Any) -> Any:
                self.reached.set()
                await self.release.wait()
                return await real(*args, **kwargs)

            setattr(service._memories, self._method, paused)
            self.result = await self._call(service)
            await session.commit()


async def _once(world: World, call: Callable[[MemoryService], Awaitable[object]]) -> object:
    async with world.factory() as session:
        result = await call(MemoryService(session))
        await session.commit()
        return result


async def _overlap(
    db_engine: AsyncEngine,
    first: _Paused,
    second: Callable[[], Coroutine[Any, Any, object]],
) -> tuple[BaseException | None, BaseException | None]:
    """Run ``first`` (paused holding the artist lock), start ``second``, prove it blocks, release."""
    task_a = asyncio.create_task(first.run())
    await asyncio.wait_for(first.reached.wait(), 5)
    task_b = asyncio.create_task(second())
    with pytest.raises(TimeoutError):  # B cannot finish while A holds the artist lock ...
        await asyncio.wait_for(asyncio.shield(task_b), 0.5)
    await _wait_until_blocked_on_a_lock(db_engine)  # ... and the database agrees it is waiting
    assert not task_a.done()
    first.release.set()
    a, b = await asyncio.wait_for(asyncio.gather(task_a, task_b, return_exceptions=True), 10)
    return (
        a if isinstance(a, BaseException) else None,
        b if isinstance(b, BaseException) else None,
    )


def _create_call(world: World, title: str) -> Callable[[MemoryService], Awaitable[object]]:
    return lambda service: service.create(
        world.artist_id, world.user_id, PRODUCT, MemoryCreate(title=title, body="b")
    )


def _restore_call(world: World, memory_id: UUID) -> Callable[[MemoryService], Awaitable[object]]:
    return lambda service: service.patch(
        memory_id, world.user_id, PRODUCT, MemoryPatch(status=StudioMemoryStatus.ACTIVE)
    )


class TestTheCapIsHard:
    async def test_two_creates_at_199_exactly_one_succeeds(
        self, world: World, db_engine: AsyncEngine
    ) -> None:
        await world.seed(active=MAX_ACTIVE_MEMORIES_PER_ARTIST - 1)
        a = _Paused(world, "insert", _create_call(world, "first"))

        error_a, error_b = await _overlap(
            db_engine, a, lambda: _once(world, _create_call(world, "second"))
        )

        assert error_a is None  # A held the lock and took the 200th place
        assert isinstance(error_b, MemoryLimitReached)  # B counted under the lock after A committed
        assert await world.count() == MAX_ACTIVE_MEMORIES_PER_ARTIST

    async def test_two_restores_at_199_exactly_one_succeeds(
        self, world: World, db_engine: AsyncEngine
    ) -> None:
        archived = await world.seed(active=MAX_ACTIVE_MEMORIES_PER_ARTIST - 1, archived=2)
        a = _Paused(world, "update", _restore_call(world, archived[0]))

        error_a, error_b = await _overlap(
            db_engine, a, lambda: _once(world, _restore_call(world, archived[1]))
        )

        assert error_a is None
        assert isinstance(error_b, MemoryLimitReached)
        assert await world.count() == MAX_ACTIVE_MEMORIES_PER_ARTIST
        assert await world.count(StudioMemoryStatus.ARCHIVED) == 1

    async def test_a_create_and_a_restore_compete_for_the_last_place(
        self, world: World, db_engine: AsyncEngine
    ) -> None:
        archived = await world.seed(active=MAX_ACTIVE_MEMORIES_PER_ARTIST - 1, archived=1)
        a = _Paused(world, "insert", _create_call(world, "new"))

        error_a, error_b = await _overlap(
            db_engine, a, lambda: _once(world, _restore_call(world, archived[0]))
        )

        assert error_a is None
        assert isinstance(error_b, MemoryLimitReached)
        assert await world.count() == MAX_ACTIVE_MEMORIES_PER_ARTIST

    async def test_at_the_cap_a_create_is_refused_and_an_archive_makes_room(
        self, world: World
    ) -> None:
        await world.seed(active=MAX_ACTIVE_MEMORIES_PER_ARTIST)
        with pytest.raises(MemoryLimitReached):
            await _once(world, _create_call(world, "one too many"))
        async with world.factory() as session:
            first = (
                await session.execute(
                    select(StudioArtistMemory.id)
                    .where(StudioArtistMemory.artist_id == world.artist_id)
                    .limit(1)
                )
            ).scalar_one()
        await _once(
            world,
            lambda service: service.patch(
                first, world.user_id, PRODUCT, MemoryPatch(status=StudioMemoryStatus.ARCHIVED)
            ),
        )
        await _once(world, _create_call(world, "now there is room"))
        assert await world.count() == MAX_ACTIVE_MEMORIES_PER_ARTIST

    async def test_other_edits_of_a_full_artist_are_not_refused(self, world: World) -> None:
        """The cap counts new active memories only: editing an active one at 200 is fine."""
        await world.seed(active=MAX_ACTIVE_MEMORIES_PER_ARTIST)
        async with world.factory() as session:
            first = (
                await session.execute(
                    select(StudioArtistMemory.id)
                    .where(StudioArtistMemory.artist_id == world.artist_id)
                    .limit(1)
                )
            ).scalar_one()
        await _once(
            world,
            lambda service: service.patch(
                first,
                world.user_id,
                PRODUCT,
                MemoryPatch(title="renamed", status=StudioMemoryStatus.ACTIVE),
            ),
        )
        assert (await world.row(first)).title == "renamed"


class TestOverlappingPatches:
    async def test_patches_of_different_list_fields_both_survive(
        self, world: World, db_engine: AsyncEngine
    ) -> None:
        memory_id = await _once(
            world,
            lambda service: service.create(
                world.artist_id,
                world.user_id,
                PRODUCT,
                MemoryCreate(title="t", body="b", themes=["old-theme"], imagery=["old-image"]),
            ),
        )
        memory_id = memory_id.id  # type: ignore[attr-defined]
        a = _Paused(
            world,
            "update",
            lambda service: service.patch(
                memory_id, world.user_id, PRODUCT, MemoryPatch(themes=["new-theme"])
            ),
        )

        error_a, error_b = await _overlap(
            db_engine,
            a,
            lambda: _once(
                world,
                lambda service: service.patch(
                    memory_id, world.user_id, PRODUCT, MemoryPatch(imagery=["new-image"])
                ),
            ),
        )

        assert (error_a, error_b) == (None, None)  # both answered 200 ...
        row = await world.row(memory_id)
        assert list(row.themes) == ["new-theme"]  # ... and neither lost the other's field
        assert list(row.imagery) == ["new-image"]

    async def test_a_patch_after_a_committed_one_sees_its_result(self, world: World) -> None:
        created = await _once(world, _create_call(world, "t"))
        memory_id: UUID = created.id  # type: ignore[attr-defined]
        for patch in (
            MemoryPatch(themes=["t1"]),
            MemoryPatch(imagery=["i1"]),
            MemoryPatch(salience=5),
        ):
            await _once(
                world,
                lambda service, patch=patch: service.patch(
                    memory_id, world.user_id, PRODUCT, patch
                ),
            )
        row = await world.row(memory_id)
        assert (list(row.themes), list(row.imagery), row.salience) == (["t1"], ["i1"], 5)


class TestLockOrder:
    async def test_a_memory_patch_waits_for_the_artist_lock_then_both_finish(
        self, world: World, db_engine: AsyncEngine
    ) -> None:
        """An artist PATCH holds the artist row; a memory PATCH queues behind it on the *artist*
        (artist, then memory) - no deadlock, both complete."""
        created = await _once(world, _create_call(world, "t"))
        memory_id: UUID = created.id  # type: ignore[attr-defined]

        reached = asyncio.Event()
        release = asyncio.Event()

        async def artist_patch() -> None:
            async with world.factory() as session:
                service = ArtistService(session)
                real = service._repo.update

                async def paused(*args: Any, **kwargs: Any) -> Any:
                    reached.set()
                    await release.wait()
                    return await real(*args, **kwargs)

                service._repo.update = paused  # type: ignore[method-assign]
                await service.patch(
                    world.artist_id, world.user_id, PRODUCT, ArtistPatch(persona="p")
                )
                await session.commit()

        task_a = asyncio.create_task(artist_patch())
        await asyncio.wait_for(reached.wait(), 5)
        task_b = asyncio.create_task(
            _once(
                world,
                lambda service: service.patch(
                    memory_id, world.user_id, PRODUCT, MemoryPatch(title="renamed")
                ),
            )
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task_b), 0.5)
        await _wait_until_blocked_on_a_lock(db_engine)
        release.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b), 10)
        assert (await world.row(memory_id)).title == "renamed"

    async def test_a_delete_never_waits_for_the_artist(self, world: World) -> None:
        """Deleting is always allowed (P2-M5): it takes no artist lock, so it cannot queue behind
        one - nor be refused because the artist is busy."""
        created = await _once(world, _create_call(world, "t"))
        memory_id: UUID = created.id  # type: ignore[attr-defined]
        async with world.factory() as holder, holder.begin():
            from src.db.repositories.studio import StudioArtistRepository

            locked = await StudioArtistRepository(holder).lock(
                world.artist_id, user_id=world.user_id, product_id=PRODUCT
            )
            assert locked is not None
            await asyncio.wait_for(
                _once(world, lambda service: service.delete(memory_id, world.user_id, PRODUCT)), 5
            )
        async with world.factory() as session:
            assert (
                await session.execute(
                    select(func.count())
                    .select_from(StudioArtistMemory)
                    .where(StudioArtistMemory.id == memory_id)
                )
            ).scalar_one() == 0
