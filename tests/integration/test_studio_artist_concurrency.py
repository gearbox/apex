"""RF2 / RD8: overlapping artist PATCHes serialize (invariant 34).

``ArtistService.patch`` is a read-modify-write: when any list field is in the request it
validates the merge against the stored lists and writes **all three** list columns. Without a row
lock two overlapping PATCHes of different list fields both returned 200 and the later writer
restored the other's field; and a tag added by one request plus the same entry added to the avoid
list by the other could persist a ``tag_conflict`` that validation forbids.

These tests use committed rows and separate sessions (the integration ``db_session`` runs in one
rolled-back transaction, which a second connection cannot see).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.studio import ArtistCreate, ArtistPatch
from src.db.models.studio import StudioArtist
from src.db.models.user import User
from src.studio.errors import StudioDocumentInvalid
from src.studio.services.artist import ArtistService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncEngine

PRODUCT = "vex"

Factory = async_sessionmaker[AsyncSession]


@pytest.fixture
async def committed_artist(
    db_engine: AsyncEngine,
) -> AsyncGenerator[tuple[Factory, UUID, UUID]]:
    """A committed user and artist (``old-tag`` / ``old-avoid`` / ``old-ref``); cleaned up after."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"studio-artist-conc-{uuid4().hex}@example.com",
        password_hash="x",
        product_id=PRODUCT,
    )
    try:
        async with factory() as session, session.begin():
            session.add(user)
            await session.flush()
            artist = await ArtistService(session).create(
                user.id,
                PRODUCT,
                ArtistCreate(
                    name="Concurrent",
                    style_tags=["old-tag"],
                    global_avoid=["old-avoid"],
                    aesthetic_references=["old-ref"],
                ),
            )
        yield factory, user.id, artist.id
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(User).where(User.id == user.id))


async def _stored(factory: Factory, artist_id: UUID) -> dict[str, list[str]]:
    async with factory() as session:
        row = (
            await session.execute(
                select(
                    StudioArtist.style_tags,
                    StudioArtist.global_avoid,
                    StudioArtist.aesthetic_references,
                ).where(StudioArtist.id == artist_id)
            )
        ).one()
    return {
        "style_tags": list(row.style_tags),
        "global_avoid": list(row.global_avoid),
        "aesthetic_references": list(row.aesthetic_references),
    }


class _PausedPatch:
    """Request A: acquires the artist lock, then pauses just before its UPDATE."""

    def __init__(
        self, factory: Factory, user_id: UUID, artist_id: UUID, patch: ArtistPatch
    ) -> None:
        self.reached_update = asyncio.Event()
        self.release = asyncio.Event()
        self._factory = factory
        self._args = (artist_id, user_id, PRODUCT, patch)

    async def run(self) -> None:
        async with self._factory() as session:
            service = ArtistService(session)
            real_update = service._repo.update

            async def paused_update(*args: Any, **kwargs: Any) -> Any:
                self.reached_update.set()
                await self.release.wait()
                return await real_update(*args, **kwargs)

            service._repo.update = paused_update  # type: ignore[method-assign]
            await service.patch(*self._args)
            await session.commit()


async def _patch_and_commit(
    factory: Factory, user_id: UUID, artist_id: UUID, patch: ArtistPatch
) -> None:
    async with factory() as session:
        await ArtistService(session).patch(artist_id, user_id, PRODUCT, patch)
        await session.commit()


async def _wait_until_blocked_on_a_lock(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as probe:
        for _ in range(500):
            waiting = (
                await probe.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                    )
                )
            ).scalar_one()
            if waiting:
                return
            await asyncio.sleep(0.01)
    pytest.fail("the second PATCH never blocked on the artist row lock")


async def _overlap(
    db_engine: AsyncEngine,
    first: _PausedPatch,
    second: Callable[[], Coroutine[Any, Any, None]],
) -> tuple[BaseException | None, BaseException | None]:
    """Run ``first`` (paused holding the lock), start ``second``, prove it blocks, release both.

    Returns:
        The exceptions (or ``None``) of ``first`` and ``second``, in that order.
    """
    task_a = asyncio.create_task(first.run())
    await asyncio.wait_for(first.reached_update.wait(), 5)
    task_b = asyncio.create_task(second())

    # B is blocked behind A's lock: it cannot finish while A holds it ...
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task_b), 0.5)
    await _wait_until_blocked_on_a_lock(db_engine)  # ... and the database agrees it is waiting
    assert not task_a.done()

    first.release.set()
    result_a, result_b = await asyncio.wait_for(
        asyncio.gather(task_a, task_b, return_exceptions=True), 10
    )
    return (
        result_a if isinstance(result_a, BaseException) else None,
        result_b if isinstance(result_b, BaseException) else None,
    )


class TestOverlappingPatches:
    async def test_patches_of_different_list_fields_both_survive(
        self, committed_artist: tuple[Factory, UUID, UUID], db_engine: AsyncEngine
    ) -> None:
        factory, user_id, artist_id = committed_artist
        a = _PausedPatch(factory, user_id, artist_id, ArtistPatch(style_tags=["new-tag"]))

        error_a, error_b = await _overlap(
            db_engine,
            a,
            lambda: _patch_and_commit(
                factory, user_id, artist_id, ArtistPatch(aesthetic_references=["new-ref"])
            ),
        )

        assert (error_a, error_b) == (None, None)  # both answered 200 ...
        stored = await _stored(factory, artist_id)
        # ... and neither lost the other's field (the probe used to end with 'old-ref').
        assert stored["style_tags"] == ["new-tag"]
        assert stored["aesthetic_references"] == ["new-ref"]
        assert stored["global_avoid"] == ["old-avoid"]

    async def test_the_waiting_patch_validates_against_the_committed_result(
        self, committed_artist: tuple[Factory, UUID, UUID], db_engine: AsyncEngine
    ) -> None:
        # A adds tag `x`; B adds avoid `x`. With serialization exactly one of them can win: B
        # re-reads the lists after A commits and sees the conflict. Validating both against the
        # *old* lists would persist a tag_conflict that validation forbids.
        factory, user_id, artist_id = committed_artist
        a = _PausedPatch(factory, user_id, artist_id, ArtistPatch(style_tags=["x"]))

        error_a, error_b = await _overlap(
            db_engine,
            a,
            lambda: _patch_and_commit(factory, user_id, artist_id, ArtistPatch(global_avoid=["X"])),
        )

        assert error_a is None
        assert isinstance(error_b, StudioDocumentInvalid)
        assert {(i.path, i.code) for i in error_b.issues} == {("global_avoid[0]", "tag_conflict")}
        stored = await _stored(factory, artist_id)
        assert stored["style_tags"] == ["x"]
        assert stored["global_avoid"] == ["old-avoid"]  # B's rejected write left no trace
        assert {t.casefold() for t in stored["style_tags"]}.isdisjoint(
            {t.casefold() for t in stored["global_avoid"]}
        )

    async def test_a_patch_of_a_scalar_field_waits_too(
        self, committed_artist: tuple[Factory, UUID, UUID], db_engine: AsyncEngine
    ) -> None:
        # The lock is taken before anything is read, whichever fields the request carries.
        factory, user_id, artist_id = committed_artist
        a = _PausedPatch(factory, user_id, artist_id, ArtistPatch(style_tags=["new-tag"]))

        error_a, error_b = await _overlap(
            db_engine,
            a,
            lambda: _patch_and_commit(factory, user_id, artist_id, ArtistPatch(persona="renamed")),
        )

        assert (error_a, error_b) == (None, None)
        async with factory() as session:
            persona = (
                await session.execute(
                    select(StudioArtist.persona).where(StudioArtist.id == artist_id)
                )
            ).scalar_one()
        assert persona == "renamed"
        assert (await _stored(factory, artist_id))["style_tags"] == ["new-tag"]

    async def test_a_patch_after_a_committed_one_sees_its_result(
        self, committed_artist: tuple[Factory, UUID, UUID]
    ) -> None:
        # Sequential control: no overlap, no waiting, the merge is simply cumulative.
        factory, user_id, artist_id = committed_artist
        await _patch_and_commit(factory, user_id, artist_id, ArtistPatch(style_tags=["t"]))
        await _patch_and_commit(factory, user_id, artist_id, ArtistPatch(global_avoid=["g"]))
        await _patch_and_commit(
            factory, user_id, artist_id, ArtistPatch(aesthetic_references=["r"])
        )
        assert await _stored(factory, artist_id) == {
            "style_tags": ["t"],
            "global_avoid": ["g"],
            "aesthetic_references": ["r"],
        }
