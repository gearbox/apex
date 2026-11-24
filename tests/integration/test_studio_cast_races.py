"""Cast races on committed rows (S1-P3, W10.4 (f) and W10.5's caps under concurrency).

Same discipline as ``test_library_holds_races.py``: the second request is proven blocked on a
specific backend, the lock-holding block exits before anything is gathered, and the loser's
outcome is a typed error (never a 500).

Lock order under test: artist → cast member → reference / look → assets.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from src.api.schemas.studio import (
    ArtistCreate,
    ArtistPatch,
    CastLookIn,
    CastMemberPatch,
)
from src.db.models.studio import StudioArtist
from src.db.models.studio_cast import StudioCastMember
from src.studio.constraints import MAX_CAST_LOOKS, MAX_CAST_REFERENCES
from src.studio.errors import CastLimitReached, CastMemberInUse, CastMemberNotFound
from src.studio.services.artist import ArtistService
from tests.cast_support import assert_reconciled, cast_service, create_body, ref_in
from tests.db_locks import backend_pid, wait_until_blocked
from tests.library_holds_support import HoldWorld, make_upload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User

PRODUCT = "vex"


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

    async def blocked(self, engine: AsyncEngine) -> None:
        assert self.task is not None
        await wait_until_blocked(engine, self.pid)
        assert not self.task.done()

    async def outcome(self) -> Any:
        assert self.task is not None
        (result,) = await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 15)
        return result


async def _member(world: HoldWorld, user: User, *, references: int = 0, name: str = "Mira") -> UUID:
    async with world.factory() as session:
        service = cast_service(session)
        assets = [await make_upload(session, user) for _ in range(references)]
        created = await service.create(
            user.id,
            PRODUCT,
            create_body(name, references=[ref_in(f"ref-{i}", a) for i, a in enumerate(assets)]),
        )
        await session.commit()
        return created.id


async def _artist(world: HoldWorld, user: User) -> UUID:
    async with world.factory() as session:
        artist = await ArtistService(session).create(user.id, PRODUCT, ArtistCreate(name="Band"))
        await session.commit()
        return artist.id


async def _performer(world: HoldWorld, artist_id: UUID) -> UUID | None:
    async with world.factory() as session:
        return (
            await session.execute(
                select(StudioArtist.performer_cast_member_id).where(StudioArtist.id == artist_id)
            )
        ).scalar_one()


async def _member_exists(world: HoldWorld, member_id: UUID) -> bool:
    async with world.factory() as session:
        return (
            await session.execute(
                select(StudioCastMember.id).where(StudioCastMember.id == member_id)
            )
        ).scalar_one_or_none() is not None


async def _reconciled(world: HoldWorld) -> None:
    async with world.factory() as session:
        await assert_reconciled(session)


# ---------------------------------------------------------------------------------------------
# (f) cast member delete vs artist PATCH setting it as performer, both orders
# ---------------------------------------------------------------------------------------------


class TestMemberDeleteVsPerformerLink:
    async def test_the_link_commits_first_so_the_delete_is_refused_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=2)
        artist_id = await _artist(hold_world, user)
        racer = Racer(hold_world, lambda s: cast_service(s).delete(member_id, user.id, PRODUCT))
        try:
            async with hold_world.factory() as linking:
                await ArtistService(linking).patch(
                    artist_id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=member_id)
                )
                await racer.start()
                await racer.blocked(db_engine)  # the delete's member lock waits for the link's FK
                await linking.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, CastMemberInUse)
        assert outcome.detail == {"blockers": [{"code": "artists", "count": 1}]}
        assert await _member_exists(hold_world, member_id)
        assert await _performer(hold_world, artist_id) == member_id
        assert await hold_world.hold_count() == 2  # nothing was released
        await _reconciled(hold_world)

    async def test_the_delete_commits_first_so_the_link_is_a_404(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=2)
        artist_id = await _artist(hold_world, user)
        racer = Racer(
            hold_world,
            lambda s: ArtistService(s).patch(
                artist_id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=member_id)
            ),
        )
        try:
            async with hold_world.factory() as deleting:
                await cast_service(deleting).delete(member_id, user.id, PRODUCT)
                await racer.start()
                await racer.blocked(db_engine)  # the artist's FK check waits for the member row
                await deleting.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, CastMemberNotFound)  # the FK said no; mapped, never a 500
        assert not await _member_exists(hold_world, member_id)
        assert await _performer(hold_world, artist_id) is None
        assert await hold_world.hold_count() == 0  # every hold went with the member
        await _reconciled(hold_world)

    async def test_the_foreign_key_backstop_turns_a_late_link_into_the_same_409(
        self, hold_world: HoldWorld
    ) -> None:
        # The member lock makes this window unreachable for a real PATCH; if the count ever
        # misses a link (here: forced to), the FK refuses the DELETE and the answer is the 409.
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=1)
        artist_id = await _artist(hold_world, user)
        async with hold_world.factory() as session:
            await ArtistService(session).patch(
                artist_id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=member_id)
            )
            await session.commit()

        async with hold_world.factory() as session:
            service = cast_service(session)
            real = service._members.count_performed_artists
            calls = {"n": 0}

            async def misses_the_link_once(*args: Any, **kwargs: Any) -> int:
                calls["n"] += 1
                return 0 if calls["n"] == 1 else await real(*args, **kwargs)

            service._members.count_performed_artists = misses_the_link_once  # type: ignore[method-assign]
            with pytest.raises(CastMemberInUse) as raised:
                await service.delete(member_id, user.id, PRODUCT)
            await session.rollback()

        assert raised.value.detail == {"blockers": [{"code": "artists", "count": 1}]}
        assert await _member_exists(hold_world, member_id)
        await _reconciled(hold_world)

    async def test_a_foreign_key_failure_nobody_can_explain_is_not_turned_into_a_409(
        self, hold_world: HoldWorld
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        user = await hold_world.user()
        member_id = await _member(hold_world, user)
        async with hold_world.factory() as session:
            service = cast_service(session)
            service._members.delete = AsyncMock(  # type: ignore[method-assign]
                side_effect=IntegrityError("stmt", {}, Exception("something else"))
            )
            with pytest.raises(IntegrityError):
                await service.delete(member_id, user.id, PRODUCT)


# ---------------------------------------------------------------------------------------------
# The caps are hard: two parallel adds at 11 -> one 409
# ---------------------------------------------------------------------------------------------


class TestCapsUnderConcurrency:
    async def test_two_parallel_reference_adds_at_eleven_one_wins_one_is_a_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=MAX_CAST_REFERENCES - 1)
        async with hold_world.factory() as session:
            first, second = (await make_upload(session, user), await make_upload(session, user))
            await session.commit()

        racer = Racer(
            hold_world,
            lambda s: cast_service(s).add_reference(
                member_id, user.id, PRODUCT, ref_in("second", second)
            ),
        )
        try:
            async with hold_world.factory() as winning:
                added = await cast_service(winning).add_reference(
                    member_id, user.id, PRODUCT, ref_in("first", first)
                )
                assert added.position == MAX_CAST_REFERENCES - 1
                await racer.start()
                await racer.blocked(db_engine)  # the count is taken under the member lock
                await winning.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, CastLimitReached)
        assert outcome.detail == {"kind": "references", "limit": MAX_CAST_REFERENCES}
        async with hold_world.factory() as session:
            detail = await cast_service(session).get(member_id, user.id, PRODUCT)
        assert len(detail.references) == MAX_CAST_REFERENCES
        assert await hold_world.hold_count() == MAX_CAST_REFERENCES  # the loser pinned nothing
        await _reconciled(hold_world)

    async def test_two_parallel_look_creations_at_eleven_one_wins_one_is_a_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=1)
        async with hold_world.factory() as session:
            service = cast_service(session)
            only = [(await service.get(member_id, user.id, PRODUCT)).references[0].id]
            for index in range(MAX_CAST_LOOKS - 1):
                await service.create_look(
                    member_id,
                    user.id,
                    PRODUCT,
                    CastLookIn(name=f"Look {index}", reference_ids=only),
                )
            await session.commit()

        racer = Racer(
            hold_world,
            lambda s: cast_service(s).create_look(
                member_id, user.id, PRODUCT, CastLookIn(name="Second", reference_ids=only)
            ),
        )
        try:
            async with hold_world.factory() as winning:
                await cast_service(winning).create_look(
                    member_id, user.id, PRODUCT, CastLookIn(name="First", reference_ids=only)
                )
                await racer.start()
                await racer.blocked(db_engine)
                await winning.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, CastLimitReached)
        assert outcome.detail == {"kind": "looks", "limit": MAX_CAST_LOOKS}
        async with hold_world.factory() as session:
            assert len((await cast_service(session).get(member_id, user.id, PRODUCT)).looks) == (
                MAX_CAST_LOOKS
            )

    async def test_deleting_a_reference_waits_for_a_look_being_built_on_it(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        # Both mutate the member: serialized, and the look ends up with the reference removed
        # (or the look creation is refused) - never a look that names a deleted reference.
        user = await hold_world.user()
        member_id = await _member(hold_world, user, references=2)
        async with hold_world.factory() as session:
            detail = await cast_service(session).get(member_id, user.id, PRODUCT)
        kept, doomed = (r.id for r in detail.references)

        racer = Racer(
            hold_world,
            lambda s: cast_service(s).delete_reference(member_id, doomed, user.id, PRODUCT),
        )
        try:
            async with hold_world.factory() as building:
                await cast_service(building).create_look(
                    member_id,
                    user.id,
                    PRODUCT,
                    CastLookIn(name="Look", reference_ids=[kept, doomed]),
                )
                await racer.start()
                await racer.blocked(db_engine)
                await building.commit()
        finally:
            outcome = await racer.outcome()

        assert outcome is None
        async with hold_world.factory() as session:
            after = await cast_service(session).get(member_id, user.id, PRODUCT)
        assert [lk.reference_ids for lk in after.looks] == [[kept]]
        await _reconciled(hold_world)


async def test_two_members_of_one_owner_do_not_block_each_other(hold_world: HoldWorld) -> None:
    """The member lock is per member: unrelated members mutate in parallel."""
    user = await hold_world.user()
    first = await _member(hold_world, user, references=1, name="First")
    second = await _member(hold_world, user, references=1, name="Second")
    async with hold_world.factory() as one, hold_world.factory() as two:
        await cast_service(one).patch(first, user.id, PRODUCT, CastMemberPatch(description="a"))
        # does not wait for ``one``'s uncommitted lock on a different member
        await asyncio.wait_for(
            cast_service(two).patch(second, user.id, PRODUCT, CastMemberPatch(description="b")), 5
        )
        await one.commit()
        await two.commit()
