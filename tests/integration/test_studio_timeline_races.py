"""Realized timeline races on committed rows (S3a W9.2 / W9.9).

Same discipline as ``test_studio_track_audio_races.py``: the first request does its work and holds
its transaction open, the second is proven blocked on **that backend** (``tests.db_locks``), the
first commits and leaves its block, and only then is the second gathered. The loser's outcome is a
typed error or a consistent state: never a 500 and never a dangling pointer. Every test ends with
the consistency check over every timeline row and the audio-hold reconciliation.

======  =====================================================  ====================================
race    what collides                                          test class
======  =====================================================  ====================================
(a)     timeline create vs audio delete (both orders)          ``TestCreateVsAudioDelete``
(b)     timeline create vs a link-changing PATCH (both)        ``TestCreateVsLinkPatch``
(c)     ``PUT current-timeline`` vs deleting it (both)         ``TestSetCurrentVsDelete``
(d)     two creates, same parent and content                   ``TestTwinCreates``
cap     two creates at 49 -> exactly one 409; numbers          ``TestCapAndNumbering``
more    timeline delete vs audio delete (both orders)          ``TestTimelineDeleteVsAudioDelete``
more    two deletes of the same timeline                       ``TestTwinDeletes``
more    timeline create vs track delete (both orders)          ``TestCreateVsTrackDelete``
======  =====================================================  ====================================

Lock order under test: track → audio version → timeline.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select

from src.api.schemas.studio import TimelineCreate, TrackAudioPatch
from src.db.models.studio import StudioAudioTimeline, StudioTrackAudio
from src.studio.constraints import MAX_TIMELINES_PER_AUDIO
from src.studio.errors import (
    AudioHasTimelines,
    AudioVersionNotFound,
    StudioDocumentInvalid,
    StudioNotFound,
    TimelineIsCurrent,
    TimelineLimitReached,
    TimelineNotFound,
)
from src.studio.services.track import TrackService
from tests.library_holds_support import make_holds
from tests.timeline_support import (
    assert_timelines_consistent,
    committed_song,
    create_timeline,
    timeline_service,
)
from tests.track_audio_support import Racer, assert_audio_reconciled, audio_service

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.studio.timeline import TimelineDocument
    from tests.library_holds_support import HoldWorld
    from tests.timeline_support import Song

PRODUCT = "vex"


@pytest.fixture(autouse=True)
async def _every_row_stays_consistent(hold_world: HoldWorld) -> AsyncGenerator[None]:
    yield
    async with hold_world.factory() as session:
        await assert_timelines_consistent(session)
        await assert_audio_reconciled(session)


@pytest.fixture
async def song(hold_world: HoldWorld) -> Song:
    return await committed_song(hold_world, await hold_world.user())


# -- the two halves of a race ----------------------------------------------------------------------


type Work = Callable[[AsyncSession], Awaitable[Any]]


async def _race(world: HoldWorld, engine: AsyncEngine, *, first: Work, second: Work) -> Any:
    """Run ``first`` and keep its transaction open; prove ``second`` is blocked by it; commit.

    Returns the second request's result, or the exception it raised.
    """
    racer = Racer(world, second)
    try:
        async with world.factory() as holder:
            await first(holder)
            await racer.start()
            await racer.blocked(engine)
            await holder.commit()
    finally:
        outcome = await racer.outcome() if racer.task is not None else None
    return outcome


def _create(
    song: Song,
    document: TimelineDocument,
    *,
    parent_id: UUID | None = None,
    make_current: bool = False,
) -> Work:
    async def work(session: AsyncSession) -> Any:
        return await timeline_service(session).create(
            song.audio_id,
            song.user.id,
            PRODUCT,
            TimelineCreate(content=document, parent_id=parent_id, make_current=make_current),
        )

    return work


def _delete_timeline(song: Song, timeline_id: UUID) -> Work:
    async def work(session: AsyncSession) -> None:
        await timeline_service(session).delete(timeline_id, song.user.id, PRODUCT)

    return work


def _set_current(song: Song, timeline_id: UUID) -> Work:
    async def work(session: AsyncSession) -> Any:
        return await timeline_service(session).set_current(
            song.audio_id, song.user.id, PRODUCT, timeline_id
        )

    return work


def _delete_audio(song: Song) -> Work:
    async def work(session: AsyncSession) -> None:
        await audio_service(session).delete(song.audio_id, song.user.id, PRODUCT)

    return work


def _drop_lyrics(song: Song) -> Work:
    async def work(session: AsyncSession) -> Any:
        return await audio_service(session).patch(
            song.audio_id, song.user.id, PRODUCT, TrackAudioPatch(lyrics_version_id=None)
        )

    return work


def _delete_track(song: Song) -> Work:
    async def work(session: AsyncSession) -> None:
        await TrackService(session, make_holds(session)).delete(
            song.track_id, song.user.id, PRODUCT
        )

    return work


# -- state -----------------------------------------------------------------------------------------


async def _timeline_ids(world: HoldWorld, audio_id: UUID) -> list[UUID]:
    async with world.factory() as session:
        return list(
            (
                await session.execute(
                    select(StudioAudioTimeline.id)
                    .where(StudioAudioTimeline.audio_version_id == audio_id)
                    .order_by(StudioAudioTimeline.version_number)
                )
            )
            .scalars()
            .all()
        )


async def _numbers(world: HoldWorld, audio_id: UUID) -> list[int]:
    async with world.factory() as session:
        return list(
            (
                await session.execute(
                    select(StudioAudioTimeline.version_number)
                    .where(StudioAudioTimeline.audio_version_id == audio_id)
                    .order_by(StudioAudioTimeline.version_number)
                )
            )
            .scalars()
            .all()
        )


async def _audio_row(world: HoldWorld, audio_id: UUID) -> StudioTrackAudio | None:
    async with world.factory() as session:
        return await session.get(StudioTrackAudio, audio_id)


# -- (a) timeline create vs audio delete -----------------------------------------------------------


class TestCreateVsAudioDelete:
    async def test_the_create_commits_first_so_the_delete_takes_the_new_timeline(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_create(song, song.complete(), make_current=True),
            second=_delete_audio(song),
        )

        assert outcome is None, outcome
        assert await _audio_row(hold_world, song.audio_id) is None
        assert await _timeline_ids(hold_world, song.audio_id) == []

    async def test_the_delete_commits_first_so_the_create_is_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_audio(song),
            second=_create(song, song.complete(), make_current=True),
        )

        assert isinstance(outcome, AudioVersionNotFound), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == []


# -- (b) timeline create vs a link-changing PATCH --------------------------------------------------


class TestCreateVsLinkPatch:
    async def test_the_create_commits_first_so_the_patch_finds_the_links_frozen(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_create(song, song.complete()),
            second=_drop_lyrics(song),
        )

        assert isinstance(outcome, AudioHasTimelines), outcome
        assert outcome.detail == {"count": 1}
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.lyrics_version_id == song.lyrics_version_id

    async def test_the_patch_commits_first_so_the_create_validates_against_the_new_links(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        """Its lines are unknown once the lyrics are gone: a create that had read the links
        before taking the locks would have stored them."""
        outcome = await _race(
            hold_world,
            db_engine,
            first=_drop_lyrics(song),
            second=_create(song, song.complete()),
        )

        assert isinstance(outcome, StudioDocumentInvalid), outcome
        assert {issue.code for issue in outcome.issues} == {"unknown_line"}
        assert await _timeline_ids(hold_world, song.audio_id) == []

    async def test_the_patch_commits_first_and_completeness_follows_the_new_links(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        """Sections only: incomplete against the old links, complete against the new ones."""
        outcome = await _race(
            hold_world,
            db_engine,
            first=_drop_lyrics(song),
            second=_create(song, song.sections_only()),
        )

        timeline, created = outcome
        assert created is True
        assert timeline.is_complete is True
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.lyrics_version_id is None


# -- (c) PUT current-timeline vs deleting that timeline --------------------------------------------


class TestSetCurrentVsDelete:
    async def test_the_pointer_commits_first_so_the_delete_is_refused(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        timeline, _ = await create_timeline(hold_world, song, song.complete())

        outcome = await _race(
            hold_world,
            db_engine,
            first=_set_current(song, timeline.id),
            second=_delete_timeline(song, timeline.id),
        )

        assert isinstance(outcome, TimelineIsCurrent), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == [timeline.id]
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.current_timeline_id == timeline.id

    async def test_the_delete_commits_first_so_the_pointer_is_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        timeline, _ = await create_timeline(hold_world, song, song.complete())

        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_timeline(song, timeline.id),
            second=_set_current(song, timeline.id),
        )

        assert isinstance(outcome, TimelineNotFound), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == []
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.current_timeline_id is None


# -- (d) two creates with the same parent and content ----------------------------------------------


class TestTwinCreates:
    @pytest.mark.parametrize("with_parent", [False, True], ids=["roots", "children"])
    async def test_one_creates_and_the_other_replays_it(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song, with_parent: bool
    ) -> None:
        parent_id = None
        if with_parent:
            parent, _ = await create_timeline(hold_world, song, song.sections_only())
            parent_id = parent.id
        winner: list[Any] = []

        async def first(session: AsyncSession) -> None:
            winner.append(await _create(song, song.complete(), parent_id=parent_id)(session))

        outcome = await _race(
            hold_world,
            db_engine,
            first=first,
            second=_create(song, song.complete(), parent_id=parent_id, make_current=True),
        )

        ((created_out, created),) = winner
        replay_out, replay_created = outcome
        assert (created, replay_created) == (True, False)
        assert replay_out.id == created_out.id
        assert replay_out.version_number == created_out.version_number
        assert replay_out.is_current is True  # the replay still applied make_current
        expected = [1, 2] if with_parent else [1]
        assert await _numbers(hold_world, song.audio_id) == expected
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.next_timeline_number == len(expected) + 1


# -- the cap and the numbers -----------------------------------------------------------------------


class TestCapAndNumbering:
    async def test_two_creates_at_49_end_in_exactly_one_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        async with hold_world.factory() as session:
            for step in range(MAX_TIMELINES_PER_AUDIO - 1):
                await _create(song, song.variant(step))(session)
            await session.commit()

        outcome = await _race(
            hold_world,
            db_engine,
            first=_create(song, song.variant(100)),
            second=_create(song, song.variant(200)),
        )

        assert isinstance(outcome, TimelineLimitReached), outcome
        assert await _numbers(hold_world, song.audio_id) == list(range(1, 51))
        audio = await _audio_row(hold_world, song.audio_id)
        assert audio is not None
        assert audio.next_timeline_number == 51  # the refused create burned no number

    async def test_the_loser_of_the_track_lock_takes_the_next_number(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_create(song, song.variant(1)),
            second=_create(song, song.variant(2)),
        )

        second, created = outcome
        assert (created, second.version_number) == (True, 2)
        assert await _numbers(hold_world, song.audio_id) == [1, 2]

    async def test_numbers_are_gap_free_when_many_creates_arrive_at_once(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        async def one(step: int) -> int:
            out, _ = await create_timeline(hold_world, song, song.variant(step))
            return out.version_number

        numbers = await asyncio.gather(*(one(step) for step in range(8)))

        assert sorted(numbers) == list(range(1, 9))
        assert await _numbers(hold_world, song.audio_id) == list(range(1, 9))


# -- further pairs the cascade makes worth proving -------------------------------------------------


class TestTimelineDeleteVsAudioDelete:
    async def test_the_timeline_delete_commits_first(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        timeline, _ = await create_timeline(hold_world, song, song.complete())

        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_timeline(song, timeline.id),
            second=_delete_audio(song),
        )

        assert outcome is None, outcome
        assert await _audio_row(hold_world, song.audio_id) is None

    async def test_the_audio_delete_commits_first_so_the_timeline_is_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        timeline, _ = await create_timeline(hold_world, song, song.complete())

        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_audio(song),
            second=_delete_timeline(song, timeline.id),
        )

        assert isinstance(outcome, TimelineNotFound), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == []


class TestTwinDeletes:
    async def test_the_second_delete_finds_it_gone_after_waiting_for_the_locks(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        """It resolved the timeline before the first delete committed, so it must read it again
        under the locks."""
        kept, _ = await create_timeline(hold_world, song, song.complete())
        timeline, _ = await create_timeline(hold_world, song, song.sections_only())

        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_timeline(song, timeline.id),
            second=_delete_timeline(song, timeline.id),
        )

        assert isinstance(outcome, TimelineNotFound), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == [kept.id]


class TestCreateVsTrackDelete:
    async def test_the_create_commits_first_so_the_track_delete_cascades_it(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_create(song, song.complete(), make_current=True),
            second=_delete_track(song),
        )

        assert outcome is None, outcome
        assert await _timeline_ids(hold_world, song.audio_id) == []

    async def test_the_track_delete_commits_first_so_the_create_is_not_found(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        outcome = await _race(
            hold_world,
            db_engine,
            first=_delete_track(song),
            second=_create(song, song.complete()),
        )

        assert isinstance(outcome, (AudioVersionNotFound, StudioNotFound)), outcome
        assert await _timeline_ids(hold_world, song.audio_id) == []
