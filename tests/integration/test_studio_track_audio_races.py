"""Track audio races on committed rows (S2a W11.6 and the cap / numbering under concurrency).

Same discipline as ``test_studio_cast_races.py``: the second request is proven blocked on a
specific backend (``tests.db_locks``), the lock-holding block exits before anything is gathered,
and the loser's outcome is a typed error - never a 500, never an orphan hold. Every test ends with
the reconciliation check.

======  ==========================================================  ===========================
race    what collides                                               test class
======  ==========================================================  ===========================
(a)     attach vs library delete of the same asset (both orders)    ``TestAttachVsLibraryDelete``
(b)     track delete vs attach (both orders)                        ``TestTrackDeleteVsAttach``
(c)     audio delete vs ``PUT current`` at it (both orders)         ``TestAudioDeleteVsPutCurrent``
(d)     document-version delete vs attach linking it (both orders)  ``TestVersionDeleteVsAttach``
cap     two parallel attaches at 19 -> one 409                      ``TestCapAndNumbering``
======  ==========================================================  ===========================

Lock order under test: track → audio version → assets (inside the hold service).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from sqlalchemy import select

from src.api.schemas.studio import PlanVersionCreate
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import LibraryService
from src.core.enums import AssetDeleteOutcome, StudioDocumentOrigin
from src.core.enums import StudioSectionKind as K
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.db.models.studio import StudioTrack, StudioTrackAudio
from src.studio.constraints import MAX_AUDIO_VERSIONS_PER_TRACK
from src.studio.errors import (
    AssetNotFound,
    AudioVersionIsCurrent,
    AudioVersionLimitReached,
    AudioVersionNotFound,
    StudioNotFound,
    VersionHasAudio,
    VersionNotFound,
)
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService
from tests.library_holds_support import HoldWorld, make_holds
from tests.studio_support import make_plan, make_section
from tests.track_audio_support import (
    Racer,
    assert_audio_reconciled,
    attach_body,
    audio_service,
    committed_track,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.storage import UserImage
    from src.db.models.user import User

PRODUCT = "vex"


def _content_proxy() -> ContentProxyService:
    storage = MagicMock()
    storage.delete = AsyncMock()
    return ContentProxyService(storage=storage, settings=MagicMock(content_url_ttl=60))


async def _audio(world: HoldWorld, user: User, **kwargs: object) -> UserImage:
    return await world.audio(user, **kwargs)


async def _attach(
    world: HoldWorld, user: User, track_id: UUID, asset: UserImage, **links: UUID | None
) -> UUID:
    async with world.factory() as session:
        out = await audio_service(session).create(
            track_id, user.id, PRODUCT, attach_body(asset, **links)
        )
        await session.commit()
        return out.id


async def _set_current(world: HoldWorld, user: User, track_id: UUID, audio_id: UUID | None) -> None:
    async with world.factory() as session:
        await DocumentVersionService(session).set_current(
            track_id,
            user.id,
            PRODUCT,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=audio_id,
        )
        await session.commit()


async def _plan(world: HoldWorld, user: User, track_id: UUID, title: str) -> UUID:
    async with world.factory() as session:
        created, _ = await DocumentVersionService(session).create(
            track_id,
            _plan_body(title),
            user.id,
            PRODUCT,
            origin=StudioDocumentOrigin.USER,
        )
        await session.commit()
        return created.version.id


def _plan_body(title: str) -> PlanVersionCreate:
    return PlanVersionCreate(
        content=make_plan(make_section(K.VERSE), title=title), make_current=False
    )


async def _audio_rows(world: HoldWorld, track_id: UUID) -> list[int]:
    async with world.factory() as session:
        return list(
            (
                await session.execute(
                    select(StudioTrackAudio.version_number)
                    .where(StudioTrackAudio.track_id == track_id)
                    .order_by(StudioTrackAudio.version_number)
                )
            )
            .scalars()
            .all()
        )


async def _track_exists(world: HoldWorld, track_id: UUID) -> bool:
    async with world.factory() as session:
        return (
            await session.execute(select(StudioTrack.id).where(StudioTrack.id == track_id))
        ).scalar_one_or_none() is not None


async def _counter(world: HoldWorld, track_id: UUID) -> int:
    async with world.factory() as session:
        return (
            await session.execute(
                select(StudioTrack.next_audio_version_number).where(StudioTrack.id == track_id)
            )
        ).scalar_one()


async def _reconciled(world: HoldWorld) -> None:
    async with world.factory() as session:
        await assert_audio_reconciled(session)


# ---------------------------------------------------------------------------------------------
# (a) attach vs library delete of the same asset, both orders
# ---------------------------------------------------------------------------------------------


class TestAttachVsLibraryDelete:
    async def _library_delete(
        self, session: AsyncSession, user: User, asset: UserImage
    ) -> AssetDeleteOutcome:
        return await LibraryService(session=session).delete_asset(
            format_asset_ref(LibraryAssetSource.UPLOAD, asset.id),
            user.id,
            PRODUCT,
            session=session,
            content_proxy=_content_proxy(),
        )

    async def test_the_attach_commits_first_so_the_library_delete_is_refused_as_held(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        asset = await _audio(hold_world, user)
        racer = Racer(hold_world, lambda s: self._library_delete(s, user, asset))
        try:
            async with hold_world.factory() as attaching:
                await audio_service(attaching).create(
                    track_id, user.id, PRODUCT, attach_body(asset)
                )
                await racer.start()
                await racer.blocked(db_engine)  # the DELETE waits for the hold's KEY SHARE
                await attaching.commit()
        finally:
            outcome = await racer.outcome()

        assert outcome is AssetDeleteOutcome.HELD  # the foreign key said no; mapped, never a 500
        assert await hold_world.upload_exists(asset)
        assert await _audio_rows(hold_world, track_id) == [1]
        await _reconciled(hold_world)

    async def test_the_library_delete_commits_first_so_the_attach_is_a_404_and_burns_no_number(
        self, hold_world: HoldWorld
    ) -> None:
        """The library delete commits on its own (``ContentProxyService.delete_content``), so this
        order cannot be held open: the delete is finished when the attach arrives."""
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        asset = await _audio(hold_world, user)
        async with hold_world.factory() as deleting:
            outcome = await self._library_delete(deleting, user, asset)
        assert outcome is AssetDeleteOutcome.DELETED

        async with hold_world.factory() as attaching:
            with pytest.raises(AssetNotFound):
                await audio_service(attaching).create(
                    track_id, user.id, PRODUCT, attach_body(asset)
                )
            await attaching.rollback()

        assert not await hold_world.upload_exists(asset)
        assert await _audio_rows(hold_world, track_id) == []
        assert await hold_world.hold_count() == 0
        assert await _counter(hold_world, track_id) == 1  # the refused attach burned no number
        await _reconciled(hold_world)

    async def test_a_delete_waits_for_an_attach_and_proceeds_when_the_attach_rolls_back(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        """An attach in flight pins the asset row (its hold takes ``FOR KEY SHARE``): a delete that
        arrives waits for it, and once the attach is rolled back nothing holds the asset any more."""
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        asset = await _audio(hold_world, user)
        racer = Racer(hold_world, lambda s: self._library_delete(s, user, asset))
        try:
            async with hold_world.factory() as attaching:
                await audio_service(attaching).create(
                    track_id, user.id, PRODUCT, attach_body(asset)
                )
                await racer.start()
                await racer.blocked(db_engine)
                await attaching.rollback()  # the attach gives up: the delete may proceed
        finally:
            outcome = await racer.outcome()

        assert outcome is AssetDeleteOutcome.DELETED  # nothing held it any more
        assert await hold_world.hold_count() == 0
        await _reconciled(hold_world)


# ---------------------------------------------------------------------------------------------
# (b) track delete vs attach, both orders
# ---------------------------------------------------------------------------------------------


def _tracks(session: AsyncSession) -> TrackService:
    return TrackService(session, make_holds(session))


class TestTrackDeleteVsAttach:
    async def test_the_attach_commits_first_so_the_delete_releases_its_hold_too(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        existing = await _attach(hold_world, user, track_id, await _audio(hold_world, user))
        asset = await _audio(hold_world, user)
        racer = Racer(hold_world, lambda s: _tracks(s).delete(track_id, user.id, PRODUCT))
        try:
            async with hold_world.factory() as attaching:
                added = await audio_service(attaching).create(
                    track_id, user.id, PRODUCT, attach_body(asset)
                )
                assert added.version_number == 2
                await racer.start()
                await racer.blocked(db_engine)  # the delete waits for the attach's track lock
                await attaching.commit()
        finally:
            outcome = await racer.outcome()

        assert outcome is None  # the delete saw the new version and released its hold as well
        assert not await _track_exists(hold_world, track_id)
        assert await hold_world.hold_count() == 0
        assert existing is not None
        await _reconciled(hold_world)

    async def test_the_delete_commits_first_so_the_attach_is_a_404_and_pins_nothing(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        asset = await _audio(hold_world, user)
        racer = Racer(
            hold_world,
            lambda s: audio_service(s).create(track_id, user.id, PRODUCT, attach_body(asset)),
        )
        try:
            async with hold_world.factory() as deleting:
                await _tracks(deleting).delete(track_id, user.id, PRODUCT)
                await racer.start()
                await racer.blocked(db_engine)  # the attach's track lock waits for the delete
                await deleting.commit()
        finally:
            attach = await racer.outcome()

        assert isinstance(attach, StudioNotFound)
        assert await hold_world.hold_count() == 0
        assert await hold_world.upload_exists(asset)  # nothing was pinned or lost
        await _reconciled(hold_world)


# ---------------------------------------------------------------------------------------------
# (c) audio delete vs PUT current pointing at it, both orders
# ---------------------------------------------------------------------------------------------


class TestAudioDeleteVsPutCurrent:
    async def test_the_pointer_commits_first_so_the_delete_is_refused_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        audio_id = await _attach(hold_world, user, track_id, await _audio(hold_world, user))
        racer = Racer(hold_world, lambda s: audio_service(s).delete(audio_id, user.id, PRODUCT))
        try:
            async with hold_world.factory() as switching:
                await DocumentVersionService(switching).set_current(
                    track_id,
                    user.id,
                    PRODUCT,
                    plan_version_id=msgspec.UNSET,
                    lyrics_version_id=msgspec.UNSET,
                    audio_version_id=audio_id,
                )
                await racer.start()
                await racer.blocked(db_engine)  # the delete waits for the pointer's track lock
                await switching.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AudioVersionIsCurrent)
        assert await _audio_rows(hold_world, track_id) == [1]
        assert await hold_world.hold_count() == 1
        await _reconciled(hold_world)

    async def test_the_delete_commits_first_so_the_pointer_is_a_404_and_stays_empty(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        audio_id = await _attach(hold_world, user, track_id, await _audio(hold_world, user))
        racer = Racer(
            hold_world,
            lambda s: DocumentVersionService(s).set_current(
                track_id,
                user.id,
                PRODUCT,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=audio_id,
            ),
        )
        try:
            async with hold_world.factory() as deleting:
                await audio_service(deleting).delete(audio_id, user.id, PRODUCT)
                await racer.start()
                await racer.blocked(db_engine)  # the pointer's track lock waits for the delete
                await deleting.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AudioVersionNotFound)
        assert await _audio_rows(hold_world, track_id) == []
        assert await hold_world.hold_count() == 0
        async with hold_world.factory() as session:
            pointer = (
                await session.execute(
                    select(StudioTrack.current_audio_version_id).where(StudioTrack.id == track_id)
                )
            ).scalar_one()
        assert pointer is None
        await _reconciled(hold_world)


# ---------------------------------------------------------------------------------------------
# (d) document-version delete vs an attach that links that version, both orders
# ---------------------------------------------------------------------------------------------


class TestVersionDeleteVsAttach:
    async def test_the_attach_commits_first_so_the_version_delete_is_refused_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        plan_id = await _plan(hold_world, user, track_id, "Realized")
        asset = await _audio(hold_world, user)
        racer = Racer(
            hold_world, lambda s: DocumentVersionService(s).delete(plan_id, user.id, PRODUCT)
        )
        try:
            async with hold_world.factory() as attaching:
                await audio_service(attaching).create(
                    track_id, user.id, PRODUCT, attach_body(asset, plan_version_id=plan_id)
                )
                await racer.start()
                await racer.blocked(db_engine)  # the delete waits for the attach's track lock
                await attaching.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, VersionHasAudio)
        assert outcome.detail == {"count": 1}
        assert await _audio_rows(hold_world, track_id) == [1]
        await _reconciled(hold_world)

    async def test_the_version_delete_commits_first_so_the_attach_is_a_404_and_pins_nothing(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        plan_id = await _plan(hold_world, user, track_id, "Doomed")
        asset = await _audio(hold_world, user)
        racer = Racer(
            hold_world,
            lambda s: audio_service(s).create(
                track_id, user.id, PRODUCT, attach_body(asset, plan_version_id=plan_id)
            ),
        )
        try:
            async with hold_world.factory() as deleting:
                await DocumentVersionService(deleting).delete(plan_id, user.id, PRODUCT)
                await racer.start()
                await racer.blocked(db_engine)  # the attach's track lock waits for the delete
                await deleting.commit()
        finally:
            attach = await racer.outcome()

        assert isinstance(attach, VersionNotFound)
        assert await _audio_rows(hold_world, track_id) == []
        assert await hold_world.hold_count() == 0
        assert await _counter(hold_world, track_id) == 1
        await _reconciled(hold_world)


# ---------------------------------------------------------------------------------------------
# The cap is hard and the numbers are gap-free under concurrency
# ---------------------------------------------------------------------------------------------


class TestCapAndNumbering:
    async def test_two_parallel_attaches_at_nineteen_one_wins_and_one_is_a_409(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        for _ in range(MAX_AUDIO_VERSIONS_PER_TRACK - 1):
            await _attach(hold_world, user, track_id, await _audio(hold_world, user))
        first, second = await _audio(hold_world, user), await _audio(hold_world, user)
        racer = Racer(
            hold_world,
            lambda s: audio_service(s).create(track_id, user.id, PRODUCT, attach_body(second)),
        )
        try:
            async with hold_world.factory() as winning:
                added = await audio_service(winning).create(
                    track_id, user.id, PRODUCT, attach_body(first)
                )
                assert added.version_number == MAX_AUDIO_VERSIONS_PER_TRACK
                await racer.start()
                await racer.blocked(db_engine)  # the count is taken under the track lock
                await winning.commit()
        finally:
            loser = await racer.outcome()

        assert isinstance(loser, AudioVersionLimitReached)
        assert loser.detail == {"limit": MAX_AUDIO_VERSIONS_PER_TRACK}
        assert len(await _audio_rows(hold_world, track_id)) == MAX_AUDIO_VERSIONS_PER_TRACK
        assert (
            await hold_world.hold_count() == MAX_AUDIO_VERSIONS_PER_TRACK
        )  # the loser pinned nothing
        await _reconciled(hold_world)

    async def test_parallel_attaches_get_distinct_gap_free_numbers(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        assets = [await _audio(hold_world, user) for _ in range(6)]

        async def attach(asset: UserImage) -> int:
            async with hold_world.factory() as session:
                out = await audio_service(session).create(
                    track_id, user.id, PRODUCT, attach_body(asset)
                )
                await session.commit()
                return out.version_number

        numbers = await asyncio.wait_for(asyncio.gather(*(attach(a) for a in assets)), 30)

        assert sorted(numbers) == [1, 2, 3, 4, 5, 6]
        assert await _counter(hold_world, track_id) == 7
        await _reconciled(hold_world)

    async def test_two_parallel_deletes_of_the_same_version_one_succeeds_one_is_a_404(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        track_id = await committed_track(hold_world, user)
        audio_id = await _attach(hold_world, user, track_id, await _audio(hold_world, user))
        racer = Racer(hold_world, lambda s: audio_service(s).delete(audio_id, user.id, PRODUCT))
        try:
            async with hold_world.factory() as first:
                await audio_service(first).delete(audio_id, user.id, PRODUCT)
                await racer.start()
                await racer.blocked(db_engine)
                await first.commit()
        finally:
            outcome = await racer.outcome()

        assert isinstance(outcome, AudioVersionNotFound)
        assert await hold_world.hold_count() == 0
        await _reconciled(hold_world)

    async def test_the_last_release_of_a_shared_asset_rearms_it_once_under_concurrency(
        self, hold_world: HoldWorld
    ) -> None:
        """Two tracks pin one asset; deleting both versions in parallel leaves it unheld and re-armed."""
        user = await hold_world.user()
        one = await committed_track(hold_world, user, "One")
        two = await committed_track(hold_world, user, "Two")
        asset = await _audio(hold_world, user, expires_at=datetime.now(UTC) - timedelta(days=2))
        first = await _attach(hold_world, user, one, asset)
        second = await _attach(hold_world, user, two, asset)

        async def delete(audio_id: UUID) -> None:
            async with hold_world.factory() as session:
                await audio_service(session).delete(audio_id, user.id, PRODUCT)
                await session.commit()

        await asyncio.wait_for(asyncio.gather(delete(first), delete(second)), 30)

        assert await hold_world.hold_count() == 0
        assert await hold_world.upload_expires_at(asset) > datetime.now(UTC) + timedelta(days=5)
        await _reconciled(hold_world)
