"""Realized timelines against a real PostgreSQL (S3a W4 / W9.1-7).

Every test ends with :func:`~tests.timeline_support.assert_timelines_consistent` (invariants 1-3
over every row) and the audio-hold reconciliation, run by an autouse fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import msgspec
import pytest
from sqlalchemy import delete, event, func, select, update
from sqlalchemy.exc import IntegrityError
from structlog.testing import capture_logs

from src.api.schemas.studio import TimelineCreate, TrackAudioPatch
from src.core.enums import StudioTimelineOrigin
from src.db.models.studio import StudioAudioTimeline, StudioTrack, StudioTrackAudio
from src.db.models.user import User
from src.studio.constraints import MAX_TIMELINES_PER_AUDIO
from src.studio.errors import (
    AudioHasNoPlan,
    AudioHasTimelines,
    AudioVersionNotFound,
    InvalidCursor,
    StudioDocumentInvalid,
    StudioNotFound,
    TimelineHasChildren,
    TimelineIsCurrent,
    TimelineLimitReached,
    TimelineNotFound,
    TimelineUnchanged,
)
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService
from src.studio.timeline import SectionSpan, TimelineDocument, WordSpan
from tests.library_holds_support import make_holds
from tests.timeline_support import (
    DURATION_MS,
    MARK,
    Song,
    assert_timelines_consistent,
    audio_out,
    committed_song,
    create_timeline,
    timeline_service,
)
from tests.track_audio_support import assert_audio_reconciled, attach_body, audio_service

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.api.schemas.studio import TimelineOut
    from src.db.models.user import User as UserModel
    from tests.library_holds_support import HoldWorld

PRODUCT = "vex"


@pytest.fixture(autouse=True)
async def _every_row_stays_consistent(hold_world: HoldWorld) -> AsyncGenerator[None]:
    yield
    async with hold_world.factory() as session:
        await assert_timelines_consistent(session)
        await assert_audio_reconciled(session)


@pytest.fixture
async def user(hold_world: HoldWorld) -> UserModel:
    return await hold_world.user()


@pytest.fixture
async def song(hold_world: HoldWorld, user: UserModel) -> Song:
    return await committed_song(hold_world, user)


# -- helpers ------------------------------------------------------------------------------------


async def _rows(world: HoldWorld, audio_id: UUID) -> list[StudioAudioTimeline]:
    async with world.factory() as session:
        return list(
            (
                await session.execute(
                    select(StudioAudioTimeline)
                    .where(StudioAudioTimeline.audio_version_id == audio_id)
                    .order_by(StudioAudioTimeline.version_number)
                )
            )
            .scalars()
            .all()
        )


async def _counter(world: HoldWorld, audio_id: UUID) -> int:
    async with world.factory() as session:
        return (
            await session.execute(
                select(StudioTrackAudio.next_timeline_number).where(StudioTrackAudio.id == audio_id)
            )
        ).scalar_one()


async def _pointer(world: HoldWorld, audio_id: UUID) -> UUID | None:
    async with world.factory() as session:
        return (
            await session.execute(
                select(StudioTrackAudio.current_timeline_id).where(StudioTrackAudio.id == audio_id)
            )
        ).scalar_one()


async def _count_all(world: HoldWorld, user_id: UUID) -> int:
    async with world.factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(StudioAudioTimeline)
                .where(StudioAudioTimeline.user_id == user_id)
            )
        ).scalar_one()


async def _set_current(world: HoldWorld, song: Song, timeline_id: UUID | None) -> Any:
    async with world.factory() as session:
        out = await timeline_service(session).set_current(
            song.audio_id, song.user.id, PRODUCT, timeline_id
        )
        await session.commit()
        return out


async def _delete(world: HoldWorld, song: Song, timeline_id: UUID) -> None:
    async with world.factory() as session:
        await timeline_service(session).delete(timeline_id, song.user.id, PRODUCT)
        await session.commit()


async def _patch(world: HoldWorld, song: Song, **fields: Any) -> Any:
    async with world.factory() as session:
        out = await audio_service(session).patch(
            song.audio_id, song.user.id, PRODUCT, TrackAudioPatch(**fields)
        )
        await session.commit()
        return out


async def _chain(world: HoldWorld, song: Song, length: int) -> list[TimelineOut]:
    """``length`` timelines, each the parent of the next."""
    chain: list[TimelineOut] = []
    for step in range(length):
        out, _ = await create_timeline(
            world, song, song.variant(step), parent_id=chain[-1].id if chain else None
        )
        chain.append(out)
    return chain


# -- W9.1 create --------------------------------------------------------------------------------


class TestCreate:
    async def test_with_plan_and_lyrics(self, hold_world: HoldWorld, song: Song) -> None:
        out, created = await create_timeline(hold_world, song, song.complete())

        assert created is True
        assert out.version_number == 1
        assert out.parent_id is None
        assert out.origin is StudioTimelineOrigin.USER
        assert out.is_complete is True
        assert out.is_current is False
        assert out.audio_version_id == song.audio_id
        assert out.track_id == song.track_id
        assert out.content == song.complete()
        assert await _pointer(hold_world, song.audio_id) is None

    async def test_with_a_plan_only(self, hold_world: HoldWorld, user: UserModel) -> None:
        song = await committed_song(hold_world, user, with_lyrics=False)

        out, created = await create_timeline(hold_world, song, song.sections_only())

        assert created is True
        assert out.is_complete is True

    async def test_a_partial_timeline_is_stored_as_incomplete(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        sections_only, _ = await create_timeline(hold_world, song, song.sections_only())
        one_line, _ = await create_timeline(
            hold_world,
            song,
            TimelineDocument(sections=song.section_spans(), lines=song.line_spans()[:1]),
        )

        assert sections_only.is_complete is False
        assert one_line.is_complete is False
        assert [row.is_complete for row in await _rows(hold_world, song.audio_id)] == [False, False]

    async def test_the_stored_json_omits_every_default(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        await create_timeline(hold_world, song, song.complete())

        (row,) = await _rows(hold_world, song.audio_id)

        lines = row.content["lines"]
        assert "words" not in lines[1]
        assert "inferred" not in lines[0]["words"][0]
        assert lines[0]["words"][1]["inferred"] is True

    async def test_the_counter_numbers_versions_per_audio(
        self, hold_world: HoldWorld, user: UserModel, song: Song
    ) -> None:
        other = await committed_song(hold_world, user)

        first, _ = await create_timeline(hold_world, song, song.variant(1))
        second, _ = await create_timeline(hold_world, song, song.variant(2))
        elsewhere, _ = await create_timeline(hold_world, other, other.variant(1))

        assert (first.version_number, second.version_number, elsewhere.version_number) == (1, 2, 1)

    async def test_an_identical_create_is_replayed_without_a_new_number(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        first, _ = await create_timeline(hold_world, song, song.complete())

        replay, created = await create_timeline(hold_world, song, song.complete())

        assert created is False
        assert replay.id == first.id
        assert replay.version_number == 1
        assert await _counter(hold_world, song.audio_id) == 2
        assert len(await _rows(hold_world, song.audio_id)) == 1

    async def test_a_replay_applies_make_current(self, hold_world: HoldWorld, song: Song) -> None:
        first, _ = await create_timeline(hold_world, song, song.complete())

        replay, created = await create_timeline(
            hold_world, song, song.complete(), make_current=True
        )

        assert created is False
        assert replay.is_current is True
        assert await _pointer(hold_world, song.audio_id) == first.id

    async def test_make_current_on_create(self, hold_world: HoldWorld, song: Song) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete(), make_current=True)

        assert out.is_current is True
        assert await _pointer(hold_world, song.audio_id) == out.id

    async def test_without_make_current_the_pointer_never_moves(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        current, _ = await create_timeline(hold_world, song, song.variant(1), make_current=True)

        newer, _ = await create_timeline(hold_world, song, song.variant(2), parent_id=current.id)

        assert newer.is_current is False
        assert await _pointer(hold_world, song.audio_id) == current.id

    async def test_the_same_content_under_another_parent_is_a_new_version(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        root, _ = await create_timeline(hold_world, song, song.variant(1))
        child, _ = await create_timeline(hold_world, song, song.variant(2), parent_id=root.id)

        grandchild, created = await create_timeline(
            hold_world, song, song.variant(1), parent_id=child.id
        )

        assert created is True
        assert grandchild.parent_id == child.id
        assert grandchild.version_number == 3

    async def test_content_equal_to_the_parent_is_timeline_unchanged(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        parent, _ = await create_timeline(hold_world, song, song.complete())

        with pytest.raises(TimelineUnchanged):
            await create_timeline(hold_world, song, song.complete(), parent_id=parent.id)

        assert await _counter(hold_world, song.audio_id) == 2

    async def test_an_audio_without_a_plan_takes_no_timeline(
        self, hold_world: HoldWorld, user: UserModel
    ) -> None:
        song = await committed_song(hold_world, user, with_plan=False)

        with pytest.raises(AudioHasNoPlan):
            await create_timeline(hold_world, song, TimelineDocument(sections=()))

        assert await _counter(hold_world, song.audio_id) == 1

    @pytest.mark.parametrize("case", ["unknown", "foreign_user", "other_product"])
    async def test_an_audio_that_is_not_the_caller_s_is_not_found(
        self, hold_world: HoldWorld, song: Song, case: str
    ) -> None:
        audio_id, user_id, product = song.audio_id, song.user.id, PRODUCT
        if case == "unknown":
            audio_id = uuid4()
        elif case == "foreign_user":
            user_id = (await hold_world.user()).id
        else:
            product = "synthara"

        async with hold_world.factory() as session:
            with pytest.raises(AudioVersionNotFound):
                await timeline_service(session).create(
                    audio_id, user_id, product, TimelineCreate(content=song.complete())
                )

    async def test_a_parent_of_another_audio_or_unknown_is_not_found(
        self, hold_world: HoldWorld, user: UserModel, song: Song
    ) -> None:
        other = await committed_song(hold_world, user)
        elsewhere, _ = await create_timeline(hold_world, other, other.complete())

        for parent_id in (elsewhere.id, uuid4()):
            with pytest.raises(TimelineNotFound):
                await create_timeline(hold_world, song, song.complete(), parent_id=parent_id)

        assert await _counter(hold_world, song.audio_id) == 1

    async def test_a_document_that_would_not_decode_writes_nothing(
        self, hold_world: HoldWorld, song: Song, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RD7: the backstop for a limit normalization missed. A bug, never a 4xx."""
        monkeypatch.setattr(
            "src.studio.services.timeline.timeline_content",
            lambda _document: {"sections": [{"section_id": "nope"}]},
        )

        with pytest.raises(RuntimeError, match="would not decode"):
            await create_timeline(hold_world, song, song.complete())

        assert await _rows(hold_world, song.audio_id) == []
        assert await _counter(hold_world, song.audio_id) == 1


def _with_line(song: Song, index: int, **changes: Any) -> TimelineDocument:
    lines = list(song.line_spans())
    lines[index] = msgspec.structs.replace(lines[index], **changes)
    return TimelineDocument(sections=song.section_spans(), lines=tuple(lines))


def _with_words(song: Song, *words: WordSpan) -> TimelineDocument:
    return _with_line(song, 0, words=words)


def _sections(*spans: SectionSpan) -> TimelineDocument:
    return TimelineDocument(sections=spans)


_INVALID: dict[str, tuple[Callable[[Song], TimelineDocument], list[tuple[str, str]]]] = {
    "span_out_of_range": (
        lambda s: _sections(SectionSpan(section_id=s.solo, start_ms=0, end_ms=DURATION_MS + 1)),
        [("sections[0]", "span_out_of_range")],
    ),
    "unknown_section": (
        lambda _song: _sections(SectionSpan(section_id=uuid4(), start_ms=0, end_ms=10)),
        [("sections[0]", "unknown_section")],
    ),
    "duplicate_section": (
        lambda s: _sections(s.section_spans()[0], s.section_spans()[0]),
        [("sections[1]", "duplicate_section")],
    ),
    "section_order": (
        lambda s: _sections(s.section_spans()[1], s.section_spans()[0]),
        [("sections[1]", "section_order")],
    ),
    "section_overlap": (
        lambda s: _sections(
            s.section_spans()[0],
            SectionSpan(section_id=s.chorus, start_ms=19_999, end_ms=40_000),
        ),
        [("sections[1]", "section_overlap")],
    ),
    "unknown_line": (
        lambda s: _with_line(s, 3, line_index=2),
        [("lines[3]", "unknown_line")],
    ),
    "duplicate_line": (
        lambda s: TimelineDocument(
            sections=s.section_spans(), lines=(*s.line_spans(), s.line_spans()[1])
        ),
        [("lines[4]", "duplicate_line")],
    ),
    "line_section_untimed": (
        lambda s: TimelineDocument(sections=s.section_spans()[:1], lines=s.line_spans()[2:3]),
        [("lines[0]", "line_section_untimed")],
    ),
    "line_outside_section": (
        lambda s: _with_line(s, 1, end_ms=20_001),
        [("lines[1]", "line_outside_section")],
    ),
    "line_order": (
        lambda s: TimelineDocument(
            sections=s.section_spans(),
            lines=(s.line_spans()[1], s.line_spans()[0], *s.line_spans()[2:]),
        ),
        [("lines[1]", "line_order")],
    ),
    "line_overlap": (
        lambda s: _with_line(s, 1, start_ms=4_999),
        [("lines[1]", "line_overlap")],
    ),
    "word_outside_line": (
        lambda s: _with_words(s, WordSpan(text="a", start_ms=999, end_ms=2_000)),
        [("lines[0].words[0]", "word_outside_line")],
    ),
    "word_order": (
        lambda s: _with_words(
            s,
            WordSpan(text="b", start_ms=3_000, end_ms=4_000),
            WordSpan(text="a", start_ms=1_000, end_ms=2_000),
        ),
        [("lines[0].words[1]", "word_order")],
    ),
    "word_overlap": (
        lambda s: _with_words(
            s,
            WordSpan(text="a", start_ms=1_000, end_ms=2_001),
            WordSpan(text="b", start_ms=2_000, end_ms=3_000),
        ),
        [("lines[0].words[1]", "word_overlap")],
    ),
    "too_long_after_normalization": (
        lambda s: _with_words(s, WordSpan(text="שּׁ" * 34, start_ms=1_000, end_ms=2_000)),
        [("lines[0].words[0].text", "too_long_after_normalization")],
    ),
}


class TestValidationThroughTheService:
    @pytest.mark.parametrize("case", list(_INVALID))
    async def test_each_path_is_422_with_its_issues_and_writes_nothing(
        self, hold_world: HoldWorld, song: Song, case: str
    ) -> None:
        build, expected = _INVALID[case]

        with pytest.raises(StudioDocumentInvalid) as raised:
            await create_timeline(hold_world, song, build(song))

        assert [(i.path, i.code) for i in raised.value.issues] == expected
        assert raised.value.status_code == 422
        assert raised.value.detail == {
            "issues": [{"path": path, "code": code} for path, code in expected]
        }
        assert await _rows(hold_world, song.audio_id) == []
        assert await _counter(hold_world, song.audio_id) == 1

    async def test_the_limit_and_semantic_issues_are_reported_together(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        document = _with_words(song, WordSpan(text="שּׁ" * 34, start_ms=999, end_ms=2_000))

        with pytest.raises(StudioDocumentInvalid) as raised:
            await create_timeline(hold_world, song, document)

        assert [i.code for i in raised.value.issues] == [
            "too_long_after_normalization",
            "word_outside_line",
        ]

    async def test_without_lyrics_every_line_is_unknown(
        self, hold_world: HoldWorld, user: UserModel
    ) -> None:
        song = await committed_song(hold_world, user, with_lyrics=False)

        with pytest.raises(StudioDocumentInvalid) as raised:
            await create_timeline(hold_world, song, song.complete())

        assert {i.code for i in raised.value.issues} == {"unknown_line"}
        assert len(raised.value.issues) == 4

    async def test_the_duration_is_the_audio_version_s(
        self, hold_world: HoldWorld, user: UserModel
    ) -> None:
        short = await committed_song(hold_world, user, duration_ms=DURATION_MS - 1)

        with pytest.raises(StudioDocumentInvalid) as raised:
            await create_timeline(hold_world, short, short.sections_only())

        assert [(i.path, i.code) for i in raised.value.issues] == [
            ("sections[2]", "span_out_of_range")
        ]


# -- W9.2 the cap and the numbers ----------------------------------------------------------------


class TestCapAndNumbers:
    async def test_fifty_timelines_then_409_while_a_replay_still_works(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        async with hold_world.factory() as session:
            for step in range(MAX_TIMELINES_PER_AUDIO):
                await timeline_service(session).create(
                    song.audio_id,
                    song.user.id,
                    PRODUCT,
                    TimelineCreate(content=song.variant(step)),
                )
            await session.commit()

        with pytest.raises(TimelineLimitReached) as raised:
            await create_timeline(hold_world, song, song.variant(MAX_TIMELINES_PER_AUDIO))
        replay, created = await create_timeline(hold_world, song, song.variant(0))

        assert MAX_TIMELINES_PER_AUDIO == 50
        assert raised.value.detail == {"limit": 50}
        assert (created, replay.version_number) == (False, 1)
        assert await _counter(hold_world, song.audio_id) == 51

    async def test_a_delete_frees_a_slot_and_the_number_is_never_reused(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        first, _ = await create_timeline(hold_world, song, song.variant(1))
        second, _ = await create_timeline(hold_world, song, song.variant(2))
        await _delete(hold_world, song, second.id)
        await _delete(hold_world, song, first.id)

        again, created = await create_timeline(hold_world, song, song.variant(1))

        assert (created, again.version_number) == (True, 3)


# -- W9.3 the current pointer ----------------------------------------------------------------


class TestSetCurrent:
    async def test_it_moves_and_clears_the_pointer(self, hold_world: HoldWorld, song: Song) -> None:
        first, _ = await create_timeline(hold_world, song, song.complete())
        second, _ = await create_timeline(hold_world, song, song.sections_only())

        chosen = await _set_current(hold_world, song, first.id)
        moved = await _set_current(hold_world, song, second.id)
        cleared = await _set_current(hold_world, song, None)

        assert (chosen.current_timeline_id, chosen.timeline_complete) == (first.id, True)
        assert (moved.current_timeline_id, moved.timeline_complete) == (second.id, False)
        assert (cleared.current_timeline_id, cleared.timeline_complete) == (None, None)
        assert cleared.timeline_count == 2
        assert await _pointer(hold_world, song.audio_id) is None

    async def test_setting_the_same_timeline_again_writes_nothing(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete(), make_current=True)
        before = await audio_out(hold_world, song)

        after = await _set_current(hold_world, song, out.id)

        assert after.updated_at == before.updated_at

    async def test_another_audio_s_timeline_is_not_found(
        self, hold_world: HoldWorld, user: UserModel, song: Song
    ) -> None:
        other = await committed_song(hold_world, user)
        elsewhere, _ = await create_timeline(hold_world, other, other.complete())

        for timeline_id in (elsewhere.id, uuid4()):
            with pytest.raises(TimelineNotFound):
                await _set_current(hold_world, song, timeline_id)

        assert await _pointer(hold_world, song.audio_id) is None

    async def test_the_database_refuses_another_audio_s_timeline(
        self, hold_world: HoldWorld, user: UserModel, song: Song
    ) -> None:
        """The composite foreign key is the backstop behind the service check."""
        other = await committed_song(hold_world, user)
        elsewhere, _ = await create_timeline(hold_world, other, other.complete())

        async with hold_world.factory() as session:
            with pytest.raises(IntegrityError, match="fk_studio_track_audio_current_timeline"):
                await session.execute(
                    update(StudioTrackAudio)
                    .where(StudioTrackAudio.id == song.audio_id)
                    .values(current_timeline_id=elsewhere.id)
                )
                await session.commit()

    async def test_another_user_s_audio_is_not_found(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        stranger = await hold_world.user()

        async with hold_world.factory() as session:
            with pytest.raises(AudioVersionNotFound):
                await timeline_service(session).set_current(
                    song.audio_id, stranger.id, PRODUCT, None
                )


# -- W9.4 deletes -----------------------------------------------------------------------------


class TestDelete:
    async def test_the_current_timeline_is_refused(self, hold_world: HoldWorld, song: Song) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete(), make_current=True)

        with pytest.raises(TimelineIsCurrent):
            await _delete(hold_world, song, out.id)

        assert [row.id for row in await _rows(hold_world, song.audio_id)] == [out.id]

    async def test_a_parent_is_refused_and_a_leaf_is_deleted(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        root, leaf = await _chain(hold_world, song, 2)

        with pytest.raises(TimelineHasChildren):
            await _delete(hold_world, song, root.id)
        await _delete(hold_world, song, leaf.id)
        await _delete(hold_world, song, root.id)

        assert await _rows(hold_world, song.audio_id) == []

    @pytest.mark.parametrize("case", ["unknown", "foreign_user", "other_product"])
    async def test_a_timeline_that_is_not_the_caller_s_is_not_found(
        self, hold_world: HoldWorld, song: Song, case: str
    ) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete())
        timeline_id, user_id, product = out.id, song.user.id, PRODUCT
        if case == "unknown":
            timeline_id = uuid4()
        elif case == "foreign_user":
            user_id = (await hold_world.user()).id
        else:
            product = "synthara"

        async with hold_world.factory() as session:
            service = timeline_service(session)
            with pytest.raises(TimelineNotFound):
                await service.delete(timeline_id, user_id, product)
            with pytest.raises(TimelineNotFound):
                await service.get(timeline_id, user_id, product)
            with pytest.raises(TimelineNotFound):
                await service.export(timeline_id, "lrc", user_id, product)

        assert len(await _rows(hold_world, song.audio_id)) == 1

    async def test_the_database_refuses_to_delete_a_parent_on_its_own(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        root, _ = await _chain(hold_world, song, 2)

        async with hold_world.factory() as session:
            with pytest.raises(IntegrityError, match="fk_studio_audio_timelines_parent"):
                await session.execute(
                    delete(StudioAudioTimeline).where(StudioAudioTimeline.id == root.id)
                )

    async def test_an_audio_delete_cascades_every_chain_in_one_statement(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        chain = await _chain(hold_world, song, 4)
        await create_timeline(hold_world, song, song.sections_only(), parent_id=chain[1].id)
        await _set_current(hold_world, song, chain[-1].id)  # the pointer goes with its row
        statements: list[str] = []

        def on_execute(_conn: object, _cursor: object, statement: str, *_: object) -> None:
            statements.append(" ".join(statement.split()).upper())

        event.listen(db_engine.sync_engine, "before_cursor_execute", on_execute)
        try:
            async with hold_world.factory() as session:
                await audio_service(session).delete(song.audio_id, song.user.id, PRODUCT)
                await session.commit()
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", on_execute)

        assert await _count_all(hold_world, song.user.id) == 0
        deletes = [s for s in statements if s.startswith("DELETE")]
        assert not [s for s in deletes if "STUDIO_AUDIO_TIMELINES" in s], deletes
        assert len([s for s in deletes if s.startswith("DELETE FROM STUDIO_TRACK_AUDIO")]) == 1

    async def test_a_track_delete_leaves_no_timeline(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        chain = await _chain(hold_world, song, 3)
        await _set_current(hold_world, song, chain[-1].id)

        async with hold_world.factory() as session:
            await TrackService(session, make_holds(session)).delete(
                song.track_id, song.user.id, PRODUCT
            )
            await session.commit()

        assert await _count_all(hold_world, song.user.id) == 0

    async def test_a_user_hard_delete_leaves_no_timeline(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        chain = await _chain(hold_world, song, 3)
        await _set_current(hold_world, song, chain[-1].id)
        async with hold_world.factory() as session:
            await DocumentVersionService(session).set_current(
                song.track_id,
                song.user.id,
                PRODUCT,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=song.audio_id,
            )
            await session.commit()

        async with hold_world.factory() as session, session.begin():
            await session.execute(delete(User).where(User.id == song.user.id))

        assert await _count_all(hold_world, song.user.id) == 0
        async with hold_world.factory() as session:
            assert (
                await session.execute(select(StudioTrack.id).where(StudioTrack.id == song.track_id))
            ).scalar_one_or_none() is None


# -- W9.5 the link freeze (S3-5) ---------------------------------------------------------------


class TestLinkFreeze:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"lyrics_version_id": None}, id="drop-lyrics"),
            pytest.param({"plan_version_id": None, "lyrics_version_id": None}, id="drop-both"),
            pytest.param({"plan_version_id": uuid4()}, id="another-plan"),
            pytest.param({"lyrics_version_id": uuid4()}, id="another-lyrics"),
        ],
    )
    async def test_a_link_change_with_a_timeline_is_refused(
        self, hold_world: HoldWorld, song: Song, change: dict[str, Any]
    ) -> None:
        """Refused before the new link is even looked up: a frozen audio takes no new link."""
        await create_timeline(hold_world, song, song.complete())

        with pytest.raises(AudioHasTimelines) as raised:
            await _patch(hold_world, song, **change)

        assert raised.value.detail == {"count": 1}
        out = await audio_out(hold_world, song)
        assert (out.plan_version_id, out.lyrics_version_id) == (
            song.plan_version_id,
            song.lyrics_version_id,
        )

    async def test_the_count_reports_every_timeline(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        await _chain(hold_world, song, 3)

        with pytest.raises(AudioHasTimelines) as raised:
            await _patch(hold_world, song, lyrics_version_id=None)

        assert raised.value.detail == {"count": 3}

    async def test_a_refused_patch_changes_nothing_not_even_the_label(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        await create_timeline(hold_world, song, song.complete())

        with pytest.raises(AudioHasTimelines):
            await _patch(hold_world, song, label="New", lyrics_version_id=None)

        assert (await audio_out(hold_world, song)).label is None

    @pytest.mark.parametrize("what", ["label", "same_links", "label_and_same_links", "nothing"])
    async def test_label_only_and_same_link_patches_still_work(
        self, hold_world: HoldWorld, song: Song, what: str
    ) -> None:
        await create_timeline(hold_world, song, song.complete())
        same = {
            "plan_version_id": song.plan_version_id,
            "lyrics_version_id": song.lyrics_version_id,
        }
        fields: dict[str, Any] = {
            "label": {"label": "Radio edit"},
            "same_links": same,
            "label_and_same_links": {"label": "Radio edit", **same},
            "nothing": {},
        }[what]

        out = await _patch(hold_world, song, **fields)

        assert (out.plan_version_id, out.lyrics_version_id) == (
            song.plan_version_id,
            song.lyrics_version_id,
        )
        assert out.label == ("Radio edit" if "label" in fields else None)

    async def test_without_timelines_links_change_as_before(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        dropped = await _patch(hold_world, song, lyrics_version_id=None)
        relinked = await _patch(hold_world, song, lyrics_version_id=song.lyrics_version_id)

        assert dropped.lyrics_version_id is None
        assert relinked.lyrics_version_id == song.lyrics_version_id

    async def test_deleting_every_timeline_unfreezes_the_links(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete())
        await _delete(hold_world, song, out.id)

        patched = await _patch(hold_world, song, lyrics_version_id=None)

        assert patched.lyrics_version_id is None


# -- W9.6 the track audio fields --------------------------------------------------------------


class TestTrackAudioFields:
    async def test_they_follow_the_timelines_on_every_read(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        async def reads() -> list[tuple[UUID | None, int, bool | None]]:
            async with hold_world.factory() as session:
                service = audio_service(session)
                page = await service.list_page(
                    song.track_id, song.user.id, PRODUCT, limit=10, cursor=None
                )
                detail = await TrackService(session, make_holds(session)).get_detail(
                    song.track_id, song.user.id, PRODUCT
                )
                single = await service.get(song.audio_id, song.user.id, PRODUCT)
            assert detail.current_audio is not None
            return [
                (item.current_timeline_id, item.timeline_count, item.timeline_complete)
                for item in (single, page.items[0], detail.current_audio)
            ]

        async with hold_world.factory() as session:
            await DocumentVersionService(session).set_current(
                song.track_id,
                song.user.id,
                PRODUCT,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=song.audio_id,
            )
            await session.commit()

        assert await reads() == [(None, 0, None)] * 3
        complete, _ = await create_timeline(hold_world, song, song.complete())
        assert await reads() == [(None, 1, None)] * 3
        await _set_current(hold_world, song, complete.id)
        assert await reads() == [(complete.id, 1, True)] * 3
        partial, _ = await create_timeline(
            hold_world, song, song.sections_only(), make_current=True
        )
        assert await reads() == [(partial.id, 2, False)] * 3

    async def test_the_list_page_runs_a_fixed_number_of_statements(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, user: UserModel, song: Song
    ) -> None:
        async def statements_of_one_page() -> tuple[int, int]:
            statements: list[str] = []

            def on_execute(_conn: object, _cursor: object, statement: str, *_: object) -> None:
                statements.append(statement)

            event.listen(db_engine.sync_engine, "before_cursor_execute", on_execute)
            try:
                async with hold_world.factory() as session:
                    page = await audio_service(session).list_page(
                        song.track_id, user.id, PRODUCT, limit=10, cursor=None
                    )
            finally:
                event.remove(db_engine.sync_engine, "before_cursor_execute", on_execute)
            return len(page.items), len(statements)

        await create_timeline(hold_world, song, song.complete(), make_current=True)
        one_item = await statements_of_one_page()
        for step in range(3):
            asset = await hold_world.audio(user, duration_ms=DURATION_MS)
            async with hold_world.factory() as session:
                attached = await audio_service(session).create(
                    song.track_id,
                    user.id,
                    PRODUCT,
                    attach_body(asset, plan_version_id=song.plan_version_id),
                )
                for variant in range(step + 1):
                    await timeline_service(session).create(
                        attached.id,
                        user.id,
                        PRODUCT,
                        TimelineCreate(content=song.variant(variant), make_current=True),
                    )
                await session.commit()
        four_items = await statements_of_one_page()

        assert one_item[0] == 1
        assert four_items[0] == 4
        assert four_items[1] == one_item[1], "the page issued a query per item"


# -- reads --------------------------------------------------------------------------------------


class TestReads:
    async def test_the_list_is_newest_first_and_pages_by_number(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        for step in range(5):
            await create_timeline(hold_world, song, song.variant(step), make_current=step == 2)

        async with hold_world.factory() as session:
            service = timeline_service(session)
            first = await service.list_page(
                song.audio_id, song.user.id, PRODUCT, limit=2, cursor=None
            )
            second = await service.list_page(
                song.audio_id, song.user.id, PRODUCT, limit=2, cursor=first.next_cursor
            )
            third = await service.list_page(
                song.audio_id, song.user.id, PRODUCT, limit=2, cursor=second.next_cursor
            )

        assert [i.version_number for i in first.items] == [5, 4]
        assert [i.version_number for i in second.items] == [3, 2]
        assert [i.version_number for i in third.items] == [1]
        assert (third.has_more, third.next_cursor) == (False, None)
        assert [i.is_current for i in (*first.items, *second.items)] == [False, False, True, False]
        assert second.items[0].content == song.variant(2)

    async def test_a_bad_cursor_and_a_foreign_audio(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        stranger = await hold_world.user()
        async with hold_world.factory() as session:
            service = timeline_service(session)
            with pytest.raises(InvalidCursor):
                await service.list_page(
                    song.audio_id, song.user.id, PRODUCT, limit=2, cursor="nonsense"
                )
            with pytest.raises(AudioVersionNotFound):
                await service.list_page(song.audio_id, stranger.id, PRODUCT, limit=2, cursor=None)


# -- W9.7 exports -------------------------------------------------------------------------------


class TestExports:
    async def test_lrc_and_srt_of_a_complete_timeline(
        self, hold_world: HoldWorld, song: Song
    ) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete())

        async with hold_world.factory() as session:
            service = timeline_service(session)
            lrc = await service.export(out.id, "lrc", song.user.id, PRODUCT)
            srt = await service.export(out.id, "srt", song.user.id, PRODUCT)

        assert lrc.text.splitlines() == [
            f"[00:01.00]<00:01.00>{MARK}-w1 <00:02.00>{MARK}-w2",
            f"[00:05.00]{MARK}-lyric verse two",
            f"[00:21.00]{MARK}-lyric hook one",
            f"[00:25.00]{MARK}-lyric hook two",
        ]
        assert [(w.code, w.count) for w in lrc.warnings] == [("inferred_word_timings", 1)]
        assert srt.text.startswith(f"1\n00:00:01,000 --> 00:00:05,000\n{MARK}-lyric verse one\n\n")
        assert srt.warnings == []
        for text_ in (lrc.text, srt.text):
            for secret in ("label", "notes", "title", "idea", "reference"):
                assert f"{MARK}-{secret}" not in text_

    async def test_an_audio_without_lyrics_exports_nothing_with_a_warning(
        self, hold_world: HoldWorld, user: UserModel
    ) -> None:
        song = await committed_song(hold_world, user, with_lyrics=False)
        out, _ = await create_timeline(hold_world, song, song.sections_only())

        async with hold_world.factory() as session:
            lrc = await timeline_service(session).export(out.id, "lrc", user.id, PRODUCT)

        assert lrc.text == ""
        assert [(w.code, w.count) for w in lrc.warnings] == [("no_lyrics", None)]

    async def test_an_unknown_format_is_a_404(self, hold_world: HoldWorld, song: Song) -> None:
        out, _ = await create_timeline(hold_world, song, song.complete())

        async with hold_world.factory() as session:
            with pytest.raises(StudioNotFound):
                await timeline_service(session).export(out.id, "vtt", song.user.id, PRODUCT)

    async def test_it_is_one_unlocked_select(
        self, hold_world: HoldWorld, db_engine: AsyncEngine, song: Song
    ) -> None:
        """One snapshot for the times and the text they belong to, and no lock."""
        out, _ = await create_timeline(hold_world, song, song.complete())
        statements: list[str] = []

        def on_execute(_conn: object, _cursor: object, statement: str, *_: object) -> None:
            statements.append(" ".join(statement.split()).upper())

        event.listen(db_engine.sync_engine, "before_cursor_execute", on_execute)
        try:
            async with hold_world.factory() as session:
                await timeline_service(session).export(out.id, "lrc", song.user.id, PRODUCT)
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", on_execute)

        assert len(statements) == 1, statements
        assert statements[0].startswith("SELECT")
        assert "STUDIO_DOCUMENT_VERSIONS" in statements[0]
        assert "FOR UPDATE" not in statements[0]


# -- invariant 8: logs ---------------------------------------------------------------------------


async def test_a_full_journey_logs_no_lyric_word_or_label_text(
    hold_world: HoldWorld, song: Song
) -> None:
    with capture_logs() as logs:
        out, _ = await create_timeline(hold_world, song, song.complete(), make_current=True)
        await create_timeline(hold_world, song, song.complete())  # replay
        with pytest.raises(StudioDocumentInvalid):
            await create_timeline(
                hold_world,
                song,
                _with_words(song, WordSpan(text=f"{MARK}-bad", start_ms=0, end_ms=1)),
            )
        child, _ = await create_timeline(hold_world, song, song.variant(1), parent_id=out.id)
        await _set_current(hold_world, song, None)
        async with hold_world.factory() as session:
            await timeline_service(session).export(out.id, "lrc", song.user.id, PRODUCT)
            await timeline_service(session).export(out.id, "srt", song.user.id, PRODUCT)
        await _delete(hold_world, song, child.id)

    events = {entry["event"] for entry in logs}
    assert {
        "studio.timeline_created",
        "studio.timeline_replayed",
        "studio.current_timeline_set",
        "studio.timeline_exported",
        "studio.timeline_deleted",
    } <= events
    for entry in logs:
        assert MARK.lower() not in repr(entry).lower(), entry
