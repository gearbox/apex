"""Helpers shared by the realized-timeline integration tests (S3a).

:func:`committed_song` builds what a timeline needs: a track with a plan (verse, chorus and an
instrumental solo), lyrics for the verse and chorus, and a 60 s audio version linking them. Every
string a log or an export must never leak carries :data:`MARK`.

:func:`assert_timelines_consistent` is invariants 1 and 2 as a query over **every** stored row:
each timeline still decodes, validates against its audio's duration, plan and lyrics, and its
``is_complete`` and ``content_hash`` equal what its content computes to. The timeline test modules
run it after every test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.api.schemas.studio import (
    LyricsVersionCreate,
    PlanVersionCreate,
    TimelineCreate,
)
from src.core.enums import StudioDocumentOrigin, StudioTimelineOrigin
from src.core.enums import StudioSectionKind as K
from src.core.uid import new_id
from src.db.models.studio import StudioAudioTimeline, StudioDocumentVersion, StudioTrackAudio
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.services.timeline import TimelineService
from src.studio.services.version import DocumentVersionService
from src.studio.timeline import (
    LineSpan,
    SectionSpan,
    TimelineDocument,
    WordSpan,
    decode_stored_timeline,
    is_complete,
    timeline_hash,
    validate_timeline,
)
from tests.studio_support import make_lyrics, make_plan, make_section
from tests.track_audio_support import attach_body, audio_service, committed_track

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import TimelineOut, TrackAudioOut
    from src.db.models.user import User
    from src.studio.documents import LyricsDocument, PlanDocument
    from tests.library_holds_support import HoldWorld

__all__ = [
    "DURATION_MS",
    "MARK",
    "Song",
    "assert_timelines_consistent",
    "committed_song",
    "create_timeline",
    "timeline_service",
]

PRODUCT = "vex"
DURATION_MS = 60_000
MARK = "ZXQ-SECRET"


@dataclass(frozen=True)
class Song:
    """A committed track with documents and one audio version linking (some of) them."""

    user: User
    track_id: UUID
    audio_id: UUID
    plan: PlanDocument
    lyrics: LyricsDocument
    plan_version_id: UUID
    lyrics_version_id: UUID
    verse: UUID
    chorus: UUID
    solo: UUID

    def section_spans(self) -> tuple[SectionSpan, ...]:
        """Every section, end to end (touching)."""
        return (
            SectionSpan(section_id=self.verse, start_ms=0, end_ms=20_000),
            SectionSpan(section_id=self.chorus, start_ms=20_000, end_ms=40_000),
            SectionSpan(section_id=self.solo, start_ms=40_000, end_ms=DURATION_MS),
        )

    def line_spans(self) -> tuple[LineSpan, ...]:
        """Every lyric line; the first carries two words, one of them inferred."""
        return (
            LineSpan(
                section_id=self.verse,
                line_index=0,
                start_ms=1_000,
                end_ms=5_000,
                words=(
                    WordSpan(text=f"{MARK}-w1", start_ms=1_000, end_ms=2_000),
                    WordSpan(text=f"{MARK}-w2", start_ms=2_000, end_ms=5_000, inferred=True),
                ),
            ),
            LineSpan(section_id=self.verse, line_index=1, start_ms=5_000, end_ms=9_000),
            LineSpan(section_id=self.chorus, line_index=0, start_ms=21_000, end_ms=25_000),
            LineSpan(section_id=self.chorus, line_index=1, start_ms=25_000, end_ms=30_000),
        )

    def complete(self) -> TimelineDocument:
        """Everything timed (complete with the lyrics linked)."""
        return TimelineDocument(sections=self.section_spans(), lines=self.line_spans())

    def sections_only(self) -> TimelineDocument:
        """All sections, no line (complete only without lyrics)."""
        return TimelineDocument(sections=self.section_spans())

    def variant(self, step: int) -> TimelineDocument:
        """A valid partial timeline that differs for every ``step`` (0 ≤ step < 20000)."""
        return TimelineDocument(
            sections=(SectionSpan(section_id=self.verse, start_ms=0, end_ms=20_000 - step),)
        )


async def _document_version(
    session: AsyncSession,
    user: User,
    track_id: UUID,
    request: PlanVersionCreate | LyricsVersionCreate,
) -> UUID:
    created, _ = await DocumentVersionService(session).create(
        track_id, request, user.id, user.product_id, origin=StudioDocumentOrigin.USER
    )
    return created.version.id


async def committed_song(
    world: HoldWorld,
    user: User,
    *,
    with_plan: bool = True,
    with_lyrics: bool = True,
    duration_ms: int = DURATION_MS,
) -> Song:
    """A committed track, plan, lyrics and one audio version linking what was asked for."""
    verse, chorus, solo = new_id(), new_id(), new_id()
    plan = make_plan(
        make_section(K.VERSE, section_id=verse, label=f"{MARK}-label", notes=f"{MARK}-notes"),
        make_section(K.CHORUS, section_id=chorus),
        make_section(K.INSTRUMENTAL, section_id=solo),
        title=f"{MARK}-title",
        idea=f"{MARK}-idea",
        references=(f"{MARK}-reference",),
    )
    lyrics = make_lyrics(
        (verse, [f"{MARK}-lyric verse one", f"{MARK}-lyric verse two"]),
        (chorus, [f"{MARK}-lyric hook one", f"{MARK}-lyric hook two"]),
    )
    track_id = await committed_track(world, user)
    async with world.factory() as session:
        plan_version_id = await _document_version(
            session, user, track_id, PlanVersionCreate(content=plan, make_current=False)
        )
        lyrics_version_id = await _document_version(
            session,
            user,
            track_id,
            LyricsVersionCreate(
                content=lyrics, based_on_plan_version_id=plan_version_id, make_current=False
            ),
        )
        await session.commit()
    asset = await world.audio(user, duration_ms=duration_ms)
    async with world.factory() as session:
        audio = await audio_service(session).create(
            track_id,
            user.id,
            user.product_id,
            attach_body(
                asset,
                plan_version_id=plan_version_id if with_plan else None,
                lyrics_version_id=lyrics_version_id if with_plan and with_lyrics else None,
            ),
        )
        await session.commit()
    return Song(
        user=user,
        track_id=track_id,
        audio_id=audio.id,
        plan=plan,
        lyrics=lyrics,
        plan_version_id=plan_version_id,
        lyrics_version_id=lyrics_version_id,
        verse=verse,
        chorus=chorus,
        solo=solo,
    )


def timeline_service(session: AsyncSession) -> TimelineService:
    return TimelineService(session)


async def create_timeline(
    world: HoldWorld,
    song: Song,
    document: TimelineDocument,
    *,
    parent_id: UUID | None = None,
    make_current: bool = False,
) -> tuple[TimelineOut, bool]:
    """Create (or replay) a timeline of the song's audio version, committed."""
    async with world.factory() as session:
        result = await timeline_service(session).create(
            song.audio_id,
            song.user.id,
            song.user.product_id,
            TimelineCreate(content=document, parent_id=parent_id, make_current=make_current),
        )
        await session.commit()
        return result


async def audio_out(world: HoldWorld, song: Song) -> TrackAudioOut:
    async with world.factory() as session:
        return await audio_service(session).get(song.audio_id, song.user.id, PRODUCT)


async def assert_timelines_consistent(session: AsyncSession) -> None:
    """Invariants 1-3 over every stored timeline (and every audio version's counter)."""
    rows = (await session.execute(select(StudioAudioTimeline))).scalars().all()
    audio_by_id = {
        audio.id: audio
        for audio in (
            await session.execute(
                select(StudioTrackAudio).execution_options(populate_existing=True)
            )
        )
        .scalars()
        .all()
    }
    versions = {
        version.id: version
        for version in (await session.execute(select(StudioDocumentVersion))).scalars().all()
    }
    numbers: set[tuple[UUID, int]] = set()
    for row in rows:
        audio = audio_by_id[row.audio_version_id]
        assert (row.track_id, row.user_id, row.product_id) == (
            audio.track_id,
            audio.user_id,
            audio.product_id,
        )
        assert audio.plan_version_id is not None, f"timeline {row.id} on audio without a plan"
        document = decode_stored_timeline(row.schema_version, row.content)
        plan = decode_plan(versions[audio.plan_version_id])
        lyrics = (
            None
            if audio.lyrics_version_id is None
            else decode_lyrics(versions[audio.lyrics_version_id])
        )
        issues = validate_timeline(
            document, duration_ms=audio.duration_ms, plan=plan, lyrics=lyrics
        )
        assert issues == [], f"timeline {row.id} no longer validates: {issues}"
        assert row.is_complete is is_complete(document, plan=plan, lyrics=lyrics), row.id
        assert row.content_hash == timeline_hash(document), row.id
        assert row.origin == StudioTimelineOrigin.USER.value
        assert row.version_number < audio.next_timeline_number, row.id
        assert (row.audio_version_id, row.version_number) not in numbers
        numbers.add((row.audio_version_id, row.version_number))
    timeline_ids = {row.id: row.audio_version_id for row in rows}
    for audio in audio_by_id.values():
        if audio.current_timeline_id is not None:
            assert timeline_ids.get(audio.current_timeline_id) == audio.id, "dangling pointer"
