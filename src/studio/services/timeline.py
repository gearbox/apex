"""Realized timelines of an audio version (``/v1/studio/audio/{id}/timelines``, S3a).

A timeline places the plan sections - and optionally the lyric lines - that its audio version links
in time. Timelines are **immutable, numbered versions** in a parent tree, so a later aligner result
(S3b) can never overwrite the user's corrections; one of them may be the audio version's current
timeline. Create is naturally idempotent, like document versions: an identical sibling (same
parent and content) is replayed, with no new row and no new number.

Once an audio version has any timeline, its plan and lyrics links are frozen (S3-5,
``TrackAudioService.patch``). With immutable documents and an immutable audio duration, that means
every stored timeline stays valid against its audio's duration, plan and lyrics, and its
``is_complete`` flag, computed once at write time, stays true to its content.

Transactions and locks
----------------------
The service never commits or flushes; it runs inside the request transaction. **Lock order: track →
audio version → timeline** (it extends the track-audio order). Every write resolves the audio, locks
its track ``FOR UPDATE``, then the audio version ``FOR UPDATE``, and reads everything it decides on
after those locks: the links, the current pointer, the cap and the counter.

Errors raised after a write roll the request back, so a refusal never leaves a partial change.
Logs carry ids, numbers, counts and codes: **never lyric text, word text or a section label.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import (
    CursorPage,
    decode_version_number_cursor,
    encode_version_number_cursor,
)
from src.api.schemas.studio import (
    TimelineExportOut,
    TimelineExportWarningOut,
    TimelineOut,
    TimelineSummaryOut,
)
from src.core.enums import StudioTimedLyricsFormat, StudioTimelineOrigin
from src.db.integrity import violated_constraint
from src.db.models.studio import TIMELINE_PARENT_FK, TRACK_AUDIO_CURRENT_TIMELINE_FK
from src.db.repositories.studio import StudioDocumentVersionRepository, StudioTrackRepository
from src.db.repositories.studio_timeline import StudioAudioTimelineRepository
from src.db.repositories.studio_track_audio import StudioTrackAudioRepository
from src.studio.constraints import MAX_TIMELINES_PER_AUDIO
from src.studio.errors import (
    AudioHasNoPlan,
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
from src.studio.renderers.timed_lyrics import TIMED_LYRICS_RENDERERS, project_timed_lyrics
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.services.track_audio import build_audio_outs
from src.studio.timeline import (
    TIMELINE_SCHEMA_VERSION,
    TimelineDocument,
    check_timeline_limits,
    decode_stored_timeline,
    is_complete,
    normalize_timeline,
    timeline_content,
    timeline_hash,
    validate_timeline,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import TimelineCreate, TrackAudioOut
    from src.db.models.studio import StudioDocumentVersion, StudioTrackAudio
    from src.db.repositories.studio_timeline import TimelineRow, TimelineSummaryRow
    from src.studio.documents import LyricsDocument, PlanDocument

__all__ = ["TimelineService"]

logger = structlog.get_logger(__name__)


def _summary_out(row: TimelineSummaryRow) -> TimelineSummaryOut:
    return TimelineSummaryOut(
        id=row.id,
        audio_version_id=row.audio_version_id,
        track_id=row.track_id,
        version_number=row.version_number,
        parent_id=row.parent_id,
        origin=StudioTimelineOrigin(row.origin),
        is_complete=row.is_complete,
        is_current=row.is_current,
        created_at=row.created_at,
    )


def _timeline_out(row: TimelineRow) -> TimelineOut:
    summary = _summary_out(row)
    return TimelineOut(
        id=summary.id,
        audio_version_id=summary.audio_version_id,
        track_id=summary.track_id,
        version_number=summary.version_number,
        parent_id=summary.parent_id,
        origin=summary.origin,
        is_complete=summary.is_complete,
        is_current=summary.is_current,
        created_at=summary.created_at,
        content=decode_stored_timeline(row.schema_version, row.content),
    )


class TimelineService:
    """Business logic for realized timelines. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._tracks = StudioTrackRepository(session)
        self._audio = StudioTrackAudioRepository(session)
        self._versions = StudioDocumentVersionRepository(session)
        self._timelines = StudioAudioTimelineRepository(session)

    # ----------------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------------

    async def get(self, timeline_id: UUID, user_id: UUID, product_id: str) -> TimelineOut:
        row = await self._timelines.get(timeline_id, user_id=user_id, product_id=product_id)
        if row is None:
            raise TimelineNotFound
        return _timeline_out(row)

    async def list_page(
        self,
        audio_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[TimelineSummaryOut]:
        """The audio version's timelines, newest (highest ``version_number``) first, without
        their documents."""
        before: int | None = None
        if cursor is not None:
            try:
                before = decode_version_number_cursor(cursor)
            except ValueError as exc:
                raise InvalidCursor from exc
        if await self._audio.get(audio_id, user_id=user_id, product_id=product_id) is None:
            raise AudioVersionNotFound
        rows = await self._timelines.list_page(
            audio_id, user_id=user_id, product_id=product_id, limit=limit, before_number=before
        )
        has_more = len(rows) > limit
        page = rows[:limit]
        return CursorPage(
            items=[_summary_out(row) for row in page],
            limit=limit,
            has_more=has_more,
            next_cursor=encode_version_number_cursor(page[-1].version_number) if has_more else None,
        )

    async def export(
        self, timeline_id: UUID, export_format: str, user_id: UUID, product_id: str
    ) -> TimelineExportOut:
        """Render a timeline as a lyric file. A pure read: no lock, nothing written.

        Raises:
            StudioNotFound: ``export_format`` is not a known format.
            TimelineNotFound: No such timeline for the caller.
        """
        try:
            target = StudioTimedLyricsFormat(export_format)
        except ValueError as exc:
            raise StudioNotFound from exc
        found = await self._timelines.get_for_export(
            timeline_id, user_id=user_id, product_id=product_id
        )
        if found is None:
            raise TimelineNotFound
        document = decode_stored_timeline(found.timeline.schema_version, found.timeline.content)
        projection = None
        if found.lyrics_version_id is not None:
            if found.lyrics_row is None:
                # A composite NO ACTION foreign key keeps a linked version alive.
                raise RuntimeError("An audio version links a document version that is gone.")
            projection = project_timed_lyrics(document, decode_lyrics(found.lyrics_row))
        rendered = TIMED_LYRICS_RENDERERS[target].render(projection)
        logger.info(
            "studio.timeline_exported",
            timeline_id=str(timeline_id),
            format=target.value,
            warnings=len(rendered.warnings),
        )
        return TimelineExportOut(
            timeline_id=timeline_id,
            format=target,
            text=rendered.text,
            warnings=[
                TimelineExportWarningOut(code=warning.code, count=warning.count)
                for warning in rendered.warnings
            ],
        )

    # ----------------------------------------------------------------------------------------
    # Create
    # ----------------------------------------------------------------------------------------

    async def create(
        self, audio_id: UUID, user_id: UUID, product_id: str, data: TimelineCreate
    ) -> tuple[TimelineOut, bool]:
        """Save a timeline of the audio version, or replay an identical one.

        A replay (same parent and normalized content) writes no row and takes no number, and
        ``make_current`` is still applied.

        Returns:
            ``(timeline, created)``: the route answers 201 when a row was written, 200 on replay.

        Raises:
            AudioVersionNotFound: No such audio version for the caller.
            AudioHasNoPlan: The audio version links no plan.
            StudioDocumentInvalid: The timeline failed validation (all issues).
            TimelineNotFound: ``parent_id`` is not a timeline of this audio version.
            TimelineUnchanged: The content equals the parent's.
            TimelineLimitReached: The audio version holds the maximum number of timelines.
        """
        document = normalize_timeline(data.content)
        issues = check_timeline_limits(document)

        audio = await self._lock_audio(audio_id, user_id, product_id)
        if audio.plan_version_id is None:
            raise AudioHasNoPlan
        plan, lyrics = await self._linked_documents(
            audio, audio.plan_version_id, user_id, product_id
        )
        issues.extend(
            validate_timeline(document, duration_ms=audio.duration_ms, plan=plan, lyrics=lyrics)
        )
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        parent = (
            None
            if data.parent_id is None
            else await self._timeline_of(data.parent_id, audio_id, user_id, product_id)
        )

        digest = timeline_hash(document)
        existing = await self._timelines.find_sibling(
            audio_id, data.parent_id, digest, user_id=user_id, product_id=product_id
        )
        if existing is not None:
            timeline_id, created = existing, False
        else:
            if parent is not None and parent.content_hash == digest:
                raise TimelineUnchanged
            held = await self._timelines.count_for_audio(
                audio_id, user_id=user_id, product_id=product_id
            )
            if held >= MAX_TIMELINES_PER_AUDIO:
                raise TimelineLimitReached(MAX_TIMELINES_PER_AUDIO)
            content = timeline_content(document)
            self._assert_decodes(content)
            async with self._session.begin_nested():
                # The counter is inside the savepoint: a refused insert burns no number.
                timeline_id = await self._timelines.insert(
                    user_id=user_id,
                    product_id=product_id,
                    track_id=audio.track_id,
                    audio_version_id=audio_id,
                    version_number=await self._timelines.allocate_number(audio_id),
                    parent_id=data.parent_id,
                    origin=StudioTimelineOrigin.USER.value,
                    schema_version=TIMELINE_SCHEMA_VERSION,
                    content=content,
                    content_hash=digest,
                    is_complete=is_complete(document, plan=plan, lyrics=lyrics),
                )
            created = True

        if data.make_current and audio.current_timeline_id != timeline_id:
            await self._timelines.set_current(audio_id, timeline_id)
        row = await self._timelines.get(timeline_id, user_id=user_id, product_id=product_id)
        if row is None:  # pragma: no cover - the locks rule out a concurrent delete
            raise TimelineNotFound
        logger.info(
            "studio.timeline_created" if created else "studio.timeline_replayed",
            audio_id=str(audio_id),
            timeline_id=str(timeline_id),
            version_number=row.version_number,
            sections=len(document.sections),
            lines=len(document.lines),
            is_complete=row.is_complete,
            make_current=data.make_current,
        )
        return _timeline_out(row), created

    @staticmethod
    def _assert_decodes(content: dict[str, object]) -> None:
        """Invariant: every stored timeline decodes (RD7).

        Decodes the exact JSONB ``content`` that is about to be inserted. Normalization and the
        length re-check are meant to make a failure impossible; this is the backstop for a limit
        they missed, because one unreadable row would 500 every read of the audio's timelines.

        Raises:
            RuntimeError: ``content`` would not decode. A bug, never a client error; nothing has
                been written (no row, no number).
        """
        try:
            msgspec.json.decode(msgspec.json.encode(content), type=TimelineDocument)
        except msgspec.ValidationError as exc:
            raise RuntimeError("studio timeline would not decode after normalization") from exc

    # ----------------------------------------------------------------------------------------
    # Current pointer
    # ----------------------------------------------------------------------------------------

    async def set_current(
        self, audio_id: UUID, user_id: UUID, product_id: str, timeline_id: UUID | None
    ) -> TrackAudioOut:
        """Choose the audio version's current timeline; ``None`` clears it.

        Raises:
            AudioVersionNotFound: No such audio version for the caller.
            TimelineNotFound: ``timeline_id`` is not a timeline of this audio version.
        """
        audio = await self._lock_audio(audio_id, user_id, product_id)
        if timeline_id is not None:
            await self._timeline_of(timeline_id, audio_id, user_id, product_id)
        if audio.current_timeline_id != timeline_id:
            try:
                async with self._session.begin_nested():
                    await self._timelines.set_current(audio_id, timeline_id)
            except IntegrityError as exc:
                # The composite same-audio foreign key: the backstop of the check above.
                if violated_constraint(exc, frozenset({TRACK_AUDIO_CURRENT_TIMELINE_FK})):
                    raise TimelineNotFound from exc
                raise
            logger.info(
                "studio.current_timeline_set",
                audio_id=str(audio_id),
                timeline_id=str(timeline_id) if timeline_id else None,
            )
        row = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if row is None:  # pragma: no cover - the track lock rules out a concurrent delete
            raise AudioVersionNotFound
        return (await build_audio_outs(self._session, [row], user_id))[0]

    # ----------------------------------------------------------------------------------------
    # Delete
    # ----------------------------------------------------------------------------------------

    async def delete(self, timeline_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a timeline that is neither current nor another's parent.

        Its number is not reused.

        Raises:
            TimelineNotFound: No such timeline for the caller.
            TimelineIsCurrent: It is the audio version's current timeline.
            TimelineHasChildren: Other timelines list it as their parent.
        """
        found = await self._timelines.get(timeline_id, user_id=user_id, product_id=product_id)
        if found is None:
            raise TimelineNotFound
        try:
            audio = await self._lock_audio(found.audio_version_id, user_id, product_id)
        except AudioVersionNotFound as exc:
            # The audio version went while this request waited: its timelines went with it.
            raise TimelineNotFound from exc
        # The timeline may have been deleted while this request waited for the locks.
        if await self._timelines.get(timeline_id, user_id=user_id, product_id=product_id) is None:
            raise TimelineNotFound
        if audio.current_timeline_id == timeline_id:
            raise TimelineIsCurrent
        if await self._timelines.has_children(timeline_id, user_id=user_id, product_id=product_id):
            raise TimelineHasChildren
        try:
            async with self._session.begin_nested():
                await self._timelines.delete(timeline_id, user_id=user_id, product_id=product_id)
        except IntegrityError as exc:
            # Both pointers change only under the locks this request holds, so these NO ACTION
            # keys cannot fire here; if one does, say what it means.
            violated = violated_constraint(
                exc, frozenset({TRACK_AUDIO_CURRENT_TIMELINE_FK, TIMELINE_PARENT_FK})
            )
            if violated == TRACK_AUDIO_CURRENT_TIMELINE_FK:
                raise TimelineIsCurrent from exc
            if violated == TIMELINE_PARENT_FK:
                raise TimelineHasChildren from exc
            raise
        logger.info(
            "studio.timeline_deleted",
            audio_id=str(found.audio_version_id),
            timeline_id=str(timeline_id),
            version_number=found.version_number,
        )

    # ----------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------

    async def _lock_audio(self, audio_id: UUID, user_id: UUID, product_id: str) -> StudioTrackAudio:
        """Resolve the audio version, lock its track, then lock it (track → audio version).

        The version may be deleted (alone or with its track) while this request waits for the
        track, so it is read again, under its own lock, once the track is held. The returned row's
        links and current pointer are therefore the committed ones.

        Raises:
            AudioVersionNotFound: No such audio version for the caller (any more).
        """
        found = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if found is None:
            raise AudioVersionNotFound
        track = await self._tracks.lock(found.track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise AudioVersionNotFound
        locked = await self._audio.lock(audio_id, user_id=user_id, product_id=product_id)
        if locked is None:
            raise AudioVersionNotFound
        return locked

    async def _timeline_of(
        self, timeline_id: UUID, audio_id: UUID, user_id: UUID, product_id: str
    ) -> TimelineRow:
        """A timeline of **this** audio version (another one's is simply "not found")."""
        row = await self._timelines.get(timeline_id, user_id=user_id, product_id=product_id)
        if row is None or row.audio_version_id != audio_id:
            raise TimelineNotFound
        return row

    async def _linked_documents(
        self, audio: StudioTrackAudio, plan_version_id: UUID, user_id: UUID, product_id: str
    ) -> tuple[PlanDocument, LyricsDocument | None]:
        """The plan and the optional lyrics the (locked) audio version links."""
        plan = decode_plan(
            await self._linked_row(plan_version_id, audio.track_id, user_id, product_id)
        )
        if audio.lyrics_version_id is None:
            return plan, None
        return plan, decode_lyrics(
            await self._linked_row(audio.lyrics_version_id, audio.track_id, user_id, product_id)
        )

    async def _linked_row(
        self, version_id: UUID, track_id: UUID, user_id: UUID, product_id: str
    ) -> StudioDocumentVersion:
        row = await self._versions.get_on_track(
            version_id, track_id, user_id=user_id, product_id=product_id
        )
        if row is None:
            # A composite NO ACTION foreign key keeps a linked version alive, and a version
            # delete takes the track lock this request holds.
            raise RuntimeError("An audio version links a document version that is gone.")
        return row
