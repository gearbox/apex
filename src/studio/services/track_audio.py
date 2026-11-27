"""Track audio versions: a track's numbered audio files (``/v1/studio/tracks/{id}/audio``, S2a).

A track holds up to ``MAX_AUDIO_VERSIONS_PER_TRACK`` audio versions. Each one **pins its library
asset** through the library's hold mechanism (``AssetHoldService``): the asset is exempt from
retention and cannot be deleted for as long as the version exists, and its retention clock restarts
when the last hold goes. The version stores no ``asset_ref`` of its own - the asset lives on the
hold row (one source of truth) and is read through a join. A version may name the plan and lyrics
versions it *realizes*; those links are validated here (typed ``404`` / ``422``) and enforced by
composite foreign keys (the backstop). Once the version has a realized timeline (S3a) its two
links are **frozen** (S3-5): a timeline's line indexes mean nothing against other lyrics. The label
stays editable, and a deleted version takes its timelines with it.

The studio reaches assets only through ``AssetHoldService`` and the library's read builders. It
never calls ingest, R2 or the upload service: the file is uploaded first
(``POST /v1/storage/upload``) and then attached by its ``asset`` ref.

Transactions and locks
----------------------
The service never commits or flushes; it runs inside the request transaction and calls the
library's ``AssetHoldService`` in the same one. **Lock order: track → audio version → (assets,
inside the hold service).** Nothing takes them the other way round. (``TimelineService`` extends
it: track → audio version → timeline.)

* Every mutation first locks the track ``FOR UPDATE``. The 20-version cap and the version number
  (``next_audio_version_number``, never ``max()+1``) are taken under that lock, so two parallel
  creates at 19 versions end in one ``409 audio_version_limit_reached``.
* A create (hold → insert) and a delete (delete → release) are each one savepoint: a failure
  leaves neither a hold without its version nor a version without its hold. Invariant after every
  committed transaction: every version has exactly one hold, and every ``studio.track_audio`` hold
  has exactly one version.

Validation follows RD6: the label is NFC-normalized, trimmed and whitespace-collapsed, then every
length is re-checked **after** normalization against the limit declared on the wire Struct - the
only source of the limit. Logs carry ids and counts only: never labels.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import (
    CursorPage,
    decode_version_number_cursor,
    encode_version_number_cursor,
)
from src.api.schemas.studio import TrackAudioCreate, TrackAudioOut, TrackAudioPatch
from src.api.services.library_holds import (
    AssetHolder,
    AssetNotEligibleError,
    AssetNotFoundError,
)
from src.api.services.library_media import build_media_object
from src.core.enums import MediaKind, StudioDocumentKind, StudioTrackAudioOrigin
from src.core.library_ref import AssetRef, LibraryAssetSource, parse_asset_ref
from src.core.uid import new_id
from src.db.integrity import violated_constraint
from src.db.models.studio import (
    TRACK_AUDIO_HOLDER_KIND,
    TRACK_AUDIO_LYRICS_FK,
    TRACK_AUDIO_PLAN_FK,
)
from src.db.repositories.output import OutputRepository
from src.db.repositories.studio import StudioDocumentVersionRepository, StudioTrackRepository
from src.db.repositories.studio_timeline import StudioAudioTimelineRepository
from src.db.repositories.studio_track_audio import StudioTrackAudioRepository, TrackAudioRow
from src.db.repositories.user_image import UserImageRepository
from src.studio.constraints import MAX_AUDIO_VERSIONS_PER_TRACK, collect_length_issues
from src.studio.errors import (
    AssetNotEligible,
    AssetNotFound,
    AudioHasTimelines,
    AudioVersionIsCurrent,
    AudioVersionLimitReached,
    AudioVersionNotFound,
    DocumentIssue,
    InvalidCursor,
    LyricsIncompatibleWithPlan,
    RealizedPlanRequired,
    StudioDocumentInvalid,
    StudioNotFound,
    VersionKindMismatch,
    VersionNotFound,
)
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.validation import check_pair_compatible

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.media import MediaObject
    from src.api.services.library_holds import AssetHoldService
    from src.db.models.studio import StudioDocumentVersion, StudioTrack

__all__ = ["TrackAudioService", "build_audio_outs"]

logger = structlog.get_logger(__name__)

_EMPTY: Final = "empty_after_normalization"
_INVALID_ASSET: Final = "invalid_asset_ref"

_ACCEPTED_KINDS: Final = frozenset({MediaKind.AUDIO})
"""A track audio version pins audio only."""

_LINK_FIELDS: Final = (
    ("plan_version_id", StudioDocumentKind.PLAN, TRACK_AUDIO_PLAN_FK),
    ("lyrics_version_id", StudioDocumentKind.LYRICS, TRACK_AUDIO_LYRICS_FK),
)

type _MediaKey = tuple[str, UUID]


def _tidy(raw: str) -> str:
    """NFC, trimmed, inner whitespace collapsed: for labels."""
    return " ".join(unicodedata.normalize("NFC", raw).split())


async def _media(
    session: AsyncSession, rows: Collection[TrackAudioRow], user_id: UUID
) -> dict[_MediaKey, MediaObject]:
    """``MediaObject`` per ``(source, asset_id)``: **one batched query per source table**.

    Through the library's one builder (never a hand-built URL). Audio has no derivatives, so no
    derivative query is made either.
    """
    upload_ids = sorted({r.asset_id for r in rows if r.source == LibraryAssetSource.UPLOAD})
    output_ids = sorted({r.asset_id for r in rows if r.source == LibraryAssetSource.OUTPUT})
    uploads = await UserImageRepository(session).get_many(upload_ids, user_id=user_id)
    generated = await OutputRepository(session).get_many(output_ids, user_id=user_id)
    media: dict[_MediaKey, MediaObject] = {}
    for row in rows:
        asset = (uploads if row.source == LibraryAssetSource.UPLOAD else generated).get(
            row.asset_id
        )
        if asset is None:
            # A hold keeps its asset alive (NO ACTION foreign key): this cannot happen.
            raise RuntimeError("A track audio version points at a library asset that is gone.")
        media[(row.source.value, row.asset_id)] = build_media_object(
            source=row.source,
            asset_id=asset.id,
            width=asset.width,
            height=asset.height,
            content_type=asset.content_type,
            size_bytes=asset.size_bytes,
            duration_ms=asset.duration_ms,
            derivatives=[],
        )
    return media


async def build_audio_outs(
    session: AsyncSession, rows: Sequence[TrackAudioRow], user_id: UUID
) -> list[TrackAudioOut]:
    """The wire items of ``rows`` (in order), their assets built in batched queries - never one
    query per item. ``is_current`` comes from the row (computed in the same statement)."""
    media = await _media(session, rows, user_id)
    return [
        TrackAudioOut(
            id=row.id,
            track_id=row.track_id,
            version_number=row.version_number,
            label=row.label,
            origin=StudioTrackAudioOrigin(row.origin),
            asset=media[(row.source.value, row.asset_id)],
            duration_ms=row.duration_ms,
            plan_version_id=row.plan_version_id,
            lyrics_version_id=row.lyrics_version_id,
            is_current=row.is_current,
            current_timeline_id=row.current_timeline_id,
            timeline_count=row.timeline_count,
            timeline_complete=row.timeline_complete,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


class TrackAudioService:
    """Business logic for a track's audio versions. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession, holds: AssetHoldService) -> None:
        self._session = session
        self._holds = holds
        self._tracks = StudioTrackRepository(session)
        self._audio = StudioTrackAudioRepository(session)
        self._versions = StudioDocumentVersionRepository(session)
        self._timelines = StudioAudioTimelineRepository(session)

    # ----------------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------------

    async def list_page(
        self,
        track_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[TrackAudioOut]:
        """The track's audio versions, newest (highest ``version_number``) first."""
        before: int | None = None
        if cursor is not None:
            try:
                before = decode_version_number_cursor(cursor)
            except ValueError as exc:
                raise InvalidCursor from exc
        if await self._tracks.get(track_id, user_id=user_id, product_id=product_id) is None:
            raise StudioNotFound
        rows = await self._audio.list_page(
            track_id, user_id=user_id, product_id=product_id, limit=limit, before_number=before
        )
        has_more = len(rows) > limit
        page = rows[:limit]
        return CursorPage(
            items=await build_audio_outs(self._session, page, user_id),
            limit=limit,
            has_more=has_more,
            next_cursor=encode_version_number_cursor(page[-1].version_number) if has_more else None,
        )

    async def get(self, audio_id: UUID, user_id: UUID, product_id: str) -> TrackAudioOut:
        row = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if row is None:
            raise AudioVersionNotFound
        return (await build_audio_outs(self._session, [row], user_id))[0]

    # ----------------------------------------------------------------------------------------
    # Create
    # ----------------------------------------------------------------------------------------

    async def create(
        self, track_id: UUID, user_id: UUID, product_id: str, data: TrackAudioCreate
    ) -> TrackAudioOut:
        """Attach a library audio asset to the track as its next audio version, pinning it.

        Raises:
            StudioDocumentInvalid: ``asset`` is not an asset ref, or the label is empty or too
                long after normalization.
            StudioNotFound: No such track for the caller.
            AudioVersionLimitReached: The track already holds the maximum number of versions.
            VersionNotFound: A realized-version link is not a version of this track.
            VersionKindMismatch: A link names a version of the other kind.
            AssetNotFound: The asset is missing, someone else's or another product's.
            AssetNotEligible: A derivative, not audio, or an output of an unavailable job.
        """
        issues: list[DocumentIssue] = []
        asset = self._asset(data.asset, "asset", issues)
        label = self._label(data.label, issues)
        collect_length_issues(msgspec.structs.replace(data, label=label), issues)
        if issues or asset is None:
            raise StudioDocumentInvalid(tuple(issues))

        track = await self._lock_track(track_id, user_id, product_id)
        if await self._audio.count_for_track(track.id) >= MAX_AUDIO_VERSIONS_PER_TRACK:
            raise AudioVersionLimitReached(MAX_AUDIO_VERSIONS_PER_TRACK)
        await self._check_links(
            track.id,
            user_id,
            product_id,
            plan_version_id=data.plan_version_id,
            lyrics_version_id=data.lyrics_version_id,
        )

        audio_id = new_id()
        try:
            async with self._session.begin_nested():
                # The counter is inside the savepoint: a refused hold burns no number.
                version_number = await self._audio.allocate_version_number(track.id)
                hold_id = await self._hold(asset, audio_id, user_id=user_id, product_id=product_id)
                duration_ms = await self._asset_duration(asset, user_id)
                await self._audio.insert(
                    audio_id=audio_id,
                    user_id=user_id,
                    product_id=product_id,
                    track_id=track.id,
                    version_number=version_number,
                    label=label,
                    origin=StudioTrackAudioOrigin.UPLOAD.value,
                    hold_id=hold_id,
                    plan_version_id=data.plan_version_id,
                    lyrics_version_id=data.lyrics_version_id,
                    duration_ms=duration_ms,
                )
        except IntegrityError as exc:
            self._raise_link_conflict(exc)
            raise
        logger.info(
            "studio.track_audio_created",
            track_id=str(track.id),
            audio_id=str(audio_id),
            version_number=version_number,
            user_id=str(user_id),
        )
        return await self._out(audio_id, user_id, product_id)

    # ----------------------------------------------------------------------------------------
    # Patch
    # ----------------------------------------------------------------------------------------

    async def patch(
        self, audio_id: UUID, user_id: UUID, product_id: str, data: TrackAudioPatch
    ) -> TrackAudioOut:
        """Change the label and / or the realized-version links. The asset is immutable.

        Raises:
            AudioVersionNotFound: No such audio version for the caller.
            StudioDocumentInvalid: The label is empty or too long after normalization.
            AudioHasTimelines: A link would change, and the version has timelines (S3-5).
            VersionNotFound: A link is not a version of this track.
            VersionKindMismatch: A link names a version of the other kind.
        """
        issues: list[DocumentIssue] = []
        label: str | msgspec.UnsetType | None = data.label
        if isinstance(label, str):
            label = self._label(label, issues)
            collect_length_issues(msgspec.structs.replace(data, label=label), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        current = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if current is None:
            raise AudioVersionNotFound
        # Track first, then the version (the lock order); the version may have been deleted while
        # this request waited for the track.
        await self._lock_track(current.track_id, user_id, product_id)
        locked = await self._audio.lock(audio_id, user_id=user_id, product_id=product_id)
        if locked is None:
            raise AudioVersionNotFound
        plan_version_id = (
            locked.plan_version_id
            if isinstance(data.plan_version_id, msgspec.UnsetType)
            else data.plan_version_id
        )
        lyrics_version_id = (
            locked.lyrics_version_id
            if isinstance(data.lyrics_version_id, msgspec.UnsetType)
            else data.lyrics_version_id
        )
        # Re-sending the stored links changes nothing and stays allowed. The count is taken
        # under the track lock, which creating a timeline needs too.
        if (plan_version_id, lyrics_version_id) != (
            locked.plan_version_id,
            locked.lyrics_version_id,
        ):
            timelines = await self._timelines.count_for_audio(
                audio_id, user_id=user_id, product_id=product_id
            )
            if timelines:
                raise AudioHasTimelines(timelines)
        await self._check_links(
            current.track_id,
            user_id,
            product_id,
            plan_version_id=plan_version_id,
            lyrics_version_id=lyrics_version_id,
        )
        if (
            isinstance(label, msgspec.UnsetType)
            and isinstance(data.plan_version_id, msgspec.UnsetType)
            and isinstance(data.lyrics_version_id, msgspec.UnsetType)
        ):
            return await self._out(audio_id, user_id, product_id)
        try:
            async with self._session.begin_nested():
                await self._audio.update(
                    audio_id,
                    user_id=user_id,
                    product_id=product_id,
                    label=label,
                    plan_version_id=data.plan_version_id,
                    lyrics_version_id=data.lyrics_version_id,
                )
        except IntegrityError as exc:
            self._raise_link_conflict(exc)
            raise
        logger.info("studio.track_audio_updated", audio_id=str(audio_id), user_id=str(user_id))
        return await self._out(audio_id, user_id, product_id)

    # ----------------------------------------------------------------------------------------
    # Delete
    # ----------------------------------------------------------------------------------------

    async def delete(self, audio_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a version that is not the track's current audio, and release its hold.

        The row and its hold go in one savepoint; the last release of an asset re-arms its
        retention clock (inside ``AssetHoldService``). Its timelines go with it (``ON DELETE
        CASCADE``, S3a).

        Raises:
            AudioVersionNotFound: No such audio version for the caller.
            AudioVersionIsCurrent: It is the track's current audio.
        """
        row = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if row is None:
            raise AudioVersionNotFound
        track = await self._lock_track(row.track_id, user_id, product_id)
        locked = await self._audio.lock(audio_id, user_id=user_id, product_id=product_id)
        if locked is None:  # deleted while waiting for the track lock
            raise AudioVersionNotFound
        if track.current_audio_version_id == audio_id:
            raise AudioVersionIsCurrent
        try:
            async with self._session.begin_nested():
                await self._audio.delete(audio_id, user_id=user_id, product_id=product_id)
                # The row went first: it is what kept its hold from being deleted.
                await self._holds.release([locked.hold_id])
        except IntegrityError as exc:
            # The track's pointer is changed only under the track lock this request holds, so the
            # NO ACTION foreign key cannot fire here; if it does, say what it means.
            if violated_constraint(exc, frozenset({"fk_studio_tracks_current_audio_version"})):
                raise AudioVersionIsCurrent from exc
            raise
        logger.info(
            "studio.track_audio_deleted",
            track_id=str(row.track_id),
            audio_id=str(audio_id),
            user_id=str(user_id),
        )

    # ----------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------

    async def _lock_track(self, track_id: UUID, user_id: UUID, product_id: str) -> StudioTrack:
        track = await self._tracks.lock(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        return track

    @staticmethod
    def _asset(raw: str, path: str, issues: list[DocumentIssue]) -> AssetRef | None:
        try:
            return parse_asset_ref(raw)
        except ValueError:
            issues.append(DocumentIssue(path, _INVALID_ASSET))
            return None

    @staticmethod
    def _label(raw: str | None, issues: list[DocumentIssue]) -> str | None:
        """A label, normalized; ``None`` stays ``None``; recorded as an issue if nothing is left."""
        if raw is None:
            return None
        text = _tidy(raw)
        if not text:
            issues.append(DocumentIssue("label", _EMPTY))
        return text

    async def _check_links(
        self,
        track_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        plan_version_id: UUID | None,
        lyrics_version_id: UUID | None,
    ) -> None:
        """Validate the plan/lyrics links as one pair under the track lock.

        Runs under the track lock, which a version delete takes too, so a version cannot vanish
        between this check and the insert.
        """
        plan_row: StudioDocumentVersion | None = None
        if plan_version_id is not None:
            plan_row = await self._versions.get_on_track(
                plan_version_id, track_id, user_id=user_id, product_id=product_id
            )
            if plan_row is None:
                raise VersionNotFound
            if plan_row.kind != StudioDocumentKind.PLAN.value:
                raise VersionKindMismatch("plan_version_id", StudioDocumentKind.PLAN.value)

        lyrics_row: StudioDocumentVersion | None = None
        if lyrics_version_id is not None:
            lyrics_row = await self._versions.get_on_track(
                lyrics_version_id, track_id, user_id=user_id, product_id=product_id
            )
            if lyrics_row is None:
                raise VersionNotFound
            if lyrics_row.kind != StudioDocumentKind.LYRICS.value:
                raise VersionKindMismatch("lyrics_version_id", StudioDocumentKind.LYRICS.value)

        if lyrics_version_id is not None and plan_version_id is None:
            raise RealizedPlanRequired
        if plan_row is None or lyrics_row is None:
            return

        plan = decode_plan(plan_row)
        lyrics = decode_lyrics(lyrics_row)
        incompatibilities = check_pair_compatible(plan, lyrics)
        if incompatibilities:
            raise LyricsIncompatibleWithPlan(incompatibilities)

    @staticmethod
    def _raise_link_conflict(exc: IntegrityError) -> None:
        """A link FK that fires despite the check (the version vanished): the typed 404."""
        if violated_constraint(exc, frozenset(fk for _, _, fk in _LINK_FIELDS)) is not None:
            raise VersionNotFound from exc

    async def _hold(
        self, asset: AssetRef, audio_id: UUID, *, user_id: UUID, product_id: str
    ) -> UUID:
        """Pin ``asset`` for the version; the library's errors become the studio's."""
        try:
            return await self._holds.hold(
                asset,
                holder=AssetHolder(kind=TRACK_AUDIO_HOLDER_KIND, id=audio_id),
                user_id=user_id,
                product_id=product_id,
                accept=_ACCEPTED_KINDS,
            )
        except AssetNotFoundError as exc:
            raise AssetNotFound from exc
        except AssetNotEligibleError as exc:
            raise AssetNotEligible(exc.reason.value) from exc

    async def _asset_duration(self, asset: AssetRef, user_id: UUID) -> int:
        """The pinned asset's duration, read **after** the hold took its lock on the asset row.

        The row is held ``FOR KEY SHARE`` until this transaction ends, and an asset is immutable,
        so the value cannot change underneath the snapshot.
        """
        duration_ms: int | None
        if asset.source == LibraryAssetSource.UPLOAD:
            uploads = await UserImageRepository(self._session).get_many(
                [asset.asset_id], user_id=user_id
            )
            upload = uploads.get(asset.asset_id)
            duration_ms = upload.duration_ms if upload is not None else None
        else:
            outputs = await OutputRepository(self._session).get_many(
                [asset.asset_id], user_id=user_id
            )
            output = outputs.get(asset.asset_id)
            duration_ms = output.duration_ms if output is not None else None
        if duration_ms is None or duration_ms <= 0:
            # An audio asset always has its probed duration: a broken invariant, never a 4xx.
            raise RuntimeError("A held audio asset has no usable duration.")
        return duration_ms

    async def _out(self, audio_id: UUID, user_id: UUID, product_id: str) -> TrackAudioOut:
        row = await self._audio.get(audio_id, user_id=user_id, product_id=product_id)
        if row is None:  # pragma: no cover - the track lock rules out a concurrent delete
            raise AudioVersionNotFound
        return (await build_audio_outs(self._session, [row], user_id))[0]


def _given(value: UUID | msgspec.UnsetType | None) -> UUID | None:
    """The id of a link that is being **set**; ``None`` for omitted and for "clear" (no check)."""
    return None if isinstance(value, msgspec.UnsetType) else value
