"""Document versions: create (idempotent), list, read, switch current, delete.

All mutations run in the request transaction under the **track row lock**, which serializes
version numbering, the (plan, lyrics) pair and deletion against each other. Errors raised after
a write simply roll the request back (``get_db_session`` rolls back on exceptions), so a 409
never leaves a partial change behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.schemas.studio import (
    LyricsVersionDetail,
    PlanVersionCreate,
    PlanVersionDetail,
    TrackOut,
    VersionCreate,
    VersionCreateOut,
    VersionSummary,
)
from src.core.enums import StudioDocumentKind, StudioIncompatibleLyricsPolicy
from src.db.repositories.studio import StudioDocumentVersionRepository, StudioTrackRepository
from src.studio.documents import (
    CURRENT_SCHEMA_VERSION,
    LyricsDocument,
    PlanDocument,
    decode_stored,
)
from src.studio.errors import (
    DocumentIssue,
    InvalidCursor,
    LyricsIncompatibleWithPlan,
    ParentVersionNotFound,
    PlanRequired,
    PlanVersionNotFound,
    StudioDocumentInvalid,
    StudioError,
    StudioNotFound,
    VersionHasChildren,
    VersionIsCurrent,
    VersionIsLyricsBase,
    VersionLimitReached,
    VersionUnchanged,
)
from src.studio.services.mappers import (
    decode_lyrics,
    decode_plan,
    require_current_row,
    track_out,
    version_detail,
    version_summary,
)
from src.studio.validation import (
    check_pair_compatible,
    content_hash,
    normalize_lyrics,
    normalize_plan,
    to_content,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.enums import StudioDocumentOrigin
    from src.db.models.studio import StudioDocumentVersion, StudioTrack

__all__ = ["MAX_VERSIONS_PER_KIND", "DocumentVersionService"]

logger = structlog.get_logger(__name__)

MAX_VERSIONS_PER_KIND: Final = 500
"""Cap per (track, kind). Deleting a version frees room; numbers are never reused."""


class DocumentVersionService:
    """Versioned Plan / Lyrics documents of a track."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._tracks = StudioTrackRepository(session)
        self._versions = StudioDocumentVersionRepository(session)

    # -- helpers -----------------------------------------------------------------------------

    async def _lock_track(self, track_id: UUID, user_id: UUID, product_id: str) -> StudioTrack:
        track = await self._tracks.lock(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        return track

    async def _version_of(
        self,
        version_id: UUID,
        track_id: UUID,
        kind: StudioDocumentKind,
        user_id: UUID,
        product_id: str,
        *,
        missing: type[StudioError],
    ) -> StudioDocumentVersion:
        """A version of this track *and kind*, or ``missing``."""
        version = await self._versions.get_on_track(
            version_id, track_id, user_id=user_id, product_id=product_id
        )
        if version is None or version.kind != kind.value:
            raise missing
        return version

    async def _pointer_row(
        self, version_id: UUID, track: StudioTrack, user_id: UUID, product_id: str
    ) -> StudioDocumentVersion:
        """The row a non-null current pointer of ``track`` refers to.

        Raises:
            RuntimeError: There is no such row (a broken invariant, never "no document").
        """
        return require_current_row(
            await self._versions.get_on_track(
                version_id, track.id, user_id=user_id, product_id=product_id
            )
        )

    async def _current_plan_doc(
        self, track: StudioTrack, user_id: UUID, product_id: str
    ) -> PlanDocument | None:
        """The current plan, or ``None`` when the track has none (a null pointer)."""
        if track.current_plan_version_id is None:
            return None
        row = await self._pointer_row(track.current_plan_version_id, track, user_id, product_id)
        return decode_plan(row)

    async def _current_lyrics_doc(
        self, track: StudioTrack, user_id: UUID, product_id: str
    ) -> LyricsDocument | None:
        """The current lyrics, or ``None`` when the track has none (a null pointer)."""
        if track.current_lyrics_version_id is None:
            return None
        row = await self._pointer_row(track.current_lyrics_version_id, track, user_id, product_id)
        return decode_lyrics(row)

    # -- create ------------------------------------------------------------------------------

    async def create(
        self,
        track_id: UUID,
        request: VersionCreate,
        user_id: UUID,
        product_id: str,
        *,
        origin: StudioDocumentOrigin,
    ) -> tuple[VersionCreateOut, bool]:
        """Create a version — or replay it.

        Create is naturally idempotent: an identical sibling (same track, kind, parent and
        normalized content) is returned instead of duplicated, and ``make_current`` is still
        applied, so the call means "this version exists and, if asked, is current".

        Args:
            track_id: Track to write to.
            request: The plan or lyrics version to create.
            user_id: Owner scope.
            product_id: Product scope.
            origin: Who produced the content (stored on a new row; required so no caller can
                forget to say — the public route passes ``user``, the Suno importer ``import``).

        Returns:
            ``(response, created)`` — the route answers 201 when a row was written, 200 on replay.

        Raises:
            StudioNotFound: The track is not the caller's.
            ParentVersionNotFound: The parent is not a version of this track and kind.
            PlanVersionNotFound: The lyrics' base is not a plan version of this track.
            StudioDocumentInvalid: The document failed validation (all issues), or
                ``incompatible_lyrics="detach"`` was sent with ``make_current=False``
                (``requires_make_current``; raised before anything is read).
            VersionUnchanged: The content equals the parent's.
            LyricsIncompatibleWithPlan: ``make_current`` would make an incompatible pair current.
            PlanRequired: Lyrics were made current on a track without a plan.
            VersionLimitReached: The per-kind cap is reached (replays are exempt).
        """
        if (
            isinstance(request, PlanVersionCreate)
            and request.incompatible_lyrics is StudioIncompatibleLyricsPolicy.DETACH
            and not request.make_current
        ):
            # "Detach" is a statement about the *current* pair; without make_current there is
            # no change to the pair, so the request contradicts itself. Refused before any read.
            raise StudioDocumentInvalid(
                (DocumentIssue("incompatible_lyrics", "requires_make_current"),)
            )
        track = await self._lock_track(track_id, user_id, product_id)
        kind = (
            StudioDocumentKind.PLAN
            if isinstance(request, PlanVersionCreate)
            else StudioDocumentKind.LYRICS
        )

        parent: StudioDocumentVersion | None = None
        if request.parent_version_id is not None:
            parent = await self._version_of(
                request.parent_version_id,
                track_id,
                kind,
                user_id,
                product_id,
                missing=ParentVersionNotFound,
            )

        # Each branch keeps its narrowly typed document; the shared tail only needs the union.
        document: PlanDocument | LyricsDocument
        based_on: UUID | None
        detach_lyrics = False
        if isinstance(request, PlanVersionCreate):
            based_on = None
            plan_document = normalize_plan(request.content)
            document = plan_document
            digest = content_hash(kind, based_on, document)
            if parent is not None and parent.content_hash == digest:
                raise VersionUnchanged
            # Decide the resulting current pair *before* writing anything.
            if request.make_current:
                detach_lyrics = await self._check_plan_against_current_lyrics(
                    track, request.incompatible_lyrics, plan_document, user_id, product_id
                )
        else:
            based_on = request.based_on_plan_version_id
            base_plan = await self._version_of(
                based_on,
                track_id,
                StudioDocumentKind.PLAN,
                user_id,
                product_id,
                missing=PlanVersionNotFound,
            )
            lyrics_document = normalize_lyrics(request.content, decode_plan(base_plan))
            document = lyrics_document
            digest = content_hash(kind, based_on, document)
            if parent is not None and parent.content_hash == digest:
                raise VersionUnchanged
            if request.make_current:
                await self._check_lyrics_against_current_plan(
                    track, lyrics_document, user_id, product_id
                )

        existing = await self._versions.find_sibling(
            track_id=track_id,
            kind=kind,
            parent_version_id=request.parent_version_id,
            content_hash=digest,
        )
        if existing is not None:
            row, created = existing, False
        else:
            if await self._versions.count(track_id, kind) >= MAX_VERSIONS_PER_KIND:
                raise VersionLimitReached
            content = to_content(document)
            self._assert_decodes(kind, content)
            row, created = await self._versions.insert_or_get_sibling(
                user_id=user_id,
                product_id=product_id,
                track_id=track_id,
                kind=kind,
                version_number=await self._tracks.allocate_version_number(track_id, kind),
                parent_version_id=request.parent_version_id,
                based_on_plan_version_id=based_on,
                schema_version=CURRENT_SCHEMA_VERSION[kind],
                origin=origin.value,
                content=content,
                content_hash=digest,
            )

        if request.make_current:
            track = await self._apply_current(track, row, detach_lyrics=detach_lyrics)

        logger.info(
            "studio.version_created" if created else "studio.version_replayed",
            track_id=str(track_id),
            version_id=str(row.id),
            kind=kind.value,
            version_number=row.version_number,
            make_current=request.make_current,
            lyrics_detached=detach_lyrics,
        )
        return (
            VersionCreateOut(
                version=version_detail(row, track, document), lyrics_detached=detach_lyrics
            ),
            created,
        )

    @staticmethod
    def _assert_decodes(kind: StudioDocumentKind, content: dict[str, object]) -> None:
        """Invariant: every stored version decodes (RD7).

        Decodes the exact JSONB ``content`` that is about to be inserted, with the strict read
        path. Normalization + the length re-check are meant to make a failure impossible; this is
        the backstop for a limit they missed — one unreadable row would 500 every read of its
        version, its track and its export for good.

        Raises:
            RuntimeError: ``content`` would not decode. A bug, never a client error; nothing has
                been written (no row, no version number).
        """
        try:
            decode_stored(kind, CURRENT_SCHEMA_VERSION[kind], content)
        except msgspec.ValidationError as exc:
            raise RuntimeError("studio document would not decode after normalization") from exc

    async def _check_plan_against_current_lyrics(
        self,
        track: StudioTrack,
        policy: StudioIncompatibleLyricsPolicy,
        plan: PlanDocument,
        user_id: UUID,
        product_id: str,
    ) -> bool:
        """Whether the current lyrics must be detached for ``plan`` to become current.

        Raises:
            LyricsIncompatibleWithPlan: Incompatible lyrics and the policy is ``reject``.
        """
        lyrics = await self._current_lyrics_doc(track, user_id, product_id)
        if lyrics is None:
            return False
        issues = check_pair_compatible(plan, lyrics)
        if not issues:
            return False
        if policy is StudioIncompatibleLyricsPolicy.DETACH:
            return True
        raise LyricsIncompatibleWithPlan(issues)

    async def _check_lyrics_against_current_plan(
        self, track: StudioTrack, lyrics: LyricsDocument, user_id: UUID, product_id: str
    ) -> None:
        plan = await self._current_plan_doc(track, user_id, product_id)
        if plan is None:
            raise PlanRequired
        issues = check_pair_compatible(plan, lyrics)
        if issues:
            raise LyricsIncompatibleWithPlan(issues)

    async def _apply_current(
        self, track: StudioTrack, row: StudioDocumentVersion, *, detach_lyrics: bool
    ) -> StudioTrack:
        """Point the track at ``row`` — a no-op write when it already is (idempotent replay)."""
        if row.kind == StudioDocumentKind.PLAN.value:
            if track.current_plan_version_id == row.id and not detach_lyrics:
                return track
            return await self._tracks.set_current(
                track.id,
                plan_version_id=row.id,
                lyrics_version_id=None if detach_lyrics else msgspec.UNSET,
            )
        if track.current_lyrics_version_id == row.id:
            return track
        return await self._tracks.set_current(track.id, lyrics_version_id=row.id)

    # -- read --------------------------------------------------------------------------------

    async def list_page(
        self,
        track_id: UUID,
        kind: StudioDocumentKind | None,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[VersionSummary]:
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        track = await self._tracks.get(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound
        rows = await self._versions.list_page(
            track_id,
            kind,
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
        return CursorPage(
            items=[version_summary(v, track) for v in page],
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def get(
        self, version_id: UUID, user_id: UUID, product_id: str
    ) -> PlanVersionDetail | LyricsVersionDetail:
        version = await self._versions.get(version_id, user_id=user_id, product_id=product_id)
        if version is None:
            raise StudioNotFound
        track = await self._tracks.get(version.track_id, user_id=user_id, product_id=product_id)
        if track is None:  # pragma: no cover - versions cascade with their track
            raise StudioNotFound
        return version_detail(version, track)

    # -- switch current ----------------------------------------------------------------------

    async def set_current(
        self,
        track_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        plan_version_id: UUID | msgspec.UnsetType,
        lyrics_version_id: UUID | msgspec.UnsetType | None,
    ) -> TrackOut:
        """Switch the track's current plan and/or lyrics.

        Omitted arguments keep their pointer; ``lyrics_version_id=None`` detaches the lyrics.
        The resulting pair must be compatible.

        Raises:
            StudioNotFound: The track, or a given version of the matching kind, is not found.
            PlanRequired: Lyrics would be current without a plan.
            LyricsIncompatibleWithPlan: The resulting pair is incompatible.
        """
        track = await self._lock_track(track_id, user_id, product_id)

        plan_id = track.current_plan_version_id
        plan_row: StudioDocumentVersion | None = None
        if not isinstance(plan_version_id, msgspec.UnsetType):
            plan_row = await self._version_of(
                plan_version_id,
                track_id,
                StudioDocumentKind.PLAN,
                user_id,
                product_id,
                missing=StudioNotFound,
            )
            plan_id = plan_row.id

        lyrics_id = track.current_lyrics_version_id
        lyrics_row: StudioDocumentVersion | None = None
        if not isinstance(lyrics_version_id, msgspec.UnsetType):
            if lyrics_version_id is None:
                lyrics_id = None
            else:
                lyrics_row = await self._version_of(
                    lyrics_version_id,
                    track_id,
                    StudioDocumentKind.LYRICS,
                    user_id,
                    product_id,
                    missing=StudioNotFound,
                )
                lyrics_id = lyrics_row.id

        if lyrics_id is not None:
            if plan_id is None:
                raise PlanRequired
            # A row that was not given in this call is the one the track's pointer names; if it
            # cannot be read the invariant is broken and the compatibility check must not be
            # skipped (RD5).
            plan_doc = decode_plan(
                plan_row
                if plan_row is not None
                else await self._pointer_row(plan_id, track, user_id, product_id)
            )
            lyrics_doc = decode_lyrics(
                lyrics_row
                if lyrics_row is not None
                else await self._pointer_row(lyrics_id, track, user_id, product_id)
            )
            issues = check_pair_compatible(plan_doc, lyrics_doc)
            if issues:
                raise LyricsIncompatibleWithPlan(issues)

        if plan_id != track.current_plan_version_id or lyrics_id != track.current_lyrics_version_id:
            track = await self._tracks.set_current(
                track_id,
                plan_version_id=msgspec.UNSET
                if plan_id == track.current_plan_version_id
                else plan_id,
                lyrics_version_id=msgspec.UNSET
                if lyrics_id == track.current_lyrics_version_id
                else lyrics_id,
            )
            logger.info(
                "studio.current_set",
                track_id=str(track_id),
                plan_version_id=str(plan_id) if plan_id else None,
                lyrics_version_id=str(lyrics_id) if lyrics_id else None,
            )
        return track_out(track)

    # -- delete ------------------------------------------------------------------------------

    async def delete(self, version_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a version that is not current, has no children and is not a lyrics base.

        Raises:
            StudioNotFound: No such version for the caller.
            VersionIsCurrent: It is the track's current plan or lyrics.
            VersionHasChildren: Other versions list it as their parent.
            VersionIsLyricsBase: Lyrics versions are based on this plan version.
        """
        version = await self._versions.get(version_id, user_id=user_id, product_id=product_id)
        if version is None:
            raise StudioNotFound
        track = await self._lock_track(version.track_id, user_id, product_id)
        # The version may have been deleted while this request waited for the lock.
        version = await self._versions.get(version_id, user_id=user_id, product_id=product_id)
        if version is None:
            raise StudioNotFound

        await self._refuse_if_blocked(version, track)
        try:
            async with self._session.begin_nested():
                await self._versions.delete(version_id, user_id=user_id, product_id=product_id)
        except IntegrityError:
            # The NO ACTION FKs are the backstop for a pointer / child that appeared after the
            # check; re-evaluate to report the precise reason.
            track = await self._lock_track(version.track_id, user_id, product_id)
            await self._refuse_if_blocked(version, track)
            raise
        logger.info(
            "studio.version_deleted",
            track_id=str(version.track_id),
            version_id=str(version_id),
            kind=version.kind,
            version_number=version.version_number,
        )

    async def _refuse_if_blocked(self, version: StudioDocumentVersion, track: StudioTrack) -> None:
        if version.id in (track.current_plan_version_id, track.current_lyrics_version_id):
            raise VersionIsCurrent
        blockers = await self._versions.deletion_blockers(
            version.id, track_id=version.track_id, kind=StudioDocumentKind(version.kind)
        )
        if blockers.child_count:
            raise VersionHasChildren(blockers.child_count)
        if blockers.lyrics_base_count:
            raise VersionIsLyricsBase(blockers.lyrics_base_count)
