"""Importing a Suno song (Style / Exclude / Lyrics text) as a new plan and lyrics version.

The inverse of the export: ``POST /v1/studio/tracks/{id}/imports/suno``. The parsing and the
decisions about ids and carry-over are pure (``src/studio/importers``); this service loads the
track's current state, runs the validators, reports, and — unless ``dry_run`` — persists through
:class:`~src.studio.services.version.DocumentVersionService`, so an imported version is
indistinguishable from a typed one except for ``origin = import``.

A real import runs in the request transaction under the **track row lock**, taken before the
current pointers are read, so nothing can move between the decision and the writes; an error at
any step rolls the whole import back. A dry run takes no lock and writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import structlog

from src.api.schemas.studio import (
    ImportWarningOut,
    LyricsVersionCreate,
    PlanVersionCreate,
    RoundTripOut,
    SunoImportOut,
    SunoImportRequest,
    VersionSummary,
)
from src.core.enums import (
    StudioDocumentKind,
    StudioDocumentOrigin,
    StudioExportTarget,
    StudioIncompatibleLyricsPolicy,
)
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioDocumentVersionRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from src.studio.constraints import collect_count_issues
from src.studio.errors import DocumentIssue, StudioDocumentInvalid, StudioNotFound, VersionUnchanged
from src.studio.importers import roundtrip
from src.studio.importers.suno import parse_suno
from src.studio.importers.suno_build import build_documents, to_import_invalid
from src.studio.renderers import RENDERERS
from src.studio.renderers.base import RenderContext
from src.studio.renderers.projection import to_artist_render_input, to_renderable_plan
from src.studio.services.mappers import (
    decode_lyrics,
    decode_plan,
    require_current_row,
    version_summary,
)
from src.studio.validation import content_hash, normalize_lyrics, normalize_plan

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioArtist, StudioDocumentVersion, StudioTrack
    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.importers.suno_build import BuiltImport, Provenance
    from src.studio.services.version import DocumentVersionService

__all__ = ["SunoImportService"]

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Applied:
    """What the writes of a real import did."""

    plan_unchanged: bool
    lyrics_unchanged: bool
    created: bool
    plan_version: VersionSummary
    lyrics_version: VersionSummary | None


class SunoImportService:
    """Parses pasted Suno text and stores it as the track's current plan and lyrics."""

    def __init__(self, session: AsyncSession, *, versions: DocumentVersionService) -> None:
        self._versions = versions
        self._tracks = StudioTrackRepository(session)
        self._rows = StudioDocumentVersionRepository(session)
        self._profiles = StudioProjectRepository(session)
        self._artists = StudioArtistRepository(session)

    async def import_suno(
        self, track_id: UUID, request: SunoImportRequest, user_id: UUID, product_id: str
    ) -> SunoImportOut:
        """Import the pasted text into ``track_id``.

        Imported versions are **children of the track's current plan / lyrics**; content equal to
        the current version is reported as ``unchanged`` and writes no row (never a 409).

        Args:
            track_id: Track to import into.
            request: The pasted boxes and options.
            user_id: Owner scope.
            product_id: Product scope.

        Returns:
            The documents, warnings and round-trip report; on a real import also the version
            summaries.

        Raises:
            StudioNotFound: The track is not the caller's.
            ImportInvalid: The text cannot become valid documents (all issues, with the pasted
                lines they come from).
            StudioError: Any error of the version create / set-current paths (the whole import
                is rolled back).
        """
        track = await self._load_track(track_id, user_id, product_id, lock=not request.dry_run)
        plan_row = await self._current_row(
            track.current_plan_version_id, track, user_id, product_id
        )
        lyrics_row = await self._current_row(
            track.current_lyrics_version_id, track, user_id, product_id
        )
        parent_plan = decode_plan(plan_row) if plan_row is not None else None
        parent_lyrics = decode_lyrics(lyrics_row) if lyrics_row is not None else None
        artist = await self._project_artist(track, user_id, product_id)
        artist_input = to_artist_render_input(artist)

        parsed = parse_suno(style=request.style, exclude=request.exclude, lyrics=request.lyrics)
        built = build_documents(
            parsed,
            track_id=track_id,
            track_title=track.title,
            title=request.title,
            instrumental=request.instrumental,
            language=(
                request.language
                or (parent_lyrics.language if parent_lyrics is not None else None)
                or (artist.default_language if artist is not None else None)
            ),
            parent_plan=parent_plan,
            parent_lyrics=parent_lyrics,
            artist=artist_input,
        )
        plan, lyrics = self._normalize(built)

        rendered = RENDERERS[StudioExportTarget.SUNO].render(
            RenderContext(plan=to_renderable_plan(plan), lyrics=lyrics, artist=artist_input)
        )
        fields = rendered.fields
        report = roundtrip.compare(
            pasted=(request.style, request.exclude, request.lyrics),
            rendered=(fields["style"].text, fields["exclude"].text, fields["lyrics"].text),
        )

        plan_unchanged = plan_row is not None and plan_row.content_hash == content_hash(
            StudioDocumentKind.PLAN, None, plan
        )
        lyrics_unchanged = self._lyrics_unchanged(
            lyrics, lyrics_row, plan_row if plan_unchanged else None
        )
        out = SunoImportOut(
            dry_run=request.dry_run,
            plan=plan,
            lyrics=lyrics,
            plan_unchanged=plan_unchanged,
            lyrics_unchanged=lyrics_unchanged,
            warnings=[ImportWarningOut(code=w.code, line=w.line) for w in built.warnings],
            round_trip=RoundTripOut(
                style=report.style, exclude=report.exclude, lyrics=report.lyrics
            ),
            plan_version=None,
            lyrics_version=None,
            created=False,
        )
        if request.dry_run:
            self._log("studio.suno_import_dry_run", track_id, built, out)
            return out

        result = await self._apply(
            track, plan, lyrics, lyrics_row, user_id=user_id, product_id=product_id
        )
        out = msgspec.structs.replace(
            out,
            plan_unchanged=result.plan_unchanged,
            lyrics_unchanged=result.lyrics_unchanged,
            plan_version=result.plan_version,
            lyrics_version=result.lyrics_version,
            created=result.created,
        )
        self._log("studio.suno_imported", track_id, built, out)
        return out

    # -- loading -----------------------------------------------------------------------------

    async def _load_track(
        self, track_id: UUID, user_id: UUID, product_id: str, *, lock: bool
    ) -> StudioTrack:
        """The track; a real import locks it first so the pointers read next cannot move."""
        track = (
            await self._tracks.lock(track_id, user_id=user_id, product_id=product_id)
            if lock
            else await self._tracks.get(track_id, user_id=user_id, product_id=product_id)
        )
        if track is None:
            raise StudioNotFound
        return track

    async def _current_row(
        self, version_id: UUID | None, track: StudioTrack, user_id: UUID, product_id: str
    ) -> StudioDocumentVersion | None:
        """The row a current pointer names; a non-null pointer without a row is a broken invariant."""
        if version_id is None:
            return None
        return require_current_row(
            await self._rows.get_on_track(
                version_id, track.id, user_id=user_id, product_id=product_id
            )
        )

    async def _project_artist(
        self, track: StudioTrack, user_id: UUID, product_id: str
    ) -> StudioArtist | None:
        profile = await self._profiles.get(
            track.library_project_id, user_id=user_id, product_id=product_id
        )
        if profile is None or profile.artist_id is None:
            return None
        return await self._artists.get(profile.artist_id, user_id=user_id, product_id=product_id)

    # -- deciding ----------------------------------------------------------------------------

    @staticmethod
    def _normalize(built: BuiltImport) -> tuple[PlanDocument, LyricsDocument | None]:
        """Run the P1 validators; their issues come back addressed to the pasted lines.

        A document built in code never went through msgspec's decoding, so the declared array
        sizes (at most 40 tags, 64 sections, 64 lines per section...) are checked here too, and
        reported together with the validators' own issues.
        """
        plan = SunoImportService._validated(built.plan, normalize_plan, "plan", built.provenance)
        if built.lyrics is None:
            return plan, None
        return plan, SunoImportService._validated(
            built.lyrics, lambda lyrics: normalize_lyrics(lyrics, plan), "lyrics", built.provenance
        )

    @staticmethod
    def _validated[D: PlanDocument | LyricsDocument](
        document: D,
        normalize: Callable[[D], D],
        name: Literal["plan", "lyrics"],
        provenance: Provenance,
    ) -> D:
        issues: list[DocumentIssue] = []
        collect_count_issues(document, issues)
        normalized: D | None = None
        try:
            normalized = normalize(document)
        except StudioDocumentInvalid as exc:
            issues.extend(exc.issues)
        if issues or normalized is None:
            raise to_import_invalid(issues, name, provenance)
        return normalized

    @staticmethod
    def _lyrics_unchanged(
        lyrics: LyricsDocument | None,
        lyrics_row: StudioDocumentVersion | None,
        unchanged_plan_row: StudioDocumentVersion | None,
    ) -> bool:
        """Whether the lyrics equal the current ones.

        The hash includes the plan version the lyrics are based on, so unchanged lyrics need an
        unchanged plan (``unchanged_plan_row``: its version is the one that stays current). No
        lines and no current lyrics is unchanged too.
        """
        if lyrics is None:
            return lyrics_row is None
        if lyrics_row is None or unchanged_plan_row is None:
            return False
        return lyrics_row.content_hash == content_hash(
            StudioDocumentKind.LYRICS, unchanged_plan_row.id, lyrics
        )

    # -- writing -----------------------------------------------------------------------------

    async def _apply(
        self,
        track: StudioTrack,
        plan: PlanDocument,
        lyrics: LyricsDocument | None,
        lyrics_row: StudioDocumentVersion | None,
        *,
        user_id: UUID,
        product_id: str,
    ) -> _Applied:
        """Write the plan, then the lyrics, as children of the current versions.

        Only :class:`VersionUnchanged` is caught (it means "nothing to write for this kind");
        every other error propagates and the request transaction rolls back, so an import is
        never half-applied.
        """
        plan_created = lyrics_created = False
        plan_unchanged = False
        # No lyric lines and no current lyrics: nothing to write, nothing changes.
        lyrics_unchanged = lyrics is None and lyrics_row is None

        try:
            out, plan_created = await self._versions.create(
                track.id,
                PlanVersionCreate(
                    content=plan,
                    parent_version_id=track.current_plan_version_id,
                    make_current=True,
                    incompatible_lyrics=StudioIncompatibleLyricsPolicy.DETACH,
                ),
                user_id,
                product_id,
                origin=StudioDocumentOrigin.IMPORT,
            )
            plan_version_id = out.version.id
        except VersionUnchanged:
            plan_unchanged = True
            assert track.current_plan_version_id is not None  # noqa: S101 — unchanged means a parent
            plan_version_id = track.current_plan_version_id

        lyrics_version_id: UUID | None = None
        if lyrics is not None:
            try:
                lyrics_out, lyrics_created = await self._versions.create(
                    track.id,
                    LyricsVersionCreate(
                        content=lyrics,
                        based_on_plan_version_id=plan_version_id,
                        parent_version_id=lyrics_row.id if lyrics_row is not None else None,
                        make_current=True,
                    ),
                    user_id,
                    product_id,
                    origin=StudioDocumentOrigin.IMPORT,
                )
                lyrics_version_id = lyrics_out.version.id
            except VersionUnchanged:
                lyrics_unchanged = True
                assert lyrics_row is not None  # noqa: S101 — unchanged means a parent
                lyrics_version_id = lyrics_row.id
        elif lyrics_row is not None:
            # The imported song has no lyrics: the current pair must say so.
            await self._versions.set_current(
                track.id,
                user_id,
                product_id,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=None,
            )

        fresh = await self._tracks.get(track.id, user_id=user_id, product_id=product_id)
        assert fresh is not None  # noqa: S101 — locked by this transaction
        return _Applied(
            plan_unchanged=plan_unchanged,
            lyrics_unchanged=lyrics_unchanged,
            created=plan_created or lyrics_created,
            plan_version=await self._summary(plan_version_id, fresh, user_id, product_id),
            lyrics_version=(
                await self._summary(lyrics_version_id, fresh, user_id, product_id)
                if lyrics_version_id is not None
                else None
            ),
        )

    async def _summary(
        self, version_id: UUID, track: StudioTrack, user_id: UUID, product_id: str
    ) -> VersionSummary:
        row = await self._rows.get_on_track(
            version_id, track.id, user_id=user_id, product_id=product_id
        )
        return version_summary(require_current_row(row), track)

    @staticmethod
    def _log(event: str, track_id: UUID, built: BuiltImport, out: SunoImportOut) -> None:
        """Ids, counts and codes only — never Style, Exclude, Lyrics or titles."""
        logger.info(
            event,
            track_id=str(track_id),
            plan_version_id=str(out.plan_version.id) if out.plan_version else None,
            lyrics_version_id=str(out.lyrics_version.id) if out.lyrics_version else None,
            sections=len(out.plan.sections),
            lyric_lines=sum(len(s.lines) for s in out.lyrics.sections) if out.lyrics else 0,
            plan_unchanged=out.plan_unchanged,
            lyrics_unchanged=out.lyrics_unchanged,
            created=out.created,
            warnings=[w.code for w in built.warnings],
        )
