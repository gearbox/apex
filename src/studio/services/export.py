"""Rendering a track for an external tool (``/v1/studio/tracks/{id}/exports/suno``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.schemas.studio import ExportFieldOut, ExportWarningOut, SunoExportOut
from src.core.enums import StudioDocumentKind, StudioExportTarget
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioDocumentVersionRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from src.studio.errors import LyricsIncompatibleWithPlan, PlanRequired, StudioNotFound
from src.studio.renderers import RENDERERS
from src.studio.renderers.base import RenderContext
from src.studio.renderers.projection import to_artist_render_input, to_renderable_plan
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.validation import check_pair_compatible

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioDocumentVersion
    from src.studio.renderers.base import RenderedExport

__all__ = ["ExportService"]

logger = structlog.get_logger(__name__)


class ExportService:
    """Pure render of a (plan, lyrics) pair: reads only, never writes, never fails on length."""

    def __init__(self, session: AsyncSession) -> None:
        self._tracks = StudioTrackRepository(session)
        self._versions = StudioDocumentVersionRepository(session)
        self._profiles = StudioProjectRepository(session)
        self._artists = StudioArtistRepository(session)

    async def render_suno(
        self,
        track_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        plan_version_id: UUID | None = None,
        lyrics_version_id: UUID | None = None,
    ) -> SunoExportOut:
        """Render the track's Suno fields.

        The plan and lyrics form a **pair**; which pair is rendered depends on what was asked:

        ===============  ==========================  ===============================
        ``plan_version`` ``lyrics_version``          Rendered
        ===============  ==========================  ===============================
        none             none                        the current plan + current lyrics
        given            none                        that plan alone (section skeleton)
        none             given                       the current plan + those lyrics
        given            given                       exactly that pair
        ===============  ==========================  ===============================

        Current lyrics are used only when *no* plan id is given: they were written for the
        current plan, so pairing them with an explicitly chosen other plan would silently render
        lyrics for a different plan. Any explicit lyrics are checked against the chosen plan.

        Args:
            track_id: Track to export.
            user_id: Owner scope.
            product_id: Product scope.
            plan_version_id: Plan version to render; defaults to the track's current plan.
            lyrics_version_id: Lyrics version to render. When omitted: the track's current
                lyrics (which may be none) if no plan was given, otherwise no lyrics.

        Raises:
            StudioNotFound: Track or a requested version not found (or of the wrong kind).
            PlanRequired: No plan was requested and the track has no current plan.
            LyricsIncompatibleWithPlan: The chosen lyrics do not fit the chosen plan.
        """
        track = await self._tracks.get(track_id, user_id=user_id, product_id=product_id)
        if track is None:
            raise StudioNotFound

        plan_id: UUID | None
        lyrics_id: UUID | None
        if plan_version_id is None and lyrics_version_id is None:
            plan_id, lyrics_id = track.current_plan_version_id, track.current_lyrics_version_id
        elif lyrics_version_id is None:  # explicit plan only -> that plan alone
            plan_id, lyrics_id = plan_version_id, None
        else:  # explicit lyrics, with or without an explicit plan
            plan_id = plan_version_id or track.current_plan_version_id
            lyrics_id = lyrics_version_id
        if plan_id is None:
            raise PlanRequired

        plan_row = await self._version_of(
            plan_id, track_id, StudioDocumentKind.PLAN, user_id, product_id
        )
        plan = decode_plan(plan_row)
        lyrics_row: StudioDocumentVersion | None = None
        lyrics = None
        if lyrics_id is not None:
            lyrics_row = await self._version_of(
                lyrics_id, track_id, StudioDocumentKind.LYRICS, user_id, product_id
            )
            lyrics = decode_lyrics(lyrics_row)
            issues = check_pair_compatible(plan, lyrics)
            if issues:
                raise LyricsIncompatibleWithPlan(issues)

        profile = await self._profiles.get(
            track.library_project_id, user_id=user_id, product_id=product_id
        )
        artist = None
        if profile is not None and profile.artist_id is not None:
            artist = await self._artists.get(
                profile.artist_id, user_id=user_id, product_id=product_id
            )

        rendered = RENDERERS[StudioExportTarget.SUNO].render(
            RenderContext(
                plan=to_renderable_plan(plan),
                lyrics=lyrics,
                artist=to_artist_render_input(artist),
            )
        )
        logger.info(
            "studio.export_rendered",
            track_id=str(track_id),
            target=rendered.target.value,
            renderer_version=rendered.renderer_version,
            plan_version_id=str(plan_row.id),
            lyrics_version_id=str(lyrics_row.id) if lyrics_row else None,
            warnings=len(rendered.warnings),
        )
        return _suno_out(rendered, plan_row.id, lyrics_row.id if lyrics_row else None)

    async def _version_of(
        self,
        version_id: UUID,
        track_id: UUID,
        kind: StudioDocumentKind,
        user_id: UUID,
        product_id: str,
    ) -> StudioDocumentVersion:
        version = await self._versions.get_on_track(
            version_id, track_id, user_id=user_id, product_id=product_id
        )
        if version is None or version.kind != kind.value:
            raise StudioNotFound
        return version


def _suno_out(
    rendered: RenderedExport, plan_version_id: UUID, lyrics_version_id: UUID | None
) -> SunoExportOut:
    fields = rendered.fields
    return SunoExportOut(
        target=rendered.target,
        renderer_version=rendered.renderer_version,
        plan_version_id=plan_version_id,
        lyrics_version_id=lyrics_version_id,
        title=fields["title"].text,
        style=fields["style"].text,
        exclude=fields["exclude"].text,
        lyrics=fields["lyrics"].text,
        instrumental=rendered.instrumental,
        fields={
            name: ExportFieldOut(
                length=field.report.length,
                limit=field.report.limit,
                within_limit=field.report.within_limit,
            )
            for name, field in fields.items()
        },
        warnings=[
            ExportWarningOut(code=w.code, section_id=w.section_id) for w in rendered.warnings
        ],
    )
