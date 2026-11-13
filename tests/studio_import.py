"""Pure helpers for importer tests: render a song, paste it back, with no database."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from src.core.enums import StudioExportTarget
from src.studio.importers.suno import parse_suno
from src.studio.importers.suno_build import BuiltImport, build_documents
from src.studio.renderers import RENDERERS
from src.studio.renderers.base import RenderContext
from src.studio.renderers.projection import to_renderable_plan
from src.studio.services.importer import SunoImportService

if TYPE_CHECKING:
    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.renderers.base import ArtistRenderInput, RenderedExport

TRACK_ID = UUID("00000000-0000-7000-8000-0000000000aa")


def render_song(
    plan: PlanDocument, lyrics: LyricsDocument | None, artist: ArtistRenderInput | None
) -> RenderedExport:
    return RENDERERS[StudioExportTarget.SUNO].render(
        RenderContext(plan=to_renderable_plan(plan), lyrics=lyrics, artist=artist)
    )


def texts(export: RenderedExport) -> dict[str, str]:
    return {name: field.text for name, field in export.fields.items()}


def import_text(
    *,
    style: str = "",
    exclude: str = "",
    lyrics: str = "",
    title: str = "Song",
    instrumental: bool = False,
    language: str | None = "en",
    parent_plan: PlanDocument | None = None,
    parent_lyrics: LyricsDocument | None = None,
    artist: ArtistRenderInput | None = None,
    track_id: UUID = TRACK_ID,
) -> tuple[PlanDocument, LyricsDocument | None, BuiltImport]:
    """Parse, build and validate pasted text exactly as the service does (minus the database)."""
    built = build_documents(
        parse_suno(style=style, exclude=exclude, lyrics=lyrics),
        track_id=track_id,
        title=title,
        instrumental=instrumental,
        language=language,
        parent_plan=parent_plan,
        parent_lyrics=parent_lyrics,
        artist=artist,
    )
    plan, lyrics_doc = SunoImportService._normalize(built)
    return plan, lyrics_doc, built


def reimport(
    plan: PlanDocument,
    lyrics: LyricsDocument | None,
    artist: ArtistRenderInput | None,
    *,
    parent: bool,
) -> tuple[PlanDocument, LyricsDocument | None, BuiltImport]:
    """Render ``(plan, lyrics)`` and import the text back, optionally over the song itself."""
    shown = texts(render_song(plan, lyrics, artist))
    return import_text(
        style=shown["style"],
        exclude=shown["exclude"],
        lyrics=shown["lyrics"],
        title=plan.title,
        instrumental=plan.instrumental,
        language=lyrics.language if lyrics is not None else "en",
        parent_plan=plan if parent else None,
        parent_lyrics=lyrics if parent else None,
        artist=artist,
    )
