"""The only place where a stored plan and an artist row are turned into renderer input.

Everything a renderer must not see is dropped here: ``idea``, ``references``, section ``notes``
and the artist's persona / voice notes / aesthetic references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.studio.renderers.base import ArtistRenderInput, RenderablePlan, RenderableSection

if TYPE_CHECKING:
    from src.db.models.studio import StudioArtist
    from src.studio.documents import PlanDocument

__all__ = ["to_artist_render_input", "to_renderable_plan"]


def to_renderable_plan(plan: PlanDocument) -> RenderablePlan:
    """Project a plan onto what renderers may see."""
    return RenderablePlan(
        title=plan.title,
        tempo_bpm=plan.tempo_bpm,
        global_style=plan.global_style,
        sections=tuple(
            RenderableSection(
                id=section.id,
                kind=section.kind,
                label=section.label,
                duration_seconds=section.duration_seconds,
                style=section.style,
                delivery=section.delivery,
                instrumental=section.instrumental,
            )
            for section in plan.sections
        ),
    )


def to_artist_render_input(artist: StudioArtist | None) -> ArtistRenderInput | None:
    """Project an artist row onto its exportable style (``None`` when the project has none)."""
    if artist is None:
        return None
    return ArtistRenderInput(
        style_tags=tuple(artist.style_tags), global_avoid=tuple(artist.global_avoid)
    )
