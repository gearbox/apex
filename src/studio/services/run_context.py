"""Building a run's context: the database-backed artist provider and the snapshot builder.

The planner only knows the ``ArtistContextProvider`` port; this module is its P2a adapter
(project artist + concept, no memories) and assembles the :class:`RunContextSnapshot` stored on a
run when it starts. Everything read here is the owner's own data, scoped by user and product.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.product_registry import get_product_config_by_slug
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from src.studio.planner.context import (
    ArtistContext,
    ArtistSnapshot,
    PlanSectionSummary,
    PlanSummary,
    RunContextRequest,
    RunContextSnapshot,
)
from src.studio.services.mappers import decode_plan, require_current_rows

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioRun
    from src.studio.documents import PlanDocument
    from src.studio.planner.context import ArtistContextProvider
    from src.studio.run_requests import RunRequest

__all__ = ["ProfileArtistContextProvider", "RunContextBuilder"]


class ProfileArtistContextProvider:
    """The project's artist and concept; never memories (P2a-2)."""

    async def for_run(self, request: RunContextRequest, *, session: AsyncSession) -> ArtistContext:
        """The artist (if the project has one) and the project concept for ``request``."""
        profile = await StudioProjectRepository(session).get(
            request.library_project_id, user_id=request.user_id, product_id=request.product_id
        )
        if profile is None:
            return ArtistContext(artist=None, concept=None)
        artist = None
        if profile.artist_id is not None:
            row = await StudioArtistRepository(session).get(
                profile.artist_id, user_id=request.user_id, product_id=request.product_id
            )
            if row is not None:
                artist = ArtistSnapshot(
                    id=row.id,
                    name=row.name,
                    persona=row.persona,
                    voice_notes=row.voice_notes,
                    default_language=row.default_language,
                    style_tags=tuple(row.style_tags),
                    global_avoid=tuple(row.global_avoid),
                    aesthetic_references=tuple(row.aesthetic_references),
                    updated_at=row.updated_at,
                )
        return ArtistContext(artist=artist, concept=profile.concept)


def _summary(plan: PlanDocument) -> PlanSummary:
    return PlanSummary(
        title=plan.title,
        tempo_bpm=plan.tempo_bpm,
        instrumental=plan.instrumental,
        tags=plan.global_style.tags,
        avoid=plan.global_style.avoid,
        sections=tuple(
            PlanSectionSummary(kind=s.kind, label=s.label, instrumental=s.instrumental)
            for s in plan.sections
        ),
    )


class RunContextBuilder:
    """Reads what a run needs to know and freezes it into a :class:`RunContextSnapshot`."""

    def __init__(self, provider: ArtistContextProvider[AsyncSession]) -> None:
        self._provider = provider

    async def build(
        self, session: AsyncSession, run: StudioRun, request: RunRequest
    ) -> RunContextSnapshot:
        """The snapshot for ``run`` (its owner's track, artist, concept and current documents).

        ``plan_version_id`` / ``lyrics_version_id`` are the track's *current* pair as of this read;
        what the run works on is added by the kind's handler (``input_*``, ``language``).

        Raises:
            RuntimeError: The track is missing, or a current pointer names a row that is not
                there - an invariant violation (runs cascade with their track; the pointers and
                their rows are read in one statement), never a client error.
        """
        found = await StudioTrackRepository(session).get_with_current_versions(
            run.track_id, user_id=run.user_id, product_id=run.product_id
        )
        if found is None:
            raise RuntimeError("studio run without a track")
        track = found[0]
        plan_row, lyrics_row = require_current_rows(track, found[1], found[2])
        context = await self._provider.for_run(
            RunContextRequest(
                track_id=track.id,
                user_id=run.user_id,
                product_id=run.product_id,
                library_project_id=track.library_project_id,
                memory_ids=request.memory_ids,
                auto_fill=request.auto_fill,
            ),
            session=session,
        )
        rating = get_product_config_by_slug(run.product_id).content_policy.rating
        return RunContextSnapshot(
            track_title=track.title,
            artist=context.artist,
            concept=context.concept,
            content_rating=rating.value,
            plan_version_id=plan_row.id if plan_row else None,
            plan_content_hash=plan_row.content_hash if plan_row else None,
            current_plan=_summary(decode_plan(plan_row)) if plan_row else None,
            lyrics_version_id=lyrics_row.id if lyrics_row else None,
            lyrics_content_hash=lyrics_row.content_hash if lyrics_row else None,
            memories=context.memories,
        )
