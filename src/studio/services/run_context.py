"""Building a run's context: the database-backed artist provider and the snapshot builder.

The planner only knows the ``ArtistContextProvider`` port; this module is its adapter (the
project's artist and concept, and the artist memories the run uses) and assembles the
:class:`RunContextSnapshot` stored on a run when it starts. Everything read here is the owner's own
data, scoped by user and product; a memory carries that scope on its own row.

Which memories a run gets is decided here, at claim, from the rows as they are *now* (P2b-2): a pin
that was archived or deleted since the run was accepted is skipped and reported, never an error.
The ordering and the budget are the pure ``select_memories``.
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
    PlanSectionSummary,
    PlanSummary,
    RunContextRequest,
    RunContextSnapshot,
)
from src.studio.services.mappers import artist_snapshot, decode_plan, require_current_rows
from src.studio.services.memory_selection import ArtistMemorySelector

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioRun
    from src.studio.documents import PlanDocument
    from src.studio.planner.context import ArtistContextProvider, MemorySnapshot
    from src.studio.run_requests import RunRequest

__all__ = ["MemoryArtistContextProvider", "RunContextBuilder"]


class MemoryArtistContextProvider:
    """The project's artist and concept, and the artist memories the run uses (P2-M2)."""

    async def for_run(self, request: RunContextRequest, *, session: AsyncSession) -> ArtistContext:
        """The artist (if the project has one), the concept and the selected memories.

        Memories: the request's pins that are still active memories of the project's artist, in
        request order; then, with ``auto_fill``, the artist's other active memories by
        ``salience DESC, occurred_on DESC NULLS LAST, id`` - all inside the character budget,
        stopping at the first memory that does not fit. A pin that is missing, archived, another
        artist's (the project's artist changed), or no longer fits is **skipped** and reported in
        ``unavailable_memory_ids``: it never fails the run.
        """
        profile = await StudioProjectRepository(session).get(
            request.library_project_id, user_id=request.user_id, product_id=request.product_id
        )
        if profile is None:
            return ArtistContext(artist=None, concept=None)
        artist = None
        memories: tuple[MemorySnapshot, ...] = ()
        unavailable: tuple[UUID, ...] = request.memory_ids
        if profile.artist_id is not None:
            row = await StudioArtistRepository(session).get(
                profile.artist_id, user_id=request.user_id, product_id=request.product_id
            )
            if row is not None:
                artist = artist_snapshot(row)
                selection = await ArtistMemorySelector().select(
                    session,
                    row.id,
                    user_id=request.user_id,
                    product_id=request.product_id,
                    pins=request.memory_ids,
                    auto_fill=request.auto_fill,
                )
                memories, unavailable = selection.memories, selection.unavailable_ids
        return ArtistContext(
            artist=artist,
            concept=profile.concept,
            memories=memories,
            unavailable_memory_ids=unavailable,
        )


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
            unavailable_memory_ids=context.unavailable_memory_ids,
        )
