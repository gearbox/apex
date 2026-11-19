"""What a run is told about the artist and the project - and the snapshot that records it.

``ArtistContextProvider`` is the planner's port to that knowledge. P2a ships one adapter
(``ProfileArtistContextProvider``, ``src/studio/services/run_context.py``): the project's artist
and concept, with **no memories** (P2a-2). The request already carries ``memory_ids`` and
``auto_fill`` so artist memories (P2b) plug in without changing the run pipeline (P2-M3).

``RunContextSnapshot`` is stored on the run when it starts: the exact context the model was given,
immune to later edits of the artist or the plan. It holds user text, so it is never logged or
returned by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol
from uuid import UUID

from msgspec import Struct

from src.core.enums import StudioSectionKind

__all__ = [
    "ArtistContext",
    "ArtistContextProvider",
    "ArtistSnapshot",
    "MemorySnapshot",
    "PlanSectionSummary",
    "PlanSummary",
    "RunContextRequest",
    "RunContextSnapshot",
]


class ArtistSnapshot(Struct, frozen=True, kw_only=True):
    """Every planner-relevant field of the artist, as it was when the run started."""

    id: UUID
    name: str
    persona: str | None
    voice_notes: str | None
    default_language: str | None
    style_tags: tuple[str, ...]
    global_avoid: tuple[str, ...]
    aesthetic_references: tuple[str, ...]
    updated_at: datetime


class MemorySnapshot(Struct, frozen=True, kw_only=True):
    """One artist memory as it was when the run started (P2-M4). Unused until P2b."""

    id: UUID
    title: str
    body: str
    themes: tuple[str, ...]
    imagery: tuple[str, ...]
    occurred_on: date | None
    updated_at: datetime


class PlanSectionSummary(Struct, frozen=True, kw_only=True):
    kind: StudioSectionKind
    label: str | None
    instrumental: bool


class PlanSummary(Struct, frozen=True, kw_only=True):
    """A compact view of the track's current plan - enough to plan *around* it, not to copy it."""

    title: str
    tempo_bpm: int | None
    instrumental: bool
    tags: tuple[str, ...]
    avoid: tuple[str, ...]
    sections: tuple[PlanSectionSummary, ...]


class RunContextSnapshot(Struct, frozen=True, kw_only=True):
    """The context of one run.

    ``plan_version_id`` / ``lyrics_version_id`` (and their hashes) are the track's **current**
    pair at claim - the parent a ``lyrics`` result is saved under. What the run actually *works on*
    is recorded in the ``input_*`` fields (the versions named in the run's inputs); they, and
    ``language``, are ``None`` for a ``plan`` run. Documents travel in memory, not in the snapshot:
    versions are immutable and a version an active run uses cannot be deleted, so ids and hashes
    are enough for the record.
    """

    snapshot_version: Literal[2] = 2
    track_title: str
    artist: ArtistSnapshot | None
    concept: str | None
    content_rating: str
    plan_version_id: UUID | None
    plan_content_hash: str | None
    current_plan: PlanSummary | None
    lyrics_version_id: UUID | None
    lyrics_content_hash: str | None
    memories: tuple[MemorySnapshot, ...] = ()
    """Always empty in P2a (P2a-2)."""
    input_plan_version_id: UUID | None = None
    input_plan_content_hash: str | None = None
    input_lyrics_version_id: UUID | None = None
    input_lyrics_content_hash: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class RunContextRequest:
    """What a context provider needs to know about the run."""

    track_id: UUID
    user_id: UUID
    product_id: str
    library_project_id: UUID
    memory_ids: tuple[UUID, ...]
    auto_fill: bool


@dataclass(frozen=True, slots=True)
class ArtistContext:
    """What a provider returns: the artist, the project concept and any memories."""

    artist: ArtistSnapshot | None
    concept: str | None
    memories: tuple[MemorySnapshot, ...] = ()


class ArtistContextProvider[S](Protocol):
    """The planner's port to artist knowledge, generic over the unit of work it reads through.

    ``S`` is the adapter's session type (``AsyncSession`` for the database-backed provider); the
    planner never names it, so it needs no database import.
    """

    async def for_run(self, request: RunContextRequest, *, session: S) -> ArtistContext:
        """The artist context for one run.

        Args:
            request: The run being started.
            session: The adapter's unit of work (the claim transaction).
        """
        ...
