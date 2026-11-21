"""What a run is told about the artist and the project - and the snapshot that records it.

``ArtistContextProvider`` is the planner's port to that knowledge. Its adapter
(``MemoryArtistContextProvider``, ``src/studio/services/run_context.py``) returns the project's
artist and concept and the artist memories the run uses: the pinned ones the request names
(``memory_ids``), then - with ``auto_fill`` - active ones in a fixed order, within a character
budget (P2-M2, P2-M3). The selection itself is the pure ``src/studio/planner/memories.py``.

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
    """One artist memory as it was when the run started (P2-M4): its full text.

    ``salience`` is part of what the model is shown (``render_memory``), so it is part of the
    record. The snapshot holds user text: it is never logged or returned by the API.
    """

    id: UUID
    title: str
    body: str
    themes: tuple[str, ...]
    imagery: tuple[str, ...]
    occurred_on: date | None
    salience: int
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

    snapshot_version: Literal[3] = 3
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
    """The memories the model was given, in prompt order (full text, P2-M4)."""
    unavailable_memory_ids: tuple[UUID, ...] = ()
    """Pinned memories that were archived or deleted between submit and claim: skipped, not
    failed (P2b-2). The executor reports them as the run warning ``pinned_memory_unavailable``."""
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
    """What a provider returns: the artist, the project concept and the selected memories."""

    artist: ArtistSnapshot | None
    concept: str | None
    memories: tuple[MemorySnapshot, ...] = ()
    unavailable_memory_ids: tuple[UUID, ...] = ()


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
