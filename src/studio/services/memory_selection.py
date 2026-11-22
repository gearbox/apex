"""Which artist memories a model is given: the one artist-keyed selector (S1-P2c, P2c-4).

Planner runs and the artist chat both show the model the artist's memories, chosen by the **same**
rule: the pins first (in request order, only those that are still active memories of this artist),
then - with ``auto_fill`` - the artist's other active memories by ``salience DESC, occurred_on DESC
NULLS LAST, id``, all inside the 12,000-character budget, stopping at the first memory that does
not fit (``src/studio/planner/memories.py::select_memories``, pure). A run passes the request's
pins and its own ``auto_fill``; the chat passes no pins and ``auto_fill = True``.

Extracted from ``MemoryArtistContextProvider`` as a move, not a rewrite: the run's behaviour and
the rendered prompts are byte-identical to before.

Reads only the owner's own rows (every repository call filters ``user_id`` / ``product_id``); no
memory text is logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.enums import StudioMemoryStatus
from src.db.repositories.studio import StudioArtistMemoryRepository
from src.studio.constraints import MAX_ACTIVE_MEMORIES_PER_ARTIST
from src.studio.planner.memories import select_memories
from src.studio.services.mappers import memory_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.studio.planner.context import MemorySnapshot

__all__ = ["ArtistMemorySelector", "MemorySelection"]


@dataclass(frozen=True, slots=True)
class MemorySelection:
    """The memories chosen, in prompt order, and the pinned ids that did not make it."""

    memories: tuple[MemorySnapshot, ...]
    unavailable_ids: tuple[UUID, ...]
    """Pinned ids that are missing, archived, another artist's or no longer fit - in pin order."""


class ArtistMemorySelector:
    """Chooses the memories of one artist for one model call. Stateless."""

    async def select(
        self,
        session: AsyncSession,
        artist_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        pins: Sequence[UUID],
        auto_fill: bool,
    ) -> MemorySelection:
        """The memories for a call, and the pinned ids that were skipped.

        Args:
            session: The unit of work to read through.
            artist_id: The artist whose memories these are.
            user_id: Owner scope.
            product_id: Product scope.
            pins: Memory ids the caller insists on, in order (``()`` for none).
            auto_fill: Whether to add the artist's other active memories after the pins.
        """
        repo = StudioArtistMemoryRepository(session)
        active = {
            memory.id: memory
            for memory in await repo.get_many_owned(
                pins, artist_id=artist_id, user_id=user_id, product_id=product_id
            )
            if memory.status == StudioMemoryStatus.ACTIVE.value
        }
        pinned = tuple(memory_snapshot(active[i]) for i in pins if i in active)
        candidates: tuple[MemorySnapshot, ...] = ()
        if auto_fill:
            candidates = tuple(
                memory_snapshot(memory)
                for memory in await repo.active_in_selection_order(
                    artist_id,
                    user_id=user_id,
                    product_id=product_id,
                    exclude=pins,
                    limit=MAX_ACTIVE_MEMORIES_PER_ARTIST,
                )
            )
        chosen = select_memories(pinned, candidates, auto_fill=auto_fill)
        used = {memory.id for memory in chosen}
        return MemorySelection(
            memories=chosen, unavailable_ids=tuple(i for i in pins if i not in used)
        )
