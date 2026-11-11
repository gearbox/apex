"""Port through which other bounded contexts can veto deleting a library project (D12).

The library owns this module and imports nothing from the contexts that implement it. A context
(e.g. the studio) provides a :class:`ProjectDeletionGuard`; dependency injection hands the guards
to ``LibraryProjectService``. This replaces mapping raw database errors, which would mean parsing
constraint names, with typed blockers a client can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["ProjectDeletionBlocker", "ProjectDeletionGuard"]


@dataclass(frozen=True, slots=True)
class ProjectDeletionBlocker:
    """One reason a project cannot be deleted right now."""

    code: str
    """Stable machine code, e.g. ``"studio_tracks"``."""
    count: int
    """How many dependent records hold the project (always > 0 when reported)."""


class ProjectDeletionGuard(Protocol):
    """Reports what, if anything, blocks deleting ``project_id``.

    Implementations run in the caller's transaction (``session``) and must not write.
    """

    async def blockers(
        self, project_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> Sequence[ProjectDeletionBlocker]: ...
