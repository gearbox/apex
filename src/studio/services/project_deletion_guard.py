"""Studio's implementation of the library-owned ``ProjectDeletionGuard`` port (D12).

The library package defines the port and never imports this module; DI wires the two together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.library_project_guards import ProjectDeletionBlocker
from src.db.repositories.studio import StudioTrackRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["STUDIO_TRACKS_BLOCKER", "StudioProjectDeletionGuard"]

STUDIO_TRACKS_BLOCKER = "studio_tracks"


class StudioProjectDeletionGuard:
    """A library project that still has studio tracks cannot be deleted.

    A profile *without* tracks does not block: it goes with the project (``ON DELETE CASCADE``).
    """

    async def blockers(
        self, project_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
    ) -> Sequence[ProjectDeletionBlocker]:
        count = await StudioTrackRepository(session).count_for_project(
            project_id, user_id=user_id, product_id=product_id
        )
        return [ProjectDeletionBlocker(STUDIO_TRACKS_BLOCKER, count)] if count > 0 else []
