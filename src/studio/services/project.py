"""Studio profile of a library project (``/v1/studio/projects``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.db.repositories.library_project import LibraryProjectRepository
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from src.studio.errors import ArtistNotFound, InvalidCursor, ProjectHasTracks, StudioNotFound
from src.studio.services.mappers import project_out
from src.studio.validation import normalize_block

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import ProjectProfileOut, ProjectProfilePut

__all__ = ["StudioProjectService"]

logger = structlog.get_logger(__name__)


class StudioProjectService:
    """Enables studio on a library project, and lists / reads / disables it."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._profiles = StudioProjectRepository(session)
        self._artists = StudioArtistRepository(session)
        self._tracks = StudioTrackRepository(session)
        self._library_projects = LibraryProjectRepository(session)

    async def upsert_profile(
        self, library_project_id: UUID, user_id: UUID, product_id: str, data: ProjectProfilePut
    ) -> tuple[ProjectProfileOut, bool]:
        """Create or replace the studio profile.

        Returns:
            ``(profile, created)`` — the route answers 201 / 200.

        Raises:
            StudioNotFound: The library project does not exist or is not the caller's.
            ArtistNotFound: ``artist_id`` is not one of the caller's artists.
        """
        project = await self._library_projects.get(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if project is None:
            raise StudioNotFound
        if data.artist_id is not None:
            artist = await self._artists.get(data.artist_id, user_id=user_id, product_id=product_id)
            if artist is None:
                raise ArtistNotFound
        try:
            async with self._session.begin_nested():
                _, created = await self._profiles.upsert(
                    library_project_id,
                    user_id=user_id,
                    product_id=product_id,
                    artist_id=data.artist_id,
                    release_kind=data.release_kind.value,
                    concept=normalize_block(data.concept),
                )
        except IntegrityError as exc:  # the artist was deleted after the check
            raise ArtistNotFound from exc
        row = await self._profiles.get_row(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if row is None:  # pragma: no cover - just written in this transaction
            raise StudioNotFound
        logger.info(
            "studio.project_enabled" if created else "studio.project_updated",
            project_id=str(library_project_id),
            user_id=str(user_id),
            release_kind=data.release_kind.value,
        )
        return project_out(row), created

    async def get_profile(
        self, library_project_id: UUID, user_id: UUID, product_id: str
    ) -> ProjectProfileOut:
        row = await self._profiles.get_row(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if row is None:
            raise StudioNotFound
        return project_out(row)

    async def list_profiles(
        self, user_id: UUID, product_id: str, *, limit: int, cursor: str | None
    ) -> CursorPage[ProjectProfileOut]:
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        rows = await self._profiles.list_page(
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        next_cursor = (
            encode_cursor(page[-1].profile.created_at, page[-1].profile.library_project_id)
            if has_more and page
            else None
        )
        return CursorPage(
            items=[project_out(row) for row in page],
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def disable(self, library_project_id: UUID, user_id: UUID, product_id: str) -> None:
        """Remove the studio profile. Refused while the project still has tracks."""
        profile = await self._profiles.lock(
            library_project_id, user_id=user_id, product_id=product_id
        )
        if profile is None:
            raise StudioNotFound
        if (
            await self._tracks.count_for_project(
                library_project_id, user_id=user_id, product_id=product_id
            )
            > 0
        ):
            raise ProjectHasTracks
        await self._profiles.delete(library_project_id, user_id=user_id, product_id=product_id)
        logger.info(
            "studio.project_disabled", project_id=str(library_project_id), user_id=str(user_id)
        )
