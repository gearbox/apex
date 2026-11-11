"""Artist CRUD (``/v1/studio/artists``)."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.owner_scoped_names import normalize_owner_scoped_name
from src.db.repositories.studio import StudioArtistRepository
from src.studio.errors import (
    ArtistInUse,
    ArtistLimitReached,
    ArtistNameConflict,
    ArtistNotFound,
    DocumentIssue,
    InvalidCursor,
    StudioDocumentInvalid,
)
from src.studio.services.mappers import artist_out
from src.studio.validation import normalize_artist_style, normalize_block

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import ArtistCreate, ArtistOut, ArtistPatch

__all__ = ["MAX_ARTISTS_PER_OWNER", "ArtistService"]

logger = structlog.get_logger(__name__)

MAX_ARTISTS_PER_OWNER: Final = 50
_NAME_MAX: Final = 100


def _normalize_name(raw: str) -> str:
    try:
        return normalize_owner_scoped_name(unicodedata.normalize("NFC", raw), max_length=_NAME_MAX)
    except ValueError as exc:
        raise StudioDocumentInvalid((DocumentIssue("name", "empty_after_normalization"),)) from exc


class ArtistService:
    """Business logic for artists. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = StudioArtistRepository(session)

    async def list_page(
        self, user_id: UUID, product_id: str, *, limit: int, cursor: str | None
    ) -> CursorPage[ArtistOut]:
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        rows = await self._repo.list_page(
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
        return CursorPage(
            items=[artist_out(a) for a in page],
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def get(self, artist_id: UUID, user_id: UUID, product_id: str) -> ArtistOut:
        artist = await self._repo.get(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound
        return artist_out(artist)

    async def create(self, user_id: UUID, product_id: str, data: ArtistCreate) -> ArtistOut:
        name = _normalize_name(data.name)
        style = normalize_artist_style(
            style_tags=data.style_tags,
            global_avoid=data.global_avoid,
            aesthetic_references=data.aesthetic_references,
        )
        # A soft cap: two simultaneous creates at 49 artists may both pass this check. That is
        # accepted — the cap bounds abuse, it is not an integrity rule, so no lock is taken.
        if (
            await self._repo.count_for_owner(user_id=user_id, product_id=product_id)
            >= MAX_ARTISTS_PER_OWNER
        ):
            raise ArtistLimitReached
        try:
            async with self._session.begin_nested():
                artist = await self._repo.insert(
                    user_id=user_id,
                    product_id=product_id,
                    name=name,
                    persona=normalize_block(data.persona),
                    voice_notes=normalize_block(data.voice_notes),
                    default_language=data.default_language,
                    aesthetic_references=style.aesthetic_references,
                    style_tags=style.style_tags,
                    global_avoid=style.global_avoid,
                )
        except IntegrityError as exc:
            raise ArtistNameConflict from exc
        logger.info("studio.artist_created", artist_id=str(artist.id), user_id=str(user_id))
        return artist_out(artist)

    async def patch(
        self, artist_id: UUID, user_id: UUID, product_id: str, data: ArtistPatch
    ) -> ArtistOut:
        artist = await self._repo.get(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound

        changes: dict[str, object] = {}
        if data.name is not msgspec.UNSET:
            changes["name"] = _normalize_name(data.name)
        if data.persona is not msgspec.UNSET:
            changes["persona"] = normalize_block(data.persona)
        if data.voice_notes is not msgspec.UNSET:
            changes["voice_notes"] = normalize_block(data.voice_notes)
        if data.default_language is not msgspec.UNSET:
            changes["default_language"] = data.default_language

        # tags / avoid are validated as a pair: a new tag may now clash with the stored avoid list
        if (
            data.style_tags is not msgspec.UNSET
            or data.global_avoid is not msgspec.UNSET
            or data.aesthetic_references is not msgspec.UNSET
        ):
            style = normalize_artist_style(
                style_tags=artist.style_tags
                if data.style_tags is msgspec.UNSET
                else data.style_tags,
                global_avoid=artist.global_avoid
                if data.global_avoid is msgspec.UNSET
                else data.global_avoid,
                aesthetic_references=artist.aesthetic_references
                if data.aesthetic_references is msgspec.UNSET
                else data.aesthetic_references,
            )
            changes["style_tags"] = list(style.style_tags)
            changes["global_avoid"] = list(style.global_avoid)
            changes["aesthetic_references"] = list(style.aesthetic_references)

        try:
            async with self._session.begin_nested():
                updated = await self._repo.update(
                    artist_id, user_id=user_id, product_id=product_id, changes=changes
                )
        except IntegrityError as exc:
            raise ArtistNameConflict from exc
        if updated is None:  # deleted between the read and the update
            raise ArtistNotFound
        logger.info(
            "studio.artist_updated",
            artist_id=str(artist_id),
            user_id=str(user_id),
            fields=sorted(changes),
        )
        return artist_out(updated)

    async def delete(self, artist_id: UUID, user_id: UUID, product_id: str) -> None:
        artist = await self._repo.get(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound
        in_use = await self._repo.count_profiles_using(
            artist_id, user_id=user_id, product_id=product_id
        )
        if in_use:
            raise ArtistInUse(in_use)
        try:
            async with self._session.begin_nested():
                deleted = await self._repo.delete(artist_id, user_id=user_id, product_id=product_id)
        except IntegrityError as exc:
            # A project adopted the artist after the count: the FK is the backstop. If the
            # recount still finds none, the failure is unexplained -- let it surface rather than
            # blame a project that is not there with a misleading 409.
            in_use = await self._repo.count_profiles_using(
                artist_id, user_id=user_id, product_id=product_id
            )
            if not in_use:
                raise
            raise ArtistInUse(in_use) from exc
        if not deleted:
            raise ArtistNotFound
        logger.info("studio.artist_deleted", artist_id=str(artist_id), user_id=str(user_id))
