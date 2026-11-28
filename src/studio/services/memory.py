"""Artist memories CRUD (``/v1/studio/artists/{id}/memories``, ``/v1/studio/memories/{id}``).

A memory belongs to an artist and carries the artist's ``user_id`` / ``product_id`` on its own row
(a composite FK makes any other pair unrepresentable), so every query is scoped to the caller twice:
the artist is resolved through the caller's scope first, and each memory read filters the memory's
own scope. An unknown, foreign or cross-product id is the same 404.

**Lock order is artist, then memory - never the reverse.** A create and an un-archive take the
artist row (``FOR UPDATE``) and count the active memories under that lock, so the 200-active cap is
a hard rule: two overlapping requests at 199 cannot both succeed. A PATCH takes the artist, then the
memory, so overlapping PATCHes of different list fields never lose each other's field (RD8). A
delete takes neither lock: it is always allowed (P2-M5) because a run keeps its own copy of the text
it used.

Logs carry ids and field **names** only - never a title, body, theme or image (P2-M9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import structlog

from src.api.schemas.pagination import CursorPage, decode_memory_cursor, encode_memory_cursor
from src.core.enums import StudioMemoryDraftStatus, StudioMemoryOrigin, StudioMemoryStatus
from src.db.repositories.studio import StudioArtistMemoryRepository, StudioArtistRepository
from src.db.repositories.studio_chat import StudioChatRepository, StudioMemoryDraftRepository
from src.studio.constraints import MAX_ACTIVE_MEMORIES_PER_ARTIST, collect_text_issues
from src.studio.errors import (
    ArtistNotFound,
    DocumentIssue,
    InvalidCursor,
    MemoryLimitReached,
    MemoryNotFound,
    StudioDocumentInvalid,
)
from src.studio.services.mappers import memory_out
from src.studio.validation import normalize_block, normalize_text, normalize_tokens

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import MemoryCreate, MemoryOut, MemoryPatch

__all__ = ["MemoryService"]

logger = structlog.get_logger(__name__)

_EMPTY = "empty_after_normalization"


def _title(raw: str, issues: list[DocumentIssue]) -> str:
    """The normalized title (a single line); an empty one is recorded as an issue."""
    title = normalize_text(raw)
    if not title:
        issues.append(DocumentIssue("title", _EMPTY))
    return title


def _body(raw: str, issues: list[DocumentIssue]) -> str:
    """The normalized body; an empty one is recorded as an issue."""
    body = normalize_block(raw)
    if body is None:
        issues.append(DocumentIssue("body", _EMPTY))
        return ""
    return body


class MemoryService:
    """Business logic for artist memories. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._memories = StudioArtistMemoryRepository(session)
        self._artists = StudioArtistRepository(session)
        self._drafts = StudioMemoryDraftRepository(session)
        self._chats = StudioChatRepository(session)

    async def list_page(
        self,
        artist_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        status: StudioMemoryStatus,
        limit: int,
        cursor: str | None,
    ) -> CursorPage[MemoryOut]:
        """The artist's memories of ``status`` in diary order.

        Diary order is ``occurred_on DESC NULLS LAST, created_at DESC, id DESC``: the episode date
        (undated last), then the newest written. Changing a memory's ``occurred_on`` while a client
        pages can move that one row across a page boundary; no other row is duplicated or skipped.

        Raises:
            InvalidCursor: ``cursor`` is not one this endpoint issued.
            ArtistNotFound: The artist is not the caller's.
        """
        cursor_on, cursor_ts, cursor_id = None, None, None
        if cursor is not None:
            try:
                cursor_on, cursor_ts, cursor_id = decode_memory_cursor(cursor)
            except ValueError as exc:
                raise InvalidCursor from exc
        if await self._artists.get(artist_id, user_id=user_id, product_id=product_id) is None:
            raise ArtistNotFound
        rows = await self._memories.list_page(
            artist_id,
            user_id=user_id,
            product_id=product_id,
            status=status,
            limit=limit,
            cursor_occurred_on=cursor_on,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        last = page[-1] if has_more and page else None
        return CursorPage(
            items=[memory_out(m) for m in page],
            limit=limit,
            has_more=has_more,
            next_cursor=(
                encode_memory_cursor(last.occurred_on, last.created_at, last.id)
                if last is not None
                else None
            ),
        )

    async def create(
        self, artist_id: UUID, user_id: UUID, product_id: str, data: MemoryCreate
    ) -> MemoryOut:
        """Create an ``active`` memory: ``origin = user``, or ``chat`` with a valid ``from_draft_id``.

        The artist is locked first and the active memories are counted under that lock, so the
        cap holds under concurrent creates. ``from_draft_id`` is checked after that lock (P2c-29):
        it only says *where the text came from* - the memory's fields are always the ones the
        client sent (it has edited them), normalized and limited like any memory's.

        Raises:
            ArtistNotFound: The artist is not the caller's.
            StudioDocumentInvalid: With every issue found (an empty title or body, a bad theme or
                image, a value NFC pushed past its limit; a ``from_draft_id`` that is unknown,
                not ``complete`` or from another artist's chat).
            MemoryLimitReached: The artist already has 200 active memories.
        """
        artist = await self._artists.lock(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound
        issues: list[DocumentIssue] = []
        origin = StudioMemoryOrigin.USER
        if data.from_draft_id is not None:
            draft_issue = await self._check_draft(
                data.from_draft_id, artist_id, user_id, product_id
            )
            if draft_issue is not None:
                issues.append(draft_issue)  # reported together with every other issue
            else:
                origin = StudioMemoryOrigin.CHAT
        title = _title(data.title, issues)
        body = _body(data.body, issues)
        themes = normalize_tokens(data.themes, "themes", issues, check_reserved=True)
        imagery = normalize_tokens(data.imagery, "imagery", issues, check_reserved=True)
        # RD6: NFC can expand a value past the limit the wire already checked.
        collect_text_issues(
            msgspec.structs.replace(
                data, title=title, body=body, themes=list(themes), imagery=list(imagery)
            ),
            issues,
        )
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        if (
            await self._memories.count_active(
                artist_id, user_id=artist.user_id, product_id=artist.product_id
            )
            >= MAX_ACTIVE_MEMORIES_PER_ARTIST
        ):
            raise MemoryLimitReached
        # The locked artist row's owner, not the caller's: what the composite FK checks against.
        memory = await self._memories.insert(
            user_id=artist.user_id,
            product_id=artist.product_id,
            artist_id=artist_id,
            title=title,
            body=body,
            occurred_on=data.occurred_on,
            themes=themes,
            imagery=imagery,
            salience=data.salience,
            origin=origin,
        )
        logger.info(
            "studio.memory_created",
            memory_id=str(memory.id),
            artist_id=str(artist_id),
            user_id=str(user_id),
            origin=origin.value,
        )
        return memory_out(memory)

    async def _check_draft(
        self, draft_id: UUID, artist_id: UUID, user_id: UUID, product_id: str
    ) -> DocumentIssue | None:
        """Why a draft cannot back a memory of ``artist_id``, or ``None`` when it can.

        A draft that is not the caller's is the same ``draft_not_found`` as an unknown one (no
        probing). Read-only: neither the draft nor its chat is locked or changed.
        """
        draft = await self._drafts.get(draft_id, user_id=user_id, product_id=product_id)
        if draft is None:
            return DocumentIssue("from_draft_id", "draft_not_found")
        if draft.status != StudioMemoryDraftStatus.COMPLETE.value:
            return DocumentIssue("from_draft_id", "draft_not_complete")
        chat = await self._chats.get(draft.chat_id, user_id=user_id, product_id=product_id)
        if chat is None or chat.artist_id != artist_id:
            return DocumentIssue("from_draft_id", "draft_other_artist")
        return None

    async def get(self, memory_id: UUID, user_id: UUID, product_id: str) -> MemoryOut:
        """One memory.

        Raises:
            MemoryNotFound: No such memory for the caller.
        """
        memory = await self._memories.get(memory_id, user_id=user_id, product_id=product_id)
        if memory is None:
            raise MemoryNotFound
        return memory_out(memory)

    async def patch(
        self, memory_id: UUID, user_id: UUID, product_id: str, data: MemoryPatch
    ) -> MemoryOut:
        """Apply a PATCH as a locked read-modify-write (RD8): omitted fields keep their stored value.

        Lock order: the memory's artist, then the memory. The artist is found by an unlocked read
        first (the lock needs its id); both locks are then taken before anything is read to merge
        or count, and held until the request transaction commits. Bringing an archived memory back
        (``status = active``) counts the active ones under the artist lock.

        Raises:
            MemoryNotFound: No such memory for the caller.
            StudioDocumentInvalid: With every issue found.
            MemoryLimitReached: Un-archiving would pass 200 active memories.
        """
        found = await self._memories.get(memory_id, user_id=user_id, product_id=product_id)
        if found is None:
            raise MemoryNotFound
        artist_id = found.artist_id
        if await self._artists.lock(artist_id, user_id=user_id, product_id=product_id) is None:
            raise MemoryNotFound  # the artist was deleted while this request waited
        memory = await self._memories.lock(memory_id, user_id=user_id, product_id=product_id)
        if memory is None:
            raise MemoryNotFound  # deleted while this request waited for the artist lock

        issues: list[DocumentIssue] = []
        changes: dict[str, object] = {}
        if data.title is not msgspec.UNSET:
            changes["title"] = _title(data.title, issues)
        if data.body is not msgspec.UNSET:
            changes["body"] = _body(data.body, issues)
        if data.occurred_on is not msgspec.UNSET:
            changes["occurred_on"] = data.occurred_on
        if data.themes is not msgspec.UNSET:
            changes["themes"] = list(
                normalize_tokens(data.themes, "themes", issues, check_reserved=True)
            )
        if data.imagery is not msgspec.UNSET:
            changes["imagery"] = list(
                normalize_tokens(data.imagery, "imagery", issues, check_reserved=True)
            )
        if data.salience is not msgspec.UNSET:
            changes["salience"] = data.salience
        if data.status is not msgspec.UNSET:
            changes["status"] = data.status.value

        # RD6: only what the client sent is re-checked; stored values were checked when written.
        sent = {k: v for k, v in changes.items() if k in {"title", "body", "themes", "imagery"}}
        collect_text_issues(msgspec.structs.replace(data, **sent), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        was_archived = memory.status == StudioMemoryStatus.ARCHIVED.value
        if (
            data.status is StudioMemoryStatus.ACTIVE
            and was_archived
            and await self._memories.count_active(artist_id, user_id=user_id, product_id=product_id)
            >= MAX_ACTIVE_MEMORIES_PER_ARTIST
        ):
            raise MemoryLimitReached

        updated = await self._memories.update(
            memory_id,
            artist_id=artist_id,
            user_id=user_id,
            product_id=product_id,
            changes=changes,
        )
        if updated is None:  # pragma: no cover - the row lock above rules out a concurrent delete
            raise MemoryNotFound
        fields = [f for f in data.__struct_fields__ if getattr(data, f) is not msgspec.UNSET]
        logger.info(
            "studio.memory_updated",
            memory_id=str(memory_id),
            artist_id=str(artist_id),
            user_id=str(user_id),
            fields=fields,
        )
        if data.status is StudioMemoryStatus.ARCHIVED and not was_archived:
            logger.info(
                "studio.memory_archived",
                memory_id=str(memory_id),
                artist_id=str(artist_id),
                user_id=str(user_id),
            )
        return memory_out(updated)

    async def delete(self, memory_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a memory. Always allowed (P2-M5): a run keeps its own copy of what it used.

        Raises:
            MemoryNotFound: No such memory for the caller.
        """
        artist_id = await self._memories.delete(memory_id, user_id=user_id, product_id=product_id)
        if artist_id is None:
            raise MemoryNotFound
        logger.info(
            "studio.memory_deleted",
            memory_id=str(memory_id),
            artist_id=str(artist_id),
            user_id=str(user_id),
        )
