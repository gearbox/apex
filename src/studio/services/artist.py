"""Artist CRUD (``/v1/studio/artists``)."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.services.owner_scoped_names import normalize_owner_scoped_name
from src.db.integrity import violated_constraint
from src.db.models.studio import CAST_PERFORMER_FK
from src.db.repositories.studio import StudioArtistRepository
from src.db.repositories.studio_cast import StudioCastMemberRepository
from src.db.repositories.studio_chat import (
    StudioChatMessageRepository,
    StudioChatRepository,
    StudioMemoryDraftRepository,
)
from src.studio.constraints import TOO_LONG, collect_length_issues
from src.studio.errors import (
    ArtistBusy,
    ArtistInUse,
    ArtistLimitReached,
    ArtistNameConflict,
    ArtistNotFound,
    CastMemberNotFound,
    DocumentIssue,
    InvalidCursor,
    StudioDocumentInvalid,
)
from src.studio.services.mappers import artist_out
from src.studio.validation import collect_artist_style, normalize_block

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.studio import ArtistCreate, ArtistOut, ArtistPatch

__all__ = ["MAX_ARTISTS_PER_OWNER", "ArtistService"]

logger = structlog.get_logger(__name__)

MAX_ARTISTS_PER_OWNER: Final = 50
_NAME_MAX: Final = 100
_EMPTY: Final = "empty_after_normalization"


def _normalize_name(raw: str, issues: list[DocumentIssue]) -> str:
    """The normalized name, or ``""`` after recording why it is not acceptable.

    ``normalize_owner_scoped_name`` re-checks the length *after* NFC, so an expanding character
    cannot push a name past ``varchar(100)``; the two ways it can fail get their own code.
    """
    composed = unicodedata.normalize("NFC", raw)
    try:
        return normalize_owner_scoped_name(composed, max_length=_NAME_MAX)
    except ValueError:
        issues.append(DocumentIssue("name", TOO_LONG if composed.split() else _EMPTY))
        return ""


def _artist_write_error(exc: IntegrityError) -> CastMemberNotFound | ArtistNameConflict:
    """Which typed error an integrity failure of an artist insert / update is.

    The performer foreign key fails when the cast member vanished between the existence check
    and the write (it was deleted concurrently); every other failure is the name's unique index.
    The constraint name is matched, never assumed.
    """
    if violated_constraint(exc, frozenset({CAST_PERFORMER_FK})) is not None:
        return CastMemberNotFound()
    return ArtistNameConflict()


class ArtistService:
    """Business logic for artists. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = StudioArtistRepository(session)
        self._cast = StudioCastMemberRepository(session)
        self._chats = StudioChatRepository(session)
        self._messages = StudioChatMessageRepository(session)
        self._drafts = StudioMemoryDraftRepository(session)

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
        issues: list[DocumentIssue] = []
        name = _normalize_name(data.name, issues)
        style = collect_artist_style(
            style_tags=data.style_tags,
            global_avoid=data.global_avoid,
            aesthetic_references=data.aesthetic_references,
            issues=issues,
        )
        persona = normalize_block(data.persona)
        voice_notes = normalize_block(data.voice_notes)
        # RD6: NFC can expand a value past the limit the wire already checked.
        collect_length_issues(
            msgspec.structs.replace(
                data,
                name=name,
                persona=persona,
                voice_notes=voice_notes,
                aesthetic_references=list(style.aesthetic_references),
                style_tags=list(style.style_tags),
                global_avoid=list(style.global_avoid),
            ),
            issues,
        )
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        # A soft cap: two simultaneous creates at 49 artists may both pass this check. That is
        # accepted — the cap bounds abuse, it is not an integrity rule, so no lock is taken.
        if (
            await self._repo.count_for_owner(user_id=user_id, product_id=product_id)
            >= MAX_ARTISTS_PER_OWNER
        ):
            raise ArtistLimitReached
        if data.performer_cast_member_id is not None:
            await self._require_cast_member(data.performer_cast_member_id, user_id, product_id)
        try:
            async with self._session.begin_nested():
                artist = await self._repo.insert(
                    user_id=user_id,
                    product_id=product_id,
                    name=name,
                    persona=persona,
                    voice_notes=voice_notes,
                    default_language=data.default_language,
                    aesthetic_references=style.aesthetic_references,
                    style_tags=style.style_tags,
                    global_avoid=style.global_avoid,
                    performer_cast_member_id=data.performer_cast_member_id,
                )
        except IntegrityError as exc:
            raise _artist_write_error(exc) from exc
        logger.info("studio.artist_created", artist_id=str(artist.id), user_id=str(user_id))
        return artist_out(artist)

    async def patch(
        self, artist_id: UUID, user_id: UUID, product_id: str, data: ArtistPatch
    ) -> ArtistOut:
        """Apply a PATCH as a locked read-modify-write (RD8).

        The artist row is locked (``SELECT … FOR UPDATE``) before the stored lists are read, the
        merge is validated and the columns are written, and the lock is held until the request
        transaction commits. Without it two overlapping PATCHes of different list fields would
        each write all three list columns from a stale read: the later writer would revert the
        other's field, and a tag added by one and an avoid entry added by the other could persist
        a ``tag_conflict`` that validation forbids.

        Lock order: this locks only the artist row, and no studio path locks an artist and then
        anything else (profiles reference artists, never the reverse), so no cycle is possible.

        Raises:
            ArtistNotFound: No such artist for the caller.
            StudioDocumentInvalid: With every issue found (including values that normalization
                pushed past their limit).
            ArtistNameConflict: The new name collides case-insensitively.
        """
        artist = await self._repo.lock(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound

        issues: list[DocumentIssue] = []
        changes: dict[str, object] = {}
        if data.name is not msgspec.UNSET:
            changes["name"] = _normalize_name(data.name, issues)
        if data.persona is not msgspec.UNSET:
            changes["persona"] = normalize_block(data.persona)
        if data.voice_notes is not msgspec.UNSET:
            changes["voice_notes"] = normalize_block(data.voice_notes)
        if data.default_language is not msgspec.UNSET:
            changes["default_language"] = data.default_language
        sent = dict(changes)  # what the client set, normalized: the RD6 length re-check input
        performer = data.performer_cast_member_id
        if performer is not msgspec.UNSET:
            if performer is not None:
                await self._require_cast_member(performer, user_id, product_id)
            changes["performer_cast_member_id"] = performer

        # tags / avoid are validated as a pair: a new tag may now clash with the stored avoid list
        if (
            data.style_tags is not msgspec.UNSET
            or data.global_avoid is not msgspec.UNSET
            or data.aesthetic_references is not msgspec.UNSET
        ):
            style = collect_artist_style(
                style_tags=artist.style_tags
                if data.style_tags is msgspec.UNSET
                else data.style_tags,
                global_avoid=artist.global_avoid
                if data.global_avoid is msgspec.UNSET
                else data.global_avoid,
                aesthetic_references=artist.aesthetic_references
                if data.aesthetic_references is msgspec.UNSET
                else data.aesthetic_references,
                issues=issues,
            )
            for column, merged in (
                ("style_tags", style.style_tags),
                ("global_avoid", style.global_avoid),
                ("aesthetic_references", style.aesthetic_references),
            ):
                changes[column] = list(merged)
                if getattr(data, column) is not msgspec.UNSET:
                    sent[column] = list(merged)

        # RD6: only the fields the client sent are re-checked; stored values were checked on write.
        collect_length_issues(msgspec.structs.replace(data, **sent), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        try:
            async with self._session.begin_nested():
                updated = await self._repo.update(
                    artist_id, user_id=user_id, product_id=product_id, changes=changes
                )
        except IntegrityError as exc:
            raise _artist_write_error(exc) from exc
        if updated is None:  # pragma: no cover - the row lock above rules out a concurrent delete
            raise ArtistNotFound
        logger.info(
            "studio.artist_updated",
            artist_id=str(artist_id),
            user_id=str(user_id),
            fields=[f for f in data.__struct_fields__ if getattr(data, f) is not msgspec.UNSET],
        )
        return artist_out(updated)

    async def _require_cast_member(self, member_id: UUID, user_id: UUID, product_id: str) -> None:
        """404 ``cast_member_not_found`` unless the caller has this cast member.

        The artist does not lock the cast row: the foreign key's own ``KEY SHARE`` is enough
        (and keeps the lock order artist → cast member), and a member deleted after this read
        fails the FK, which ``_artist_write_error`` maps to the same 404.
        """
        if not await self._cast.exists_for_owner(member_id, user_id=user_id, product_id=product_id):
            raise CastMemberNotFound

    async def delete(self, artist_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete an artist; its memories, chats, messages and memory drafts go with it.

        Refused while paid work is in flight (R1): a pending chat reply or memory draft has
        reserved tokens and a task is about to write it, and the cascade
        ``artist → chats → messages / drafts`` would delete the row that the supervisor's and the
        sweep's guarded ``pending → failed`` update (and with it the refund) hang off. The
        precedents are ``TrackService.delete`` (``RunInProgress``) and ``ChatService.delete``
        (``ChatBusy``): refuse, and the user retries in a few seconds.

        Lock order is artist → chats (→ messages), as everywhere: the artist row, then every
        chat of the artist in id order. A send or a draft request holds its chat's lock until it
        commits, so this waits for it and then sees the pending row; one that arrives later waits
        for this transaction and finds no chat (``404``). Without the chat locks a send that
        commits between the pending check and the ``DELETE`` would be cascaded away with its
        reservation. Pending work is checked **before** the project check, so a busy artist that
        is also in use reports ``artist_busy``.

        Raises:
            ArtistNotFound: No such artist for the caller.
            ArtistBusy: A chat reply or a memory draft of the artist is pending.
            ArtistInUse: A project uses the artist.
        """
        artist = await self._repo.lock(artist_id, user_id=user_id, product_id=product_id)
        if artist is None:
            raise ArtistNotFound
        chat_ids = await self._chats.lock_for_artist(
            artist_id, user_id=user_id, product_id=product_id
        )
        if chat_ids and (
            await self._messages.has_pending_reply_for_artist(
                artist_id, user_id=user_id, product_id=product_id
            )
            or await self._drafts.has_pending_for_artist(
                artist_id, user_id=user_id, product_id=product_id
            )
        ):
            raise ArtistBusy
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
        if not deleted:  # pragma: no cover - the row lock above rules out a concurrent delete
            raise ArtistNotFound
        logger.info("studio.artist_deleted", artist_id=str(artist_id), user_id=str(user_id))
