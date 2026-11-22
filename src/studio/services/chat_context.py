"""What a chat reply is told: the artist, its memories, its discography and the conversation (C7).

``ArtistChatContextAssembler.assemble`` runs **inside the send transaction**, so every read is one
consistent view; the result is plain data (:class:`AssembledContext`) that travels in memory to the
reply task - no session crosses the model call.

Everything read is the owner's own data, scoped by user and product. Chat text comes from **this
chat only** (the history query is by ``chat_id``), and the planner never reads chat (P2c-6): the
only way a chat's content reaches a run is a memory the user saved.

* **Artist** - the planner's ``ArtistSnapshot`` mapping.
* **Memories** - ``ArtistMemorySelector`` (``pins=()``, ``auto_fill=True``), rendered by the one
  memory renderer: the same rule, order and 12,000-character budget as a run.
* **Discography** - the artist's tracks (through the project profiles that name the artist), most
  recently touched first, as plain text within 6,000 characters (``render_discography``).
* **Window** - the chat's last ``complete`` messages, within 20 messages and 16,000 characters.

Logs: none here (the callers log ids and counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.enums import StudioChatRole
from src.core.product_registry import get_product_config_by_slug
from src.db.repositories.studio import StudioArtistRepository, StudioTrackRepository
from src.db.repositories.studio_chat import StudioChatMessageRepository
from src.llm.port import LlmMessage
from src.studio.chat.discography import DiscographySection, DiscographyTrack, render_discography
from src.studio.chat.prompts.chat_v1 import build_system_message
from src.studio.chat.window import select_window
from src.studio.constraints import CHAT_WINDOW_MESSAGES, DISCOGRAPHY_TITLES_MAX
from src.studio.errors import ChatNotFound
from src.studio.section_names import rendered_labels
from src.studio.services.mappers import artist_snapshot, decode_lyrics, decode_plan
from src.studio.services.memory_selection import ArtistMemorySelector

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioDocumentVersion, StudioTrack
    from src.db.models.studio_chat import StudioArtistChat

__all__ = ["ArtistChatContextAssembler", "AssembledContext"]


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The conversation to send, and which memories it carries."""

    messages: tuple[LlmMessage, ...]
    """The system message, then the history window, the new user message last."""
    memory_ids_used: tuple[UUID, ...]
    """The memories in the system message, in prompt order."""


def discography_tracks(
    rows: Sequence[tuple[StudioTrack, StudioDocumentVersion | None, StudioDocumentVersion | None]],
) -> tuple[DiscographyTrack, ...]:
    """The artist's tracks as a listener knows them: a title and, where there are lyrics, the
    sections by name. Plans, notes, ideas and references never appear: only the lyrics' own lines.
    """
    tracks: list[DiscographyTrack] = []
    for track, plan_row, lyrics_row in rows:
        sections: list[DiscographySection] = []
        if plan_row is not None and lyrics_row is not None:
            plan = decode_plan(plan_row)
            lyrics = decode_lyrics(lyrics_row)
            labels = dict(
                zip(
                    (section.id for section in plan.sections),
                    rendered_labels((section.kind, section.label) for section in plan.sections),
                    strict=True,
                )
            )
            sections = [
                DiscographySection(
                    name=labels[section.section_id],
                    lines=tuple(line.text for line in section.lines),
                )
                for section in lyrics.sections
                if section.section_id in labels
            ]
        tracks.append(DiscographyTrack(title=track.title, sections=tuple(sections)))
    return tuple(tracks)


class ArtistChatContextAssembler:
    """Reads what a reply needs and freezes it into an :class:`AssembledContext`."""

    async def assemble(
        self,
        session: AsyncSession,
        *,
        chat: StudioArtistChat,
        new_message: str,
        new_position: int,
    ) -> AssembledContext:
        """The conversation for the reply to ``new_message``.

        Args:
            session: The send transaction (reads see one consistent state).
            chat: The locked chat.
            new_message: The (normalized) user message being answered.
            new_position: The position the new user message takes (history is what is before it).

        Raises:
            ChatNotFound: The chat's artist is gone (cannot happen while the chat is locked; a
                guard against a broken invariant, not a client error).
        """
        artist_row = await StudioArtistRepository(session).get(
            chat.artist_id, user_id=chat.user_id, product_id=chat.product_id
        )
        if artist_row is None:
            raise ChatNotFound
        selection = await ArtistMemorySelector().select(
            session,
            artist_row.id,
            user_id=chat.user_id,
            product_id=chat.product_id,
            pins=(),
            auto_fill=True,
        )
        discography = render_discography(
            discography_tracks(
                await StudioTrackRepository(session).list_recent_for_artist_with_current_versions(
                    artist_row.id,
                    user_id=chat.user_id,
                    product_id=chat.product_id,
                    limit=DISCOGRAPHY_TITLES_MAX,
                )
            )
        )
        history = await StudioChatMessageRepository(session).complete_history_before(
            chat.id,
            new_position,
            user_id=chat.user_id,
            product_id=chat.product_id,
            limit=CHAT_WINDOW_MESSAGES - 1,
        )
        window = select_window(
            (
                ("user" if m.role == StudioChatRole.USER.value else "assistant", m.content)
                for m in history
            ),
            new_message,
        )
        system = build_system_message(
            artist=artist_snapshot(artist_row),
            memories=selection.memories,
            discography=discography,
            rating=get_product_config_by_slug(chat.product_id).content_policy.rating,
        )
        return AssembledContext(
            messages=(LlmMessage("system", system), *window),
            memory_ids_used=tuple(memory.id for memory in selection.memories),
        )
