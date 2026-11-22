"""Deleting an artist must not cascade away paid, pending chat jobs (S1-P2c remediation R1).

An artist that no project uses can still own chats, and the cascade ``artist → chats → messages /
drafts`` deletes a pending reply or memory draft **without refunding it**: the supervisor and the
sweep refund only after a guarded ``pending → failed`` update, and that update would hit nothing.
So ``ArtistService.delete`` locks the artist, then every chat of the artist (artist → chat, the
order of the whole chat), and refuses with ``409 artist_busy`` while a reply or a draft is pending -
the precedent of ``RunInProgress`` and ``ChatBusy``.

Committed rows and separate sessions, over the real PostgreSQL (the integration ``db_session`` is
one rolled-back transaction, which a second connection cannot see).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from src.api.schemas.studio import ArtistCreate
from src.api.schemas.studio_chat import MessageCreate
from src.db.models.studio import StudioArtist
from src.db.models.studio_chat import (
    StudioArtistChat,
    StudioArtistChatMessage,
    StudioMemoryDraft,
)
from src.llm.port import LlmUnavailable
from src.studio.chat.draft import MemoryDraftDraft
from src.studio.errors import ArtistBusy, ArtistInUse, ChatNotFound, StudioError
from src.studio.services.artist import ArtistService
from tests.integration.test_studio_artist_concurrency import _wait_until_blocked_on_a_lock
from tests.llm_support import FakeLlm
from tests.studio_chat_support import reply
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

LONG = "I remember the rooftop, the neon sign and the rain on the asphalt that night."
_START = 20
"""The balance every actor starts with; what is missing at the end is what is held."""


@pytest.fixture
async def actor(chat_world: ChatWorld) -> Actor:
    return await chat_world.add_actor(balance=_START, artist=True)


def _draft() -> MemoryDraftDraft:
    return MemoryDraftDraft(
        refusal=None,
        title="Learning to sing on the roof",
        body="I learned to sing on a rooftop, under a humming neon sign.",
        themes=["home"],
        imagery=["humming neon sign"],
    )


async def _spare_artist(world: ChatWorld, actor: Actor, name: str = "Spare") -> UUID:
    """An artist **no project uses** (the actor's own artist is the project's): deletable but for
    its chats."""
    async with world.factory() as session, session.begin():
        created = await ArtistService(session).create(
            actor.user_id, PRODUCT, ArtistCreate(name=name, default_language="en")
        )
    return created.id


async def _delete(world: ChatWorld, actor: Actor, artist_id: UUID) -> None:
    async with world.factory() as session:
        await ArtistService(session).delete(artist_id, actor.user_id, PRODUCT)
        await session.commit()


async def _delete_outcome(world: ChatWorld, actor: Actor, artist_id: UUID) -> StudioError | None:
    """What the route would answer: the error, or ``None`` when the delete went through."""
    try:
        await _delete(world, actor, artist_id)
    except StudioError as error:
        return error
    return None


async def _artist_exists(world: ChatWorld, artist_id: UUID) -> bool:
    async with world.factory() as session:
        return (
            await session.execute(
                select(func.count()).select_from(StudioArtist).where(StudioArtist.id == artist_id)
            )
        ).scalar_one() == 1


async def _count(world: ChatWorld, model: type, chat_id: UUID) -> int:
    """Rows of a chat table that belong to ``chat_id`` (messages and drafts)."""
    async with world.factory() as session:
        return (
            await session.execute(
                select(func.count()).select_from(model).where(model.chat_id == chat_id)  # type: ignore[attr-defined]
            )
        ).scalar_one()


async def _held(world: ChatWorld, actor: Actor) -> int:
    """Tokens reserved and not refunded: ``sum(reserved) - sum(refunded)``."""
    return _START - await world.balance(actor)


async def _completed_replies(world: ChatWorld, chat_id: UUID) -> int:
    async with world.factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(StudioArtistChatMessage)
                .where(
                    StudioArtistChatMessage.chat_id == chat_id,
                    StudioArtistChatMessage.role == "artist",
                    StudioArtistChatMessage.status == "complete",
                )
            )
        ).scalar_one()


class TestAPendingJobBlocksTheDelete:
    async def test_deleting_an_artist_with_a_pending_reply_is_refused_and_keeps_the_money(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        spare = await _spare_artist(chat_world, actor)
        chat_id = await chat_world.new_chat(actor, artist=spare)
        write = await chat_world.send(actor, chat_id, dispatch=False)  # paid; the task has not run
        assert await _held(chat_world, actor) == 1

        error = await _delete_outcome(chat_world, actor, spare)
        assert isinstance(error, ArtistBusy)
        assert (error.status_code, error.code) == (409, "artist_busy")
        # the artist, the chat, both messages and the reservation are all still there
        assert await _artist_exists(chat_world, spare)
        assert (await chat_world.chat(chat_id)).message_count == 2
        assert [m.status for m in await chat_world.messages(chat_id)] == ["complete", "pending"]
        assert await chat_world.refunds_and_debits(actor) == (1, 0)
        assert await _held(chat_world, actor) == 1

        # let the task finish: the reply is delivered and charged once, and now the delete works
        chat_world.chat_llm.add(reply(LONG))
        async with chat_world.factory() as session:
            await chat_world.chat_service(session).dispatch(write)
        await chat_world.settle()
        assert (await chat_world.message(write.artist_message.id)).status == "complete"

        await _delete(chat_world, actor, spare)
        assert not await _artist_exists(chat_world, spare)
        assert await _count(chat_world, StudioArtistChatMessage, chat_id) == 0
        assert await chat_world.refunds_and_debits(actor) == (1, 0)  # charged once, for the reply

    async def test_deleting_an_artist_with_a_pending_memory_draft_is_refused_and_keeps_the_money(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        spare = await _spare_artist(chat_world, actor)
        chat_id = await chat_world.new_chat(actor, artist=spare)
        chat_world.chat_llm.add(reply(LONG))
        _, artist_message = await chat_world.ask(actor, chat_id, "Where did you learn to sing?")
        assert artist_message.status == "complete"  # the reply is done: only the draft is pending
        write = await chat_world.request_draft(actor, artist_message.id, dispatch=False)
        assert await _held(chat_world, actor) == 2  # the reply and the draft

        error = await _delete_outcome(chat_world, actor, spare)
        assert isinstance(error, ArtistBusy)
        assert await _artist_exists(chat_world, spare)
        assert (await chat_world.draft(write.draft.id)).status == "pending"
        assert await _held(chat_world, actor) == 2
        assert await chat_world.refunds_and_debits(actor) == (2, 0)

        chat_world.draft_llm = FakeLlm(_draft())
        async with chat_world.factory() as session:
            await chat_world.draft_service(session).dispatch(write)
        await chat_world.settle()
        assert (await chat_world.draft(write.draft.id)).status == "complete"

        await _delete(chat_world, actor, spare)
        assert not await _artist_exists(chat_world, spare)
        assert await _count(chat_world, StudioMemoryDraft, chat_id) == 0
        assert await chat_world.refunds_and_debits(actor) == (2, 0)

    async def test_a_busy_artist_that_a_project_uses_reports_artist_busy(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Pending work is checked before the project check: ``artist_busy`` wins."""
        assert actor.artist_id is not None  # the project's artist
        chat_id = await chat_world.new_chat(actor)
        write = await chat_world.send(actor, chat_id, dispatch=False)

        error = await _delete_outcome(chat_world, actor, actor.artist_id)
        assert isinstance(error, ArtistBusy)

        chat_world.chat_llm.add(reply(LONG))
        async with chat_world.factory() as session:
            await chat_world.chat_service(session).dispatch(write)
        await chat_world.settle()

    async def test_without_pending_work_a_used_artist_still_reports_artist_in_use(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        assert actor.artist_id is not None
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, await chat_world.new_chat(actor), "hello")  # all finished
        error = await _delete_outcome(chat_world, actor, actor.artist_id)
        assert isinstance(error, ArtistInUse)

    async def test_a_reply_that_failed_and_was_refunded_does_not_keep_the_artist_busy(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        spare = await _spare_artist(chat_world, actor)
        chat_id = await chat_world.new_chat(actor, artist=spare)
        chat_world.chat_llm.add([LlmUnavailable("down")])
        _, artist_message = await chat_world.ask(actor, chat_id)
        assert artist_message.status == "failed"
        assert await chat_world.refunds_and_debits(actor) == (1, 1)

        await _delete(chat_world, actor, spare)  # only *pending* work blocks
        assert not await _artist_exists(chat_world, spare)
        assert await _held(chat_world, actor) == 0

    async def test_pending_work_of_another_artist_does_not_block(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        busy = await _spare_artist(chat_world, actor, "Busy")
        idle = await _spare_artist(chat_world, actor, "Idle")
        await chat_world.send(actor, await chat_world.new_chat(actor, artist=busy), dispatch=False)
        await chat_world.new_chat(actor, artist=idle)  # a chat with nothing pending

        await _delete(chat_world, actor, idle)
        assert not await _artist_exists(chat_world, idle)
        assert await _artist_exists(chat_world, busy)
        assert await _held(chat_world, actor) == 1  # the other artist's paid reply is untouched

        chat_world.chat_llm.add(reply(LONG))
        await chat_world.settle()


class TestTheCascadeStillWorks:
    async def test_deleting_an_artist_without_pending_work_deletes_its_chats(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        spare = await _spare_artist(chat_world, actor)
        first = await chat_world.new_chat(actor, "one", artist=spare)
        second = await chat_world.new_chat(actor, "two", artist=spare)
        chat_world.chat_llm.add(reply(LONG), reply(LONG))
        await chat_world.ask(actor, first, "hello")
        _, artist_message = await chat_world.ask(actor, second, "hello")
        chat_world.draft_llm = FakeLlm(_draft())
        await chat_world.draft_of(actor, artist_message.id)  # a finished draft
        assert await chat_world.count(StudioArtistChat, actor) == 2
        assert await chat_world.count(StudioArtistChatMessage, actor) == 4
        assert await chat_world.count(StudioMemoryDraft, actor) == 1

        await _delete(chat_world, actor, spare)

        assert not await _artist_exists(chat_world, spare)
        assert await chat_world.count(StudioArtistChat, actor) == 0
        assert await chat_world.count(StudioArtistChatMessage, actor) == 0
        assert await chat_world.count(StudioMemoryDraft, actor) == 0
        # finished work was charged and stays charged: the ledger is not a chat row
        assert await chat_world.refunds_and_debits(actor) == (3, 0)

    async def test_an_artist_without_chats_is_deleted_as_before(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        spare = await _spare_artist(chat_world, actor)
        await _delete(chat_world, actor, spare)
        assert not await _artist_exists(chat_world, spare)


class TestRacesWithASend:
    @pytest.mark.parametrize("order", ["send-first", "delete-first"])
    async def test_an_artist_delete_and_a_send_never_lose_a_reservation(
        self, chat_world: ChatWorld, actor: Actor, db_engine: AsyncEngine, order: str
    ) -> None:
        """Without the chat locks a send that commits between the delete's pending check and its
        ``DELETE`` would be cascaded away with its reservation. With them, in both orders,
        ``sum(reserved) - sum(refunded) == price * completed replies``."""
        spare = await _spare_artist(chat_world, actor)
        chat_id = await chat_world.new_chat(actor, artist=spare)
        chat_world.chat_llm.add(reply(LONG))
        message = MessageCreate(client_message_id=uuid4(), content="Tell me about the rooftop.")
        price = chat_world.settings.studio_chat_message_price

        if order == "send-first":
            # A holds the chat lock mid-send; B deletes the artist and must wait for it.
            async with chat_world.factory() as session_a:
                write = await chat_world.chat_service(session_a).send(
                    chat_id, actor.user_id, PRODUCT, message
                )
                delete = asyncio.create_task(_delete_outcome(chat_world, actor, spare))
                await _wait_until_blocked_on_a_lock(db_engine)
                assert not delete.done()
                await session_a.commit()
            error = await asyncio.wait_for(delete, 10)

            assert isinstance(error, ArtistBusy)  # it saw the pending row the send committed
            assert await _artist_exists(chat_world, spare)
            async with chat_world.factory() as session:
                await chat_world.chat_service(session).dispatch(write)
            await chat_world.settle()
            assert (await chat_world.message(write.artist_message.id)).status == "complete"
            assert await _completed_replies(chat_world, chat_id) == 1
            assert await _held(chat_world, actor) == price * 1
            await _delete(chat_world, actor, spare)  # and once it is done the delete goes through
            assert not await _artist_exists(chat_world, spare)
        else:
            # B (the delete) holds the artist and chat locks and has deleted the chat; A's send
            # must wait for B and then find no chat.
            async with chat_world.factory() as session_b:
                await ArtistService(session_b).delete(spare, actor.user_id, PRODUCT)
                send = asyncio.create_task(_send_outcome(chat_world, actor, chat_id, message))
                await _wait_until_blocked_on_a_lock(db_engine)
                assert not send.done()
                await session_b.commit()
            error = await asyncio.wait_for(send, 10)

            assert isinstance(error, ChatNotFound)  # 404 chat_not_found
            assert not await _artist_exists(chat_world, spare)
            assert await chat_world.refunds_and_debits(actor) == (0, 0)  # nothing was reserved
            assert await _completed_replies(chat_world, chat_id) == 0
            assert await _held(chat_world, actor) == price * 0


async def _send_outcome(
    world: ChatWorld, actor: Actor, chat_id: UUID, message: MessageCreate
) -> StudioError | None:
    """A send as the route runs it (send, commit): the error, or ``None``."""
    try:
        async with world.factory() as session:
            await world.chat_service(session).send(chat_id, actor.user_id, PRODUCT, message)
            await session.commit()
    except StudioError as error:
        return error
    return None
