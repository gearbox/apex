"""The artist chat on the real database: chats, sending, the reply task, failures and refunds.

Every test names the invariant it holds (see the prompt, section 3). The model is scripted
(``ScriptedChat``); the supervisor, the services and the repositories are the real ones, so the reply
task opens its own short transactions exactly as in production. Nothing sleeps: tasks are awaited,
and a reply that must stay in flight waits on a ``Gate``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.api.schemas.studio_chat import ChatCreate, ChatPatch, MessageCreate
from src.api.services.billing_errors import InsufficientBalanceError
from src.core.enums import StudioChatMessageStatus, StudioChatRole
from src.db.integrity import violated_constraint
from src.db.models.studio_chat import (
    CHAT_ONE_PENDING_REPLY,
    StudioArtistChat,
    StudioArtistChatMessage,
)
from src.llm.port import LlmUnavailable
from src.studio.errors import (
    ArtistNotFound,
    ChatBusy,
    ChatFull,
    ChatLimitReached,
    ChatNotFound,
    ChatRateLimited,
    ChatReplyInProgress,
    IdempotencyConflict,
    InvalidCursor,
    MessageNotFound,
    StudioDocumentInvalid,
)
from tests.llm_support import FakeLlm
from tests.studio_chat_support import Gate, reply
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

LONG = "I remember the rooftop, the neon sign and the rain on the asphalt that night."
"""Longer than the sentinel buffer, so the first delta goes out on its own."""


def _artist(actor: Actor) -> UUID:
    """The actor's artist (``add_actor(artist=True)``)."""
    assert actor.artist_id is not None
    return actor.artist_id


@pytest.fixture
async def actor(chat_world: ChatWorld) -> Actor:
    return await chat_world.add_actor(balance=20, artist=True)


# ---------------------------------------------------------------------------------------------
# chats
# ---------------------------------------------------------------------------------------------


class TestChats:
    async def test_create_get_rename_list_delete(self, chat_world: ChatWorld, actor: Actor) -> None:
        chat_id = await chat_world.new_chat(actor, "  First   interview ")
        async with chat_world.factory() as session:
            service = chat_world.chat_service(session)
            got = await service.get(chat_id, actor.user_id, PRODUCT)
            assert got.title == "First interview"  # normalized like every title
            assert got.message_count == 0
            assert got.artist_id == _artist(actor)

            renamed = await service.patch(
                chat_id, actor.user_id, PRODUCT, ChatPatch(title="Second interview")
            )
            assert renamed.title == "Second interview"
            await session.commit()

        async with chat_world.factory() as session:
            page = await chat_world.chat_service(session).list_page(
                _artist(actor), actor.user_id, PRODUCT, limit=10, cursor=None
            )
            assert [c.id for c in page.items] == [chat_id]

        async with chat_world.factory() as session:
            await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)
            await session.commit()
        async with chat_world.factory() as session:
            with pytest.raises(ChatNotFound):
                await chat_world.chat_service(session).get(chat_id, actor.user_id, PRODUCT)

    async def test_an_empty_title_is_a_validation_error(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        async with chat_world.factory() as session:
            with pytest.raises(StudioDocumentInvalid) as caught:
                await chat_world.chat_service(session).create(
                    _artist(actor), actor.user_id, PRODUCT, ChatCreate(title="   ")
                )
        assert [(i.path, i.code) for i in caught.value.issues] == [
            ("title", "empty_after_normalization")
        ]

    async def test_nfc_expansion_is_caught_on_the_title(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        # U+0958 decomposes into two code points: 61 of them are 122 after NFC (RD6)
        async with chat_world.factory() as session:
            with pytest.raises(StudioDocumentInvalid) as caught:
                await chat_world.chat_service(session).create(
                    _artist(actor), actor.user_id, PRODUCT, ChatCreate(title="क़" * 61)
                )
        assert [(i.path, i.code) for i in caught.value.issues] == [
            ("title", "too_long_after_normalization")
        ]

    async def test_a_chat_of_an_unknown_artist_is_artist_not_found(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        async with chat_world.factory() as session:
            with pytest.raises(ArtistNotFound):
                await chat_world.chat_service(session).create(
                    uuid4(), actor.user_id, PRODUCT, ChatCreate(title="x")
                )

    async def test_an_artist_holds_at_most_fifty_chats(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        async with chat_world.factory() as session, session.begin():
            service = chat_world.chat_service(session)
            for index in range(50):
                await service.create(
                    _artist(actor), actor.user_id, PRODUCT, ChatCreate(title=f"chat {index}")
                )
        async with chat_world.factory() as session:
            with pytest.raises(ChatLimitReached):
                await chat_world.chat_service(session).create(
                    _artist(actor), actor.user_id, PRODUCT, ChatCreate(title="the 51st")
                )

    async def test_the_list_is_most_recently_active_first_and_keyset_paged(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        first = await chat_world.new_chat(actor, "first")
        second = await chat_world.new_chat(actor, "second")
        third = await chat_world.new_chat(actor, "third")
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, first)  # activity moves "first" to the top

        async with chat_world.factory() as session:
            service = chat_world.chat_service(session)
            page = await service.list_page(
                _artist(actor), actor.user_id, PRODUCT, limit=2, cursor=None
            )
            assert [c.id for c in page.items] == [first, third]
            assert page.has_more
            rest = await service.list_page(
                _artist(actor), actor.user_id, PRODUCT, limit=2, cursor=page.next_cursor
            )
            assert [c.id for c in rest.items] == [second]
            assert not rest.has_more

    async def test_a_rename_is_not_activity(self, chat_world: ChatWorld, actor: Actor) -> None:
        older = await chat_world.new_chat(actor, "older")
        await chat_world.new_chat(actor, "newer")
        before = (await chat_world.chat(older)).updated_at
        async with chat_world.factory() as session:
            await chat_world.chat_service(session).patch(
                older, actor.user_id, PRODUCT, ChatPatch(title="renamed")
            )
            await session.commit()
        assert (await chat_world.chat(older)).updated_at == before

    async def test_a_bad_cursor_is_invalid_cursor(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        async with chat_world.factory() as session:
            with pytest.raises(InvalidCursor):
                await chat_world.chat_service(session).list_page(
                    _artist(actor), actor.user_id, PRODUCT, limit=5, cursor="not-a-cursor"
                )

    async def test_deleting_the_artist_deletes_its_chats_and_messages(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)
        from src.studio.services.artist import ArtistService

        # the project names the artist (NO ACTION): detach it first, as a user would
        async with chat_world.factory() as session, session.begin():
            await session.execute(
                text("UPDATE studio_project_profiles SET artist_id = NULL WHERE user_id = :u"),
                {"u": actor.user_id},
            )
            await ArtistService(session).delete(_artist(actor), actor.user_id, PRODUCT)
        assert await chat_world.count(StudioArtistChat, actor) == 0
        assert await chat_world.count(StudioArtistChatMessage, actor) == 0


# ---------------------------------------------------------------------------------------------
# scoping (invariant 10)
# ---------------------------------------------------------------------------------------------


class TestScoping:
    """Every chat, message and draft of another user or product is a 404."""

    async def test_another_users_chat_is_not_found_everywhere(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        user_message, artist_message = await chat_world.ask(actor, chat_id)
        stranger = await chat_world.add_actor(balance=20, artist=True)

        async with chat_world.factory() as session:
            service = chat_world.chat_service(session)
            with pytest.raises(ChatNotFound):
                await service.get(chat_id, stranger.user_id, PRODUCT)
            with pytest.raises(ChatNotFound):
                await service.patch(chat_id, stranger.user_id, PRODUCT, ChatPatch(title="x"))
            with pytest.raises(ChatNotFound):
                await service.delete(chat_id, stranger.user_id, PRODUCT)
            with pytest.raises(ChatNotFound):
                await service.list_messages(
                    chat_id, stranger.user_id, PRODUCT, before=None, limit=10
                )
            with pytest.raises(ChatNotFound):
                await service.send(
                    chat_id,
                    stranger.user_id,
                    PRODUCT,
                    MessageCreate(client_message_id=uuid4(), content="hi"),
                )
            with pytest.raises(MessageNotFound):
                await service.get_message(artist_message.id, stranger.user_id, PRODUCT)
            with pytest.raises(ArtistNotFound):
                await service.list_page(
                    _artist(actor), stranger.user_id, PRODUCT, limit=5, cursor=None
                )
            with pytest.raises(ArtistNotFound):
                await service.create(
                    _artist(actor), stranger.user_id, PRODUCT, ChatCreate(title="x")
                )
        assert user_message.chat_id == chat_id

    async def test_another_product_is_a_404_too(self, chat_world: ChatWorld, actor: Actor) -> None:
        chat_id = await chat_world.new_chat(actor)
        async with chat_world.factory() as session:
            service = chat_world.chat_service(session)
            with pytest.raises(ChatNotFound):
                await service.get(chat_id, actor.user_id, "synthara")
            with pytest.raises(ArtistNotFound):
                await service.create(
                    _artist(actor), actor.user_id, "synthara", ChatCreate(title="x")
                )

    async def test_the_database_refuses_a_row_whose_owner_differs_from_its_parents(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        stranger = await chat_world.add_actor(balance=1, artist=True)
        chat_id = await chat_world.new_chat(actor)

        # a chat of the stranger's, on the actor's artist
        async with chat_world.factory() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "INSERT INTO studio_artist_chats (id, user_id, product_id, artist_id, title) "
                        "VALUES (:id, :user, 'vex', :artist, 't')"
                    ),
                    {"id": uuid4(), "user": stranger.user_id, "artist": _artist(actor)},
                )
        # a message of the stranger's, in the actor's chat
        async with chat_world.factory() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "INSERT INTO studio_artist_chat_messages (id, user_id, product_id, "
                        "chat_id, position, role, status, client_message_id, finished_at) "
                        "VALUES (:id, :user, 'vex', :chat, 1, 'user', 'complete', :cid, now())"
                    ),
                    {
                        "id": uuid4(),
                        "user": stranger.user_id,
                        "chat": chat_id,
                        "cid": uuid4(),
                    },
                )
        # the right user but another product
        async with chat_world.factory() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "INSERT INTO studio_artist_chat_messages (id, user_id, product_id, "
                        "chat_id, position, role, status, client_message_id, finished_at) "
                        "VALUES (:id, :user, 'synthara', :chat, 1, 'user', 'complete', :cid, now())"
                    ),
                    {"id": uuid4(), "user": actor.user_id, "chat": chat_id, "cid": uuid4()},
                )


# ---------------------------------------------------------------------------------------------
# sending: the happy path (invariants 1, 2, 6)
# ---------------------------------------------------------------------------------------------


class TestSending:
    async def test_the_pair_is_written_and_the_reply_completes_and_is_charged_once(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(
            reply("I remember ", "the rooftop, ", "the neon sign, ", "the rain.")
        )

        write = await chat_world.send(actor, chat_id, "Tell me about the rooftop.")
        assert write.changed
        assert write.user_message.position == 1
        assert write.artist_message.position == 2
        assert write.user_message.status == "complete"
        assert write.artist_message.status == "pending"
        assert write.artist_message.content == ""
        assert write.artist_message.price_tokens == 1
        assert write.artist_message.prompt_version == "chat-v1"
        assert write.artist_message.model_key == "grok"

        await chat_world.settle()

        done = await chat_world.message(write.artist_message.id)
        assert done.status == "complete"
        assert done.content == "I remember the rooftop, the neon sign, the rain."
        assert done.revision == 2
        assert done.finished_at is not None
        assert done.model_id == "served-model"
        assert (done.prompt_tokens, done.completion_tokens) == (120, 40)
        assert done.truncated is False
        assert done.error_code is None
        assert await chat_world.refunds_and_debits(actor) == (1, 0)
        assert await chat_world.balance(actor) == 19

        chat = await chat_world.chat(chat_id)
        assert chat.message_count == 2
        assert chat.next_position == 3

    async def test_a_client_that_never_opens_the_stream_still_gets_a_charged_reply(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Invariant 1: nothing in the task depends on the HTTP connection or on a subscriber.

        The bus here has no subscriber at all and the "connection" is long gone: the reply is
        complete, stored and charged, and ``GET`` reads it.
        """
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        async with chat_world.factory() as session:
            out = await chat_world.chat_service(session).get_message(
                write.artist_message.id, actor.user_id, PRODUCT
            )
        assert out.status.value == "complete"
        assert out.content == LONG
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_the_events_are_pending_then_deltas_then_complete(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        kinds = [t.value for _, t, _ in chat_world.bus.events if t.value.startswith("studio.chat")]
        assert kinds == [
            "studio.chat.message.updated",  # pending
            "studio.chat.delta",
            "studio.chat.message.updated",  # complete
        ]
        statuses = [(u.status, u.revision) for u in chat_world.updates(write.artist_message.id)]
        assert statuses == [("pending", 1), ("complete", 2)]
        owners = {user for user, t, _ in chat_world.bus.events if t.value.startswith("studio.chat")}
        assert owners == {actor.user_id}  # the owner's channel only

    async def test_delta_seq_is_gapless_and_the_deltas_rebuild_the_stored_text(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Invariant 6."""
        chat_id = await chat_world.new_chat(actor)
        chunks = [f"word{i} " * 12 for i in range(30)]  # many chunks, forced out by characters
        chat_world.chat_llm.add(reply(*chunks))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        deltas = chat_world.deltas(write.artist_message.id)
        assert [d.seq for d in deltas] == list(range(1, len(deltas) + 1))
        assert len(deltas) > 2
        stored = (await chat_world.message(write.artist_message.id)).content
        assert "".join(d.text for d in deltas) == stored
        assert stored == "".join(chunks)

    async def test_the_message_is_normalized_and_sent_to_the_model_as_the_last_turn(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id, "  \r\nWhere did you grow up?\r\n  ")
        await chat_world.settle()

        assert write.user_message.content == "Where did you grow up?"
        call = chat_world.chat_llm.calls[0]
        assert [m.role for m in call] == ["system", "user"]
        assert call[-1].content == "Where did you grow up?"
        assert chat_world.chat_llm.max_output_tokens == [700]

    async def test_an_empty_message_is_a_validation_error(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        with pytest.raises(StudioDocumentInvalid) as caught:
            await chat_world.send(actor, chat_id, "  \n  ")
        assert [(i.path, i.code) for i in caught.value.issues] == [
            ("content", "empty_after_normalization")
        ]
        assert await chat_world.balance(actor) == 20

    async def test_a_reply_cut_at_the_output_limit_is_complete_and_truncated(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG, finish="length"))
        _, artist = await chat_world.ask(actor, chat_id)

        assert artist.status == "complete"
        assert artist.truncated is True
        assert artist.content == LONG
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_short_reply_that_is_not_the_sentinel_is_delivered(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Invariant 5, the other half: under 32 characters, not the sentinel."""
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply("Yes."))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        assert (await chat_world.message(write.artist_message.id)).content == "Yes."
        assert [d.text for d in chat_world.deltas(write.artist_message.id)] == ["Yes."]

    async def test_a_bus_that_fails_never_stops_the_reply(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """The events are best-effort UX: the row is what is durable, and it is still charged."""

        class _DownBus:
            async def publish(self, **_: object) -> None:
                raise ConnectionError("redis is down")

            async def publish_balance(self, _event: object) -> None:
                raise ConnectionError("redis is down")

        chat_world.bus = _DownBus()  # type: ignore[assignment]
        supervisor = chat_world.new_supervisor()
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        async with chat_world.factory() as session:
            service = chat_world.chat_service(session, supervisor=supervisor)
            write = await service.send(
                chat_id,
                actor.user_id,
                PRODUCT,
                MessageCreate(client_message_id=uuid4(), content="Hello"),
            )
            await session.commit()
        task = supervisor.start_reply(write.job)  # type: ignore[arg-type]
        assert task is not None
        await task

        done = await chat_world.message(write.artist_message.id)
        assert (done.status, done.content) == ("complete", LONG)
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_free_message_writes_no_ledger_rows(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_message_price=0)
        actor = await world.add_actor(balance=0, artist=True)  # not even a balance is needed
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        _, artist = await world.ask(actor, chat_id)

        assert artist.status == "complete"
        assert artist.price_tokens == 0
        assert await world.ledger(actor) == []

    async def test_the_chat_model_is_the_configured_one(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_model="qwen")
        actor = await world.add_actor(balance=5, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        _, artist = await world.ask(actor, chat_id)
        assert artist.model_key == "qwen"


# ---------------------------------------------------------------------------------------------
# idempotency (invariant 3)
# ---------------------------------------------------------------------------------------------


class TestIdempotency:
    async def test_a_replay_returns_the_stored_pair_with_no_second_call_or_charge(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        client_id = uuid4()
        first = await chat_world.send(actor, chat_id, "Hello", client_message_id=client_id)
        await chat_world.settle()

        again = await chat_world.send(actor, chat_id, "Hello", client_message_id=client_id)
        await chat_world.settle()

        assert again.changed is False
        assert again.job is None
        assert again.user_message.id == first.user_message.id
        assert again.artist_message.id == first.artist_message.id
        assert again.artist_message.status == "complete"  # the pair as it is now
        assert len(chat_world.chat_llm.calls) == 1
        assert await chat_world.refunds_and_debits(actor) == (1, 0)
        assert (await chat_world.chat(chat_id)).message_count == 2
        # nothing was published for the replay: the events of the first send only
        assert len(chat_world.updates(first.artist_message.id)) == 2

    async def test_a_replay_while_the_reply_is_pending_returns_the_pending_pair(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        client_id = uuid4()
        first = await chat_world.send(actor, chat_id, "Hello", client_message_id=client_id)
        await gate.reached.wait()

        again = await chat_world.send(actor, chat_id, "Hello", client_message_id=client_id)
        assert again.changed is False
        assert again.artist_message.id == first.artist_message.id
        assert again.artist_message.status == "pending"
        gate.release()
        await chat_world.settle()

    async def test_other_content_with_the_same_id_is_a_conflict(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        client_id = uuid4()
        await chat_world.send(actor, chat_id, "Hello", client_message_id=client_id)
        await chat_world.settle()

        with pytest.raises(IdempotencyConflict):
            await chat_world.send(actor, chat_id, "Something else", client_message_id=client_id)
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_the_same_id_in_another_chat_is_a_different_message(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        first_chat = await chat_world.new_chat(actor, "a")
        second_chat = await chat_world.new_chat(actor, "b")
        chat_world.chat_llm.add(reply(LONG), reply(LONG))
        client_id = uuid4()
        one = await chat_world.send(actor, first_chat, "Hello", client_message_id=client_id)
        await chat_world.settle()
        two = await chat_world.send(actor, second_chat, "Hello", client_message_id=client_id)
        await chat_world.settle()

        assert one.user_message.id != two.user_message.id
        assert await chat_world.refunds_and_debits(actor) == (2, 0)


# ---------------------------------------------------------------------------------------------
# one reply in flight (invariant 4), limits
# ---------------------------------------------------------------------------------------------


class TestOneReplyAtATime:
    async def test_a_second_send_while_a_reply_is_pending_is_refused(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        await chat_world.send(actor, chat_id, "first")
        await gate.reached.wait()

        with pytest.raises(ChatReplyInProgress):
            await chat_world.send(actor, chat_id, "second")
        # nothing of the refused send was written or charged
        assert (await chat_world.chat(chat_id)).message_count == 2
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

        gate.release()
        await chat_world.settle()
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id, "third")  # once it is answered the chat goes on

    async def test_two_concurrent_sessions_cannot_both_start_a_reply(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])

        results = await asyncio.gather(
            chat_world.send(actor, chat_id, "a"),
            chat_world.send(actor, chat_id, "b"),
            return_exceptions=True,
        )
        started = [r for r in results if not isinstance(r, BaseException)]
        refused = [r for r in results if isinstance(r, ChatReplyInProgress)]
        assert len(started) == 1
        assert len(refused) == 1
        gate.release()
        await chat_world.settle()
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_the_partial_unique_index_is_the_backstop(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        await chat_world.send(actor, chat_id)
        await gate.reached.wait()

        from src.db.repositories.studio_chat import StudioChatMessageRepository

        async with chat_world.factory() as session:
            with pytest.raises(IntegrityError) as caught:
                await StudioChatMessageRepository(session).insert_pending_reply(
                    message_id=uuid4(),
                    user_id=actor.user_id,
                    product_id=PRODUCT,
                    chat_id=chat_id,
                    position=99,
                    price_tokens=0,
                    model_key="grok",
                    prompt_version="chat-v1",
                    memory_ids_used=(),
                    billing_account_id=None,
                )
        assert violated_constraint(caught.value, frozenset({CHAT_ONE_PENDING_REPLY})) == (
            CHAT_ONE_PENDING_REPLY
        )
        gate.release()
        await chat_world.settle()


class TestLimits:
    async def test_pending_replies_across_chats_are_capped_per_user(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        gates = [Gate() for _ in range(3)]
        chats = [await chat_world.new_chat(actor, f"c{i}") for i in range(4)]
        for gate in gates:
            chat_world.chat_llm.add([gate, *reply(LONG)])
        for chat_id in chats[:3]:
            await chat_world.send(actor, chat_id)
        for gate in gates:
            await gate.reached.wait()

        with pytest.raises(ChatRateLimited):
            await chat_world.send(actor, chats[3])
        assert await chat_world.refunds_and_debits(actor) == (3, 0)

        for gate in gates:
            gate.release()
        await chat_world.settle()
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chats[3])  # room again

    async def test_a_pending_draft_counts_against_the_same_cap(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_max_pending_per_user=1)
        actor = await world.add_actor(balance=10, artist=True)
        chat_id = await world.new_chat(actor)
        other = await world.new_chat(actor, "other")
        world.chat_llm.add(reply(LONG))
        _, artist = await world.ask(actor, chat_id)

        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        world.draft_llm = FakeLlm(LlmUnavailable("down"), before_call=blocked)
        await world.request_draft(actor, artist.id)
        await gate.reached.wait()

        with pytest.raises(ChatRateLimited):
            await world.send(actor, other)
        gate.release()
        await world.settle()

    async def test_messages_and_drafts_per_hour_are_capped_in_the_database(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=2)
        actor = await world.add_actor(balance=10, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        await world.ask(actor, chat_id, "one")
        await world.ask(actor, chat_id, "two")

        with pytest.raises(ChatRateLimited):
            await world.send(actor, chat_id, "three")
        assert await world.refunds_and_debits(actor) == (2, 0)

    async def test_an_old_message_no_longer_counts(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=1)
        actor = await world.add_actor(balance=10, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        await world.ask(actor, chat_id, "one")
        await world.age_usage(actor)  # the limit counts the usage ledger (R2), not the messages
        await world.ask(actor, chat_id, "two")

    async def test_a_chat_holds_at_most_two_hundred_messages(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        async with chat_world.factory() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE studio_artist_chats SET message_count = 199, next_position = 200 "
                    "WHERE id = :id"
                ),
                {"id": chat_id},
            )
        with pytest.raises(ChatFull):  # 199 + the pair would be 201
            await chat_world.send(actor, chat_id)
        async with chat_world.factory() as session, session.begin():
            await session.execute(
                text("UPDATE studio_artist_chats SET message_count = 198 WHERE id = :id"),
                {"id": chat_id},
            )
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)  # 198 + 2 = 200 is the last pair
        assert (await chat_world.chat(chat_id)).message_count == 200

    async def test_an_insufficient_balance_is_a_402_and_leaves_nothing_behind(
        self, chat_world: ChatWorld
    ) -> None:
        poor = await chat_world.add_actor(balance=0, artist=True)
        chat_id = await chat_world.new_chat(poor)
        with pytest.raises(InsufficientBalanceError):
            await chat_world.send(poor, chat_id)

        assert await chat_world.messages(chat_id) == []
        chat = await chat_world.chat(chat_id)
        assert (chat.message_count, chat.next_position) == (0, 1)
        assert await chat_world.ledger(poor) == []
        assert chat_world.chat_llm.calls == []
        assert chat_world.supervisor.in_flight == 0


# ---------------------------------------------------------------------------------------------
# reading messages
# ---------------------------------------------------------------------------------------------


class TestReadingMessages:
    async def test_newest_first_by_position_with_a_keyset(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG), reply(LONG), reply(LONG))
        for question in ("one", "two", "three"):
            await chat_world.ask(actor, chat_id, question)

        async with chat_world.factory() as session:
            service = chat_world.chat_service(session)
            page = await service.list_messages(
                chat_id, actor.user_id, PRODUCT, before=None, limit=4
            )
            assert [m.position for m in page.items] == [6, 5, 4, 3]
            assert page.has_more
            assert page.next_cursor == "3"
            rest = await service.list_messages(
                chat_id, actor.user_id, PRODUCT, before=int(page.next_cursor), limit=4
            )
            assert [m.position for m in rest.items] == [2, 1]
            assert not rest.has_more
            assert [m.role for m in rest.items] == [StudioChatRole.ARTIST, StudioChatRole.USER]

    async def test_a_pending_artist_message_has_no_text_yet(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()

        async with chat_world.factory() as session:
            out = await chat_world.chat_service(session).get_message(
                write.artist_message.id, actor.user_id, PRODUCT
            )
        assert out.status is StudioChatMessageStatus.PENDING
        assert out.content == ""
        assert out.finished_at is None
        gate.release()
        await chat_world.settle()


# ---------------------------------------------------------------------------------------------
# deleting (invariant 11)
# ---------------------------------------------------------------------------------------------


class TestDeleting:
    async def test_a_chat_with_a_pending_reply_cannot_be_deleted(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        await chat_world.send(actor, chat_id)
        await gate.reached.wait()

        async with chat_world.factory() as session:
            with pytest.raises(ChatBusy):
                await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)

        gate.release()
        await chat_world.settle()
        async with chat_world.factory() as session:
            await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)
            await session.commit()
        assert await chat_world.messages(chat_id) == []  # the messages went with it

    async def test_a_failed_reply_does_not_block_a_delete(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add([LlmUnavailable("down")])
        await chat_world.ask(actor, chat_id)
        async with chat_world.factory() as session:
            await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)
            await session.commit()


# ---------------------------------------------------------------------------------------------
# failures and refunds (invariant 2, 5)
# ---------------------------------------------------------------------------------------------


class TestFailures:
    async def _fails(
        self, chat_world: ChatWorld, actor: Actor, script: list[object], code: str, **kw: object
    ) -> StudioArtistChatMessage:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(script)
        _, artist = await chat_world.ask(actor, chat_id, **kw)  # type: ignore[arg-type]
        assert artist.status == "failed"
        assert artist.error_code == code
        assert artist.finished_at is not None
        assert artist.content == ""
        assert artist.revision == 2
        # refunded exactly once, and the user's own message is still there
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        assert await chat_world.balance(actor) == 20
        messages = await chat_world.messages(chat_id)
        assert [m.status for m in messages] == ["complete", "failed"]
        return artist

    async def test_a_provider_error_is_llm_unavailable(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await self._fails(chat_world, actor, [LlmUnavailable("down")], "llm_unavailable")

    async def test_an_error_after_some_text_is_llm_unavailable_and_the_text_is_not_kept(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await self._fails(chat_world, actor, [LONG, LlmUnavailable("cut")], "llm_unavailable")

    async def test_the_sentinel_publishes_nothing_and_is_a_content_policy_refusal(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Invariant 5."""
        artist = await self._fails(
            chat_world, actor, reply("⟦decl", "ined⟧"), "content_policy_refused"
        )
        assert chat_world.deltas(artist.id) == []
        assert [u.status for u in chat_world.updates(artist.id)] == ["pending", "failed"]

    async def test_the_sentinel_followed_by_prose_is_still_a_refusal(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        artist = await self._fails(
            chat_world,
            actor,
            reply("⟦declined⟧ I'm sorry, but I can't talk about that."),
            "content_policy_refused",
        )
        assert chat_world.deltas(artist.id) == []

    async def test_the_providers_own_content_filter_is_the_same_failure(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await self._fails(
            chat_world,
            actor,
            reply("Some words", finish="content_filter"),
            "content_policy_refused",
        )

    @pytest.mark.parametrize("script", [reply(), reply("   ", "\n"), reply("", finish="stop")])
    async def test_an_empty_reply_is_reply_empty(
        self, chat_world: ChatWorld, actor: Actor, script: list[object]
    ) -> None:
        await self._fails(chat_world, actor, script, "reply_empty")

    async def test_a_stream_without_an_end_is_llm_unavailable(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await self._fails(chat_world, actor, [LONG], "llm_unavailable")

    async def test_the_deadline_is_reply_timeout(self, chat_world: ChatWorld, actor: Actor) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()  # the model never answers
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id, deadline_seconds=0.05)
        await chat_world.settle()

        artist = await chat_world.message(write.artist_message.id)
        assert (artist.status, artist.error_code) == ("failed", "reply_timeout")
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        gate.release()

    async def test_an_unexpected_exception_leaves_the_row_to_the_sweep(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add([RuntimeError("a bug")])
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        stuck = await chat_world.message(write.artist_message.id)
        assert stuck.status == "pending"  # not failed by the task...
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

        await chat_world.age(message_id=stuck.id)
        assert await chat_world.sweeper().sweep() == 1  # ...the sweep does it
        swept = await chat_world.message(stuck.id)
        assert (swept.status, swept.error_code) == ("failed", "reply_timeout")
        assert await chat_world.refunds_and_debits(actor) == (1, 1)

    async def test_a_failed_message_is_terminal_and_a_new_message_may_follow(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add([LlmUnavailable("down")], reply(LONG))
        await chat_world.ask(actor, chat_id, "one")
        _, second = await chat_world.ask(actor, chat_id, "two")
        assert second.status == "complete"
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_a_failure_is_published_with_its_revision(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add([LlmUnavailable("down")])
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()
        assert [(u.status, u.revision) for u in chat_world.updates(write.artist_message.id)] == [
            ("pending", 1),
            ("failed", 2),
        ]
        assert any(e.transaction_type == "refund" for e in chat_world.bus.balance_events)


# ---------------------------------------------------------------------------------------------
# the sweep (invariants 2 and 12) and shutdown (invariant 13)
# ---------------------------------------------------------------------------------------------


class TestTheSweep:
    async def test_it_fails_and_refunds_an_orphan_older_than_the_deadline(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        await chat_world.age(message_id=write.artist_message.id)

        assert await chat_world.sweeper().sweep() == 1
        swept = await chat_world.message(write.artist_message.id)
        assert (swept.status, swept.error_code, swept.revision) == ("failed", "reply_timeout", 2)
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        assert [u.status for u in chat_world.updates(write.artist_message.id)] == [
            "pending",
            "failed",
        ]

        # a second sweep finds nothing: terminal once, refunded once
        assert await chat_world.sweeper().sweep() == 0
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        gate.release()
        await chat_world.settle()

    async def test_a_task_that_finishes_after_the_sweep_changes_nothing(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """The race of invariant 2: the sweep got there first; the late task must not charge,
        complete or refund again."""
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        await chat_world.age(message_id=write.artist_message.id)
        await chat_world.sweeper().sweep()

        gate.release()  # the task now finishes its reply...
        await chat_world.settle()

        message = await chat_world.message(write.artist_message.id)
        assert (message.status, message.error_code) == ("failed", "reply_timeout")
        assert message.content == ""
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        assert await chat_world.balance(actor) == 20

    async def test_a_row_the_task_already_finished_is_left_alone(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        _, artist = await chat_world.ask(actor, chat_id)
        await chat_world.age(message_id=artist.id)  # old, but complete

        assert await chat_world.sweeper().sweep() == 0
        assert (await chat_world.message(artist.id)).status == "complete"
        assert await chat_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_recent_pending_row_is_left_alone(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        assert await chat_world.sweeper().sweep() == 0  # younger than deadline + 30 s
        gate.release()
        await chat_world.settle()

    async def test_a_row_just_past_the_deadline_but_inside_the_grace_is_left_alone(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        async with chat_world.factory() as session, session.begin():
            await session.execute(
                text("UPDATE studio_artist_chat_messages SET created_at = :at WHERE id = :id"),
                {"at": datetime.now(UTC) - timedelta(seconds=130), "id": write.artist_message.id},
            )
        assert await chat_world.sweeper().sweep() == 0  # 130 s < 120 + 30
        gate.release()
        await chat_world.settle()


class TestShutdown:
    async def test_it_fails_and_refunds_what_is_in_flight(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        assert chat_world.supervisor.in_flight == 1

        await chat_world.supervisor.shutdown()

        message = await chat_world.message(write.artist_message.id)
        assert (message.status, message.error_code) == ("failed", "server_shutdown")
        assert await chat_world.refunds_and_debits(actor) == (1, 1)
        assert chat_world.supervisor.in_flight == 0
        assert [u.status for u in chat_world.updates(write.artist_message.id)] == [
            "pending",
            "failed",
        ]

    async def test_after_shutdown_no_task_is_started_and_the_row_is_the_sweeps(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        await chat_world.supervisor.shutdown()
        chat_world.chat_llm.add(reply(LONG))

        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        assert (await chat_world.message(write.artist_message.id)).status == "pending"
        await chat_world.age(message_id=write.artist_message.id)
        assert await chat_world.sweeper().sweep() == 1
        assert await chat_world.refunds_and_debits(actor) == (1, 1)

    async def test_shutdown_with_nothing_in_flight_is_a_no_op(self, chat_world: ChatWorld) -> None:
        await chat_world.supervisor.shutdown()
