"""Memory drafts: ask for one, read it, save it as a memory with ``origin = chat`` (invariant 14).

A draft is an asynchronous, priced LLM job over one chat message, on the chat's own task machinery
(its own table, ``studio_runs`` untouched). Nothing is written to the artist's memories until the
user saves the (edited) fields through the normal memory create with ``from_draft_id``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from structlog.testing import capture_logs

from src.api.schemas.studio import ArtistCreate, MemoryCreate
from src.api.services.billing_errors import InsufficientBalanceError
from src.db.models.studio import StudioArtistMemory, StudioRun
from src.db.models.studio_chat import StudioMemoryDraft
from src.llm.port import LlmOutputInvalid, LlmUnavailable, LlmUsage
from src.studio.chat.draft import MemoryDraftDraft, MemoryDraftRefusal
from src.studio.errors import (
    ChatBusy,
    ChatRateLimited,
    IdempotencyConflict,
    IdempotencyKeyRequired,
    MemoryDraftInProgress,
    MemoryDraftNotFound,
    MemoryLimitReached,
    MessageNotComplete,
    MessageNotFound,
    StudioDocumentInvalid,
)
from src.studio.services.artist import ArtistService
from src.studio.services.memory import MemoryService
from tests.llm_support import FakeLlm
from tests.studio_chat_support import Gate, reply
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from src.db.models.studio_chat import StudioArtistChatMessage
    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

LONG = "I learned to sing on the rooftop, with the neon sign humming and the rain on the roof."


def good(**overrides: object) -> MemoryDraftDraft:
    base: dict[str, object] = {
        "refusal": None,
        "title": "Learning to sing on the roof",
        "body": "I learned to sing on a rooftop, under a humming neon sign.",
        "themes": ["home", "beginnings"],
        "imagery": ["humming neon sign", "rain on tar paper"],
    }
    return MemoryDraftDraft(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.fixture
async def actor(chat_world: ChatWorld) -> Actor:
    return await chat_world.add_actor(balance=20, artist=True)


async def _message(
    world: ChatWorld, actor: Actor, text_: str = LONG
) -> tuple[UUID, StudioArtistChatMessage]:
    """A completed artist reply in a new chat: ``(chat id, the artist message)``."""
    chat_id = await world.new_chat(actor)
    world.chat_llm.add(reply(text_))
    _, artist = await world.ask(actor, chat_id, "Where did you learn to sing?")
    assert artist.status == "complete"
    return chat_id, artist


async def _memories(world: ChatWorld, actor: Actor) -> int:
    return await world.count(StudioArtistMemory, actor)


# ---------------------------------------------------------------------------------------------
# the draft job
# ---------------------------------------------------------------------------------------------


class TestAskingForADraft:
    async def test_a_draft_completes_is_charged_once_and_creates_no_memory(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good(), usage=LlmUsage(200, 80))

        write = await chat_world.request_draft(actor, artist.id)
        assert write.changed
        assert write.draft.status == "pending"
        assert write.draft.draft is None
        assert write.draft.price_tokens == 1
        assert write.draft.source_message_id == artist.id
        await chat_world.settle()

        done = await chat_world.draft(write.draft.id)
        assert done.status == "complete"
        assert done.draft == {
            "title": "Learning to sing on the roof",
            "body": "I learned to sing on a rooftop, under a humming neon sign.",
            "themes": ["home", "beginnings"],
            "imagery": ["humming neon sign", "rain on tar paper"],
        }
        assert done.revision == 2
        assert done.finished_at is not None
        assert done.error_code is None
        assert (done.prompt_tokens, done.completion_tokens) == (200, 80)
        assert done.model_id == "fake-model"
        assert done.prompt_version == "memory-draft-v1"
        # one charge for the reply and one for the draft; no refund; and NO memory
        assert await chat_world.refunds_and_debits(actor) == (2, 0)
        assert await chat_world.balance(actor) == 18
        assert await _memories(chat_world, actor) == 0
        assert [u.status for u in chat_world.draft_updates(write.draft.id)] == [
            "pending",
            "complete",
        ]

    async def test_the_draft_is_asked_with_the_message_the_one_before_it_and_the_artist(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        await chat_world.draft_of(actor, artist.id)

        (call,) = chat_world.draft_llm.calls
        assert [m.role for m in call] == ["system", "user"]
        user = call[1].content
        assert "name: Nova" in user
        assert "<previous_message>\nfrom: user\nWhere did you learn to sing?" in user
        assert f"<source_message>\nfrom: artist\n{LONG}" in user
        assert "Adult themes are allowed." in call[0].content  # the product's content rules

    async def test_a_user_message_can_be_the_source(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """P2c-30: in a diary-style interview the user often tells the story."""
        chat_id, _ = await _message(chat_world, actor)
        user_message = (await chat_world.messages(chat_id))[0]
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, user_message.id)

        assert draft.status == "complete"
        (call,) = chat_world.draft_llm.calls
        assert "<source_message>\nfrom: user\nWhere did you learn to sing?" in call[1].content
        assert "<previous_message>" not in call[1].content  # it is the first message

    async def test_an_invalid_draft_is_fixed_once(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good(themes=["ok", "a[b]"]), good())
        draft = await chat_world.draft_of(actor, artist.id)

        assert draft.status == "complete"
        assert chat_world.draft_llm.call_count == 2
        fix = chat_world.draft_llm.calls[1]
        assert [m.role for m in fix] == ["system", "user", "assistant", "user"]
        assert "themes[1]" in fix[3].content  # the problem as a path...
        assert "a[b]" not in fix[3].content  # ...never as text
        assert await chat_world.refunds_and_debits(actor) == (2, 0)

    async def test_two_invalid_drafts_fail_and_refund(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good(title=" "), good(body=""))
        draft = await chat_world.draft_of(actor, artist.id)

        assert (draft.status, draft.error_code) == ("failed", "memory_draft_invalid")
        assert draft.draft is None
        assert chat_world.draft_llm.call_count == 2  # the cap: a call and one fix call
        assert await chat_world.refunds_and_debits(actor) == (2, 1)
        assert (draft.prompt_tokens, draft.completion_tokens) == (200, 100)

    async def test_an_undecodable_reply_is_asked_for_again_once(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(LlmOutputInvalid("no", usage=LlmUsage(5, 5)), good())
        draft = await chat_world.draft_of(actor, artist.id)
        assert draft.status == "complete"
        assert chat_world.draft_llm.call_count == 2
        assert [m.role for m in chat_world.draft_llm.calls[1]][-1] == "user"
        assert len(chat_world.draft_llm.calls[1]) == 3  # the undecodable reply is not echoed back

    async def test_a_content_policy_refusal_fails_and_refunds(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(
            good(
                refusal=MemoryDraftRefusal.CONTENT_POLICY, title="", body="", themes=[], imagery=[]
            )
        )
        draft = await chat_world.draft_of(actor, artist.id)
        assert (draft.status, draft.error_code) == ("failed", "content_policy_refused")
        assert chat_world.draft_llm.call_count == 1
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_nothing_to_remember_is_an_invalid_draft(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(
            good(refusal=MemoryDraftRefusal.NOTHING_TO_REMEMBER, title="", body="")
        )
        draft = await chat_world.draft_of(actor, artist.id)
        assert (draft.status, draft.error_code) == ("failed", "memory_draft_invalid")
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_a_provider_error_fails_and_refunds(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(LlmUnavailable("down"))
        draft = await chat_world.draft_of(actor, artist.id)
        assert (draft.status, draft.error_code) == ("failed", "llm_unavailable")
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_a_draft_that_never_finishes_is_failed_and_refunded_by_the_sweep(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        chat_world.draft_llm = FakeLlm(good(), before_call=blocked)
        write = await chat_world.request_draft(actor, artist.id)
        await gate.reached.wait()
        await chat_world.age(draft_id=write.draft.id)

        assert await chat_world.sweeper().sweep() == 1
        swept = await chat_world.draft(write.draft.id)
        assert (swept.status, swept.error_code) == ("failed", "reply_timeout")
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

        gate.release()  # the late task finishes: it must not complete, charge or refund again
        await chat_world.settle()
        again = await chat_world.draft(write.draft.id)
        assert again.status == "failed"
        assert again.draft is None
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_the_deadline_fails_a_draft(self, chat_world: ChatWorld, actor: Actor) -> None:
        _, artist = await _message(chat_world, actor)
        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        chat_world.draft_llm = FakeLlm(good(), before_call=blocked)
        write = await chat_world.request_draft(actor, artist.id, dispatch=False)
        assert write.job is not None
        job = dataclasses.replace(write.job, deadline_seconds=0.05)
        chat_world.supervisor.start_memory_draft(job)
        await chat_world.settle()
        draft = await chat_world.draft(write.draft.id)
        assert (draft.status, draft.error_code) == ("failed", "reply_timeout")
        gate.release()

    async def test_shutdown_fails_and_refunds_an_in_flight_draft(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        chat_world.draft_llm = FakeLlm(good(), before_call=blocked)
        write = await chat_world.request_draft(actor, artist.id)
        await gate.reached.wait()
        await chat_world.supervisor.shutdown()

        draft = await chat_world.draft(write.draft.id)
        assert (draft.status, draft.error_code) == ("failed", "server_shutdown")
        assert await chat_world.refunds_and_debits(actor) == (2, 1)

    async def test_the_draft_never_touches_the_runs(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        await chat_world.draft_of(actor, artist.id)
        assert await chat_world.count(StudioRun, actor) == 0


class TestTheDraftRequest:
    async def test_the_key_is_required(self, chat_world: ChatWorld, actor: Actor) -> None:
        _, artist = await _message(chat_world, actor)
        for key in ("", "   ", "k" * 129):
            with pytest.raises(IdempotencyKeyRequired):
                await chat_world.request_draft(actor, artist.id, key)

    async def test_a_replay_returns_the_stored_draft_and_charges_once(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        first = await chat_world.request_draft(actor, artist.id, "same-key")
        await chat_world.settle()

        again = await chat_world.request_draft(actor, artist.id, "same-key")
        await chat_world.settle()
        assert again.changed is False
        assert again.job is None
        assert again.draft.id == first.draft.id
        assert again.draft.status == "complete"
        assert chat_world.draft_llm.call_count == 1
        assert await chat_world.refunds_and_debits(actor) == (2, 0)

    async def test_the_same_key_for_another_message_is_a_conflict(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_id = await chat_world.new_chat(actor, "another")
        chat_world.chat_llm.add(reply(LONG))
        _, other = await chat_world.ask(actor, chat_id)
        chat_world.draft_llm = FakeLlm(good())
        await chat_world.draft_of(actor, artist.id, "shared-key")

        with pytest.raises(IdempotencyConflict):
            await chat_world.request_draft(actor, other.id, "shared-key")

    async def test_a_second_draft_of_the_same_message_while_one_is_pending_is_refused(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        chat_world.draft_llm = FakeLlm(good(), good(), before_call=blocked)
        await chat_world.request_draft(actor, artist.id, "one")
        await gate.reached.wait()
        with pytest.raises(MemoryDraftInProgress):
            await chat_world.request_draft(actor, artist.id, "two")
        gate.release()
        await chat_world.settle()
        # once the first is done a new draft of the same message is allowed (a second opinion)
        again = await chat_world.draft_of(actor, artist.id, "three")
        assert again.status == "complete"

    async def test_only_a_complete_message_can_be_a_source(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)], [LlmUnavailable("down")])
        pending = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        with pytest.raises(MessageNotComplete):
            await chat_world.request_draft(actor, pending.artist_message.id)
        gate.release()
        await chat_world.settle()

        _, failed = await chat_world.ask(actor, chat_id, "again")
        assert failed.status == "failed"
        with pytest.raises(MessageNotComplete):
            await chat_world.request_draft(actor, failed.id)
        assert await chat_world.count(StudioMemoryDraft, actor) == 0
        assert await chat_world.refunds_and_debits(actor) == (2, 1)  # nothing charged for drafts

    async def test_the_draft_is_priced_and_a_402_leaves_nothing(
        self, chat_world: ChatWorld
    ) -> None:
        poor = await chat_world.add_actor(balance=1, artist=True)
        _, artist = await _message(chat_world, poor)  # spends the only token
        with pytest.raises(InsufficientBalanceError):
            await chat_world.request_draft(poor, artist.id)
        assert await chat_world.count(StudioMemoryDraft, poor) == 0

    async def test_the_draft_price_is_its_own_setting(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_memory_draft_price=3)
        actor = await world.add_actor(balance=10, artist=True)
        _, artist = await _message(world, actor)
        world.draft_llm = FakeLlm(good())
        draft = await world.draft_of(actor, artist.id)
        assert draft.price_tokens == 3
        assert await world.balance(actor) == 10 - 1 - 3

    async def test_drafts_share_the_hourly_budget_with_messages(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=2)
        actor = await world.add_actor(balance=10, artist=True)
        _, artist = await _message(world, actor)  # one message
        world.draft_llm = FakeLlm(good())
        await world.draft_of(actor, artist.id)  # two
        with pytest.raises(ChatRateLimited):
            await world.request_draft(actor, artist.id)

    async def test_a_chat_with_a_pending_draft_cannot_be_deleted(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id, artist = await _message(chat_world, actor)
        gate = Gate()

        async def blocked() -> None:
            await gate.wait()

        chat_world.draft_llm = FakeLlm(good(), before_call=blocked)
        write = await chat_world.request_draft(actor, artist.id)
        await gate.reached.wait()
        async with chat_world.factory() as session:
            with pytest.raises(ChatBusy):
                await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)
        gate.release()
        await chat_world.settle()

        async with chat_world.factory() as session:
            await chat_world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)
            await session.commit()
        assert await chat_world.count(StudioMemoryDraft, actor) == 0  # the draft went with its chat
        assert write.draft.id is not None


class TestScoping:
    async def test_another_users_message_and_draft_are_not_found(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)
        stranger = await chat_world.add_actor(balance=10, artist=True)

        with pytest.raises(MessageNotFound):
            await chat_world.request_draft(stranger, artist.id)
        async with chat_world.factory() as session:
            with pytest.raises(MemoryDraftNotFound):
                await chat_world.draft_service(session).get(draft.id, stranger.user_id, PRODUCT)
            with pytest.raises(MemoryDraftNotFound):
                await chat_world.draft_service(session).get(draft.id, actor.user_id, "synthara")
            out = await chat_world.draft_service(session).get(draft.id, actor.user_id, PRODUCT)
        assert out.draft is not None
        assert out.draft.title == "Learning to sing on the roof"

    async def test_the_database_refuses_a_draft_whose_owner_differs_from_its_messages(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id, artist = await _message(chat_world, actor)
        stranger = await chat_world.add_actor(balance=1, artist=True)
        async with chat_world.factory() as session:
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        "INSERT INTO studio_memory_drafts (id, user_id, product_id, chat_id, "
                        "source_message_id, status, idempotency_key, price_tokens) VALUES "
                        "(:id, :user, 'vex', :chat, :msg, 'pending', 'k', 0)"
                    ),
                    {"id": uuid4(), "user": stranger.user_id, "chat": chat_id, "msg": artist.id},
                )


# ---------------------------------------------------------------------------------------------
# saving a draft as a memory (invariant 14)
# ---------------------------------------------------------------------------------------------


async def _create(
    world: ChatWorld, actor: Actor, artist_id: UUID | None, **fields: object
) -> object:
    data = {"title": "Edited title", "body": "Edited body", **fields}
    async with world.factory() as session, session.begin():
        return await MemoryService(session).create(
            artist_id or actor.artist_id,  # type: ignore[arg-type]
            actor.user_id,
            PRODUCT,
            MemoryCreate(**data),  # type: ignore[arg-type]
        )


class TestSavingADraftAsAMemory:
    async def test_a_memory_made_from_a_draft_is_origin_chat_and_keeps_the_clients_fields(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)

        memory = await _create(
            chat_world, actor, None, title="My own title", body="Edited by the user",
            themes=["kept"], from_draft_id=draft.id,
        )  # fmt: skip
        assert memory.origin.value == "chat"  # type: ignore[attr-defined]
        # the client still sends the (edited) fields: nothing is copied from the draft
        assert memory.title == "My own title"  # type: ignore[attr-defined]
        assert memory.body == "Edited by the user"  # type: ignore[attr-defined]
        assert memory.themes == ["kept"]  # type: ignore[attr-defined]
        assert await _memories(chat_world, actor) == 1

    async def test_without_a_draft_the_origin_is_user(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        memory = await _create(chat_world, actor, None)
        assert memory.origin.value == "user"  # type: ignore[attr-defined]

    async def test_the_origin_is_never_taken_from_the_client(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="origin"):
            msgspec.json.decode(b'{"title": "t", "body": "b", "origin": "chat"}', type=MemoryCreate)

    async def test_a_memory_made_from_a_draft_is_an_ordinary_memory(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """It is normalized, limited and selected like any other - and told to the next reply."""
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)
        with pytest.raises(StudioDocumentInvalid) as caught:
            await _create(chat_world, actor, None, title="  ", from_draft_id=draft.id)
        assert [(i.path, i.code) for i in caught.value.issues] == [
            ("title", "empty_after_normalization")
        ]
        assert await _memories(chat_world, actor) == 0

        await _create(chat_world, actor, None, title="Saved", from_draft_id=draft.id)
        chat_id = await chat_world.new_chat(actor, "later")
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)
        assert "Saved" in chat_world.chat_llm.system_messages[-1]

    @pytest.mark.parametrize(
        ("break_it", "code"),
        [
            ("unknown", "draft_not_found"),
            ("pending", "draft_not_complete"),
            ("failed", "draft_not_complete"),
            ("other_artist", "draft_other_artist"),
            ("other_user", "draft_not_found"),
        ],
    )
    async def test_a_draft_that_cannot_back_the_memory_is_a_422(
        self, chat_world: ChatWorld, actor: Actor, break_it: str, code: str
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)
        draft_id: UUID = draft.id
        target_artist: UUID | None = None
        caller = actor
        if break_it == "unknown":
            draft_id = uuid4()
        elif break_it in {"pending", "failed"}:
            async with chat_world.factory() as session, session.begin():
                if break_it == "pending":
                    await session.execute(
                        text(
                            "UPDATE studio_memory_drafts SET status = 'pending', draft = NULL, "
                            "finished_at = NULL WHERE id = :id"
                        ),
                        {"id": draft.id},
                    )
                else:
                    await session.execute(
                        text(
                            "UPDATE studio_memory_drafts SET status = 'failed', draft = NULL, "
                            "error_code = 'llm_unavailable' WHERE id = :id"
                        ),
                        {"id": draft.id},
                    )
        elif break_it == "other_artist":
            async with chat_world.factory() as session, session.begin():
                second = await ArtistService(session).create(
                    actor.user_id, PRODUCT, ArtistCreate(name="A Second Artist")
                )
            target_artist = second.id  # the draft is from a chat of Nova, not of this artist
        else:
            caller = await chat_world.add_actor(balance=1, artist=True)

        with pytest.raises(StudioDocumentInvalid) as caught:
            await _create(chat_world, caller, target_artist, from_draft_id=draft_id)
        assert [(i.path, i.code) for i in caught.value.issues] == [("from_draft_id", code)]
        assert await _memories(chat_world, actor) == 0

    async def test_the_two_hundred_cap_still_holds_for_a_memory_from_a_draft(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)
        async with chat_world.factory() as session, session.begin():
            await session.execute(
                text(
                    "INSERT INTO studio_artist_memories (id, user_id, product_id, artist_id, "
                    "title, body) SELECT gen_random_uuid(), :u, 'vex', :a, 't' || g, 'b' "
                    "FROM generate_series(1, 200) g"
                ),
                {"u": actor.user_id, "a": actor.artist_id},
            )
        with pytest.raises(MemoryLimitReached):
            await _create(chat_world, actor, None, from_draft_id=draft.id)


# ---------------------------------------------------------------------------------------------
# privacy (invariant 15)
# ---------------------------------------------------------------------------------------------


class TestPrivacy:
    async def test_no_log_record_carries_the_message_or_the_draft(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        secret_reply = "PRIVATE-REPLY-WORDS-1187 and a few more words to open the gate."
        secret_draft = "PRIVATE-DRAFT-BODY-6620"
        _, artist = await _message(chat_world, actor, secret_reply)
        chat_world.draft_llm = FakeLlm(
            good(title="PRIVATE-DRAFT-TITLE-3391", body=secret_draft, themes=["a[b]"]),
            good(title="PRIVATE-DRAFT-TITLE-3391", body=secret_draft),
        )
        with capture_logs() as logs:
            await chat_world.draft_of(actor, artist.id)

        rendered = json.dumps(logs, default=str)
        for secret in ("PRIVATE-REPLY-WORDS", secret_draft, "PRIVATE-DRAFT-TITLE"):
            assert secret not in rendered, secret
        assert {"studio.memory_draft_requested", "studio.memory_draft_finished"} <= {
            log["event"] for log in logs
        }

    async def test_the_draft_events_carry_ids_and_a_status_only(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        _, artist = await _message(chat_world, actor)
        chat_world.draft_llm = FakeLlm(good())
        draft = await chat_world.draft_of(actor, artist.id)

        for payload in chat_world.draft_updates(draft.id):
            encoded = json.dumps({f: str(getattr(payload, f)) for f in payload.__struct_fields__})
            assert "humming neon sign" not in encoded
            assert "Learning to sing" not in encoded
