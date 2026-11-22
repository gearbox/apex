"""Hourly limits must not be reset by deleting data (S1-P2c remediation R2).

The hourly limits cap provider spend whatever a job costs (a price of 0 bills nothing). They used to
count the surviving rows of the jobs - chat messages, memory drafts, planner runs - so
send → complete → delete the chat (or the track) → repeat ran any number of jobs per hour. They now
count ``studio_llm_usage``: an append-only ledger, written in the savepoint of every accepted job,
that no studio delete reaches (its only foreign key is to ``users``).

Committed rows over the real PostgreSQL: the ledger is read, purged and written under the per-user
advisory quota lock by separate request transactions.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.api.services.billing_errors import InsufficientBalanceError
from src.core.enums import StudioLlmUsageKind
from src.db.models.studio_chat import StudioArtistChatMessage, StudioMemoryDraft
from src.studio.chat.draft import MemoryDraftDraft
from src.studio.errors import ChatRateLimited, RunRateLimited
from src.studio.services.track import TrackService
from tests.llm_support import FakeLlm
from tests.studio_chat_support import reply
from tests.studio_llm import make_draft
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

LONG = "I remember the rooftop, the neon sign and the rain on the asphalt that night."
CHAT_MESSAGE = StudioLlmUsageKind.CHAT_MESSAGE.value
MEMORY_DRAFT = StudioLlmUsageKind.MEMORY_DRAFT.value
RUN = StudioLlmUsageKind.RUN.value


def _memory_draft() -> MemoryDraftDraft:
    return MemoryDraftDraft(
        refusal=None,
        title="Learning to sing on the roof",
        body="I learned to sing on a rooftop, under a humming neon sign.",
        themes=["home"],
        imagery=["humming neon sign"],
    )


async def _delete_chat(world: ChatWorld, actor: Actor, chat_id: UUID) -> None:
    async with world.factory() as session, session.begin():
        await world.chat_service(session).delete(chat_id, actor.user_id, PRODUCT)


async def _delete_track(world: ChatWorld, actor: Actor, track_id: UUID) -> None:
    async with world.factory() as session, session.begin():
        await TrackService(session).delete(track_id, actor.user_id, PRODUCT)


async def _insert_usage(world: ChatWorld, actor: Actor, kind: str, *, age: timedelta) -> None:
    """A ledger row written directly, as if the job had been accepted ``age`` ago."""
    async with world.factory() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO studio_llm_usage (id, user_id, product_id, kind, created_at) "
                "VALUES (:id, :u, :p, :kind, now() - CAST(:age AS interval))"
            ),
            {"id": uuid4(), "u": actor.user_id, "p": PRODUCT, "kind": kind, "age": age},
        )


class TestDeletingChatsDoesNotResetTheChatLimit:
    async def test_deleting_chats_does_not_reset_the_hourly_chat_limit(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=3)
        actor = await world.add_actor(balance=20, artist=True)
        first = await world.new_chat(actor, "one")
        second = await world.new_chat(actor, "two")
        world.chat_llm.add(reply(LONG), reply(LONG), reply(LONG))
        await world.ask(actor, first, "a")
        await world.ask(actor, first, "b")
        await world.ask(actor, second, "c")  # three messages across two chats, all complete

        await _delete_chat(world, actor, first)
        await _delete_chat(world, actor, second)
        assert await world.count(StudioArtistChatMessage, actor) == 0  # the rows are really gone
        assert await world.usage(actor, CHAT_MESSAGE) == 3  # ... the ledger is not

        third = await world.new_chat(actor, "three")
        with pytest.raises(ChatRateLimited) as limited:
            await world.send(actor, third)
        assert (limited.value.status_code, limited.value.code) == (429, "chat_rate_limited")
        assert await world.refunds_and_debits(actor) == (3, 0)  # nothing was charged for the 429

    async def test_deleting_chats_does_not_reset_the_limit_that_drafts_share_with_messages(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=3)
        actor = await world.add_actor(balance=20, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        await world.ask(actor, chat_id, "a")
        _, artist_message = await world.ask(actor, chat_id, "b")  # two messages
        world.draft_llm = FakeLlm(_memory_draft())
        await world.draft_of(actor, artist_message.id)  # ... and a draft: three jobs

        await _delete_chat(world, actor, chat_id)  # the messages and the draft cascade away
        assert await world.count(StudioMemoryDraft, actor) == 0
        assert await world.usage(actor, CHAT_MESSAGE, MEMORY_DRAFT) == 3

        with pytest.raises(ChatRateLimited):
            await world.send(actor, await world.new_chat(actor, "again"))

    async def test_a_draft_request_reads_the_same_ledger(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        """The draft path has its own limit check: with row counting, deleting the other chat
        would have made room for the draft."""
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=2)
        actor = await world.add_actor(balance=20, artist=True)
        kept = await world.new_chat(actor, "kept")
        other = await world.new_chat(actor, "other")
        world.chat_llm.add(reply(LONG), reply(LONG))
        _, kept_reply = await world.ask(actor, kept, "a")
        await world.ask(actor, other, "b")  # two jobs: the budget is spent
        await _delete_chat(world, actor, other)

        with pytest.raises(ChatRateLimited):
            await world.request_draft(actor, kept_reply.id)
        assert await world.usage(actor, MEMORY_DRAFT) == 0


class TestDeletingTracksDoesNotResetTheRunLimit:
    async def test_deleting_tracks_does_not_reset_the_hourly_run_limit(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_runs_per_user_per_hour=2)
        actor = await world.add_actor(balance=20)
        track = await world.new_track(actor, "first")
        other = await world.new_track(actor, "other")
        await world.submit(actor, track)
        await world.drive(FakeLlm(make_draft(title="One")))
        await world.submit(actor, track)  # a second run on the same track, once the first is done
        await world.drive(FakeLlm(make_draft(title="Two")))
        assert await world.count_runs(actor) == 2

        await _delete_track(world, actor, track)  # its finished runs cascade away
        assert await world.count_runs(actor) == 0
        assert await world.usage(actor, RUN) == 2

        with pytest.raises(RunRateLimited) as limited:
            await world.submit(actor, other)
        assert (limited.value.status_code, limited.value.code) == (429, "run_rate_limited")
        assert await world.count_runs(actor) == 0

    async def test_a_cancelled_run_still_counts(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        """Every *accepted* job counts, whatever became of it - exactly as the rows did."""
        world = make_chat_world(studio_runs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20)
        track = await world.new_track(actor)
        write = await world.submit(actor, track)
        async with world.factory() as session:
            await world.service(session).cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        assert await world.refunds_and_debits(actor) == (1, 1)  # refunded in full ...
        with pytest.raises(RunRateLimited):  # ... and still one of this hour's runs
            await world.submit(actor, track)


class TestWhatCounts:
    async def test_a_replay_and_a_402_record_no_usage(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world()
        actor = await world.add_actor(balance=20, artist=True)

        # chat message: a replay of the same client_message_id writes nothing ...
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        client_id = uuid4()
        first = await world.send(actor, chat_id, "hello", client_message_id=client_id)
        await world.settle()
        again = await world.send(actor, chat_id, "hello", client_message_id=client_id)
        assert (first.changed, again.changed) == (True, False)
        assert await world.usage(actor, CHAT_MESSAGE) == 1
        # ... and a 402 rolls the ledger row back with the pair
        poor = await world.add_actor(balance=0, artist=True)
        with pytest.raises(InsufficientBalanceError):
            await world.send(poor, await world.new_chat(poor))
        assert await world.usage(poor) == 0
        assert await world.count(StudioArtistChatMessage, poor) == 0

        # memory draft: the same
        world.draft_llm = FakeLlm(_memory_draft())
        draft = await world.request_draft(actor, first.artist_message.id, "draft-key")
        await world.settle()
        replayed = await world.request_draft(actor, first.artist_message.id, "draft-key")
        assert (draft.changed, replayed.changed) == (True, False)
        assert await world.usage(actor, MEMORY_DRAFT) == 1
        broke = await world.add_actor(balance=1, artist=True)  # exactly one reply, then nothing
        world.chat_llm.add(reply(LONG))
        _, broke_reply = await world.ask(broke, await world.new_chat(broke))
        with pytest.raises(InsufficientBalanceError):
            await world.request_draft(broke, broke_reply.id)
        assert await world.usage(broke, MEMORY_DRAFT) == 0
        assert await world.usage(broke, CHAT_MESSAGE) == 1

        # run: the same
        track = await world.new_track(actor)
        submitted = await world.submit(actor, track, key="run-key")
        resubmitted = await world.submit(actor, track, key="run-key")
        assert (submitted.changed, resubmitted.changed) == (True, False)
        assert await world.usage(actor, RUN) == 1
        with pytest.raises(InsufficientBalanceError):
            await world.submit(poor, await world.new_track(poor))
        assert await world.usage(poor, RUN) == 0

    async def test_a_refused_request_records_nothing_either(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        """A 429 is not a job: the limit that refuses a send must not count the refused send."""
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        await world.ask(actor, chat_id, "one")
        for _ in range(3):
            with pytest.raises(ChatRateLimited):
                await world.send(actor, chat_id, "more")
        assert await world.usage(actor) == 1

    async def test_the_price_does_not_matter(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        """With every price 0 nothing is billed, and the hourly limits still hold."""
        world = make_chat_world(
            studio_chat_message_price=0,
            studio_memory_draft_price=0,
            studio_run_prices={"plan": 0, "lyrics": 0, "revise_lyrics": 0},
            studio_chat_jobs_per_user_per_hour=2,
            studio_runs_per_user_per_hour=2,
        )
        actor = await world.add_actor(balance=0, artist=True)  # no money at all
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        await world.ask(actor, chat_id, "one")
        await world.ask(actor, chat_id, "two")
        await _delete_chat(world, actor, chat_id)
        with pytest.raises(ChatRateLimited):
            await world.send(actor, await world.new_chat(actor, "again"))

        track = await world.new_track(actor)
        other = await world.new_track(actor, "other")
        await world.submit(actor, track)
        await world.drive(FakeLlm(make_draft(title="One")))
        await world.submit(actor, track)
        await world.drive(FakeLlm(make_draft(title="Two")))
        await _delete_track(world, actor, track)
        with pytest.raises(RunRateLimited):
            await world.submit(actor, other)
        assert await world.refunds_and_debits(actor) == (0, 0)  # nothing was ever billed

    async def test_chat_and_run_usage_are_counted_separately(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(
            studio_chat_jobs_per_user_per_hour=1, studio_runs_per_user_per_hour=1
        )
        actor = await world.add_actor(balance=20, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        await world.ask(actor, chat_id, "one")
        await world.submit(
            actor, await world.new_track(actor)
        )  # the chat budget is spent, not this
        with pytest.raises(ChatRateLimited):
            await world.send(actor, chat_id, "two")
        with pytest.raises(RunRateLimited):
            await world.submit(actor, await world.new_track(actor, "other"))

    async def test_two_sends_racing_at_the_limit_cannot_both_pass(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        """Purge, count and record all happen under the per-user lock, so two requests of one user
        in different chats serialize: one is accepted, the other is limited."""
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20, artist=True)
        first = await world.new_chat(actor, "a")
        second = await world.new_chat(actor, "b")
        world.chat_llm.add(reply(LONG))
        results = await asyncio.gather(
            world.send(actor, first), world.send(actor, second), return_exceptions=True
        )
        await world.settle()
        assert sorted(type(r).__name__ for r in results) == ["ChatRateLimited", "ChatWrite"]
        assert await world.usage(actor) == 1


class TestRetention:
    async def test_usage_older_than_an_hour_is_purged_and_not_counted(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=2)
        actor = await world.add_actor(balance=20, artist=True)
        bystander = await world.add_actor(balance=20)
        for _ in range(5):  # stale rows would more than exhaust the limit if they counted
            await _insert_usage(world, actor, CHAT_MESSAGE, age=timedelta(hours=2))
        await _insert_usage(world, actor, MEMORY_DRAFT, age=timedelta(hours=3))
        await _insert_usage(
            world, actor, RUN, age=timedelta(hours=2)
        )  # another kind: not this check's
        await _insert_usage(
            world, bystander, CHAT_MESSAGE, age=timedelta(hours=2)
        )  # another user's
        await _insert_usage(world, actor, CHAT_MESSAGE, age=timedelta(minutes=59))  # still counts

        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG))
        await world.ask(actor, chat_id, "one")  # one limit check: purges, then counts 1 < 2

        # the chat check purged this user's stale chat rows and nothing else
        assert await world.usage(actor, CHAT_MESSAGE) == 2  # the 59-minute row and the new one
        assert await world.usage(actor, MEMORY_DRAFT) == 0
        assert await world.usage(actor, RUN) == 1
        assert await world.usage(bystander, CHAT_MESSAGE) == 1
        with pytest.raises(ChatRateLimited):  # 59 minutes ago still counts: 2 >= 2
            await world.send(actor, chat_id, "two")

    async def test_old_run_usage_is_purged_by_the_run_check_only(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_runs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20)
        for _ in range(3):
            await _insert_usage(world, actor, RUN, age=timedelta(minutes=90))
        await _insert_usage(world, actor, CHAT_MESSAGE, age=timedelta(minutes=90))

        await world.submit(actor, await world.new_track(actor))  # purge, count 0 < 1, accept
        assert await world.usage(actor, RUN) == 1
        assert await world.usage(actor, CHAT_MESSAGE) == 1  # not this check's kind

    async def test_an_hour_later_the_budget_is_back(
        self, make_chat_world: Callable[..., ChatWorld]
    ) -> None:
        world = make_chat_world(studio_chat_jobs_per_user_per_hour=1)
        actor = await world.add_actor(balance=20, artist=True)
        chat_id = await world.new_chat(actor)
        world.chat_llm.add(reply(LONG), reply(LONG))
        await world.ask(actor, chat_id, "one")
        with pytest.raises(ChatRateLimited):
            await world.send(actor, chat_id, "two")
        await world.age_usage(actor, age=timedelta(minutes=61))
        await world.ask(actor, chat_id, "two")
        assert await world.usage(actor) == 1  # the old row was purged by that check
