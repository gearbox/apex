"""What a reply is told, and what never reaches a log or the planner (invariants 7, 8, 9, 15, 17).

The memories come only through ``ArtistMemorySelector`` and fit its budget; the history is this
chat's, within its window; the discography is the artist's own lyrics, within its budget; chat text
never reaches a planner run, a log line or the Suno export.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec
import pytest
from structlog.testing import capture_logs

from src.core.enums import StudioSectionKind
from src.llm.port import LlmUnavailable
from src.studio.constraints import (
    CHAT_WINDOW_BUDGET,
    CHAT_WINDOW_MESSAGES,
    DISCOGRAPHY_BUDGET,
    MEMORY_CONTEXT_BUDGET,
)
from src.studio.services.export import ExportService
from tests.llm_support import FakeLlm
from tests.studio_chat_support import Gate, reply
from tests.studio_llm import make_draft
from tests.studio_runs_support import PRODUCT
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from tests.studio_chat_support import ChatWorld
    from tests.studio_runs_support import Actor

LONG = "I remember the rooftop, the neon sign and the rain on the asphalt that night."


@pytest.fixture
async def actor(chat_world: ChatWorld) -> Actor:
    return await chat_world.add_actor(balance=100, artist=True)


def _block(system: str, tag: str) -> str | None:
    """The body of ``<tag>`` in a system message; ``None`` when the block is not there."""
    opening = f"<{tag}>\n"
    if opening not in system:
        return None
    return system.split(opening, 1)[1].split(f"\n</{tag}>", 1)[0]


# ---------------------------------------------------------------------------------------------
# the system message
# ---------------------------------------------------------------------------------------------


class TestTheSystemMessage:
    async def test_the_artist_is_told_who_it_is_and_the_product_rules(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)

        system = chat_world.chat_llm.system_messages[0]
        assert system.startswith("You are Nova, a music artist.")
        assert "persona: A SECRET-PERSONA synth-pop singer" in system
        assert "voice notes: breathy" in system
        assert "style tags: dreamy" in system
        assert "language: en" in system
        assert "reply with exactly ⟦declined⟧ and nothing else" in system
        # vex is permissive; the planner-only fields are not told to a chat
        assert "Adult themes are allowed." in system
        assert "neon noir" not in system
        assert "SECRET-CONCEPT" not in system  # the project's concept is the planner's

    async def test_without_memories_and_tracks_there_are_no_such_blocks(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        system = chat_world.chat_llm.system_messages[0]
        assert _block(system, "artist_memories") is None
        assert _block(system, "discography") is None
        assert write.artist_message.memory_ids_used == []

    async def test_the_prompt_version_and_the_model_are_recorded_on_the_reply(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG, usage=(321, 45), model_id="the-model-that-answered"))
        _, artist = await chat_world.ask(actor, chat_id)

        assert artist.prompt_version == "chat-v1"
        assert artist.model_key == "grok"
        assert artist.model_id == "the-model-that-answered"
        assert (artist.prompt_tokens, artist.completion_tokens) == (321, 45)
        assert artist.role == "artist"


# ---------------------------------------------------------------------------------------------
# memories (invariant 8)
# ---------------------------------------------------------------------------------------------


class TestMemories:
    async def test_the_reply_is_given_the_memories_in_selection_order_and_records_them(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        low = await chat_world.add_memory(actor, title="A minor day", salience=1)
        high = await chat_world.add_memory(actor, title="The night of the fire", salience=5)
        mid = await chat_world.add_memory(actor, title="First guitar", salience=3)
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        block = _block(chat_world.chat_llm.system_messages[0], "artist_memories")
        assert block is not None
        assert block.index("The night of the fire") < block.index("First guitar")
        assert block.index("First guitar") < block.index("A minor day")
        assert block.startswith("- m1: The night of the fire")
        # the ids are the prompt order, and they are on the pending row from the start
        assert write.artist_message.memory_ids_used == [high, mid, low]
        assert (await chat_world.message(write.artist_message.id)).memory_ids_used == [
            high,
            mid,
            low,
        ]
        assert write.job is not None
        assert write.job.memory_ids_used == (high, mid, low)

    async def test_archived_memories_and_other_artists_memories_are_not_told(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        kept = await chat_world.add_memory(actor, title="Kept memory")
        archived = await chat_world.add_memory(actor, title="Archived memory")
        await chat_world.set_memory(actor, archived, status="archived")
        stranger = await chat_world.add_actor(balance=5, artist=True)
        await chat_world.add_memory(stranger, title="Someone else's memory")
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        system = chat_world.chat_llm.system_messages[0]
        assert "Kept memory" in system
        assert "Archived memory" not in system
        assert "Someone else's memory" not in system
        assert write.artist_message.memory_ids_used == [kept]

    async def test_the_memories_fit_the_budget_and_the_first_that_does_not_ends_the_selection(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        for number in range(12):  # about 1,990 characters each: 6 fit in 12,000
            await chat_world.add_memory(
                actor,
                title=f"Memory number {number:02d}",
                body="x" * 1_990,
                salience=5 - number // 4,
            )
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        write = await chat_world.send(actor, chat_id)
        await chat_world.settle()

        block = _block(chat_world.chat_llm.system_messages[0], "artist_memories")
        assert block is not None
        assert len(block) <= MEMORY_CONTEXT_BUDGET
        assert 1 < len(write.artist_message.memory_ids_used) < 12

    async def test_a_memory_archived_after_sending_does_not_change_what_was_sent(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        memory = await chat_world.add_memory(actor, title="Rooftop night")
        chat_id = await chat_world.new_chat(actor)
        gate = Gate()
        chat_world.chat_llm.add([gate, *reply(LONG)])
        write = await chat_world.send(actor, chat_id)
        await gate.reached.wait()
        await chat_world.set_memory(actor, memory, status="archived")
        gate.release()
        await chat_world.settle()

        assert "Rooftop night" in chat_world.chat_llm.system_messages[0]
        assert (await chat_world.message(write.artist_message.id)).memory_ids_used == [memory]


# ---------------------------------------------------------------------------------------------
# the discography
# ---------------------------------------------------------------------------------------------


class TestTheDiscography:
    async def _track_with_lyrics(
        self, world: ChatWorld, actor: Actor, title: str, *lines: str, notes: str | None = None
    ) -> None:
        section = make_section(StudioSectionKind.VERSE, notes=notes)
        plan = make_plan(section, title=title, idea="A SECRET-IDEA about the track")
        track_id = await world.new_track(actor, title, plan=plan)
        row = await world.track_row(track_id)
        assert row.current_plan_version_id is not None
        await world.create_lyrics(
            actor,
            track_id,
            make_lyrics((section.id, list(lines))),
            plan_version_id=row.current_plan_version_id,
        )

    async def test_the_artists_tracks_and_lyrics_are_told_as_plain_text(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await self._track_with_lyrics(
            chat_world, actor, "Neon Rain", "Rain on the neon sign", "I hum along",
            notes="SECRET-PRODUCTION-NOTE",
        )  # fmt: skip
        await chat_world.new_track(actor, "A Track Without Lyrics")
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)

        system = chat_world.chat_llm.system_messages[0]
        block = _block(system, "discography")
        assert block is not None
        assert block.startswith("Tracks: ")
        assert "Neon Rain" in block
        assert "A Track Without Lyrics" in block
        assert "\n  Verse\n" in block  # one verse: named without a number
        assert "    Rain on the neon sign\n    I hum along" in block
        # a plan's idea and notes are the planner's, never told to a chat
        assert "SECRET-IDEA" not in system
        assert "SECRET-PRODUCTION-NOTE" not in system
        assert len(block) <= DISCOGRAPHY_BUDGET

    async def test_another_artists_tracks_are_not_told(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        stranger = await chat_world.add_actor(balance=5, artist=True)
        await self._track_with_lyrics(chat_world, stranger, "Stranger Song", "not for you")
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)

        system = chat_world.chat_llm.system_messages[0]
        assert "Stranger Song" not in system
        assert "not for you" not in system

    async def test_a_long_discography_fits_its_budget(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        for number in range(6):
            await self._track_with_lyrics(
                chat_world, actor, f"Song {number}", *[f"line {number}-{i} " + "w" * 200 for i in range(12)]
            )  # fmt: skip
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id)

        block = _block(chat_world.chat_llm.system_messages[0], "discography")
        assert block is not None
        assert len(block) <= DISCOGRAPHY_BUDGET
        assert "Song 5" in block.split("\n\n", 1)[0]  # every title is in the Tracks line
        assert block.count("- Song") < 6  # not every track's lyrics fit


# ---------------------------------------------------------------------------------------------
# the window (invariants 7 and 9)
# ---------------------------------------------------------------------------------------------


class TestTheWindow:
    async def test_the_reply_sees_this_chats_history_in_order_with_the_new_message_last(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply("Answer one is long enough to open."), reply(LONG))
        await chat_world.ask(actor, chat_id, "Question one")
        await chat_world.ask(actor, chat_id, "Question two")

        call = chat_world.chat_llm.calls[1]
        assert [(m.role, m.content) for m in call[1:]] == [
            ("user", "Question one"),
            ("assistant", "Answer one is long enough to open."),
            ("user", "Question two"),
        ]

    async def test_another_chat_of_the_same_artist_is_never_in_the_context(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        """Invariant 7: the assembled context holds text from this chat only."""
        first = await chat_world.new_chat(actor, "first")
        second = await chat_world.new_chat(actor, "second")
        chat_world.chat_llm.add(reply("The secret of chat one, told in full."), reply(LONG))
        await chat_world.ask(actor, first, "SECRET-QUESTION-OF-CHAT-ONE")
        await chat_world.ask(actor, second, "A question in the second chat")

        whole = json.dumps([[m.content for m in call] for call in chat_world.chat_llm.calls[1:]])
        assert "SECRET-QUESTION-OF-CHAT-ONE" not in whole
        assert "The secret of chat one" not in whole

    async def test_a_failed_reply_is_not_history(self, chat_world: ChatWorld, actor: Actor) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add([LlmUnavailable("down")], reply(LONG))
        await chat_world.ask(actor, chat_id, "Question one")
        await chat_world.ask(actor, chat_id, "Question two")

        call = chat_world.chat_llm.calls[1]
        assert [(m.role, m.content) for m in call[1:]] == [
            ("user", "Question one"),
            ("user", "Question two"),
        ]

    async def test_the_window_holds_twenty_messages_and_sixteen_thousand_characters(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        for index in range(12):  # 24 messages of history
            chat_world.chat_llm.add(reply(f"Answer {index}: " + "a" * 60))
            await chat_world.ask(actor, chat_id, f"Question {index}")
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id, "The last question")

        window = chat_world.chat_llm.calls[-1][1:]
        assert len(window) == CHAT_WINDOW_MESSAGES
        assert window[-1].content == "The last question"
        # 19 messages of history: the newest nine pairs and the artist's answer before them
        assert (window[0].role, window[0].content) == ("assistant", "Answer 2: " + "a" * 60)
        assert sum(len(m.content) for m in window) <= CHAT_WINDOW_BUDGET

    async def test_a_long_history_is_cut_by_characters(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        for index in range(5):
            chat_world.chat_llm.add(reply("B" * 3_900))
            await chat_world.ask(actor, chat_id, f"{index} " + "Q" * 3_900)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id, "Z" * 4_000)

        window = chat_world.chat_llm.calls[-1][1:]
        total = sum(len(m.content) for m in window)
        assert total <= CHAT_WINDOW_BUDGET
        assert window[-1].content == "Z" * 4_000
        assert len(window) < CHAT_WINDOW_MESSAGES


# ---------------------------------------------------------------------------------------------
# the planner never reads chat (invariant 7) and the export is untouched (invariant 17)
# ---------------------------------------------------------------------------------------------


class TestTheStudioIsUntouched:
    async def test_a_planner_run_never_contains_chat_text(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply("THE-ARTISTS-CHAT-ANSWER told at length to the user."))
        await chat_world.ask(actor, chat_id, "THE-USERS-CHAT-QUESTION")

        track_id = await chat_world.new_track(actor, "Planned")
        llm = FakeLlm(make_draft())
        await chat_world.submit(actor, track_id)
        await chat_world.drive(llm)

        sent = json.dumps([[m.content for m in call] for call in llm.calls])
        assert "THE-USERS-CHAT-QUESTION" not in sent
        assert "THE-ARTISTS-CHAT-ANSWER" not in sent
        runs = await chat_world.runs(track_id)
        assert "THE-USERS-CHAT-QUESTION" not in json.dumps(runs[0].context_snapshot)
        assert "THE-ARTISTS-CHAT-ANSWER" not in json.dumps(runs[0].context_snapshot)

    async def test_the_suno_export_is_byte_identical_with_and_without_chats(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        section = make_section(StudioSectionKind.VERSE)
        track_id = await chat_world.new_track(
            actor, "Exported", plan=make_plan(section, title="Exported", tags=["synth"])
        )
        row = await chat_world.track_row(track_id)
        assert row.current_plan_version_id is not None
        await chat_world.create_lyrics(
            actor,
            track_id,
            make_lyrics((section.id, ["one line", "another line"])),
            plan_version_id=row.current_plan_version_id,
        )

        async def export() -> bytes:
            async with chat_world.factory() as session:
                out = await ExportService(session).render_suno(track_id, actor.user_id, PRODUCT)
            return msgspec.json.encode(out)

        before = await export()
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        await chat_world.ask(actor, chat_id, "Say something about Exported")
        await chat_world.add_memory(actor, title="A memory made after")
        after = await export()

        assert before == after
        assert b"Say something" not in after


# ---------------------------------------------------------------------------------------------
# privacy: no content in any log record (invariant 15)
# ---------------------------------------------------------------------------------------------

QUESTION = "PRIVATE-QUESTION-TEXT-7731"
ANSWER = "PRIVATE-ANSWER-TEXT-9042 told in a long sentence."
MEMORY_TEXT = "PRIVATE-MEMORY-BODY-5518"
TITLE = "PRIVATE-CHAT-TITLE-2204"


class TestPrivacy:
    async def test_no_log_record_carries_message_title_or_memory_text(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        await chat_world.add_memory(actor, title="Mem title", body=MEMORY_TEXT)
        with capture_logs() as logs:
            chat_id = await chat_world.new_chat(actor, TITLE)
            chat_world.chat_llm.add(
                reply(ANSWER),  # complete
                [LlmUnavailable("down")],  # failed
                reply("⟦declined⟧"),  # refused
                [RuntimeError(QUESTION)],  # an unexpected exception whose message holds user text
            )
            await chat_world.ask(actor, chat_id, QUESTION)
            await chat_world.ask(actor, chat_id, QUESTION + " again")
            await chat_world.ask(actor, chat_id, QUESTION + " third")
            write = await chat_world.send(actor, chat_id, QUESTION + " fourth")
            await chat_world.settle()
            await chat_world.age(message_id=write.artist_message.id)
            await chat_world.sweeper().sweep()

        rendered = json.dumps(logs, default=str)
        for secret in (QUESTION, "PRIVATE-ANSWER-TEXT", MEMORY_TEXT, TITLE):
            assert secret not in rendered, secret
        events = {log["event"] for log in logs}
        assert {
            "studio.chat_created",
            "studio.chat_message_sent",
            "studio.chat_reply_finished",
            "studio.chat_task_crashed",
            "studio.chat_reply_swept",
        } <= events

    async def test_the_reply_log_has_the_documented_fields(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG, usage=(11, 7)))
        with capture_logs() as logs:
            _, artist = await chat_world.ask(actor, chat_id)

        (finished,) = [log for log in logs if log["event"] == "studio.chat_reply_finished"]
        assert finished["message_id"] == str(artist.id)
        assert finished["status"] == "complete"
        assert finished["error_code"] is None
        assert finished["model_key"] == "grok"
        assert (finished["prompt_tokens"], finished["completion_tokens"]) == (11, 7)
        assert finished["truncated"] is False
        assert finished["memory_count"] == 0
        assert isinstance(finished["duration_ms"], int)

    async def test_the_sent_log_has_ids_and_lengths_only(
        self, chat_world: ChatWorld, actor: Actor
    ) -> None:
        chat_id = await chat_world.new_chat(actor)
        chat_world.chat_llm.add(reply(LONG))
        with capture_logs() as logs:
            await chat_world.ask(actor, chat_id, QUESTION)
        (sent,) = [log for log in logs if log["event"] == "studio.chat_message_sent"]
        assert sent["content_length"] == len(QUESTION)
        assert QUESTION not in json.dumps(sent, default=str)
        assert uuid4().hex not in json.dumps(sent, default=str)
