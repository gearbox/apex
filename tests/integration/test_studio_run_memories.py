"""Artist memories in planner runs: pins at submit, selection at claim, the snapshot, privacy.

Invariants 2-5, 8-10, 13 and 17 of S1-P2b, driven through the service and the executor
(``world.submit`` / ``world.drive``, never sleeps). The executor opens its own sessions, so the
world commits real rows.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import httpx
import msgspec
import pytest
from structlog.testing import capture_logs

from src.api.schemas.studio import ArtistCreate
from src.llm.factory import LlmClientFactory
from src.studio.constraints import MEMORY_CONTEXT_BUDGET
from src.studio.errors import StudioDocumentInvalid
from src.studio.planner.context import MemorySnapshot
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.memories import rendered_size
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from src.studio.services.artist import ArtistService
from src.studio.services.mappers import run_out
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft, make_lyrics_draft, refusal_draft, section_lines
from tests.studio_lyrics_support import PLAN, S1, S2, S4, doc
from tests.studio_memory_support import make_memory
from tests.studio_runs_support import PRODUCT

if TYPE_CHECKING:
    from src.db.models.studio import StudioRun
    from tests.studio_runs_support import Actor, RunWorld

MARK = "ZXQ-MEMORY-SECRET"
FOUR = "la la la la"


async def _world(world: RunWorld, *, balance: int = 20) -> tuple[Actor, UUID]:
    """An actor with an artist, and a plan-less track."""
    actor = await world.add_actor(balance=balance, artist=True)
    return actor, await world.new_track(actor, "Night Drive")


async def _submit(world: RunWorld, actor: Actor, track_id: UUID, **fields: Any) -> UUID:
    write = await world.submit(actor, track_id, request=PlanRunRequest(brief="rain", **fields))
    return write.run.id


async def _issues(
    world: RunWorld, actor: Actor, track_id: UUID, request: Any
) -> list[tuple[str, str]]:
    """The issues of a submit that is refused - and proof that nothing was written or charged."""
    before = (
        await world.balance(actor),
        len(await world.ledger(actor)),
        await world.count_runs(actor),
    )
    with pytest.raises(StudioDocumentInvalid) as caught:
        await world.submit(actor, track_id, request=request)
    after = (
        await world.balance(actor),
        len(await world.ledger(actor)),
        await world.count_runs(actor),
    )
    assert after == before
    return [(issue.path, issue.code) for issue in caught.value.issues]


async def _cancel(world: RunWorld, actor: Actor, run_id: UUID) -> None:
    async with world.factory() as session:
        service = world.service(session)
        cancelled = await service.cancel(run_id, user_id=actor.user_id, product_id=PRODUCT)
        await session.commit()
    await service.publish(cancelled)


def _used(run: StudioRun) -> list[UUID]:
    assert run.context_snapshot is not None
    return [UUID(memory["id"]) for memory in run.context_snapshot["memories"]]


def _snapshot_memories(run: StudioRun) -> list[MemorySnapshot]:
    assert run.context_snapshot is not None
    return [msgspec.convert(m, MemorySnapshot) for m in run.context_snapshot["memories"]]


async def _other_artist(world: RunWorld, actor: Actor) -> UUID:
    async with world.factory() as session, session.begin():
        artist = await ArtistService(session).create(
            actor.user_id, PRODUCT, ArtistCreate(name=f"Other {uuid4().hex[:6]}")
        )
    return artist.id


# ---------------------------------------------------------------------------
# Pins at submit (P2b-1; invariants 2, 3, 4, 17)
# ---------------------------------------------------------------------------


class TestPinsAtSubmit:
    async def test_an_unknown_id_is_memory_not_found(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=(uuid4(),))
        )
        assert issues == [("memory_ids[0]", "memory_not_found")]

    async def test_another_artists_or_users_memory_is_indistinguishable_from_an_unknown_id(
        self, run_world: RunWorld
    ) -> None:
        """Ownership is not probed: the three failures give the same code at the same path."""
        actor, track_id = await _world(run_world)
        other_artist = await run_world.add_memory(
            actor, artist_id=await _other_artist(run_world, actor)
        )
        stranger = await run_world.add_actor(artist=True)
        strangers = await run_world.add_memory(stranger)

        results = [
            await _issues(run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=(i,)))
            for i in (uuid4(), other_artist, strangers)
        ]
        assert results == [[("memory_ids[0]", "memory_not_found")]] * 3

    async def test_an_archived_pin_is_refused(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        memory = await run_world.add_memory(actor)
        await run_world.set_memory(actor, memory, status="archived")
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=(memory,))
        )
        assert issues == [("memory_ids[0]", "memory_archived")]

    async def test_a_repeated_pin_is_refused_at_its_second_place(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        memory = await run_world.add_memory(actor)
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=(memory, memory))
        )
        assert issues == [("memory_ids[1]", "duplicate_memory")]

    async def test_a_project_without_an_artist_has_nothing_to_pin(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=(uuid4(),))
        )
        assert issues == [("memory_ids", "no_artist")]

    async def test_pins_that_alone_exceed_the_budget_are_refused_and_never_truncated(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        pins = tuple([await run_world.add_memory(actor, body="w" * 1900) for _ in range(7)])
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=pins)
        )
        assert issues == [("memory_ids", "memories_over_budget")]
        # six of them fit: the budget is measured as the prompt renders them, not as raw lengths
        write = await run_world.submit(
            actor, track_id, request=PlanRunRequest(brief="x", memory_ids=pins[:6])
        )
        assert write.changed is True

    async def test_every_problem_is_reported_at_once(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        good = await run_world.add_memory(actor)
        archived = await run_world.add_memory(actor)
        await run_world.set_memory(actor, archived, status="archived")
        issues = await _issues(
            run_world,
            actor,
            track_id,
            PlanRunRequest(brief="x", memory_ids=(uuid4(), archived, good, good)),
        )
        assert issues == [
            ("memory_ids[0]", "memory_not_found"),
            ("memory_ids[1]", "memory_archived"),
            ("memory_ids[3]", "duplicate_memory"),
        ]

    async def test_pins_are_accepted_for_every_kind_and_stay_in_the_request_not_in_the_inputs(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(artist=True)
        track_id = await run_world.new_track(actor, "Night Drive", PLAN)
        plan_id = (await run_world.track_row(track_id)).current_plan_version_id
        assert plan_id is not None
        await run_world.create_lyrics(
            actor, track_id, doc((S1, [FOUR]), (S2, [FOUR]), (S4, [FOUR])), plan_version_id=plan_id
        )
        pin = await run_world.add_memory(actor)

        requests = {
            "plan": PlanRunRequest(brief="rain", memory_ids=(pin,)),
            "lyrics": LyricsRunRequest(language="en", memory_ids=(pin,)),
            "revise_lyrics": ReviseLyricsRunRequest(
                instruction="warmer", targets=ReviseTargets(sections=(S2,)), memory_ids=(pin,)
            ),
        }
        for kind, request in requests.items():
            write = await run_world.submit(actor, track_id, request=request)
            run = await run_world.run(write.run.id)
            assert run.kind == kind
            assert run.request["memory_ids"] == [str(pin)]  # the request, as sent
            assert "memory_ids" not in (run.inputs or {})  # never in the inputs
            await _cancel(run_world, actor, run.id)  # one active run per track: free the track

    async def test_a_replay_is_the_same_run_even_after_the_pin_was_archived(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        pin = await run_world.add_memory(actor)
        request = PlanRunRequest(brief="rain", memory_ids=(pin,))
        first = await run_world.submit(actor, track_id, request=request, key="same-key")
        await run_world.set_memory(actor, pin, status="archived")
        again = await run_world.submit(actor, track_id, request=request, key="same-key")
        assert (again.changed, again.run.id) == (False, first.run.id)


# ---------------------------------------------------------------------------
# Selection at claim (P2-M2, P2b-3; invariants 3, 9, 13)
# ---------------------------------------------------------------------------


class TestSelectionAtClaim:
    async def test_pins_come_first_in_request_order_then_the_rest_by_salience_recency_and_id(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        add = run_world.add_memory
        low = await add(actor, salience=1)
        mid = await add(actor, salience=3)
        old_high = await add(actor, salience=5, occurred_on=date(2010, 1, 1))
        new_high = await add(actor, salience=5, occurred_on=date(2020, 1, 1))
        undated_a = await add(actor, salience=5)
        undated_b = await add(actor, salience=5)
        archived = await add(actor, salience=5)
        await run_world.set_memory(actor, archived, status="archived")
        run_id = await _submit(run_world, actor, track_id, memory_ids=(mid, low))

        await run_world.drive(FakeLlm(make_draft()))

        run = await run_world.run(run_id)
        assert _used(run) == [
            mid,
            low,
            new_high,
            old_high,
            undated_a,
            undated_b,
        ]  # (invariants 9, 3)
        assert archived not in _used(run)
        assert run_out(run).memory_ids_used == _used(run)  # invariant 13

    async def test_auto_fill_off_uses_only_the_pins(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        pin = await run_world.add_memory(actor, salience=1)
        await run_world.add_memory(actor, salience=5)
        run_id = await _submit(run_world, actor, track_id, memory_ids=(pin,), auto_fill=False)
        await run_world.drive(FakeLlm(make_draft()))
        assert _used(await run_world.run(run_id)) == [pin]

    async def test_a_run_without_memories_sends_no_memory_block(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        await run_world.add_memory(actor)
        run_id = await _submit(run_world, actor, track_id, auto_fill=False)
        llm = FakeLlm(make_draft())
        await run_world.drive(llm)

        run = await run_world.run(run_id)
        assert _used(run) == []
        assert run_out(run).memory_ids_used == []
        system, user = llm.calls[0]
        assert "artist_memories" not in system.content
        assert "artist_memories" not in user.content

    async def test_the_budget_holds_and_selection_stops_at_the_first_memory_that_does_not_fit(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        big = [await run_world.add_memory(actor, salience=5, body="b" * 1900) for _ in range(6)]
        too_big = await run_world.add_memory(actor, salience=4, body="c" * 1900)
        tiny = await run_world.add_memory(actor, salience=3, body="fits easily")
        run_id = await _submit(run_world, actor, track_id)

        await run_world.drive(FakeLlm(make_draft()))

        run = await run_world.run(run_id)
        used = _snapshot_memories(run)
        assert [m.id for m in used] == big  # seven do not fit; ...
        assert too_big not in _used(run)
        assert tiny not in _used(run)  # ... and the tiny one behind them is not taken instead
        assert rendered_size(used) <= MEMORY_CONTEXT_BUDGET

    async def test_a_project_without_an_artist_simply_gets_no_memories(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        run_id = await _submit(run_world, actor, track_id)
        await run_world.drive(FakeLlm(make_draft()))
        run = await run_world.run(run_id)
        assert (run.status, _used(run)) == ("succeeded", [])

    async def test_the_model_is_given_the_block_and_the_rule_and_the_run_records_the_version(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        await run_world.add_memory(actor, title="Rooftop night", body="We watched the trains.")
        run_id = await _submit(run_world, actor, track_id)
        llm = FakeLlm(make_draft())
        await run_world.drive(llm)

        system, user = llm.calls[0]
        assert system.content.count("- <artist_memories> are episodes") == 1
        assert user.content.count("<artist_memories>") == 1
        assert "- m1: Rooftop night (salience 3)\n  We watched the trains." in user.content
        run = await run_world.run(run_id)
        assert run.prompt_version == "plan-v2"
        assert run.context_snapshot is not None
        assert run.context_snapshot["snapshot_version"] == 3

    async def test_a_lyrics_run_uses_memories_too(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor(artist=True)
        track_id = await run_world.new_track(actor, "Night Drive", PLAN)
        pin = await run_world.add_memory(actor, title="Platform seven")
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(language="en", memory_ids=(pin,))
        )
        llm = FakeLlm(
            make_lyrics_draft(
                section_lines("s1", FOUR, FOUR),
                section_lines("s2", FOUR),
                section_lines("s4", FOUR),
            )
        )
        await run_world.drive(llm)

        run = await run_world.run(write.run.id)
        assert (run.status, run.prompt_version) == ("succeeded", "lyrics-v2")
        assert _used(run) == [pin]
        assert "- m1: Platform seven (salience 3)" in llm.calls[0][1].content


# ---------------------------------------------------------------------------
# A pin that is gone by the time the run starts (P2b-2; invariant 10)
# ---------------------------------------------------------------------------


class TestPinsGoneBeforeClaim:
    async def test_a_deleted_or_archived_pin_is_skipped_and_reported_with_a_count(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        keep = await run_world.add_memory(actor)
        deleted = await run_world.add_memory(actor)
        archived = await run_world.add_memory(actor)
        run_id = await _submit(run_world, actor, track_id, memory_ids=(deleted, keep, archived))
        await run_world.delete_memory(actor, deleted)  # allowed at any time (P2-M5)
        await run_world.set_memory(actor, archived, status="archived")

        await run_world.drive(FakeLlm(make_draft()))

        run = await run_world.run(run_id)
        assert run.status == "succeeded"  # the user's own action does not fail their run
        assert run.warnings[0] == {"code": "pinned_memory_unavailable", "count": 2}
        assert _used(run) == [keep]
        assert run.context_snapshot is not None
        assert run.context_snapshot["unavailable_memory_ids"] == [str(deleted), str(archived)]
        assert run.result_plan_version_id is not None  # a paid result is saved
        assert await run_world.refunds_and_debits(actor) == (1, 0)
        warning = run_out(run).warnings[0]
        assert (warning.code, warning.count) == ("pinned_memory_unavailable", 2)

    async def test_every_pin_gone_still_succeeds(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        only = await run_world.add_memory(actor)
        run_id = await _submit(run_world, actor, track_id, memory_ids=(only,), auto_fill=False)
        await run_world.delete_memory(actor, only)
        await run_world.drive(FakeLlm(make_draft()))
        run = await run_world.run(run_id)
        assert (run.status, _used(run)) == ("succeeded", [])
        assert run.warnings == [{"code": "pinned_memory_unavailable", "count": 1}]

    async def test_it_is_generic_for_every_kind(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor(artist=True)
        track_id = await run_world.new_track(actor, "Night Drive", PLAN)
        pin = await run_world.add_memory(actor)
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(language="en", memory_ids=(pin,))
        )
        await run_world.delete_memory(actor, pin)
        await run_world.drive(
            FakeLlm(
                make_lyrics_draft(
                    section_lines("s1", FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR)
                )
            )
        )
        run = await run_world.run(write.run.id)
        assert run.status == "succeeded"
        assert run.warnings[0] == {"code": "pinned_memory_unavailable", "count": 1}

    async def test_a_pin_that_no_longer_fits_the_budget_is_skipped_not_sent(
        self, run_world: RunWorld
    ) -> None:
        """Edited after submit so that the pins together pass the budget: the rendered block never
        does (invariant 9), and what was left out is reported rather than the run failed."""
        actor, track_id = await _world(run_world)
        pins = tuple([await run_world.add_memory(actor, body="w" * 1900) for _ in range(6)])
        run_id = await _submit(run_world, actor, track_id, memory_ids=pins, auto_fill=False)
        await run_world.set_memory(
            actor,
            pins[2],
            themes=[f"{n:02d}" + "t" * 58 for n in range(12)],
            imagery=[f"{n:02d}" + "i" * 118 for n in range(20)],
        )

        await run_world.drive(FakeLlm(make_draft()))

        run = await run_world.run(run_id)
        used = _snapshot_memories(run)
        assert run.status == "succeeded"
        assert rendered_size(used) <= MEMORY_CONTEXT_BUDGET
        # rendered sizes: ~1.9k, ~1.9k, ~5.0k (grown), ~1.9k fit (10.8k); the fifth would pass 12k
        assert [m.id for m in used] == list(pins[:4])
        assert run.warnings[0] == {"code": "pinned_memory_unavailable", "count": 2}


# ---------------------------------------------------------------------------
# The snapshot is the record (P2-M4, P2-M5; invariant 5)
# ---------------------------------------------------------------------------


class TestSnapshot:
    async def test_it_holds_the_full_text_and_survives_edits_and_deletes(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        memory = await run_world.add_memory(
            actor,
            title="Rooftop night",
            body="We watched the trains.",
            themes=["loss"],
            imagery=["wet asphalt"],
            occurred_on=date(2019, 6, 4),
            salience=4,
        )
        run_id = await _submit(run_world, actor, track_id)
        executor = run_world.executor(FakeLlm(make_draft()))
        (claimed,) = await executor.claim(1)

        before = (await run_world.run(run_id)).context_snapshot
        assert before is not None
        (stored,) = before["memories"]
        assert stored == {
            "id": str(memory),
            "title": "Rooftop night",
            "body": "We watched the trains.",
            "themes": ["loss"],
            "imagery": ["wet asphalt"],
            "occurred_on": "2019-06-04",
            "salience": 4,
            "updated_at": stored["updated_at"],
        }

        await run_world.set_memory(actor, memory, title="Edited", body="Rewritten.")
        assert (await run_world.run(run_id)).context_snapshot == before
        await run_world.delete_memory(actor, memory)
        assert (await run_world.run(run_id)).context_snapshot == before

        await executor.execute(claimed)  # the run itself finishes untouched by what happened
        run = await run_world.run(run_id)
        assert run.status == "succeeded"
        assert run.context_snapshot == before
        assert run_out(run).memory_ids_used == [memory]


# ---------------------------------------------------------------------------
# A memory can be the cause of a refusal (P2b-6, P2-M8)
# ---------------------------------------------------------------------------


class TestPolicy:
    async def test_a_refusal_the_model_gives_because_of_a_memory_fails_the_run_refunded(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id = await _world(run_world)
        await run_world.add_memory(actor, body="something these rules forbid")
        run_id = await _submit(run_world, actor, track_id)

        await run_world.drive(FakeLlm(refusal_draft(DraftRefusal.CONTENT_POLICY)))

        run = await run_world.run(run_id)
        assert (run.status, run.error_code) == ("failed", "content_policy_refused")
        assert len(_used(run)) == 1  # the record says what the model was given
        assert await run_world.refunds_and_debits(actor) == (1, 1)
        assert await run_world.versions(track_id) == []


class TestARefusalNamesItsMemories:
    """R5 (P2b-10): the model names the memories that caused a content-policy refusal by key."""

    async def _refused(
        self,
        run_world: RunWorld,
        reason: DraftRefusal,
        keys: list[str],
        *,
        memories: int = 2,
    ) -> tuple[Actor, StudioRun, list[UUID]]:
        """A run with ``memories`` pinned memories (so ``m1`` is the first) whose model refuses."""
        actor, track_id = await _world(run_world)
        pins = [await run_world.add_memory(actor, title=f"Entry {n}") for n in range(memories)]
        run_id = await _submit(run_world, actor, track_id, memory_ids=tuple(pins))
        await run_world.drive(FakeLlm(refusal_draft(reason, keys)))
        return actor, await run_world.run(run_id), pins

    async def test_a_refusal_names_the_memories_that_caused_it(self, run_world: RunWorld) -> None:
        actor, run, pins = await self._refused(run_world, DraftRefusal.CONTENT_POLICY, ["m2"])

        assert (run.status, run.error_code) == ("failed", "content_policy_refused")
        assert run.error_detail == {"memory_ids": [str(pins[1])]}  # the 2nd memory, by id
        assert run_out(run).error_detail == {"memory_ids": [str(pins[1])]}
        assert _used(run) == pins  # the keys count up through what the model was given
        assert await run_world.refunds_and_debits(actor) == (1, 1)  # and the run is refunded

    async def test_the_ids_are_listed_in_prompt_order(self, run_world: RunWorld) -> None:
        _, run, pins = await self._refused(
            run_world, DraftRefusal.CONTENT_POLICY, ["m3", "m1"], memories=3
        )

        assert run.error_detail == {"memory_ids": [str(pins[0]), str(pins[2])]}

    async def test_unknown_or_repeated_memory_keys_are_dropped(self, run_world: RunWorld) -> None:
        _, run, pins = await self._refused(
            run_world, DraftRefusal.CONTENT_POLICY, ["m9", "x", "m1", "m1"]
        )

        assert run.error_detail == {"memory_ids": [str(pins[0])]}

    async def test_a_refusal_without_memory_keys_has_no_detail(self, run_world: RunWorld) -> None:
        _, run, _ = await self._refused(run_world, DraftRefusal.CONTENT_POLICY, [])

        assert (run.status, run.error_code) == ("failed", "content_policy_refused")
        assert run.error_detail is None
        assert run_out(run).error_detail is None

    async def test_keys_that_name_nothing_the_model_was_shown_leave_no_detail(
        self, run_world: RunWorld
    ) -> None:
        _, run, _ = await self._refused(run_world, DraftRefusal.CONTENT_POLICY, ["m7", "zz"])

        assert run.error_detail is None

    async def test_brief_unusable_ignores_any_keys(self, run_world: RunWorld) -> None:
        _, run, _ = await self._refused(run_world, DraftRefusal.BRIEF_UNUSABLE, ["m1", "m2"])

        assert (run.status, run.error_code) == ("failed", "brief_unusable")
        assert run.error_detail is None

    async def test_a_memory_deleted_after_the_refusal_is_still_named_by_id(
        self, run_world: RunWorld
    ) -> None:
        """Best-effort: the id outlives the memory (the UI finds it gone with a 404)."""
        actor, run, pins = await self._refused(run_world, DraftRefusal.CONTENT_POLICY, ["m1"])

        await run_world.delete_memory(actor, pins[0])

        refetched = await run_world.run(run.id)
        assert refetched.error_detail == {"memory_ids": [str(pins[0])]}

    async def test_the_model_is_shown_every_memory_under_its_key(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        pins = [await run_world.add_memory(actor, title=f"Entry {n}") for n in range(2)]
        await _submit(run_world, actor, track_id, memory_ids=tuple(pins))
        llm = FakeLlm(make_draft())

        await run_world.drive(llm)

        system, user = llm.calls[0]
        assert "- m1: Entry 0 (salience 3)" in user.content
        assert "- m2: Entry 1 (salience 3)" in user.content
        assert 'list that memory\'s key in "refusal_memories"' in system.content

    async def test_a_run_without_memories_is_not_asked_for_keys(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        await run_world.add_memory(actor)
        await _submit(run_world, actor, track_id, auto_fill=False)
        llm = FakeLlm(make_draft())

        await run_world.drive(llm)

        system, user = llm.calls[0]
        assert "refusal_memories" not in system.content
        assert "m1" not in user.content


class TestTheSizeAtSubmitIsTheSizeAtClaim:
    async def test_pins_that_exactly_fill_the_budget_are_accepted_and_all_selected(
        self, run_world: RunWorld
    ) -> None:
        """The key is part of what the budget counts, at submit and at claim alike."""
        actor, track_id = await _world(run_world)
        sizes = [1900] * 6
        # tune the last body so the block is exactly MEMORY_CONTEXT_BUDGET characters
        last = 1
        while True:
            candidates = [
                make_memory(n + 1, title=f"T{n}", body="b" * size)
                for n, size in enumerate([*sizes, last])
            ]
            total = rendered_size(candidates)
            if total >= MEMORY_CONTEXT_BUDGET or last >= 2000:
                break
            last += 1
        assert total == MEMORY_CONTEXT_BUDGET, total  # the search can land on it exactly
        pins = [
            await run_world.add_memory(actor, title=f"T{n}", body="b" * size)
            for n, size in enumerate([*sizes, last])
        ]
        run_id = await _submit(run_world, actor, track_id, memory_ids=tuple(pins), auto_fill=False)

        await run_world.drive(FakeLlm(make_draft()))

        run = await run_world.run(run_id)
        assert run.status == "succeeded"
        assert _used(run) == pins  # none was skipped at claim
        assert rendered_size(_snapshot_memories(run)) == MEMORY_CONTEXT_BUDGET
        assert run_out(run).warnings == []  # no pinned_memory_unavailable

    async def test_one_character_more_is_refused_at_submit(self, run_world: RunWorld) -> None:
        actor, track_id = await _world(run_world)
        pins = [await run_world.add_memory(actor, body="b" * 1900) for _ in range(6)]
        pins.append(await run_world.add_memory(actor, body="b" * 2000))
        issues = await _issues(
            run_world, actor, track_id, PlanRunRequest(brief="x", memory_ids=tuple(pins))
        )
        assert issues == [("memory_ids", "memories_over_budget")]


# ---------------------------------------------------------------------------
# Privacy (P2-M9; invariant 8)
# ---------------------------------------------------------------------------


class TestPrivacy:
    @pytest.fixture(autouse=True)
    def _capture_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)

    async def test_memory_text_reaches_the_provider_and_no_log_record(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world(run_world)
        with capture_logs() as logs:
            memory = await run_world.add_memory(
                actor,
                title=f"{MARK}-title",
                body=f"{MARK}-body",
                themes=[f"{MARK}-theme"],
                imagery=[f"{MARK}-image"],
            )
            await run_world.set_memory(actor, memory, salience=5, title=f"{MARK}-renamed")
            run_id = await _submit(run_world, actor, track_id, memory_ids=(memory,))
            sent: list[dict[str, Any]] = []

            def handler(request: httpx.Request) -> httpx.Response:
                sent.append(json.loads(request.content))
                reply = msgspec.json.encode(make_draft()).decode()
                return httpx.Response(
                    200,
                    json={
                        "model": "served",
                        "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
                    },
                )

            executor = run_world.executor(FakeLlm())
            executor._llm_factory = LlmClientFactory(
                run_world.registry, httpx.AsyncClient(transport=httpx.MockTransport(handler))
            )
            for claimed in await executor.claim(1):
                await executor.execute(claimed)
            await run_world.set_memory(actor, memory, status="archived")
            await run_world.delete_memory(actor, memory)

        # the provider got it ...
        assert len(sent) == 1
        assert f"{MARK}-body" in json.dumps(sent[0])
        # ... and nothing else did: no log record, no RunOut, no error detail
        events = {entry["event"] for entry in logs}
        assert {
            "studio.memory_created",
            "studio.memory_updated",
            "studio.memory_archived",
            "studio.memory_deleted",
            "studio.run.finished",
        } <= events
        for entry in logs:
            assert MARK.lower() not in repr(entry).lower(), entry
        records = "\n".join(record.getMessage() for record in caplog.records)
        assert MARK not in records
        run = await run_world.run(run_id)
        assert run.status == "succeeded"
        assert MARK not in msgspec.json.encode(run_out(run)).decode()
        assert MARK not in json.dumps(run.error_detail)

    async def test_the_memory_service_logs_field_names_not_values(
        self, run_world: RunWorld
    ) -> None:
        actor, _ = await _world(run_world)
        memory = await run_world.add_memory(actor, title=f"{MARK}-a")
        with capture_logs() as logs:
            await run_world.set_memory(actor, memory, title=f"{MARK}-b", salience=2)
        (entry,) = [e for e in logs if e["event"] == "studio.memory_updated"]
        assert entry["fields"] == ["title", "salience"]
        assert set(entry) <= {"event", "log_level", "memory_id", "artist_id", "user_id", "fields"}

    async def test_a_refusal_that_names_memories_carries_and_logs_ids_only(
        self, run_world: RunWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, track_id = await _world(run_world)
        with capture_logs() as logs:
            memory = await run_world.add_memory(
                actor,
                title=f"{MARK}-title",
                body=f"{MARK}-body",
                themes=[f"{MARK}-theme"],
                imagery=[f"{MARK}-image"],
            )
            run_id = await _submit(run_world, actor, track_id, memory_ids=(memory,))
            await run_world.drive(FakeLlm(refusal_draft(DraftRefusal.CONTENT_POLICY, ["m1"])))

        run = await run_world.run(run_id)
        assert run.error_detail == {"memory_ids": [str(memory)]}  # the id, and nothing else
        shown = msgspec.json.encode(run_out(run)).decode()
        assert str(memory) in shown
        assert MARK not in shown
        assert MARK not in json.dumps(run.error_detail)
        for entry in logs:
            assert MARK.lower() not in repr(entry).lower(), entry
        assert MARK not in "\n".join(record.getMessage() for record in caplog.records)
