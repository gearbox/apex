"""``lyrics`` runs against PostgreSQL: inputs at submit, lineage, the delete guard, the budget.

Invariants 3-7, 12, 13, 17 (the run is driven through the service and the executor, never with
sleeps). The executor opens its own sessions, so the world commits real rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from src.llm.port import LlmOutputInvalid
from src.studio.documents import SyllableRange
from src.studio.errors import (
    IdempotencyConflict,
    PlanRequired,
    PlanVersionNotFound,
    StudioDocumentInvalid,
    VersionHasChildren,
    VersionInUseByRun,
    VersionIsLyricsBase,
)
from src.studio.planner.drafts import DraftRefusal
from src.studio.run_requests import LyricsRunRequest, PlanRunRequest
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    refusal_lyrics_draft,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S4, doc
from tests.studio_runs_support import PRODUCT
from tests.studio_support import make_plan, make_section

if TYPE_CHECKING:
    from uuid import UUID

    from src.db.models.studio import StudioRun
    from tests.studio_runs_support import Actor, RunWorld

FOUR = "la la la la"
LONG = "la la la la la la la la la la"


def _good() -> Any:
    return make_lyrics_draft(
        section_lines("s1", FOUR, FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR, FOUR)
    )


async def _track(world: RunWorld, actor: Actor, plan: Any = PLAN) -> tuple[UUID, UUID]:
    """A track whose current plan is ``plan``; returns (track id, plan version id)."""
    track_id = await world.new_track(actor, "Night Drive", plan)
    row = await world.track_row(track_id)
    assert row.current_plan_version_id is not None
    return track_id, row.current_plan_version_id


async def _lyrics_runs(world: RunWorld, track_id: UUID) -> list[StudioRun]:
    return [run for run in await world.runs(track_id) if run.kind == "lyrics"]


async def _refused(world: RunWorld, actor: Actor, track_id: UUID, request: Any) -> Any:
    """Submit ``request`` and expect it to be refused before anything is written."""
    before = await world.balance(actor)
    ledger = len(await world.ledger(actor))
    runs = await world.count_runs(actor)
    with pytest.raises((StudioDocumentInvalid, PlanRequired, PlanVersionNotFound)) as caught:
        await world.submit(actor, track_id, request=request)
    assert await world.balance(actor) == before
    assert len(await world.ledger(actor)) == ledger
    assert await world.count_runs(actor) == runs
    return caught.value


# -- inputs are resolved at submit (P2a-40) ------------------------------------------------------


class TestInputs:
    async def test_the_current_plan_and_the_language_are_stored_on_the_run(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)

        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))

        run = await run_world.run(write.run.id)
        assert (run.kind, run.status) == ("lyrics", "queued")
        assert run.inputs == {
            "plan_version_id": str(plan_id),
            "language": "en",
            "lyrics_version_id": None,
        }
        assert run.request["kind"] == "lyrics"
        assert run.request["plan_version_id"] is None  # the request stays as sent: no plan named

    async def test_a_named_plan_is_used_even_when_it_is_not_current(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, first = await _track(run_world, actor)
        await run_world.create_plan(actor, track_id, make_plan(make_section(), title="Other"))

        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        assert (await run_world.run(write.run.id)).inputs["plan_version_id"] == str(first)  # type: ignore[index]

    @pytest.mark.parametrize("which", ["random", "lyrics_version", "another_tracks_plan"])
    async def test_a_plan_that_is_not_a_plan_of_this_track_is_404(
        self, run_world: RunWorld, which: str
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)
        other_track, other_plan = await _track(run_world, actor)
        lyrics_id = await run_world.create_lyrics(
            actor, track_id, doc((S1, [FOUR])), plan_version_id=plan_id
        )
        wrong = {"random": uuid4(), "lyrics_version": lyrics_id, "another_tracks_plan": other_plan}[
            which
        ]
        assert other_track != track_id

        error = await _refused(
            run_world, actor, track_id, LyricsRunRequest(plan_version_id=wrong, language="en")
        )
        assert isinstance(error, PlanVersionNotFound)

    async def test_without_a_current_plan_it_is_409(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor)
        error = await _refused(run_world, actor, track_id, LyricsRunRequest(language="en"))
        assert isinstance(error, PlanRequired)

    @pytest.mark.parametrize(
        "plan",
        [
            make_plan(make_section(), title="Flagged", instrumental=True),
            make_plan(
                make_section(instrumental=True), make_section(instrumental=True), title="Silent"
            ),
            make_plan(title="Empty", empty=True),
        ],
        ids=["track-instrumental", "no-vocal-section", "no-sections"],
    )
    async def test_a_plan_with_nothing_to_sing_is_422(self, run_world: RunWorld, plan: Any) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor, plan)
        error = await _refused(run_world, actor, track_id, LyricsRunRequest(language="en"))
        assert isinstance(error, StudioDocumentInvalid)
        assert [(i.path, i.code) for i in error.issues] == [
            ("plan_version_id", "plan_has_no_vocal_sections")
        ]

    async def test_the_language_comes_from_the_request_then_the_lyrics_then_the_artist(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(artist=True)  # the artist's default is "en"
        track_id, plan_id = await _track(run_world, actor)

        async def language(request: LyricsRunRequest) -> str:
            write = await run_world.submit(actor, track_id, request=request)
            run = await run_world.run(write.run.id)
            async with run_world.factory() as session:
                await run_world.service(session).cancel(
                    run.id, user_id=actor.user_id, product_id=PRODUCT
                )
                await session.commit()
            assert run.inputs is not None
            return str(run.inputs["language"])

        assert await language(LyricsRunRequest()) == "en"  # the artist
        await run_world.create_lyrics(
            actor, track_id, doc((S1, [FOUR]), language="ru"), plan_version_id=plan_id
        )
        assert await language(LyricsRunRequest()) == "ru"  # the current lyrics beat the artist
        assert await language(LyricsRunRequest(language="uk")) == "uk"  # the request beats both

    async def test_with_no_language_anywhere_it_is_422(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()  # no artist
        track_id, _ = await _track(run_world, actor)
        error = await _refused(run_world, actor, track_id, LyricsRunRequest())
        assert isinstance(error, StudioDocumentInvalid)
        assert [(i.path, i.code) for i in error.issues] == [("language", "language_required")]

    async def test_both_problems_are_reported_together(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor, make_plan(title="Empty", empty=True))
        error = await _refused(run_world, actor, track_id, LyricsRunRequest())
        assert isinstance(error, StudioDocumentInvalid)
        assert {(i.path, i.code) for i in error.issues} == {
            ("plan_version_id", "plan_has_no_vocal_sections"),
            ("language", "language_required"),
        }

    async def test_a_bad_body_is_422_before_anything_is_read(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        error = await _refused(
            run_world,
            actor,
            track_id,
            LyricsRunRequest(language="en", line_syllables=SyllableRange(min=9, max=2)),
        )
        assert isinstance(error, StudioDocumentInvalid)
        assert [i.code for i in error.issues] == ["syllable_range_inverted"]

    async def test_inputs_are_fixed_when_the_run_is_accepted_invariant_3(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, first = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        second = await run_world.create_plan(actor, track_id, make_plan(make_section(), title="B"))

        await run_world.drive(FakeLlm(_good()))

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded", run.error_code
        assert (await run_world.track_row(track_id)).current_plan_version_id == second
        (version,) = [v for v in await run_world.versions(track_id) if v.kind == "lyrics"]
        assert version.based_on_plan_version_id == first  # the plan the run was accepted with

    async def test_a_replay_is_the_same_run_after_the_pointers_moved_invariant_4(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, first = await _track(run_world, actor)
        request = LyricsRunRequest(language="en")
        original = await run_world.submit(actor, track_id, request=request, key="same")
        await run_world.create_plan(actor, track_id, make_plan(make_section(), title="B"))

        replay = await run_world.submit(actor, track_id, request=request, key="same")

        assert (replay.run.id, replay.changed) == (original.run.id, False)
        assert (await run_world.run(original.run.id)).inputs["plan_version_id"] == str(first)  # type: ignore[index]
        assert await run_world.count_runs(actor) == 1

    async def test_the_same_key_with_another_body_still_conflicts(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"), key="k")
        with pytest.raises(IdempotencyConflict):
            await run_world.submit(
                actor, track_id, request=LyricsRunRequest(language="ru"), key="k"
            )

    async def test_the_request_hash_ignores_the_resolved_inputs(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        explicit = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(language="en"), key="a"
        )
        async with run_world.factory() as session:
            await run_world.service(session).cancel(
                explicit.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        plan = await run_world.run(explicit.run.id)
        # a plan run with the same words hashes differently: the kind is part of what was sent
        other = await run_world.submit(actor, track_id, request=PlanRunRequest(brief="x"), key="b")
        assert plan.request_hash != (await run_world.run(other.run.id)).request_hash


# -- running ---------------------------------------------------------------------------------


class TestRunning:
    async def test_a_run_creates_exactly_one_lyrics_version_invariant_7(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)
        current = await run_world.create_lyrics(
            actor, track_id, doc((S1, ["old"]), language="en"), plan_version_id=plan_id
        )
        before = await run_world.track_row(track_id)
        write = await run_world.submit(
            actor,
            track_id,
            request=LyricsRunRequest(language="en", line_syllables=SyllableRange(min=4, max=6)),
        )

        await run_world.drive(FakeLlm(_good()))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code, run.warnings) == ("succeeded", None, [])
        assert (run.llm_calls, run.stage) == (1, None)
        assert run.result_plan_version_id is None
        versions = [v for v in await run_world.versions(track_id) if v.kind == "lyrics"]
        assert len(versions) == 2
        result = next(v for v in versions if v.id == run.result_lyrics_version_id)
        assert (result.origin, result.based_on_plan_version_id) == ("llm", plan_id)
        assert result.parent_version_id == current  # the lyrics current at claim
        assert result.content["language"] == "en"
        assert result.content["line_syllables"] == {"min": 4, "max": 6}
        after = await run_world.track_row(track_id)
        assert (after.current_plan_version_id, after.current_lyrics_version_id) == (
            before.current_plan_version_id,
            before.current_lyrics_version_id,
        )  # a run never moves a pointer
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            9,
            (1, 0),
        )

    async def test_with_no_current_lyrics_the_result_is_a_root(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        await run_world.drive(FakeLlm(_good()))
        (version,) = [v for v in await run_world.versions(track_id) if v.kind == "lyrics"]
        assert version.parent_version_id is None

    async def test_the_snapshot_records_ids_and_hashes_not_documents(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor(artist=True)
        track_id, plan_id = await _track(run_world, actor)
        other = await run_world.create_plan(actor, track_id, make_plan(make_section(), title="B"))
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=plan_id, brief="BRIEF-WORDS")
        )
        await run_world.drive(FakeLlm(_good()))

        run = await run_world.run(write.run.id)
        snapshot = run.context_snapshot
        assert snapshot is not None
        assert (snapshot["snapshot_version"], snapshot["language"]) == (3, "en")
        assert snapshot["input_plan_version_id"] == str(plan_id)
        assert snapshot["input_plan_content_hash"] is not None
        assert snapshot["input_lyrics_version_id"] is None
        assert snapshot["plan_version_id"] == str(other)  # the current pair at claim
        assert run.prompt_version == "lyrics-v2"
        assert run.model_id == "model-a"
        text = str(snapshot)
        assert "BRIEF-WORDS" not in text  # the brief lives on the request only
        assert "Neon Rain" not in text  # the plan's own text is not in the snapshot

    async def test_the_stages_are_published_in_order(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        bad = make_lyrics_draft(
            section_lines("s1", "oh [no]", FOUR),
            section_lines("s2", FOUR),
            section_lines("s4", FOUR),
        )
        await run_world.drive(FakeLlm(bad, make_lyrics_patch(make_line_patch("s1", 1, FOUR))))
        stages = [(e.status, e.stage) for e in run_world.bus.run_events]
        assert stages == [
            ("queued", None),
            ("running", "context"),
            ("running", "drafting"),
            ("running", "validating"),
            ("running", "revising"),
            ("running", "validating"),
            ("running", "saving"),
            ("succeeded", None),
        ]

    async def test_a_result_equal_to_its_parent_is_the_parent(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)
        same = doc((S1, [FOUR, FOUR]), (S2, [FOUR]), (S4, [FOUR, FOUR]))
        current = await run_world.create_lyrics(actor, track_id, same, plan_version_id=plan_id)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))

        await run_world.drive(FakeLlm(_good()))

        run = await run_world.run(write.run.id)
        assert (run.status, run.result_lyrics_version_id) == ("succeeded", current)
        assert len([v for v in await run_world.versions(track_id) if v.kind == "lyrics"]) == 1

    async def test_if_the_parent_vanishes_the_result_goes_under_the_current_lyrics(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)
        old = await run_world.create_lyrics(
            actor, track_id, doc((S1, ["old"])), plan_version_id=plan_id
        )
        newer: list[UUID] = []

        async def switch_and_delete() -> None:
            newer.append(
                await run_world.create_lyrics(
                    actor, track_id, doc((S1, ["newer"])), plan_version_id=plan_id
                )
            )
            await run_world.delete_version(actor, old)

        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        await run_world.drive(FakeLlm(_good(), before_call=switch_and_delete))

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded", run.error_code
        assert run.warnings == [{"code": "parent_version_replaced"}]
        result = next(
            v for v in await run_world.versions(track_id) if v.id == run.result_lyrics_version_id
        )
        assert result.parent_version_id == newer[0]
        assert result.based_on_plan_version_id == plan_id

    async def test_if_the_parent_vanishes_and_nothing_is_current_the_result_is_a_root(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id = await _track(run_world, actor)
        old = await run_world.create_lyrics(
            actor, track_id, doc((S1, ["old"])), plan_version_id=plan_id
        )

        async def detach_and_delete() -> None:
            await run_world.set_current(actor, track_id, lyrics_version_id=None)
            await run_world.delete_version(actor, old)

        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        await run_world.drive(FakeLlm(_good(), before_call=detach_and_delete))

        run = await run_world.run(write.run.id)
        assert run.warnings == [{"code": "parent_version_replaced"}]
        result = next(
            v for v in await run_world.versions(track_id) if v.id == run.result_lyrics_version_id
        )
        assert result.parent_version_id is None


# -- the delete guard (P2a-41, invariant 6) -----------------------------------------------------


class TestDeleteGuard:
    async def _two_plans(self, world: RunWorld, actor: Actor) -> tuple[UUID, UUID, UUID]:
        """(track, the plan a run will use, another plan that is current)."""
        track_id, first = await _track(world, actor)
        second = await world.create_plan(actor, track_id, make_plan(make_section(), title="B"))
        return track_id, first, second

    async def test_an_input_plan_cannot_be_deleted_while_the_run_is_queued(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        with pytest.raises(VersionInUseByRun):
            await run_world.delete_version(actor, first)

    async def test_nor_while_it_is_running(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        outcomes: list[str] = []

        async def try_to_delete() -> None:
            assert (await run_world.run(write.run.id)).status == "running"
            try:
                await run_world.delete_version(actor, first)
            except VersionInUseByRun:
                outcomes.append("blocked")

        await run_world.drive(
            FakeLlm(refusal_lyrics_draft(DraftRefusal.CONTENT_POLICY), before_call=try_to_delete)
        )
        assert outcomes == ["blocked"]

    async def test_it_can_be_deleted_once_the_run_is_terminal(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        await run_world.drive(FakeLlm(refusal_lyrics_draft(DraftRefusal.CONTENT_POLICY)))
        assert (await run_world.run(write.run.id)).status == "failed"

        await run_world.delete_version(actor, first)  # nothing holds it any more
        remaining = {v.id for v in await run_world.versions(track_id)}
        assert first not in remaining

    async def test_a_cancelled_run_releases_its_input(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        async with run_world.factory() as session:
            await run_world.service(session).cancel(
                write.run.id, user_id=actor.user_id, product_id=PRODUCT
            )
            await session.commit()
        await run_world.delete_version(actor, first)

    async def test_a_version_the_run_does_not_use_is_deletable_meanwhile(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, first, second = await self._two_plans(run_world, actor)
        await run_world.set_current(actor, track_id, plan_version_id=first)
        await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        await run_world.delete_version(actor, second)  # not the run's input

    async def test_the_guard_comes_after_the_older_refusals(self, run_world: RunWorld) -> None:
        """A lyrics base or a parent is refused for its own reason first."""
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        await run_world.create_lyrics(
            actor, track_id, doc((S1, [FOUR])), plan_version_id=first, make_current=False
        )
        await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        with pytest.raises((VersionIsLyricsBase, VersionHasChildren)):
            await run_world.delete_version(actor, first)

    async def test_a_plan_run_does_not_guard_anything(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, first, _ = await self._two_plans(run_world, actor)
        await run_world.submit(actor, track_id, request=PlanRunRequest(brief="x"))
        assert (await run_world.run((await run_world.runs(track_id))[0].id)).inputs is None
        await run_world.delete_version(actor, first)


# -- the budget and the failure modes -----------------------------------------------------------


class TestOutcomes:
    async def test_hard_issues_after_the_cap_fail_refund_and_use_plan_paths_invariant_13(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        stubborn = make_lyrics_draft(
            section_lines("s1", FOUR),
            section_lines("s2", FOUR, "oh [no] no"),
            section_lines("s4", FOUR),
        )
        keep = make_lyrics_patch(make_line_patch("s2", 2, "still [bad]"))
        await run_world.drive(FakeLlm(stubborn, keep, keep, keep))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "lyrics_invalid")
        assert run.error_detail == {
            "issues": [{"path": "sections[1].lines[1].text", "code": "reserved_characters"}]
        }
        assert run.llm_calls == 4
        assert [v for v in await run_world.versions(track_id) if v.kind == "lyrics"] == []
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            10,
            (1, 1),
        )
        assert "oh [no] no" not in str(run.error_detail)

    async def test_lyrics_with_a_nul_fail_as_lyrics_invalid_and_refund(
        self, run_world: RunWorld
    ) -> None:
        """N3: PostgreSQL would refuse the INSERT; the walker refuses the reply first."""
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        stubborn = make_lyrics_draft(
            section_lines("s1", FOUR),
            section_lines("s2", FOUR, "SECRET-LINE\x00tail"),
            section_lines("s4", FOUR),
        )
        keep = make_lyrics_patch(make_line_patch("s2", 2, "still\x00here"))
        await run_world.drive(FakeLlm(stubborn, keep, keep, keep))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "lyrics_invalid")
        assert run.error_detail == {
            "issues": [{"path": "sections[1].lines[1].text", "code": "invalid_character"}]
        }
        assert [v for v in await run_world.versions(track_id) if v.kind == "lyrics"] == []
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            10,
            (1, 1),
        )

    async def test_soft_misses_after_the_cap_succeed_with_warnings_into_the_result(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(
            actor,
            track_id,
            request=LyricsRunRequest(
                language="en", line_syllables=SyllableRange(min=4, max=6), avoid_words=("baby",)
            ),
        )
        flawed = make_lyrics_draft(
            section_lines("s1", FOUR, "oh baby baby"),
            section_lines("s2", FOUR),
            section_lines("s4", FOUR, LONG),
        )
        patches = [
            make_lyrics_patch(
                make_line_patch("s1", 2, "oh baby baby"), make_line_patch("s4", 2, LONG)
            )
        ] * 3
        await run_world.drive(FakeLlm(flawed, *patches))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("succeeded", None)
        assert run.warnings == [
            {"code": "avoid_word_used", "path": "sections[0].lines[1]"},
            {"code": "syllable_range_missed", "path": "sections[2].lines[1]", "count": 10},
        ]
        result = next(
            v for v in await run_world.versions(track_id) if v.id == run.result_lyrics_version_id
        )
        # the paths address THIS document: section 2 of the document is the plan's s4
        assert result.content["sections"][2]["lines"][1]["text"] == LONG
        assert "baby" not in str(run.warnings)
        assert (await run_world.refunds_and_debits(actor)) == (1, 0)

    @pytest.mark.parametrize("reason", list(DraftRefusal))
    async def test_a_refusal_fails_refunds_and_writes_no_version_invariant_17(
        self, run_world: RunWorld, reason: DraftRefusal
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        await run_world.drive(FakeLlm(refusal_lyrics_draft(reason)))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == (
            "failed",
            "content_policy_refused" if reason is DraftRefusal.CONTENT_POLICY else "brief_unusable",
        )
        assert [v for v in await run_world.versions(track_id) if v.kind == "lyrics"] == []
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            10,
            (1, 1),
        )

    async def test_the_call_cap_is_five_whatever_the_model_does_invariant_12(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        llm = FakeLlm(*[LlmOutputInvalid("garbled")] * 5)
        await run_world.drive(llm)

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code, run.llm_calls) == ("failed", "llm_output_invalid", 5)
        assert llm.call_count == 5
        assert (await run_world.balance(actor)) == 10

    async def test_a_crashed_lyrics_run_is_swept_and_refunded_never_rerun(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, _ = await _track(run_world, actor)
        write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
        executor = run_world.executor(FakeLlm())
        await executor.claim(1)
        await run_world.expire_lease(write.run.id)
        assert await executor.sweep() == 1
        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "run_timeout")
        assert await run_world.balance(actor) == 10
        assert await executor.claim(1) == []

    async def test_a_plan_deleted_before_the_claim_fails_the_run_and_refunds(
        self, run_world: RunWorld
    ) -> None:
        """Cannot happen through the API (the guard); if rows are removed anyway the claim is safe."""
        from sqlalchemy import text

        actor = await run_world.add_actor()
        track_id, first = await _track(run_world, actor)
        await run_world.create_plan(actor, track_id, make_plan(make_section(), title="B"))
        write = await run_world.submit(
            actor, track_id, request=LyricsRunRequest(plan_version_id=first, language="en")
        )
        async with run_world.factory() as session, session.begin():
            await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
            await session.execute(
                text("DELETE FROM studio_document_versions WHERE id = :id"), {"id": first}
            )

        await run_world.drive(FakeLlm())

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert await run_world.balance(actor) == 10
