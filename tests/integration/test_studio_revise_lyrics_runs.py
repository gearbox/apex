"""``revise_lyrics`` runs against PostgreSQL: inputs at submit, lineage, untouched lines, the budget.

Invariants 3, 5, 6, 8-10, 12, 13, 16, 17 and the P2a-47 refusal of lyrics that are no longer valid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from src.llm.port import LlmOutputInvalid
from src.studio.errors import (
    LyricsRequired,
    LyricsVersionNotFound,
    StudioDocumentInvalid,
    VersionInUseByRun,
    VersionIsLyricsBase,
)
from src.studio.planner.drafts import DraftRefusal
from src.studio.run_requests import LineTarget, ReviseLyricsRunRequest, ReviseTargets
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_patch,
    make_lyrics_revision,
    refusal_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S3, S4, doc
from tests.studio_runs_support import PRODUCT
from tests.studio_support import make_plan, make_section

if TYPE_CHECKING:
    from uuid import UUID

    from src.db.models.studio import StudioDocumentVersion
    from tests.studio_runs_support import Actor, RunWorld

FOUR = "la la la la"
LONG = "la la la la la la la la la la"


def _input() -> Any:
    """s1 has 3 lines (the second is the user's own, too long for the range), s2 has 2, s4 has 1."""
    return doc((S1, [FOUR, LONG, FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR]))


def _request(
    *,
    sections: tuple[UUID, ...] = (),
    lines: tuple[tuple[UUID, int], ...] = (),
    **kwargs: Any,
) -> ReviseLyricsRunRequest:
    targets = ReviseTargets(
        sections=sections, lines=tuple(LineTarget(section_id=s, line=n) for s, n in lines)
    )
    return ReviseLyricsRunRequest(instruction="make it warmer", targets=targets, **kwargs)


async def _world(run_world: RunWorld, **actor_kwargs: Any) -> tuple[Actor, UUID, UUID, UUID]:
    """(actor, track, plan version, lyrics version): a track with a plan and lyrics, both current."""
    actor = await run_world.add_actor(**actor_kwargs)
    track_id = await run_world.new_track(actor, "Night Drive", PLAN)
    plan_id = (await run_world.track_row(track_id)).current_plan_version_id
    assert plan_id is not None
    lyrics_id = await run_world.create_lyrics(actor, track_id, _input(), plan_version_id=plan_id)
    return actor, track_id, plan_id, lyrics_id


async def _refused(run_world: RunWorld, actor: Actor, track_id: UUID, request: Any) -> Any:
    before = await run_world.balance(actor)
    ledger = len(await run_world.ledger(actor))
    runs = await run_world.count_runs(actor)
    with pytest.raises((StudioDocumentInvalid, LyricsRequired, LyricsVersionNotFound)) as caught:
        await run_world.submit(actor, track_id, request=request)
    assert await run_world.balance(actor) == before
    assert len(await run_world.ledger(actor)) == ledger
    assert await run_world.count_runs(actor) == runs
    return caught.value


def _lines(version: StudioDocumentVersion) -> list[list[str]]:
    return [[line["text"] for line in s["lines"]] for s in version.content["sections"]]


async def _lyrics_versions(run_world: RunWorld, track_id: UUID) -> list[StudioDocumentVersion]:
    return [v for v in await run_world.versions(track_id) if v.kind == "lyrics"]


# -- inputs are resolved at submit ---------------------------------------------------------------


class TestInputs:
    async def test_the_current_lyrics_and_their_plan_are_stored_on_the_run(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))

        run = await run_world.run(write.run.id)
        assert (run.kind, run.status) == ("revise_lyrics", "queued")
        assert run.inputs == {
            "plan_version_id": str(plan_id),
            "lyrics_version_id": str(lyrics_id),
            "language": "en",
        }
        assert run.request["lyrics_version_id"] is None  # as sent: no version named

    async def test_the_plan_is_the_lyrics_own_not_the_tracks_current_one(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        newer = await run_world.create_plan(
            actor, track_id, make_plan(make_section(), title="Moved on"), make_current=False
        )
        await run_world.set_current(actor, track_id, plan_version_id=newer, lyrics_version_id=None)
        write = await run_world.submit(
            actor, track_id, request=_request(sections=(S2,), lyrics_version_id=lyrics_id)
        )
        inputs = (await run_world.run(write.run.id)).inputs
        assert inputs is not None
        assert inputs["plan_version_id"] == str(plan_id)  # not `newer`

    @pytest.mark.parametrize("which", ["random", "plan_version", "another_tracks_lyrics"])
    async def test_lyrics_that_are_not_lyrics_of_this_track_are_404(
        self, run_world: RunWorld, which: str
    ) -> None:
        actor, track_id, plan_id, _ = await _world(run_world)
        _, other_track, _, other_lyrics = await _world(run_world)
        wrong = {"random": uuid4(), "plan_version": plan_id, "another_tracks_lyrics": other_lyrics}[
            which
        ]
        assert other_track != track_id
        error = await _refused(
            run_world, actor, track_id, _request(sections=(S2,), lyrics_version_id=wrong)
        )
        assert isinstance(error, LyricsVersionNotFound)

    async def test_without_lyrics_it_is_409(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, "T", PLAN)
        error = await _refused(run_world, actor, track_id, _request(sections=(S2,)))
        assert isinstance(error, LyricsRequired)

    async def test_lyrics_that_are_not_valid_today_are_refused_before_the_model_is_paid(
        self, run_world: RunWorld
    ) -> None:
        """P2a-47: an older, laxer release stored a bracket; fix the lyrics first."""
        actor, track_id, plan_id, _ = await _world(run_world)
        stale = await run_world.insert_raw_lyrics(
            actor,
            track_id,
            plan_version_id=plan_id,
            make_current=True,
            content={
                "language": "en",
                "line_syllables": None,
                "sections": [
                    {
                        "section_id": str(S1),
                        "lines": [
                            {"text": "fine", "delivery": []},
                            {"text": "has [brackets]", "delivery": []},
                        ],
                    }
                ],
            },
        )
        error = await _refused(run_world, actor, track_id, _request(sections=(S2,)))
        assert isinstance(error, StudioDocumentInvalid)
        assert [(i.path, i.code) for i in error.issues] == [
            ("lyrics.sections[0].lines[1].text", "reserved_characters")
        ]
        assert "brackets" not in str(error.detail)
        assert stale is not None

    async def test_every_target_problem_is_reported_together(self, run_world: RunWorld) -> None:
        actor, track_id, _, _ = await _world(run_world)
        request = _request(
            sections=(uuid4(), S3),
            lines=((S1, 3), (S2, 2), (uuid4(), 0)),
        )
        error = await _refused(run_world, actor, track_id, request)
        assert isinstance(error, StudioDocumentInvalid)
        assert [(i.path, i.code) for i in error.issues] == [
            ("targets.sections[0]", "unknown_section"),
            ("targets.sections[1]", "instrumental_section"),
            ("targets.lines[0]", "line_out_of_range"),  # s1 has lines 0..2
            ("targets.lines[1]", "line_out_of_range"),  # s2 has lines 0..1
            ("targets.lines[2]", "line_out_of_range"),  # an unknown section has no lines
        ]

    async def test_a_line_of_an_instrumental_section_is_out_of_range_not_a_crash(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        error = await _refused(run_world, actor, track_id, _request(lines=((S3, 0),)))
        assert [(i.path, i.code) for i in error.issues] == [
            ("targets.lines[0]", "line_out_of_range")
        ]

    async def test_a_section_target_may_name_a_vocal_section_without_lines(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, "T", PLAN)
        plan_id = (await run_world.track_row(track_id)).current_plan_version_id
        assert plan_id is not None
        await run_world.create_lyrics(actor, track_id, doc((S1, [FOUR])), plan_version_id=plan_id)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        assert write.run.status == "queued"

    @pytest.mark.parametrize(
        ("request_", "path", "code"),
        [
            (_request(), "targets", "targets_required"),
            (_request(sections=(S1, S1)), "targets.sections[1]", "duplicate_target"),
            (_request(sections=(S1,), lines=((S1, 0),)), "targets.lines[0]", "overlapping_targets"),
        ],
    )
    async def test_a_bad_body_is_refused_with_nothing_written(
        self, run_world: RunWorld, request_: Any, path: str, code: str
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        error = await _refused(run_world, actor, track_id, request_)
        assert [(i.path, i.code) for i in error.issues] == [(path, code)]

    async def test_inputs_are_fixed_when_the_run_is_accepted_invariant_3(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        write = await run_world.submit(
            actor, track_id, request=_request(lines=((S2, 0),), line_syllables=None)
        )
        replacement = await run_world.create_lyrics(
            actor, track_id, doc((S1, ["different"])), plan_version_id=plan_id
        )
        await run_world.drive(
            FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 1, "new")]))
        )

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded", run.error_code
        assert (await run_world.track_row(track_id)).current_lyrics_version_id == replacement
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert result.parent_version_id == lyrics_id  # the input, not the newer current one
        assert _lines(result)[1] == ["new", FOUR]

    async def test_a_replay_is_the_same_run_after_the_pointers_moved(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        request = _request(sections=(S2,))
        original = await run_world.submit(actor, track_id, request=request, key="same")
        await run_world.create_lyrics(
            actor, track_id, doc((S1, ["different"])), plan_version_id=plan_id
        )
        replay = await run_world.submit(actor, track_id, request=request, key="same")
        assert (replay.run.id, replay.changed) == (original.run.id, False)
        inputs = (await run_world.run(original.run.id)).inputs
        assert inputs is not None
        assert inputs["lyrics_version_id"] == str(lyrics_id)


# -- running ---------------------------------------------------------------------------------


class TestRunning:
    async def test_untouched_lines_are_byte_identical_and_the_lineage_is_right_invariant_8(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        before = await run_world.track_row(track_id)
        write = await run_world.submit(
            actor, track_id, request=_request(sections=(S2,), lines=((S1, 0), (S4, 0)))
        )
        reply = make_lyrics_revision(
            sections=[section_lines("s2", "a new chorus line", "and another", "and a third")],
            lines=[make_line_patch("s1", 1, "a new first line"), make_line_patch("s4", 1, "last")],
        )
        await run_world.drive(FakeLlm(reply))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("succeeded", None)
        assert run.result_plan_version_id is None
        (result,) = [
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        ]
        original = next(v for v in await _lyrics_versions(run_world, track_id) if v.id == lyrics_id)
        assert (result.origin, result.parent_version_id, result.based_on_plan_version_id) == (
            "llm",
            lyrics_id,
            plan_id,
        )
        assert _lines(result) == [
            ["a new first line", LONG, FOUR],  # only line 0 of s1 changed
            ["a new chorus line", "and another", "and a third"],  # the section may grow
            ["last"],
        ]
        assert (
            original.content == (await _lyrics_versions(run_world, track_id))[0].content
        )  # the input is untouched
        after = await run_world.track_row(track_id)
        assert after.current_lyrics_version_id == before.current_lyrics_version_id
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            9,
            (1, 0),
        )

    async def test_the_result_is_based_on_the_lyrics_plan_even_when_it_is_not_current(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        newer = await run_world.create_plan(
            actor, track_id, make_plan(make_section(), title="B"), make_current=False
        )
        await run_world.set_current(actor, track_id, plan_version_id=newer, lyrics_version_id=None)
        write = await run_world.submit(
            actor, track_id, request=_request(lines=((S2, 1),), lyrics_version_id=lyrics_id)
        )
        await run_world.drive(
            FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 2, "x y")]))
        )

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded", run.error_code
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert result.based_on_plan_version_id == plan_id  # the plan the model saw, not `newer`
        assert _lines(result)[1] == [FOUR, "x y"]

    async def test_a_line_target_never_changes_the_number_of_lines_invariant_9(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(lines=((S1, 2),)))
        await run_world.drive(
            FakeLlm(make_lyrics_revision(lines=[make_line_patch("s1", 3, "end")]))
        )
        run = await run_world.run(write.run.id)
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert [len(s) for s in _lines(result)] == [3, 2, 1]
        assert _lines(result)[0] == [FOUR, LONG, "end"]

    async def test_a_section_without_lines_is_written_in_plan_order_invariant_9(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, "T", PLAN)
        plan_id = (await run_world.track_row(track_id)).current_plan_version_id
        assert plan_id is not None
        await run_world.create_lyrics(
            actor, track_id, doc((S1, [FOUR]), (S4, [FOUR])), plan_version_id=plan_id
        )
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        await run_world.drive(
            FakeLlm(make_lyrics_revision(sections=[section_lines("s2", "chorus")]))
        )
        run = await run_world.run(write.run.id)
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert [s["section_id"] for s in result.content["sections"]] == [str(S1), str(S2), str(S4)]

    async def test_a_revision_that_changes_nothing_returns_the_input_invariant_10(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, lyrics_id = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(lines=((S2, 0),)))
        await run_world.drive(FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 1, FOUR)])))

        run = await run_world.run(write.run.id)
        assert (run.status, run.result_lyrics_version_id) == ("succeeded", lyrics_id)
        assert run.warnings == [{"code": "revision_unchanged"}]
        assert len(await _lyrics_versions(run_world, track_id)) == 1  # nothing was written
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_only_the_lines_it_changed_are_judged_invariant_16(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        from src.studio.documents import SyllableRange

        write = await run_world.submit(
            actor,
            track_id,
            request=_request(
                lines=((S2, 0),), line_syllables=SyllableRange(min=4, max=6), avoid_words=("baby",)
            ),
        )
        await run_world.drive(
            FakeLlm(
                make_lyrics_revision(
                    lines=[make_line_patch("s2", 1, f"{FOUR} la")]
                )
            )
        )
        run = await run_world.run(write.run.id)
        # the user's own ten-syllable line (s1 line 2) is not the model's to judge
        assert (run.status, run.warnings, run.llm_calls) == ("succeeded", [], 1)

    async def test_a_fix_round_asks_for_the_changed_line_only(self, run_world: RunWorld) -> None:
        actor, track_id, _, _ = await _world(run_world)
        from src.studio.documents import SyllableRange

        write = await run_world.submit(
            actor,
            track_id,
            request=_request(sections=(S2,), line_syllables=SyllableRange(min=4, max=6)),
        )
        llm = FakeLlm(
            make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),
            make_lyrics_patch(make_line_patch("s2", 2, "la la la la la")),
        )
        await run_world.drive(llm)

        run = await run_world.run(write.run.id)
        assert (run.status, run.llm_calls, run.warnings) == ("succeeded", 2, [])
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert _lines(result)[1] == [FOUR, "la la la la la"]
        assert "- s2 line 2:" in llm.calls[1][-1].content

    async def test_the_snapshot_records_ids_and_hashes_not_text(self, run_world: RunWorld) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world, artist=True)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        await run_world.drive(FakeLlm(make_lyrics_revision(sections=[section_lines("s2", "x y")])))

        run = await run_world.run(write.run.id)
        snapshot = run.context_snapshot
        assert snapshot is not None
        assert snapshot["snapshot_version"] == 3
        assert (snapshot["input_plan_version_id"], snapshot["input_lyrics_version_id"]) == (
            str(plan_id),
            str(lyrics_id),
        )
        assert snapshot["input_lyrics_content_hash"] is not None
        assert snapshot["language"] == "en"
        assert run.prompt_version == "revise-lyrics-v2"
        assert "warmer" not in str(snapshot)  # the instruction lives on the request only
        assert "la la la" not in str(snapshot)  # no lyric line in the snapshot

    async def test_the_stages_are_published_in_order(self, run_world: RunWorld) -> None:
        actor, track_id, _, _ = await _world(run_world)
        from src.studio.documents import SyllableRange

        await run_world.submit(
            actor,
            track_id,
            request=_request(sections=(S2,), line_syllables=SyllableRange(min=4, max=6)),
        )
        await run_world.drive(
            FakeLlm(
                make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),
                make_lyrics_patch(make_line_patch("s2", 2, FOUR)),
            )
        )
        assert [(e.status, e.stage) for e in run_world.bus.run_events] == [
            ("queued", None),
            ("running", "context"),
            ("running", "drafting"),
            ("running", "validating"),
            ("running", "revising"),
            ("running", "validating"),
            ("running", "saving"),
            ("succeeded", None),
        ]


# -- the budget and the failure modes -----------------------------------------------------------


class TestOutcomes:
    async def test_hard_issues_after_the_cap_fail_refund_and_use_plan_paths_invariant_13(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        bad = make_lyrics_revision(sections=[section_lines("s2", FOUR, "oh [no] no")])
        keep = make_lyrics_patch(make_line_patch("s2", 2, "still [bad]"))
        await run_world.drive(FakeLlm(bad, keep, keep))

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code, run.llm_calls) == ("failed", "lyrics_invalid", 3)
        assert run.error_detail == {
            "issues": [{"path": "sections[1].lines[1].text", "code": "reserved_characters"}]
        }
        assert len(await _lyrics_versions(run_world, track_id)) == 1  # only the input
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            10,
            (1, 1),
        )

    async def test_soft_misses_after_the_cap_succeed_with_document_paths(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        from src.studio.documents import SyllableRange

        write = await run_world.submit(
            actor,
            track_id,
            request=_request(sections=(S2,), line_syllables=SyllableRange(min=4, max=6)),
        )
        flawed = make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)])
        keep = make_lyrics_patch(make_line_patch("s2", 2, LONG))
        await run_world.drive(FakeLlm(flawed, keep, keep))

        run = await run_world.run(write.run.id)
        assert run.status == "succeeded"
        assert run.warnings == [
            {"code": "syllable_range_missed", "path": "sections[1].lines[1]", "count": 10}
        ]

    async def test_a_revision_that_breaks_the_targets_is_asked_for_again(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        llm = FakeLlm(
            make_lyrics_revision(
                sections=[section_lines("s2", "ok"), section_lines("s4", "sneaky")]
            ),
            make_lyrics_revision(sections=[section_lines("s2", "ok")]),
        )
        await run_world.drive(llm)
        run = await run_world.run(write.run.id)
        assert (run.status, run.llm_calls) == ("succeeded", 2)
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert _lines(result)[2] == [FOUR]  # s4 stayed as it was

    @pytest.mark.parametrize("reason", list(DraftRefusal))
    async def test_a_refusal_fails_refunds_and_writes_no_version_invariant_17(
        self, run_world: RunWorld, reason: DraftRefusal
    ) -> None:
        actor, track_id, _, _ = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        await run_world.drive(FakeLlm(refusal_lyrics_revision(reason)))
        run = await run_world.run(write.run.id)
        assert run.status == "failed"
        assert run.error_code == (
            "content_policy_refused" if reason is DraftRefusal.CONTENT_POLICY else "brief_unusable"
        )
        assert len(await _lyrics_versions(run_world, track_id)) == 1
        assert (await run_world.balance(actor), await run_world.refunds_and_debits(actor)) == (
            10,
            (1, 1),
        )

    async def test_the_call_cap_is_three_invariant_12(self, run_world: RunWorld) -> None:
        actor, track_id, _, _ = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        llm = FakeLlm(*[LlmOutputInvalid("garbled")] * 3)
        await run_world.drive(llm)
        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code, run.llm_calls) == ("failed", "llm_output_invalid", 3)
        assert llm.call_count == 3


# -- the delete guard -------------------------------------------------------------------------


class TestDeleteGuard:
    async def test_the_input_lyrics_cannot_be_deleted_while_the_run_is_active_invariant_6(
        self, run_world: RunWorld
    ) -> None:
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        later = await run_world.create_plan(
            actor, track_id, make_plan(make_section(), title="B"), make_current=False
        )
        await run_world.set_current(actor, track_id, plan_version_id=later, lyrics_version_id=None)
        write = await run_world.submit(
            actor, track_id, request=_request(sections=(S2,), lyrics_version_id=lyrics_id)
        )
        with pytest.raises(VersionInUseByRun):
            await run_world.delete_version(actor, lyrics_id)

        seen: list[str] = []

        async def probe() -> None:  # the run is `running` now
            try:
                await run_world.delete_version(actor, lyrics_id)
            except VersionInUseByRun:
                seen.append("blocked")

        await run_world.drive(
            FakeLlm(refusal_lyrics_revision(DraftRefusal.CONTENT_POLICY), before_call=probe)
        )
        assert seen == ["blocked"]
        assert (await run_world.run(write.run.id)).status == "failed"

        await run_world.delete_version(actor, lyrics_id)  # terminal: nothing holds it any more
        await run_world.delete_version(actor, plan_id)  # and now its plan is free as well

    async def test_the_plan_of_the_input_lyrics_is_refused_for_its_older_reason_first(
        self, run_world: RunWorld
    ) -> None:
        """The plan is the base of the input lyrics; that refusal is older than the run guard."""
        actor, track_id, plan_id, lyrics_id = await _world(run_world)
        later = await run_world.create_plan(
            actor, track_id, make_plan(make_section(), title="B"), make_current=False
        )
        await run_world.set_current(actor, track_id, plan_version_id=later, lyrics_version_id=None)
        await run_world.submit(
            actor, track_id, request=_request(sections=(S2,), lyrics_version_id=lyrics_id)
        )
        with pytest.raises(VersionIsLyricsBase):
            await run_world.delete_version(actor, plan_id)

    async def test_a_revision_never_loses_its_parent(self, run_world: RunWorld) -> None:
        """The parent *is* an input: the save has no fallback because the delete is refused."""
        actor, track_id, _, lyrics_id = await _world(run_world)
        write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
        attempts: list[str] = []

        async def try_to_replace_and_delete() -> None:
            await run_world.set_current(
                actor,
                track_id,
                lyrics_version_id=await run_world.create_lyrics(
                    actor,
                    track_id,
                    doc((S1, ["x"])),
                    plan_version_id=(await run_world.track_row(track_id)).current_plan_version_id,  # type: ignore[arg-type]
                ),
            )
            try:
                await run_world.delete_version(actor, lyrics_id)
            except VersionInUseByRun:
                attempts.append("blocked")

        await run_world.drive(
            FakeLlm(
                make_lyrics_revision(sections=[section_lines("s2", "ok")]),
                before_call=try_to_replace_and_delete,
            )
        )
        run = await run_world.run(write.run.id)
        assert (run.status, attempts) == ("succeeded", ["blocked"])
        assert run.warnings == []  # never parent_version_replaced for a revision
        result = next(
            v
            for v in await _lyrics_versions(run_world, track_id)
            if v.id == run.result_lyrics_version_id
        )
        assert result.parent_version_id == lyrics_id


async def test_the_product_scope_is_the_runs_own(run_world: RunWorld) -> None:
    actor, track_id, _, _ = await _world(run_world)
    write = await run_world.submit(actor, track_id, request=_request(sections=(S2,)))
    run = await run_world.run(write.run.id)
    assert run.product_id == PRODUCT
