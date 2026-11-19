"""A fix round that fails at the port never discards a validated result (invariants 22 and 23, R1).

Driven through the service and the executor against PostgreSQL: the run ends ``succeeded``, writes
one version, is charged, stores ``revision_incomplete`` - and stores the number of port calls and
tokens that the executor's meter actually saw (the graph's own counters never see a failed round).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.llm.port import LlmOutputInvalid, LlmOutputTruncated, LlmUnavailable, LlmUsage
from src.studio.documents import SyllableRange
from src.studio.run_requests import (
    LineTarget,
    LyricsRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    make_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import PLAN, S1, S2, S4, doc

if TYPE_CHECKING:
    from uuid import UUID

    from tests.studio_runs_support import Actor, RunWorld

FOUR = "la la la la"
LONG = "la la la la la la la la la la"  # ten syllables: a soft miss against 4-6
RANGE = SyllableRange(min=4, max=6)
PAID = LlmUsage(prompt_tokens=7, completion_tokens=3)
"""What an unusable reply that was billed reports."""

INCOMPLETE = {"code": "revision_incomplete"}

# (what the model does on the round after the first reply, calls the port saw, prompt, completion);
# every successful reply of FakeLlm reports 100 prompt and 50 completion tokens
ROUND_FAILURES = {
    "unavailable": ([LlmUnavailable("down")], 2, 100, 50),  # no reply, so nothing was billed
    "truncated": ([LlmOutputTruncated("cut", usage=PAID)], 2, 107, 53),
    "undecodable": ([LlmOutputInvalid("bad", usage=PAID)] * 4, 5, 100 + 4 * 7, 50 + 4 * 3),
}


async def _lyrics_world(world: RunWorld) -> tuple[Actor, UUID]:
    actor = await world.add_actor()
    track_id = await world.new_track(actor, "Night Drive", PLAN)
    return actor, track_id


@pytest.mark.parametrize("failure", list(ROUND_FAILURES))
async def test_a_lyrics_run_whose_fix_round_fails_is_kept_charged_and_metered_invariants_22_23(
    run_world: RunWorld, failure: str
) -> None:
    script, calls, prompt, completion = ROUND_FAILURES[failure]
    actor, track_id = await _lyrics_world(run_world)
    write = await run_world.submit(
        actor, track_id, request=LyricsRunRequest(language="en", line_syllables=RANGE)
    )
    soft = make_lyrics_draft(
        section_lines("s1", FOUR, FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR, LONG)
    )
    llm = FakeLlm(soft, *script)
    await run_world.drive(llm)

    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("succeeded", None), run.error_code
    assert run.warnings == [
        {"code": "syllable_range_missed", "path": "sections[2].lines[1]", "count": 10},
        INCOMPLETE,
    ]
    lyrics_versions = [v for v in await run_world.versions(track_id) if v.kind == "lyrics"]
    assert [v.id for v in lyrics_versions] == [run.result_lyrics_version_id]  # exactly one
    assert lyrics_versions[0].content["sections"][2]["lines"][1]["text"] == LONG
    # charged, not refunded: the user has the lyrics
    assert (await run_world.refunds_and_debits(actor), await run_world.balance(actor)) == (
        (1, 0),
        9,
    )
    # the stored numbers are the port's, not the graph's (the graph never saw the failed round)
    assert llm.call_count == calls
    assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (calls, prompt, completion)
    assert run.error_detail is None


async def test_a_lyrics_run_with_a_hard_issue_still_fails_with_the_ports_error_and_refunds(
    run_world: RunWorld,
) -> None:
    actor, track_id = await _lyrics_world(run_world)
    write = await run_world.submit(actor, track_id, request=LyricsRunRequest(language="en"))
    hard = make_lyrics_draft(
        section_lines("s1", FOUR),
        section_lines("s2", FOUR, "oh [no] no"),
        section_lines("s4", FOUR),
    )
    await run_world.drive(FakeLlm(hard, LlmUnavailable("down")))

    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("failed", "llm_unavailable")
    assert [v for v in await run_world.versions(track_id) if v.kind == "lyrics"] == []
    assert (await run_world.refunds_and_debits(actor), await run_world.balance(actor)) == (
        (1, 1),
        10,
    )
    assert (run.llm_calls, run.prompt_tokens) == (2, 100)


async def test_a_second_round_that_fails_keeps_what_the_first_round_made(
    run_world: RunWorld,
) -> None:
    actor, track_id = await _lyrics_world(run_world)
    write = await run_world.submit(
        actor, track_id, request=LyricsRunRequest(language="en", line_syllables=RANGE)
    )
    eight = "la la la la la la la la"
    soft = make_lyrics_draft(
        section_lines("s1", FOUR), section_lines("s2", FOUR), section_lines("s4", FOUR, LONG)
    )
    await run_world.drive(
        FakeLlm(soft, make_lyrics_patch(make_line_patch("s4", 2, eight)), LlmUnavailable("down"))
    )

    run = await run_world.run(write.run.id)
    assert run.status == "succeeded"
    assert run.warnings == [
        {"code": "syllable_range_missed", "path": "sections[2].lines[1]", "count": 8},
        INCOMPLETE,
    ]
    assert run.llm_calls == 3
    (version,) = (v for v in await run_world.versions(track_id) if v.kind == "lyrics")
    assert version.content["sections"][2]["lines"][1]["text"] == eight


# -- revise_lyrics ---------------------------------------------------------------------------------


async def _revise_world(world: RunWorld) -> tuple[Actor, UUID, UUID]:
    actor = await world.add_actor()
    track_id = await world.new_track(actor, "Night Drive", PLAN)
    plan_id = (await world.track_row(track_id)).current_plan_version_id
    assert plan_id is not None
    # s1 has a long line of the user's own: a revision never judges it
    lyrics = doc((S1, [FOUR, LONG, FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR]))
    await world.create_lyrics(actor, track_id, lyrics, plan_version_id=plan_id)
    return actor, track_id, plan_id


@pytest.mark.parametrize("failure", list(ROUND_FAILURES))
async def test_a_revision_whose_fix_round_fails_is_kept_charged_and_metered_invariants_22_23(
    run_world: RunWorld, failure: str
) -> None:
    script, calls, prompt, completion = ROUND_FAILURES[failure]
    if failure == "undecodable":  # the revision's cap is three calls, not five
        script, calls = script[:2], 3
        prompt, completion = 100 + 2 * 7, 50 + 2 * 3
    actor, track_id, _ = await _revise_world(run_world)
    request: Any = ReviseLyricsRunRequest(
        instruction="make it warmer",
        targets=ReviseTargets(sections=(S2,), lines=()),
        line_syllables=RANGE,
    )
    write = await run_world.submit(actor, track_id, request=request)
    flawed = make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)])
    llm = FakeLlm(flawed, *script)
    await run_world.drive(llm)

    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("succeeded", None), run.error_code
    assert run.warnings == [
        {"code": "syllable_range_missed", "path": "sections[1].lines[1]", "count": 10},
        INCOMPLETE,
    ]
    result = next(
        v for v in await run_world.versions(track_id) if v.id == run.result_lyrics_version_id
    )
    assert result.content["sections"][1]["lines"][1]["text"] == LONG
    assert (await run_world.refunds_and_debits(actor), await run_world.balance(actor)) == (
        (1, 0),
        9,
    )
    assert llm.call_count == calls
    assert (run.llm_calls, run.prompt_tokens, run.completion_tokens) == (calls, prompt, completion)


async def test_a_revision_with_a_broken_target_still_fails_with_the_ports_error(
    run_world: RunWorld,
) -> None:
    actor, track_id, _ = await _revise_world(run_world)
    request: Any = ReviseLyricsRunRequest(
        instruction="make it warmer",
        targets=ReviseTargets(sections=(S2,), lines=(LineTarget(section_id=S1, line=0),)),
    )
    write = await run_world.submit(actor, track_id, request=request)
    missing = make_lyrics_revision(sections=[section_lines("s2", FOUR)])  # the line target is gone
    await run_world.drive(FakeLlm(missing, LlmOutputTruncated("cut", usage=PAID)))

    run = await run_world.run(write.run.id)
    assert (run.status, run.error_code) == ("failed", "llm_output_truncated")
    assert (await run_world.refunds_and_debits(actor), await run_world.balance(actor)) == (
        (1, 1),
        10,
    )
