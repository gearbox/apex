"""The ``lyrics`` graph against a scripted ``FakeLlm`` (invariants 12-14, 17; no database, no network)."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.core.enums import StudioRunKind
from src.core.enums import StudioRunStage as Stage
from src.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmResult,
    LlmUnavailable,
    LlmUsage,
)
from src.studio.documents import SyllableRange
from src.studio.errors import DocumentIssue
from src.studio.planner.drafts import (
    DraftRefusal,
    LyricLineDraft,
    LyricsDraft,
    LyricsPatchDraft,
)
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs import lyrics as lyrics_graph
from src.studio.planner.graphs.lyrics import (
    MAX_LLM_CALLS,
    LyricsDeps,
    compile_lyrics_graph,
    initial_state,
    run_lyrics_graph,
)
from src.studio.planner.lyric_checks import (
    AVOID_WORD_USED,
    SYLLABLE_CHECK_SKIPPED,
    SYLLABLE_RANGE_MISSED,
)
from src.studio.planner.outcome import (
    MAX_WARNINGS,
    REVISION_INCOMPLETE,
    LyricsReady,
    LyricsRun,
    RunRefused,
    RunRejected,
)
from src.studio.planner.prompts.blocks import REASK_MESSAGE
from src.studio.run_requests import LyricsRunRequest
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_draft,
    make_lyrics_patch,
    refusal_lyrics_draft,
    refusal_lyrics_patch,
    section_lines,
)
from tests.studio_lyrics_support import CONTEXT, PLAN, S1, S2, S4

pytestmark = pytest.mark.unit

GRAPH = compile_lyrics_graph()
RANGE = SyllableRange(min=4, max=6)
FOUR = "la la la la"  # four syllables in English


def _good() -> LyricsDraft:
    return make_lyrics_draft(
        section_lines("s1", FOUR, FOUR),
        section_lines("s2", FOUR, FOUR),
        section_lines("s4", FOUR, FOUR),
    )


def _with(section: str, *lines: str) -> LyricsDraft:
    """``_good`` with one section's lines replaced."""
    kept = {"s1": [FOUR, FOUR], "s2": [FOUR, FOUR], "s4": [FOUR, FOUR]}
    kept[section] = list(lines)
    return make_lyrics_draft(*(section_lines(key, *text) for key, text in kept.items()))


async def _run(
    llm: Any,
    stages: list[Stage] | None = None,
    *,
    request: LyricsRunRequest | None = None,
    language: str = "en",
    line_syllables: SyllableRange | None = RANGE,
    max_revisions: int = 3,
) -> LyricsRun:
    seen = stages if stages is not None else []

    async def on_stage(stage: Stage) -> None:
        seen.append(stage)

    state = initial_state(
        CONTEXT,
        request or LyricsRunRequest(),
        PLAN,
        language=language,
        line_syllables=line_syllables,
    )
    deps = LyricsDeps(llm=llm, on_stage=on_stage, max_revisions=max_revisions)
    return await run_lyrics_graph(GRAPH, state, deps)


def _ready(run: LyricsRun) -> LyricsReady:
    assert isinstance(run.outcome, LyricsReady)
    return run.outcome


# -- the happy path ---------------------------------------------------------------------------


async def test_a_valid_first_draft_is_one_call() -> None:
    llm = FakeLlm(_good())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    ready = _ready(run)
    assert (llm.call_count, run.llm_calls) == (1, 1)
    assert (run.prompt_tokens, run.completion_tokens) == (100, 50)
    assert stages == [Stage.DRAFTING, Stage.VALIDATING]
    assert ready.warnings == ()
    assert [s.section_id for s in ready.lyrics.sections] == [S1, S2, S4]
    assert (ready.lyrics.language, ready.lyrics.line_syllables) == ("en", RANGE)


async def test_the_first_call_sends_the_system_prompt_and_the_labelled_blocks() -> None:
    llm = FakeLlm(_good())
    await _run(llm, request=LyricsRunRequest(brief="about rain", avoid_words=("baby",)))
    system, user = llm.calls[0]
    assert (system.role, user.role) == ("system", "user")
    for block in (
        "<language>\nen\n</language>",
        "<plan>",
        "<line_syllables>",
        "<avoid_words>\nbaby\n</avoid_words>",
        "<brief>\nabout rain\n</brief>",
    ):
        assert block in user.content


async def test_the_model_gets_the_plan_with_its_keys_and_instrumental_marks() -> None:
    llm = FakeLlm(_good())
    await _run(llm)
    plan_block = llm.calls[0][1].content.split("<plan>")[1].split("</plan>")[0]
    assert "s1: verse (Opening); delivery: soft" in plan_block
    assert "s3: instrumental [instrumental]" in plan_block
    assert "notes: big and wide" in plan_block


# -- revisions: a patch of single lines -------------------------------------------------------


async def test_a_line_with_a_hard_problem_is_replaced_by_a_patch_and_the_rest_is_frozen() -> None:
    llm = FakeLlm(
        _with("s2", FOUR, "oh [no] no no"),
        make_lyrics_patch(make_line_patch("s2", 2, "la la la la la")),
    )
    stages: list[Stage] = []
    run = await _run(llm, stages)

    ready = _ready(run)
    assert [[line.text for line in s.lines] for s in ready.lyrics.sections] == [
        [FOUR, FOUR],
        [FOUR, "la la la la la"],
        [FOUR, FOUR],
    ]
    assert stages == [Stage.DRAFTING, Stage.VALIDATING, Stage.REVISING, Stage.VALIDATING]
    # a continuation: system, user, the model's own work, the request for exactly one line
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    request = llm.calls[1][-1].content
    assert "- s2 line 2: contains a square bracket" in request
    assert request.count("\n- ") == 1
    assert "oh [no] no no" not in request  # never the line's own text


async def test_the_second_call_shows_the_model_its_work_with_accepted_lines_frozen() -> None:
    llm = FakeLlm(
        _with("s1", "la la la la", "la"),
        make_lyrics_patch(make_line_patch("s1", 2, "la la la la")),
    )
    await _run(llm)
    echoed = msgspec.json.decode(llm.calls[1][2].content, type=LyricsDraft)
    first = echoed.sections[0]
    assert (first.key, [line.text for line in first.lines]) == ("s1", ["la la la la", "la"])


async def test_a_syllable_miss_is_asked_for_with_the_count_and_the_range() -> None:
    llm = FakeLlm(
        _with("s4", FOUR, "la la la la la la la la"),
        make_lyrics_patch(make_line_patch("s4", 2, FOUR)),
    )
    run = await _run(llm)
    assert _ready(run).warnings == ()
    request = llm.calls[1][-1].content
    assert "- s4 line 2: 8 syllables counted, allowed 4\N{EN DASH}6" in request


async def test_an_avoid_word_is_asked_for_without_repeating_it() -> None:
    llm = FakeLlm(
        _with("s1", "oh baby baby", FOUR),
        make_lyrics_patch(make_line_patch("s1", 1, FOUR)),
    )
    run = await _run(llm, request=LyricsRunRequest(avoid_words=("baby",)), line_syllables=None)
    assert _ready(run).warnings == ()
    request = llm.calls[1][-1].content
    assert "- s1 line 1: uses a phrase from <avoid_words>" in request
    assert "baby" not in request


async def test_the_soft_checks_wait_for_the_document_to_be_valid() -> None:
    """A hard problem is fixed first; the syllable miss on another line is not mentioned yet."""
    llm = FakeLlm(
        _with("s1", "oh [no] no no", "la"),
        make_lyrics_patch(make_line_patch("s1", 1, FOUR)),
        make_lyrics_patch(make_line_patch("s1", 2, FOUR)),
    )
    await _run(llm)
    assert "square bracket" in llm.calls[1][-1].content
    assert "syllables counted" not in llm.calls[1][-1].content
    assert "syllables counted" in llm.calls[2][-1].content


async def test_a_patch_for_a_line_nobody_asked_about_is_re_asked_within_the_budget() -> None:
    llm = FakeLlm(
        _with("s2", FOUR, "oh [no] no no"),
        make_lyrics_patch(make_line_patch("s1", 1, "sneaky")),  # not requested
        make_lyrics_patch(make_line_patch("s2", 2, FOUR)),
    )
    stages: list[Stage] = []
    run = await _run(llm, stages)

    ready = _ready(run)
    assert [line.text for line in ready.lyrics.sections[0].lines] == [
        FOUR,
        FOUR,
    ]  # nothing sneaked in
    assert (llm.call_count, run.llm_calls) == (3, 3)
    assert stages.count(Stage.REVISING) == 1  # the re-ask is not a revision round
    reask = llm.calls[2][-1]
    assert (reask.role, reask.content) == ("user", REASK_MESSAGE)


async def test_an_unexpected_patch_on_the_last_call_fails_the_run_as_invalid_output() -> None:
    llm = FakeLlm(
        _with("s2", FOUR, "oh [no] no no"),
        *[make_lyrics_patch(make_line_patch("s1", 1, "sneaky"))] * 4,
    )
    with pytest.raises(LlmOutputInvalid):
        await _run(llm)
    assert llm.call_count == MAX_LLM_CALLS == 5


# -- revisions: a full re-draft ---------------------------------------------------------------


async def test_a_missing_section_gets_one_full_redraft_not_a_patch() -> None:
    llm = FakeLlm(
        make_lyrics_draft(section_lines("s1", FOUR), section_lines("s2", FOUR)),  # s4 missing
        _good(),
    )
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert isinstance(run.outcome, LyricsReady)
    assert stages == [Stage.DRAFTING, Stage.VALIDATING, Stage.REVISING, Stage.VALIDATING]
    request = llm.calls[1][-1].content
    assert "return the full corrected JSON object" in request
    assert "- s4: missing_section_lines" in request


async def test_a_redraft_names_every_hard_problem_not_only_the_structural_one() -> None:
    llm = FakeLlm(
        make_lyrics_draft(section_lines("s1", FOUR, "oh [no] no"), section_lines("s2", FOUR)),
        _good(),
    )
    run = await _run(llm)
    assert isinstance(run.outcome, LyricsReady)
    request = llm.calls[1][-1].content
    assert "- s4: missing_section_lines" in request  # the structural problem
    assert "- s1 line 2: reserved_characters" in request  # and the line the reply also got wrong


async def test_an_unknown_key_and_an_instrumental_section_with_lines_are_redrafted() -> None:
    llm = FakeLlm(
        make_lyrics_draft(
            section_lines("s1", FOUR),
            section_lines("s2", FOUR),
            section_lines("s3", "sung"),
            section_lines("s4", FOUR),
            section_lines("s9", FOUR),
        ),
        _good(),
    )
    run = await _run(llm)
    assert isinstance(run.outcome, LyricsReady)
    request = llm.calls[1][-1].content
    assert "- s3: instrumental_section_has_lines" in request
    assert "- (the whole reply): unknown_section_key" in request


async def test_a_section_over_the_line_limit_is_redrafted() -> None:
    llm = FakeLlm(_with("s1", *[FOUR] * 65), _good())
    run = await _run(llm)
    assert isinstance(run.outcome, LyricsReady)
    assert "- s1: too_many_items" in llm.calls[1][-1].content


# -- the budget -------------------------------------------------------------------------------


async def test_hard_issues_after_the_cap_fail_with_plan_section_paths_invariant_13() -> None:
    stubborn = _with("s2", FOUR, "oh [no] no no")
    llm = FakeLlm(
        stubborn,
        *[make_lyrics_patch(make_line_patch("s2", 2, "still [bad]"))] * 3,
    )
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert run.outcome == RunRejected(
        (DocumentIssue("sections[1].lines[1].text", "reserved_characters"),),
        StudioRunKind.LYRICS,
    )
    assert (llm.call_count, run.llm_calls) == (4, 4)  # the draft and three revision rounds
    assert stages.count(Stage.REVISING) == 3


async def test_a_smaller_revision_budget_is_honoured() -> None:
    llm = FakeLlm(
        _with("s2", FOUR, "oh [no] no no"), make_lyrics_patch(make_line_patch("s2", 2, FOUR))
    )
    run = await _run(llm, max_revisions=0)
    assert llm.call_count == 1
    assert isinstance(run.outcome, RunRejected)


async def test_soft_misses_after_the_cap_succeed_with_warnings_on_the_result_document() -> None:
    long_line = "la la la la la la la la la la"
    llm = FakeLlm(
        _with("s4", FOUR, long_line),
        *[make_lyrics_patch(make_line_patch("s4", 2, long_line))] * 3,
    )
    run = await _run(llm)

    ready = _ready(run)
    assert llm.call_count == 4
    # s4 is the THIRD document section (s3 is instrumental and not in the document)
    assert [(w.code, w.path, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, "sections[2].lines[1]", 10)
    ]
    assert [s.section_id for s in ready.lyrics.sections] == [S1, S2, S4]


async def test_an_avoid_word_that_survives_is_a_warning_not_a_failure() -> None:
    stubborn = _with("s1", "oh baby baby", FOUR)
    llm = FakeLlm(stubborn, *[make_lyrics_patch(make_line_patch("s1", 1, "oh baby"))] * 3)
    run = await _run(llm, request=LyricsRunRequest(avoid_words=("baby",)), line_syllables=None)
    ready = _ready(run)
    assert [(w.code, w.path) for w in ready.warnings] == [(AVOID_WORD_USED, "sections[0].lines[0]")]
    assert "baby" not in repr(ready.warnings)


async def test_skipped_lines_and_a_language_without_a_counter_are_notes_not_misses() -> None:
    german = FakeLlm(_good())
    run = await _run(german, language="de")
    assert [(w.code, w.path, w.count) for w in _ready(run).warnings] == [
        (SYLLABLE_CHECK_SKIPPED, None, None)
    ]
    assert german.call_count == 1  # nothing to revise

    uncountable = FakeLlm(_with("s1", "room 101", FOUR))
    run = await _run(uncountable)
    assert [(w.code, w.count) for w in _ready(run).warnings] == [
        ("syllable_check_skipped_lines", 1)
    ]


async def test_no_range_asked_means_no_syllable_warnings_at_all() -> None:
    run = await _run(FakeLlm(_good()), language="de", line_syllables=None)
    assert _ready(run).warnings == ()


# -- refusals, undecodable replies, errors -----------------------------------------------------


@pytest.mark.parametrize("reason", list(DraftRefusal))
async def test_a_refusal_ends_the_graph_without_revising_invariant_17(reason: DraftRefusal) -> None:
    llm = FakeLlm(refusal_lyrics_draft(reason))
    stages: list[Stage] = []
    run = await _run(llm, stages)
    assert run.outcome == RunRefused(reason)
    assert llm.call_count == 1
    assert Stage.REVISING not in stages


async def test_a_refusal_in_a_patch_reply_ends_the_run_too() -> None:
    llm = FakeLlm(
        _with("s2", FOUR, "oh [no] no no"), refusal_lyrics_patch(DraftRefusal.CONTENT_POLICY)
    )
    run = await _run(llm)
    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY)


async def test_an_undecodable_reply_is_re_asked_without_using_a_revision() -> None:
    paid = LlmOutputInvalid("bad", usage=LlmUsage(prompt_tokens=7, completion_tokens=3))
    llm = FakeLlm(paid, _good())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert isinstance(run.outcome, LyricsReady)
    assert (llm.call_count, run.llm_calls) == (2, 2)
    assert Stage.REVISING not in stages
    assert (run.prompt_tokens, run.completion_tokens) == (100 + 7, 50 + 3)
    assert (llm.calls[1][-1].role, llm.calls[1][-1].content) == ("user", REASK_MESSAGE)


async def test_the_call_cap_includes_re_asks() -> None:
    llm = FakeLlm(*[LlmOutputInvalid("bad")] * 5)
    with pytest.raises(LlmOutputInvalid):
        await _run(llm)
    assert llm.call_count == 5


async def test_a_port_error_propagates_unchanged() -> None:
    error = LlmUnavailable("down")
    llm = FakeLlm(error)
    with pytest.raises(LlmUnavailable) as caught:
        await _run(llm)
    assert caught.value is error


async def test_an_error_during_a_revision_propagates() -> None:
    llm = FakeLlm(_with("s2", FOUR, "oh [no] no no"), LlmUnavailable("down"))
    with pytest.raises(LlmUnavailable):
        await _run(llm)


# -- a fix round that fails at the port never discards a valid result (invariant 22, R1) -------

LONG = "la la la la la la la la la la"  # ten syllables: a soft miss against RANGE
SOFT = _with("s4", FOUR, LONG)  # validates; one line misses the syllable range
HARD = _with("s2", FOUR, "oh [no] no no")  # a bracket in a line: nothing usable


def _round_fails(reason: str) -> list[Any]:
    """What the model port does on every call of the round that follows ``SOFT``."""
    return {
        "undecodable": [LlmOutputInvalid("bad")] * 4,  # the call cap is reached
        "unusable": [make_lyrics_patch(make_line_patch("s1", 1, "sneaky"))] * 4,  # not asked for
        "unavailable": [LlmUnavailable("down")],
        "truncated": [LlmOutputTruncated("cut")],
        "rejected": [LlmRejected(400)],
    }[reason]


@pytest.mark.parametrize(
    "reason", ["undecodable", "unusable", "unavailable", "truncated", "rejected"]
)
async def test_a_failed_fix_round_keeps_the_valid_draft_with_revision_incomplete_invariant_22(
    reason: str,
) -> None:
    llm = FakeLlm(SOFT, *_round_fails(reason))
    run = await _run(llm)

    ready = _ready(run)
    assert [(w.code, w.path, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, "sections[2].lines[1]", 10),
        (REVISION_INCOMPLETE, None, None),
    ]
    assert [s.section_id for s in ready.lyrics.sections] == [S1, S2, S4]
    assert ready.lyrics.sections[2].lines[1].text == LONG  # the draft as it stood, unchanged


async def test_the_other_soft_warnings_are_kept_next_to_revision_incomplete() -> None:
    avoid = LyricsRunRequest(avoid_words=("baby",))
    draft = _with("s1", "oh baby baby", FOUR)
    run = await _run(
        FakeLlm(draft, LlmUnavailable("down")),
        request=avoid,
        language="de",  # no counter
    )
    assert [w.code for w in _ready(run).warnings] == [
        AVOID_WORD_USED,
        SYLLABLE_CHECK_SKIPPED,
        REVISION_INCOMPLETE,
    ]


async def test_a_failed_second_round_keeps_the_result_of_the_first() -> None:
    # Round 1 fixes s4 line 2 to another miss (a line of eight syllables); round 2 fails at the port.
    eight = "la la la la la la la la"
    llm = FakeLlm(
        SOFT,
        make_lyrics_patch(make_line_patch("s4", 2, eight)),
        LlmUnavailable("down"),
    )
    stages: list[Stage] = []
    run = await _run(llm, stages)

    ready = _ready(run)
    assert ready.lyrics.sections[2].lines[1].text == eight  # the latest valid document, not SOFT
    assert [(w.code, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, 8),
        (REVISION_INCOMPLETE, None),
    ]
    assert stages == [
        Stage.DRAFTING,
        Stage.VALIDATING,
        Stage.REVISING,
        Stage.VALIDATING,
        Stage.REVISING,
    ]  # the graph ends at the failed round; it does not check again


async def test_a_failed_fix_round_with_a_hard_issue_fails_with_the_ports_error() -> None:
    error = LlmUnavailable("down")
    with pytest.raises(LlmUnavailable) as caught:
        await _run(FakeLlm(HARD, error))
    assert caught.value is error


async def test_a_failed_redraft_with_a_hard_issue_fails_with_the_ports_error() -> None:
    missing = make_lyrics_draft(section_lines("s1", FOUR), section_lines("s2", FOUR))  # no s4
    with pytest.raises(LlmOutputTruncated):
        await _run(FakeLlm(missing, LlmOutputTruncated("cut")))


async def test_a_refusal_on_the_fix_round_is_still_a_refusal() -> None:
    llm = FakeLlm(SOFT, refusal_lyrics_patch(DraftRefusal.CONTENT_POLICY))
    run = await _run(llm)
    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY)


async def test_only_the_ports_errors_are_kept_other_exceptions_still_fail_the_run() -> None:
    class Broken(Exception):
        pass

    with pytest.raises(Broken):
        await _run(FakeLlm(SOFT, Broken()))


async def test_revision_incomplete_survives_a_full_warnings_list() -> None:
    soft_lines = [LONG] * 20  # one syllable-range warning per line
    draft = make_lyrics_draft(
        *(section_lines(key, *soft_lines) for key in ("s1", "s2", "s4")),
    )
    run = await _run(FakeLlm(draft, LlmUnavailable("down")))  # 60 warnings: not over the cap
    assert [w.code for w in _ready(run).warnings][-1] == REVISION_INCOMPLETE

    soft_lines = [LONG] * 40
    draft = make_lyrics_draft(
        *(section_lines(key, *soft_lines) for key in ("s1", "s2", "s4")),
    )
    warnings = _ready(await _run(FakeLlm(draft, LlmUnavailable("down")))).warnings
    assert len(warnings) == MAX_WARNINGS
    assert [w.code for w in warnings][-2:] == ["warnings_truncated", REVISION_INCOMPLETE]


async def test_a_failing_stage_callback_stops_the_graph() -> None:
    class Stop(Exception):
        pass

    async def on_stage(_stage: Stage) -> None:
        raise Stop

    llm = FakeLlm(_good())
    state = initial_state(CONTEXT, LyricsRunRequest(), PLAN, language="en", line_syllables=None)
    with pytest.raises(Stop):
        await run_lyrics_graph(GRAPH, state, LyricsDeps(llm=llm, on_stage=on_stage))
    assert llm.call_count == 0


async def test_the_graph_can_be_reused_across_runs() -> None:
    first = _ready(await _run(FakeLlm(_good())))
    second = _ready(await _run(FakeLlm(_good())))
    assert first.lyrics == second.lyrics


# -- planner output gets the validation user input gets (invariant 14) --------------------------


@pytest.mark.parametrize(
    ("bad_line", "code"),
    [
        (LyricLineDraft(text="a [bracket] here", delivery=[]), "reserved_characters"),
        (LyricLineDraft(text=FOUR, delivery=["Chorus"]), "reserved_delivery_tag"),
        (LyricLineDraft(text="la " * 101, delivery=[]), "too_long_after_normalization"),
        (LyricLineDraft(text=" ", delivery=[]), "empty_after_normalization"),
        (LyricLineDraft(text=FOUR, delivery=["x", "x"]), "duplicate_tag"),
    ],
)
async def test_the_models_lyrics_are_validated_like_a_users_and_fail_after_the_cap(
    bad_line: LyricLineDraft, code: str
) -> None:
    bad = make_lyrics_draft(
        section_lines("s1", FOUR, FOUR),
        section_lines("s2", FOUR, bad_line),
        section_lines("s4", FOUR, FOUR),
    )
    keep = make_lyrics_patch(make_line_patch("s2", 2, bad_line.text, bad_line.delivery))
    run = await _run(FakeLlm(bad, keep, keep, keep))

    assert isinstance(run.outcome, RunRejected)
    assert run.outcome.kind is StudioRunKind.LYRICS
    assert code in {issue.code for issue in run.outcome.issues}
    # plan-section paths: s2 is the plan's second section, its second line is index 1
    assert {issue.path.split(".")[0] for issue in run.outcome.issues} == {"sections[1]"}
    assert all(issue.path.startswith("sections[1].lines[1]") for issue in run.outcome.issues)


async def test_a_line_over_three_hundred_characters_after_nfc_is_caught() -> None:
    # U+0958 expands to two code points under NFC: 160 of them are 320 after normalization
    expanding = "\N{DEVANAGARI LETTER QA}" * 160
    assert len(expanding) < 300
    bad = _with("s1", expanding, FOUR)
    keep = make_lyrics_patch(make_line_patch("s1", 1, expanding))
    run = await _run(FakeLlm(bad, keep, keep, keep), line_syllables=None)
    assert isinstance(run.outcome, RunRejected)
    assert (
        DocumentIssue("sections[0].lines[0].text", "too_long_after_normalization")
        in run.outcome.issues
    )


async def test_more_than_sixty_four_lines_in_a_section_fail_as_invalid_after_the_cap() -> None:
    too_many = _with("s1", *[FOUR] * 65)
    run = await _run(FakeLlm(too_many, too_many, too_many, too_many), line_syllables=None)
    assert isinstance(run.outcome, RunRejected)
    assert {issue.code for issue in run.outcome.issues} == {"too_many_items"}


# -- the call cap, whatever the model returns (invariant 12) --------------------------------------


class _ChaosLlm:
    """A model that answers anything: valid or invalid structs, undecodable replies, errors."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)  # noqa: S311 - test data, not security
        self.calls = 0

    async def complete(self, messages: Any, *, schema: Any, schema_name: str) -> LlmResult[Any]:  # noqa: ARG002
        self.calls += 1
        roll = self._random.random()
        if roll < 0.2:
            raise LlmOutputInvalid("bad", usage=LlmUsage(prompt_tokens=1, completion_tokens=1))
        return LlmResult(value=self._reply(schema), usage=LlmUsage(1, 1), model_id="chaos")

    def _line(self) -> LyricLineDraft:
        text = self._random.choice(
            [FOUR, "la", "oh [no]", "baby baby baby", "la " * 40, "", "2 4 6"]
        )
        return LyricLineDraft(text=text, delivery=self._random.choice([[], ["soft"], ["Verse"]]))

    def _reply(self, schema: type) -> Any:
        refusal = self._random.choice([None] * 12 + [DraftRefusal.CONTENT_POLICY])
        if schema is LyricsPatchDraft:
            keys = ["s1", "s2", "s3", "s4", "s9"]
            return LyricsPatchDraft(
                refusal=refusal,
                lines=[
                    make_line_patch(
                        self._random.choice(keys), self._random.randint(0, 4), "la la la la"
                    )
                    for _ in range(self._random.randint(0, 3))
                ],
            )
        sections = [
            section_lines(key, *(self._line() for _ in range(self._random.randint(0, 3))))
            for key in self._random.sample(
                ["s1", "s2", "s3", "s4", "s9"], self._random.randint(0, 5)
            )
        ]
        return LyricsDraft(refusal=refusal, sections=sections)


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(seed=st.integers(min_value=0, max_value=10_000))
async def test_the_call_cap_holds_whatever_the_model_returns_invariant_12(seed: int) -> None:
    llm = _ChaosLlm(seed)
    try:
        run = await _run(llm, request=LyricsRunRequest(avoid_words=("baby",)))
    except LlmOutputInvalid:
        pass  # the last allowed call was unusable: the run fails, still within the cap
    else:
        assert run.llm_calls == llm.calls
    assert llm.calls <= MAX_LLM_CALLS


# -- impossible states are named, not asserted --------------------------------------------------


def _runtime() -> Any:
    async def on_stage(_stage: Stage) -> None:
        return None

    return SimpleNamespace(context=LyricsDeps(llm=FakeLlm(), on_stage=on_stage))


async def test_check_before_a_draft_is_a_named_error() -> None:
    state = initial_state(CONTEXT, LyricsRunRequest(), PLAN, language="en", line_syllables=None)
    with pytest.raises(PlannerStateError) as caught:
        await lyrics_graph._check(state, _runtime())
    assert str(caught.value) == "check ran before draft"


async def test_revise_before_a_check_is_a_named_error() -> None:
    state = initial_state(CONTEXT, LyricsRunRequest(), PLAN, language="en", line_syllables=None)
    with pytest.raises(PlannerStateError) as caught:
        await lyrics_graph._revise(state, _runtime())
    assert str(caught.value) == "revise ran before check"


async def test_a_graph_that_ends_without_an_outcome_raises_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lyrics_graph, "_route", lambda _state: "done")
    broken = lyrics_graph.build_lyrics_graph().compile()
    state = initial_state(CONTEXT, LyricsRunRequest(), PLAN, language="en", line_syllables=None)

    async def on_stage(_stage: Stage) -> None:
        return None

    with pytest.raises(PlannerStateError, match="ended without an outcome"):
        await run_lyrics_graph(
            broken, state, LyricsDeps(llm=FakeLlm(_with("s2", FOUR, "[no]")), on_stage=on_stage)
        )
