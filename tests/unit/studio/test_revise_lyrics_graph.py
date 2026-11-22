"""The ``revise_lyrics`` graph against a scripted ``FakeLlm`` (invariants 8-10, 12, 13, 16, 17)."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

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
from src.studio.documents import LyricsDocument, SyllableRange
from src.studio.errors import DocumentIssue
from src.studio.planner.drafts import (
    DraftRefusal,
    LyricLineDraft,
    LyricsPatchDraft,
    LyricsRevisionDraft,
)
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs import revise_lyrics as revise_graph
from src.studio.planner.graphs.revise_lyrics import (
    MAX_LLM_CALLS,
    ReviseDeps,
    compile_revise_graph,
    initial_state,
    run_revise_graph,
)
from src.studio.planner.lyric_checks import AVOID_WORD_USED, SYLLABLE_RANGE_MISSED
from src.studio.planner.outcome import (
    REVISION_INCOMPLETE,
    LyricsReady,
    LyricsRun,
    RunRefused,
    RunRejected,
)
from src.studio.planner.prompts.blocks import REASK_MESSAGE
from src.studio.run_requests import LineTarget, ReviseLyricsRunRequest, ReviseTargets
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_line_patch,
    make_lyrics_patch,
    make_lyrics_revision,
    refusal_lyrics_patch,
    refusal_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import CONTEXT, PLAN, S1, S2, S4, doc

if TYPE_CHECKING:
    from uuid import UUID

pytestmark = pytest.mark.unit

GRAPH = compile_revise_graph()
RANGE = SyllableRange(min=4, max=6)
FOUR = "la la la la"
LONG = "la la la la la la la la la la"  # ten syllables


def _input() -> LyricsDocument:
    """What is being revised: s1 has 3 lines, s2 has 2, s4 has 1; one line is the user's own long one."""
    return doc((S1, [FOUR, LONG, FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR]))


def _targets(
    sections: tuple[UUID, ...] = (), lines: tuple[tuple[UUID, int], ...] = ()
) -> ReviseTargets:
    return ReviseTargets(
        sections=sections, lines=tuple(LineTarget(section_id=s, line=n) for s, n in lines)
    )


def _request(targets: ReviseTargets, **kwargs: Any) -> ReviseLyricsRunRequest:
    return ReviseLyricsRunRequest(instruction="make it warmer", targets=targets, **kwargs)


async def _run(
    llm: Any,
    targets: ReviseTargets,
    stages: list[Stage] | None = None,
    *,
    request: ReviseLyricsRunRequest | None = None,
    lyrics: LyricsDocument | None = None,
    line_syllables: SyllableRange | None = RANGE,
    max_fixes: int = 2,
) -> LyricsRun:
    seen = stages if stages is not None else []

    async def on_stage(stage: Stage) -> None:
        seen.append(stage)

    state = initial_state(
        CONTEXT,
        request or _request(targets),
        PLAN,
        lyrics or _input(),
        line_syllables=line_syllables,
    )
    return await run_revise_graph(
        GRAPH, state, ReviseDeps(llm=llm, on_stage=on_stage, max_fixes=max_fixes)
    )


def _ready(run: LyricsRun) -> LyricsReady:
    assert isinstance(run.outcome, LyricsReady)
    return run.outcome


def _texts(lyrics: LyricsDocument) -> list[list[str]]:
    return [[line.text for line in section.lines] for section in lyrics.sections]


# -- the happy path ---------------------------------------------------------------------------


async def test_a_section_target_is_rewritten_and_nothing_else_changes() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", "la la la la la", FOUR, FOUR)])
    )
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    ready = _ready(run)
    assert _texts(ready.lyrics) == [
        [FOUR, LONG, FOUR],
        ["la la la la la", FOUR, FOUR],  # the section may change its number of lines
        [FOUR],
    ]
    assert (llm.call_count, run.llm_calls) == (1, 1)
    assert stages == [Stage.DRAFTING, Stage.VALIDATING]
    assert ready.warnings == ()


async def test_a_line_target_replaces_one_line_and_never_the_count() -> None:
    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s1", 3, "la la la la la")]))
    ready = _ready(await _run(llm, _targets(lines=((S1, 2),))))
    assert _texts(ready.lyrics)[0] == [FOUR, LONG, "la la la la la"]


async def test_sections_and_lines_in_one_request() -> None:
    llm = FakeLlm(
        make_lyrics_revision(
            sections=[section_lines("s4", FOUR, FOUR)], lines=[make_line_patch("s2", 1, FOUR)]
        )
    )
    ready = _ready(await _run(llm, _targets(sections=(S4,), lines=((S2, 0),))))
    assert _texts(ready.lyrics)[2] == [FOUR, FOUR]


async def test_a_section_with_no_lines_yet_can_be_written() -> None:
    lyrics = doc((S1, [FOUR]), (S4, [FOUR]))
    llm = FakeLlm(make_lyrics_revision(sections=[section_lines("s2", FOUR, FOUR)]))
    ready = _ready(await _run(llm, _targets(sections=(S2,)), lyrics=lyrics))
    assert [s.section_id for s in ready.lyrics.sections] == [S1, S2, S4]


async def test_the_saved_document_carries_the_range_this_run_used() -> None:
    inputs_own = SyllableRange(min=2, max=9)
    lyrics = doc((S1, [FOUR]), line_syllables=inputs_own)
    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s1", 1, "la la la la la")]))
    ready = _ready(
        await _run(llm, _targets(lines=((S1, 0),)), lyrics=lyrics, line_syllables=inputs_own)
    )
    assert ready.lyrics.line_syllables == inputs_own

    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s1", 1, FOUR)]))
    ready = _ready(await _run(llm, _targets(lines=((S1, 0),)), lyrics=lyrics, line_syllables=RANGE))
    assert ready.lyrics.line_syllables == RANGE


async def test_the_model_sees_the_numbered_lyrics_the_targets_and_the_instruction() -> None:
    llm = FakeLlm(
        make_lyrics_revision(
            sections=[section_lines("s2", FOUR)], lines=[make_line_patch("s1", 2, FOUR)]
        )
    )
    request = _request(_targets(sections=(S2,), lines=((S1, 1),)), avoid_words=("baby",))
    await _run(llm, request.targets, request=request)

    system, user = llm.calls[0]
    assert system.role == "system"
    text = user.content
    assert (
        "<lyrics>\ns1\n1: la la la la\n2: " + LONG + "\n3: la la la la\ns2\n1: la la la la" in text
    )
    assert "<targets>\nsection s2\ns1 line 2\n</targets>" in text
    assert "<instruction>\nmake it warmer\n</instruction>" in text
    assert "<avoid_words>\nbaby\n</avoid_words>" in text
    assert "<brief>" not in text
    assert text.index("<language>") < text.index("<plan>") < text.index("<lyrics>")


# -- only what the model changed is judged (invariant 16) ---------------------------------------


async def test_the_users_own_long_line_is_never_checked_only_the_changed_ones() -> None:
    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 1, FOUR)]))
    run = await _run(llm, _targets(lines=((S2, 0),)))
    assert _ready(run).warnings == ()  # s1 line 2 has ten syllables and is not the model's
    assert llm.call_count == 1


async def test_an_avoid_word_in_an_untouched_line_is_not_flagged_either() -> None:
    lyrics = doc((S1, ["oh baby baby", FOUR]))
    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s1", 2, FOUR)]))
    request = _request(_targets(lines=((S1, 1),)), avoid_words=("baby",))
    run = await _run(llm, request.targets, request=request, lyrics=lyrics, line_syllables=None)
    assert _ready(run).warnings == ()


async def test_a_changed_line_that_misses_is_patched_and_the_patch_names_only_it() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),
        make_lyrics_patch(make_line_patch("s2", 2, FOUR)),
    )
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    assert _texts(_ready(run).lyrics)[1] == [FOUR, FOUR]
    assert stages == [Stage.DRAFTING, Stage.VALIDATING, Stage.REVISING, Stage.VALIDATING]
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    request = llm.calls[1][-1].content
    assert "- s2 line 2: 10 syllables counted, allowed 4\N{EN DASH}6" in request
    assert request.count("\n- ") == 1
    assert LONG not in request


async def test_the_follow_up_shows_the_model_its_own_work_with_the_lines_merged() -> None:
    llm = FakeLlm(
        make_lyrics_revision(
            sections=[section_lines("s2", FOUR, LONG)], lines=[make_line_patch("s1", 1, FOUR)]
        ),
        make_lyrics_patch(make_line_patch("s2", 2, FOUR)),
    )
    await _run(llm, _targets(sections=(S2,), lines=((S1, 0),)))
    echoed = llm.calls[1][2].content
    assert '"key":"s2"' in echoed
    assert LONG in echoed  # it sees what it wrote, so it can fix the right line
    assert '"line":1' in echoed


async def test_a_patch_for_a_line_that_was_not_asked_for_is_re_asked() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),
        make_lyrics_patch(make_line_patch("s2", 1, "sneaky")),  # line 1 was fine
        make_lyrics_patch(make_line_patch("s2", 2, FOUR)),
    )
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    assert _texts(_ready(run).lyrics)[1] == [FOUR, FOUR]
    assert (llm.call_count, run.llm_calls) == (3, 3)
    assert stages.count(Stage.REVISING) == 1
    assert (llm.calls[2][-1].role, llm.calls[2][-1].content) == ("user", REASK_MESSAGE)


# -- a reply that broke the targets -----------------------------------------------------------


async def test_a_missing_target_gets_one_more_full_revision() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR)]),  # the line target is missing
        make_lyrics_revision(
            sections=[section_lines("s2", FOUR)], lines=[make_line_patch("s1", 1, FOUR)]
        ),
    )
    run = await _run(llm, _targets(sections=(S2,), lines=((S1, 0),)))

    assert isinstance(run.outcome, LyricsReady)
    request = llm.calls[1][-1].content
    assert "return the full corrected JSON object" in request
    assert "- s1 line 1: missing_target" in request


async def test_an_unexpected_target_gets_one_more_full_revision() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR), section_lines("s4", "sneaky")]),
        make_lyrics_revision(sections=[section_lines("s2", FOUR)]),
    )
    run = await _run(llm, _targets(sections=(S2,)))

    ready = _ready(run)
    assert _texts(ready.lyrics)[2] == [FOUR]  # s4 never took the sneaky line
    assert "- s4: unexpected_target" in llm.calls[1][-1].content


async def test_a_full_revision_request_names_every_hard_problem() -> None:
    llm = FakeLlm(
        make_lyrics_revision(
            sections=[section_lines("s2", "oh [no]", FOUR)]
        ),  # the line target is missing
        make_lyrics_revision(
            sections=[section_lines("s2", FOUR, FOUR)], lines=[make_line_patch("s1", 1, FOUR)]
        ),
    )
    await _run(llm, _targets(sections=(S2,), lines=((S1, 0),)))
    request = llm.calls[1][-1].content
    assert "- s1 line 1: missing_target" in request
    assert "- s2 line 1: reserved_characters" in request


async def test_the_second_full_revision_is_merged_from_the_input_again() -> None:
    llm = FakeLlm(
        make_lyrics_revision(
            sections=[section_lines("s2", "first try", FOUR)]
        ),  # missing line target
        make_lyrics_revision(
            sections=[section_lines("s2", FOUR, FOUR)], lines=[make_line_patch("s1", 1, FOUR)]
        ),
    )
    ready = _ready(await _run(llm, _targets(sections=(S2,), lines=((S1, 0),))))
    assert _texts(ready.lyrics)[1] == [FOUR, FOUR]


# -- the budget -------------------------------------------------------------------------------


async def test_hard_issues_after_the_cap_fail_with_plan_section_paths_invariant_13() -> None:
    bad = make_lyrics_revision(sections=[section_lines("s2", FOUR, "oh [no] no no")])
    keep = make_lyrics_patch(make_line_patch("s2", 2, "still [bad]"))
    llm = FakeLlm(bad, keep, keep)
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    assert run.outcome == RunRejected(
        (DocumentIssue("sections[1].lines[1].text", "reserved_characters"),),
        StudioRunKind.REVISE_LYRICS,
    )
    assert (llm.call_count, run.llm_calls) == (3, 3) == (MAX_LLM_CALLS, 3)
    assert stages.count(Stage.REVISING) == 2


async def test_soft_misses_after_the_cap_succeed_with_warnings_on_the_result_document() -> None:
    keep = make_lyrics_patch(make_line_patch("s2", 2, LONG))
    llm = FakeLlm(make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]), keep, keep)
    run = await _run(llm, _targets(sections=(S2,)))

    ready = _ready(run)
    assert llm.call_count == 3
    assert [(w.code, w.path, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, "sections[1].lines[1]", 10)
    ]


async def test_an_avoid_word_in_a_changed_line_is_a_warning() -> None:
    bad = make_lyrics_revision(lines=[make_line_patch("s1", 1, "oh baby")])
    keep = make_lyrics_patch(make_line_patch("s1", 1, "oh baby"))
    request = _request(_targets(lines=((S1, 0),)), avoid_words=("baby",))
    run = await _run(
        FakeLlm(bad, keep, keep), request.targets, request=request, line_syllables=None
    )
    assert [(w.code, w.path) for w in _ready(run).warnings] == [
        (AVOID_WORD_USED, "sections[0].lines[0]")
    ]


async def test_a_revision_that_changes_nothing_still_returns_the_input_text() -> None:
    llm = FakeLlm(make_lyrics_revision(lines=[make_line_patch("s2", 1, FOUR)]))
    ready = _ready(await _run(llm, _targets(lines=((S2, 0),))))
    assert ready.lyrics.sections == _input().sections
    assert ready.lyrics.line_syllables == RANGE


async def test_a_smaller_fix_budget_is_honoured() -> None:
    bad = make_lyrics_revision(sections=[section_lines("s2", "oh [no]", FOUR)])
    llm = FakeLlm(bad, make_lyrics_patch(make_line_patch("s2", 1, FOUR)))
    run = await _run(llm, _targets(sections=(S2,)), max_fixes=0)
    assert llm.call_count == 1
    assert isinstance(run.outcome, RunRejected)


# -- refusals, undecodable replies, errors -----------------------------------------------------


@pytest.mark.parametrize("reason", list(DraftRefusal))
async def test_a_refusal_ends_the_graph_invariant_17(reason: DraftRefusal) -> None:
    llm = FakeLlm(refusal_lyrics_revision(reason))
    run = await _run(llm, _targets(sections=(S2,)))
    assert run.outcome == RunRefused(reason)
    assert llm.call_count == 1


async def test_a_refusal_in_a_fix_ends_the_run_too() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),
        refusal_lyrics_patch(DraftRefusal.BRIEF_UNUSABLE),
    )
    run = await _run(llm, _targets(sections=(S2,)))
    assert run.outcome == RunRefused(DraftRefusal.BRIEF_UNUSABLE)


async def test_an_undecodable_reply_is_re_asked_without_using_a_fix() -> None:
    llm = FakeLlm(
        LlmOutputInvalid("bad", usage=LlmUsage(prompt_tokens=7, completion_tokens=3)),
        make_lyrics_revision(sections=[section_lines("s2", FOUR)]),
    )
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    assert isinstance(run.outcome, LyricsReady)
    assert (llm.call_count, run.llm_calls) == (2, 2)
    assert Stage.REVISING not in stages
    assert (run.prompt_tokens, run.completion_tokens) == (107, 53)


async def test_the_call_cap_includes_re_asks() -> None:
    llm = FakeLlm(*[LlmOutputInvalid("bad")] * 3)
    with pytest.raises(LlmOutputInvalid):
        await _run(llm, _targets(sections=(S2,)))
    assert llm.call_count == 3


async def test_port_errors_propagate() -> None:
    error = LlmUnavailable("down")
    with pytest.raises(LlmUnavailable) as caught:
        await _run(FakeLlm(error), _targets(sections=(S2,)))
    assert caught.value is error


# -- a fix round that fails at the port never discards a valid revision (invariant 22, R1) -----

SOFT = make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)])  # one soft miss
HARD = make_lyrics_revision(sections=[section_lines("s2", FOUR, "oh [no] no")])  # a bracket


def _round_fails(reason: str) -> list[Any]:
    """What the model port does on every call of the round that follows ``SOFT``."""
    return {
        "undecodable": [LlmOutputInvalid("bad")] * 2,  # the call cap (3) is reached
        "unusable": [make_lyrics_patch(make_line_patch("s2", 1, "sneaky"))] * 2,  # not asked for
        "unavailable": [LlmUnavailable("down")],
        "truncated": [LlmOutputTruncated("cut")],
        "rejected": [LlmRejected(400)],
    }[reason]


@pytest.mark.parametrize(
    "reason", ["undecodable", "unusable", "unavailable", "truncated", "rejected"]
)
async def test_a_failed_fix_round_keeps_the_valid_revision_with_revision_incomplete_invariant_22(
    reason: str,
) -> None:
    llm = FakeLlm(SOFT, *_round_fails(reason))
    run = await _run(llm, _targets(sections=(S2,)))

    ready = _ready(run)
    assert [(w.code, w.path, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, "sections[1].lines[1]", 10),
        (REVISION_INCOMPLETE, None, None),
    ]
    assert _texts(ready.lyrics)[1] == [FOUR, LONG]  # the revision as it stood
    assert _texts(ready.lyrics)[0] == [FOUR, LONG, FOUR]  # and the user's own lines, untouched


async def test_a_failed_second_fix_keeps_the_result_of_the_first() -> None:
    eight = "la la la la la la la la"
    llm = FakeLlm(SOFT, make_lyrics_patch(make_line_patch("s2", 2, eight)), LlmUnavailable("down"))
    stages: list[Stage] = []
    run = await _run(llm, _targets(sections=(S2,)), stages)

    ready = _ready(run)
    assert _texts(ready.lyrics)[1] == [FOUR, eight]
    assert [(w.code, w.count) for w in ready.warnings] == [
        (SYLLABLE_RANGE_MISSED, 8),
        (REVISION_INCOMPLETE, None),
    ]
    assert stages[-1] == Stage.REVISING  # it ends at the failed round and does not check again


async def test_a_failed_fix_round_with_a_hard_issue_fails_with_the_ports_error() -> None:
    error = LlmUnavailable("down")
    with pytest.raises(LlmUnavailable) as caught:
        await _run(FakeLlm(HARD, error), _targets(sections=(S2,)))
    assert caught.value is error


async def test_a_failed_full_revision_round_with_a_broken_target_fails_with_the_ports_error() -> (
    None
):
    missing = make_lyrics_revision(sections=[section_lines("s1", FOUR)])  # s2 was targeted
    with pytest.raises(LlmOutputTruncated):
        await _run(FakeLlm(missing, LlmOutputTruncated("cut")), _targets(sections=(S2,)))


async def test_a_refusal_on_the_fix_round_is_still_a_refusal() -> None:
    llm = FakeLlm(SOFT, refusal_lyrics_patch(DraftRefusal.CONTENT_POLICY))
    run = await _run(llm, _targets(sections=(S2,)))
    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY)


async def test_only_the_ports_errors_are_kept_other_exceptions_still_fail_the_run() -> None:
    class Broken(Exception):
        pass

    with pytest.raises(Broken):
        await _run(FakeLlm(SOFT, Broken()), _targets(sections=(S2,)))


# -- the call cap, whatever the model returns (invariant 12) --------------------------------------


class _ChaosLlm:
    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)  # noqa: S311 - test data, not security
        self.calls = 0

    async def complete(self, messages: Any, *, schema: Any, schema_name: str) -> LlmResult[Any]:  # noqa: ARG002
        self.calls += 1
        if self._random.random() < 0.2:
            raise LlmOutputInvalid("bad", usage=LlmUsage(prompt_tokens=1, completion_tokens=1))
        return LlmResult(value=self._reply(schema), usage=LlmUsage(1, 1), model_id="chaos")

    def _text(self) -> str:
        return self._random.choice([FOUR, "la", "oh [no]", "baby baby baby", LONG, "", "2 4 6"])

    def _reply(self, schema: type) -> Any:
        refusal = self._random.choice([None] * 12 + [DraftRefusal.CONTENT_POLICY])
        keys = ["s1", "s2", "s3", "s4", "s9"]
        patches = [
            make_line_patch(self._random.choice(keys), self._random.randint(0, 4), self._text())
            for _ in range(self._random.randint(0, 3))
        ]
        if schema is LyricsPatchDraft:
            return LyricsPatchDraft(refusal=refusal, refusal_memories=[], lines=patches)
        sections = [
            section_lines(
                self._random.choice(keys),
                *(
                    LyricLineDraft(text=self._text(), delivery=[])
                    for _ in range(self._random.randint(0, 3))
                ),
            )
            for _ in range(self._random.randint(0, 2))
        ]
        return LyricsRevisionDraft(
            refusal=refusal, refusal_memories=[], sections=sections, lines=patches
        )


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(seed=st.integers(min_value=0, max_value=10_000))
async def test_the_call_cap_holds_whatever_the_model_returns_invariant_12(seed: int) -> None:
    llm = _ChaosLlm(seed)
    request = _request(_targets(sections=(S2,), lines=((S1, 0),)), avoid_words=("baby",))
    try:
        run = await _run(llm, request.targets, request=request)
    except LlmOutputInvalid:
        pass
    else:
        assert run.llm_calls == llm.calls
    assert llm.calls <= MAX_LLM_CALLS


# -- impossible states are named ----------------------------------------------------------------


def _runtime() -> Any:
    async def on_stage(_stage: Stage) -> None:
        return None

    return SimpleNamespace(context=ReviseDeps(llm=FakeLlm(), on_stage=on_stage))


async def test_check_before_a_revision_is_a_named_error() -> None:
    state = initial_state(
        CONTEXT, _request(_targets(sections=(S2,))), PLAN, _input(), line_syllables=None
    )
    with pytest.raises(PlannerStateError) as caught:
        await revise_graph._check(state, _runtime())
    assert str(caught.value) == "check ran before the revision"


async def test_fix_before_a_check_is_a_named_error() -> None:
    state = initial_state(
        CONTEXT, _request(_targets(sections=(S2,))), PLAN, _input(), line_syllables=None
    )
    with pytest.raises(PlannerStateError) as caught:
        await revise_graph._fix(state, _runtime())
    assert str(caught.value) == "fix ran before check"
