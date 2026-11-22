"""A content-policy refusal names the memories that caused it (R5, P2b-10).

Only the model knows which memory it objected to, so it names them by key (``m1``, ``m2``, ...) and
the server maps the keys back to ids: unknown and malformed keys and repeats dropped, prompt order
kept. The attribution is best-effort - it can be empty - and only ``content_policy`` carries it.
Pure: the three graphs against a scripted ``FakeLlm``, then the failure the outcome becomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec
import pytest

from src.studio.documents import SyllableRange
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.graphs.lyrics import LyricsDeps, compile_lyrics_graph, run_lyrics_graph
from src.studio.planner.graphs.lyrics import initial_state as lyrics_state
from src.studio.planner.graphs.plan import PlanDeps, compile_plan_graph, run_plan_graph
from src.studio.planner.graphs.plan import initial_state as plan_state
from src.studio.planner.graphs.revise_lyrics import (
    ReviseDeps,
    compile_revise_graph,
    run_revise_graph,
)
from src.studio.planner.graphs.revise_lyrics import initial_state as revise_state
from src.studio.planner.outcome import RunRefused, refused
from src.studio.run_failures import RunFailure, failure_of_outcome
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    ReviseTargets,
)
from tests.llm_support import FakeLlm
from tests.studio_llm import (
    make_lyrics_draft,
    make_lyrics_revision,
    refusal_draft,
    refusal_lyrics_draft,
    refusal_lyrics_patch,
    refusal_lyrics_revision,
    section_lines,
)
from tests.studio_lyrics_support import CONTEXT, PLAN, S1, S2, S4, doc
from tests.studio_memory_support import make_memory

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from src.studio.planner.outcome import RunOutcome

pytestmark = pytest.mark.unit

MEMORIES = tuple(make_memory(n) for n in (11, 22, 33))
FIRST, SECOND, THIRD = (memory.id for memory in MEMORIES)
WITH_MEMORIES = msgspec.structs.replace(CONTEXT, memories=MEMORIES)
FOUR = "la la la la"
LONG = "la la la la la la la la la la"  # too long for the 4-6 range: a soft miss
RANGE = SyllableRange(min=4, max=6)


async def _noop(_stage: object) -> None:
    return None


# One runner per kind: the keys the model names -> the outcome of a run whose FIRST reply refuses.
type Runner = Callable[[Sequence[str], DraftRefusal], Awaitable[RunOutcome]]


async def _plan(keys: Sequence[str], reason: DraftRefusal) -> RunOutcome:
    run = await run_plan_graph(
        compile_plan_graph(),
        plan_state(WITH_MEMORIES, PlanRunRequest(brief="a brief")),
        PlanDeps(llm=FakeLlm(refusal_draft(reason, keys)), on_stage=_noop, max_revisions=2),
    )
    return run.outcome


async def _lyrics(keys: Sequence[str], reason: DraftRefusal) -> RunOutcome:
    run = await run_lyrics_graph(
        compile_lyrics_graph(),
        lyrics_state(
            WITH_MEMORIES, LyricsRunRequest(brief="b"), PLAN, language="en", line_syllables=None
        ),
        LyricsDeps(
            llm=FakeLlm(refusal_lyrics_draft(reason, keys)), on_stage=_noop, max_revisions=3
        ),
    )
    return run.outcome


def _revision_state() -> Any:
    lyrics = doc((S1, [FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR]))
    request = ReviseLyricsRunRequest(
        instruction="make it warmer", targets=ReviseTargets(sections=(S2,))
    )
    return revise_state(WITH_MEMORIES, request, PLAN, lyrics, line_syllables=None)


async def _revise(keys: Sequence[str], reason: DraftRefusal) -> RunOutcome:
    run = await run_revise_graph(
        compile_revise_graph(),
        _revision_state(),
        ReviseDeps(llm=FakeLlm(refusal_lyrics_revision(reason, keys)), on_stage=_noop, max_fixes=2),
    )
    return run.outcome


KINDS = pytest.mark.parametrize("run", [_plan, _lyrics, _revise], ids=["plan", "lyrics", "revise"])


# -- the three graphs ---------------------------------------------------------------------------


@KINDS
async def test_a_refusal_names_the_memories_that_caused_it(run: Runner) -> None:
    outcome = await run(["m2"], DraftRefusal.CONTENT_POLICY)

    assert outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(SECOND,))


@KINDS
async def test_the_named_memories_come_back_in_prompt_order_not_the_models(run: Runner) -> None:
    outcome = await run(["m3", "m1"], DraftRefusal.CONTENT_POLICY)

    assert outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(FIRST, THIRD))


@KINDS
async def test_unknown_or_repeated_memory_keys_are_dropped(run: Runner) -> None:
    outcome = await run(["m9", "x", "m1", "m1", "M2", "m02", ""], DraftRefusal.CONTENT_POLICY)

    assert outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(FIRST,))


@KINDS
async def test_a_refusal_without_memory_keys_names_nothing(run: Runner) -> None:
    outcome = await run([], DraftRefusal.CONTENT_POLICY)

    assert outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=())
    assert failure_of_outcome(outcome) == RunFailure("content_policy_refused", None)


@KINDS
async def test_brief_unusable_ignores_any_keys(run: Runner) -> None:
    outcome = await run(["m1", "m2"], DraftRefusal.BRIEF_UNUSABLE)

    assert outcome == RunRefused(DraftRefusal.BRIEF_UNUSABLE, memory_ids=())
    assert failure_of_outcome(outcome) == RunFailure("brief_unusable", None)


async def test_a_run_with_no_memories_cannot_name_any() -> None:
    """Keys the model invented in a run that was shown no memories name nothing."""
    run = await run_plan_graph(
        compile_plan_graph(),
        plan_state(CONTEXT, PlanRunRequest(brief="a brief")),
        PlanDeps(
            llm=FakeLlm(refusal_draft(DraftRefusal.CONTENT_POLICY, ["m1"])),
            on_stage=_noop,
            max_revisions=2,
        ),
    )

    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY)


# -- a refusal in the middle of the run (the fix round) -----------------------------------------


async def test_a_lyrics_fix_round_that_refuses_keeps_the_keys_it_named() -> None:
    one_soft_miss = make_lyrics_draft(
        section_lines("s1", FOUR),
        section_lines("s2", FOUR),
        section_lines("s4", FOUR, LONG),
    )
    soft = FakeLlm(one_soft_miss, refusal_lyrics_patch(DraftRefusal.CONTENT_POLICY, ["m3", "m3"]))
    run = await run_lyrics_graph(
        compile_lyrics_graph(),
        lyrics_state(
            WITH_MEMORIES,
            LyricsRunRequest(),
            PLAN,
            language="en",
            line_syllables=RANGE,
        ),
        LyricsDeps(llm=soft, on_stage=_noop, max_revisions=3),
    )

    assert soft.call_count == 2  # the draft and the refusing fix
    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(THIRD,))


async def test_a_revision_fix_round_that_refuses_keeps_the_keys_it_named() -> None:
    llm = FakeLlm(
        make_lyrics_revision(sections=[section_lines("s2", FOUR, LONG)]),  # one soft miss
        refusal_lyrics_patch(DraftRefusal.CONTENT_POLICY, ["m1"]),
    )
    run = await run_revise_graph(
        compile_revise_graph(),
        revise_state(
            WITH_MEMORIES,
            ReviseLyricsRunRequest(
                instruction="make it warmer", targets=ReviseTargets(sections=(S2,))
            ),
            PLAN,
            doc((S1, [FOUR]), (S2, [FOUR, FOUR]), (S4, [FOUR])),
            line_syllables=RANGE,
        ),
        ReviseDeps(llm=llm, on_stage=_noop, max_fixes=2),
    )

    assert llm.call_count == 2
    assert run.outcome == RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(FIRST,))


# -- the pure constructor and the failure it becomes --------------------------------------------


def test_the_constructor_builds_ids_only_for_content_policy() -> None:
    assert refused(DraftRefusal.CONTENT_POLICY, ["m2"], MEMORIES) == RunRefused(
        DraftRefusal.CONTENT_POLICY, (SECOND,)
    )
    assert refused(DraftRefusal.BRIEF_UNUSABLE, ["m2"], MEMORIES) == RunRefused(
        DraftRefusal.BRIEF_UNUSABLE, ()
    )


def test_the_default_refusal_carries_no_memories() -> None:
    assert RunRefused(DraftRefusal.CONTENT_POLICY).memory_ids == ()


def test_the_failure_lists_the_ids_as_strings_and_nothing_else() -> None:
    failure = failure_of_outcome(RunRefused(DraftRefusal.CONTENT_POLICY, memory_ids=(FIRST, THIRD)))

    assert failure == RunFailure(
        "content_policy_refused", {"memory_ids": [str(UUID(int=11)), str(UUID(int=33))]}
    )
    assert set(failure.detail or {}) == {"memory_ids"}  # no text, no keys, no counts


def test_a_hand_built_brief_unusable_with_ids_still_has_no_detail() -> None:
    """Whatever builds the outcome, ``brief_unusable`` never leaks ids into the failure."""
    failure = failure_of_outcome(RunRefused(DraftRefusal.BRIEF_UNUSABLE, memory_ids=(FIRST,)))

    assert failure == RunFailure("brief_unusable", None)
