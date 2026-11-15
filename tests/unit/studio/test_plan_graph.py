"""The plan graph against a scripted ``FakeLlm`` (invariants 8-10, 24; no database, no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from src.core.enums import StudioRunStage as Stage
from src.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
    LlmUsage,
)
from src.studio.errors import DocumentIssue
from src.studio.planner.context import ArtistSnapshot, RunContextSnapshot
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs import plan as plan_graph
from src.studio.planner.graphs.plan import (
    MAX_LLM_CALLS,
    PlanDeps,
    PlanReady,
    PlanRefused,
    PlanRejected,
    PlanRun,
    build_plan_graph,
    compile_plan_graph,
    initial_state,
    run_plan_graph,
)
from src.studio.planner.prompts.plan_v1 import REASK_MESSAGE
from src.studio.run_requests import PlanRunRequest
from tests.llm_support import FakeLlm
from tests.studio_llm import make_draft, make_section_draft, refusal_draft

pytestmark = pytest.mark.unit

GRAPH = compile_plan_graph()
CONTEXT = RunContextSnapshot(
    track_title="Track",
    artist=ArtistSnapshot(
        id=UUID(int=1),
        name="Nova",
        persona=None,
        voice_notes=None,
        default_language=None,
        style_tags=(),
        global_avoid=(),
        aesthetic_references=(),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    ),
    concept=None,
    content_rating="sfw",
    plan_version_id=None,
    plan_content_hash=None,
    current_plan=None,
    lyrics_version_id=None,
    lyrics_content_hash=None,
)
REQUEST = PlanRunRequest(brief="a brief")


def _bad_draft():  # type: ignore[no-untyped-def]
    """Invalid: a tempo-like tag (and so a ``reserved_tempo_tag`` issue)."""
    return make_draft(tags=["120 BPM"])


async def _run(
    llm: FakeLlm,
    stages: list[Stage] | None = None,
    *,
    max_revisions: int = 2,
    graph: Any = GRAPH,
) -> PlanRun:
    seen = stages if stages is not None else []

    async def on_stage(stage: Stage) -> None:
        seen.append(stage)

    deps = PlanDeps(llm=llm, on_stage=on_stage, max_revisions=max_revisions)
    return await run_plan_graph(graph, initial_state(CONTEXT, REQUEST), deps)


async def test_a_valid_first_draft_is_one_call() -> None:
    llm = FakeLlm(make_draft())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert isinstance(run.outcome, PlanReady)
    assert run.outcome.plan.title == "Neon Rain"
    assert run.outcome.plan.idea == "a brief"
    assert llm.call_count == 1
    assert run.llm_calls == 1
    assert (run.prompt_tokens, run.completion_tokens) == (100, 50)
    assert stages == [Stage.DRAFTING, Stage.VALIDATING]


async def test_the_first_call_sends_the_system_prompt_and_the_user_blocks() -> None:
    llm = FakeLlm(make_draft())
    await _run(llm)
    system, user = llm.calls[0]
    assert system.role == "system"
    assert user.role == "user"
    assert "<brief>\na brief\n</brief>" in user.content


async def test_one_revision_fixes_the_plan() -> None:
    llm = FakeLlm(_bad_draft(), make_draft())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert isinstance(run.outcome, PlanReady)
    assert llm.call_count == run.llm_calls == 2
    assert stages == [Stage.DRAFTING, Stage.VALIDATING, Stage.REVISING, Stage.VALIDATING]
    # the second call is a continuation: system, user, previous draft, the rejection
    roles = [m.role for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "reserved_tempo_tag" in llm.calls[1][-1].content
    assert "tags[0]" in llm.calls[1][-1].content


async def test_the_call_cap_is_three_when_nothing_ever_validates() -> None:
    llm = FakeLlm(_bad_draft(), _bad_draft(), _bad_draft())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert llm.call_count == run.llm_calls == MAX_LLM_CALLS == 3
    assert isinstance(run.outcome, PlanRejected)
    assert stages.count(Stage.REVISING) == 2
    assert {(i.path, i.code) for i in run.outcome.issues} == {
        ("global_style.tags[0]", "reserved_tempo_tag")
    }


async def test_a_smaller_revision_budget_is_honoured() -> None:
    llm = FakeLlm(_bad_draft(), make_draft())
    run = await _run(llm, max_revisions=0)
    assert llm.call_count == 1
    assert isinstance(run.outcome, PlanRejected)


async def test_duplicate_section_keys_are_revised() -> None:
    twice = make_draft(make_section_draft("s1"), make_section_draft("s1"))
    llm = FakeLlm(twice, make_draft())
    run = await _run(llm)
    assert isinstance(run.outcome, PlanReady)
    assert "duplicate_section_key" in llm.calls[1][-1].content


@pytest.mark.parametrize("reason", list(DraftRefusal))
async def test_a_refusal_ends_the_graph_without_revising(reason: DraftRefusal) -> None:
    llm = FakeLlm(refusal_draft(reason))
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert run.outcome == PlanRefused(reason)
    assert llm.call_count == 1
    assert Stage.REVISING not in stages


async def test_an_undecodable_reply_is_re_asked_without_using_a_revision() -> None:
    llm = FakeLlm(LlmOutputInvalid("bad"), make_draft())
    stages: list[Stage] = []
    run = await _run(llm, stages)

    assert isinstance(run.outcome, PlanReady)
    assert llm.call_count == 2
    assert run.llm_calls == 2
    assert Stage.REVISING not in stages
    reask = llm.calls[1][-1]
    assert (reask.role, reask.content) == ("user", REASK_MESSAGE)


async def test_the_graph_counts_the_tokens_of_an_undecodable_reply_it_re_asks() -> None:
    """Invariant 26: a reply that was paid for but could not be used is still counted."""
    paid = LlmOutputInvalid("bad", usage=LlmUsage(prompt_tokens=7, completion_tokens=3))
    run = await _run(FakeLlm(paid, make_draft()))

    assert isinstance(run.outcome, PlanReady)
    assert (run.prompt_tokens, run.completion_tokens) == (100 + 7, 50 + 3)


async def test_an_undecodable_reply_without_usage_adds_nothing() -> None:
    run = await _run(FakeLlm(LlmOutputInvalid("bad"), make_draft()))
    assert (run.prompt_tokens, run.completion_tokens) == (100, 50)


async def test_an_undecodable_reply_on_the_last_call_propagates() -> None:
    llm = FakeLlm(_bad_draft(), _bad_draft(), LlmOutputInvalid("bad"))
    with pytest.raises(LlmOutputInvalid):
        await _run(llm)
    assert llm.call_count == 3


async def test_re_asks_share_the_call_budget() -> None:
    llm = FakeLlm(LlmOutputInvalid("a"), LlmOutputInvalid("b"), LlmOutputInvalid("c"))
    with pytest.raises(LlmOutputInvalid):
        await _run(llm)
    assert llm.call_count == 3


@pytest.mark.parametrize(
    "error",
    [LlmUnavailable("down"), LlmRejected(400), LlmOutputTruncated("cut")],
    ids=["unavailable", "rejected", "truncated"],
)
async def test_port_errors_propagate_unchanged(error: Exception) -> None:
    llm = FakeLlm(error)
    with pytest.raises(type(error)) as caught:
        await _run(llm)
    assert caught.value is error
    assert llm.call_count == 1


async def test_an_error_during_a_revision_propagates() -> None:
    llm = FakeLlm(_bad_draft(), LlmUnavailable("down"))
    with pytest.raises(LlmUnavailable):
        await _run(llm)


async def test_a_failing_stage_callback_stops_the_graph() -> None:
    class Stop(Exception):
        pass

    async def on_stage(_stage: Stage) -> None:
        raise Stop

    llm = FakeLlm(make_draft())
    deps = PlanDeps(llm=llm, on_stage=on_stage)
    with pytest.raises(Stop):
        await run_plan_graph(GRAPH, initial_state(CONTEXT, REQUEST), deps)
    assert llm.call_count == 0


async def test_the_graph_can_be_reused_across_runs() -> None:
    first = (await _run(FakeLlm(make_draft()))).outcome
    second = (await _run(FakeLlm(make_draft(title="Other")))).outcome
    assert isinstance(first, PlanReady)
    assert isinstance(second, PlanReady)
    assert second.plan.title == "Other"
    assert first.plan.sections[0].id != second.plan.sections[0].id


def test_issue_objects_in_the_state_are_plain_issues() -> None:
    assert DocumentIssue("a", "b").path == "a"


# -- invariant 24: no outcome without a plan, and a named error for an impossible state -----------


def test_a_ready_outcome_cannot_exist_without_a_plan() -> None:
    with pytest.raises(TypeError):
        PlanReady()  # type: ignore[call-arg]


def test_the_outcome_variants_are_frozen_values() -> None:
    refusal = PlanRefused(DraftRefusal.CONTENT_POLICY)
    with pytest.raises(AttributeError):
        refusal.reason = DraftRefusal.BRIEF_UNUSABLE  # type: ignore[misc]
    assert PlanRejected(()) == PlanRejected(())


async def test_a_graph_that_ends_without_an_outcome_raises_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced: the router sends an invalid first draft to the end instead of to a revision."""
    monkeypatch.setattr(plan_graph, "_route", lambda _state: "done")
    broken = build_plan_graph().compile()

    with pytest.raises(PlannerStateError, match="ended without an outcome") as caught:
        await _run(FakeLlm(_bad_draft()), graph=broken)
    assert isinstance(caught.value, RuntimeError)


def _runtime() -> Any:
    async def on_stage(_stage: Stage) -> None:
        return None

    return SimpleNamespace(context=PlanDeps(llm=FakeLlm(), on_stage=on_stage))


async def test_validate_before_a_draft_is_a_named_error() -> None:
    state = initial_state(CONTEXT, REQUEST)
    with pytest.raises(PlannerStateError) as caught:
        await plan_graph._validate(state, _runtime())
    assert str(caught.value) == "validate ran before draft"


async def test_revise_before_a_draft_is_a_named_error() -> None:
    state = initial_state(CONTEXT, REQUEST)
    with pytest.raises(PlannerStateError) as caught:
        await plan_graph._revise(state, _runtime())
    assert str(caught.value) == "revise ran before draft"


async def test_the_state_errors_name_no_data() -> None:
    state = initial_state(CONTEXT, REQUEST)
    for node in (plan_graph._validate, plan_graph._revise):
        with pytest.raises(PlannerStateError) as caught:
            await node(state, _runtime())
        assert "a brief" not in str(caught.value)
        assert "Nova" not in str(caught.value)
