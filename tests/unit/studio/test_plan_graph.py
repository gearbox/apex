"""The plan graph against a scripted ``FakeLlm`` (invariants 8-10; no database, no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from src.core.enums import StudioRunStage as Stage
from src.studio.errors import DocumentIssue
from src.studio.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.planner.context import ArtistSnapshot, RunContextSnapshot
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.graphs.plan import (
    MAX_LLM_CALLS,
    PlanDeps,
    compile_plan_graph,
    initial_state,
    run_plan_graph,
)
from src.studio.planner.prompts.plan_v1 import REASK_MESSAGE
from src.studio.run_requests import PlanRunRequest
from tests.studio_llm import FakeLlm, make_draft, make_section_draft, refusal_draft

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


async def _run(llm: FakeLlm, stages: list[Stage] | None = None, *, max_revisions: int = 2):  # type: ignore[no-untyped-def]
    seen = stages if stages is not None else []

    async def on_stage(stage: Stage) -> None:
        seen.append(stage)

    deps = PlanDeps(llm=llm, on_stage=on_stage, max_revisions=max_revisions)
    return await run_plan_graph(GRAPH, initial_state(CONTEXT, REQUEST), deps)


async def test_a_valid_first_draft_is_one_call() -> None:
    llm = FakeLlm(make_draft())
    stages: list[Stage] = []
    final = await _run(llm, stages)

    assert final["outcome"] is not None
    assert final["outcome"].status == "ok"
    assert final["plan"] is not None
    assert final["plan"].title == "Neon Rain"
    assert final["plan"].idea == "a brief"
    assert llm.call_count == 1
    assert final["llm_calls"] == 1
    assert final["revisions"] == 0
    assert (final["prompt_tokens"], final["completion_tokens"]) == (100, 50)
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
    final = await _run(llm, stages)

    assert final["outcome"].status == "ok"  # type: ignore[union-attr]
    assert final["revisions"] == 1
    assert llm.call_count == 2
    assert stages == [Stage.DRAFTING, Stage.VALIDATING, Stage.REVISING, Stage.VALIDATING]
    # the second call is a continuation: system, user, previous draft, the rejection
    roles = [m.role for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "reserved_tempo_tag" in llm.calls[1][-1].content
    assert "tags[0]" in llm.calls[1][-1].content


async def test_the_call_cap_is_three_when_nothing_ever_validates() -> None:
    llm = FakeLlm(_bad_draft(), _bad_draft(), _bad_draft())
    final = await _run(llm)

    assert llm.call_count == MAX_LLM_CALLS == 3
    assert final["outcome"].status == "invalid"  # type: ignore[union-attr]
    assert final["plan"] is None
    assert final["revisions"] == 2
    assert {(i.path, i.code) for i in final["issues"]} == {
        ("global_style.tags[0]", "reserved_tempo_tag")
    }


async def test_a_smaller_revision_budget_is_honoured() -> None:
    llm = FakeLlm(_bad_draft(), make_draft())
    final = await _run(llm, max_revisions=0)
    assert llm.call_count == 1
    assert final["outcome"].status == "invalid"  # type: ignore[union-attr]


async def test_duplicate_section_keys_are_revised() -> None:
    twice = make_draft(make_section_draft("s1"), make_section_draft("s1"))
    llm = FakeLlm(twice, make_draft())
    final = await _run(llm)
    assert final["outcome"].status == "ok"  # type: ignore[union-attr]
    assert "duplicate_section_key" in llm.calls[1][-1].content


@pytest.mark.parametrize("reason", list(DraftRefusal))
async def test_a_refusal_ends_the_graph_without_revising(reason: DraftRefusal) -> None:
    llm = FakeLlm(refusal_draft(reason))
    stages: list[Stage] = []
    final = await _run(llm, stages)

    assert final["outcome"].status == "refused"  # type: ignore[union-attr]
    assert final["outcome"].refusal is reason  # type: ignore[union-attr]
    assert final["plan"] is None
    assert llm.call_count == 1
    assert Stage.REVISING not in stages


async def test_an_undecodable_reply_is_re_asked_without_using_a_revision() -> None:
    llm = FakeLlm(LlmOutputInvalid("bad"), make_draft())
    final = await _run(llm)

    assert final["outcome"].status == "ok"  # type: ignore[union-attr]
    assert llm.call_count == 2
    assert final["revisions"] == 0
    assert final["llm_calls"] == 2
    reask = llm.calls[1][-1]
    assert (reask.role, reask.content) == ("user", REASK_MESSAGE)


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
    first = await _run(FakeLlm(make_draft()))
    second = await _run(FakeLlm(make_draft(title="Other")))
    assert first["plan"] is not None
    assert second["plan"] is not None
    assert second["plan"].title == "Other"
    assert first["plan"].sections[0].id != second["plan"].sections[0].id


def test_issue_objects_in_the_state_are_plain_issues() -> None:
    assert DocumentIssue("a", "b").path == "a"
