"""The run failure table: errors and graph outcomes -> a run's ``error_code``."""

from __future__ import annotations

import pytest

from src.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.errors import DocumentIssue, RunInProgress, VersionLimitReached
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.outcome import PlanReady
from src.studio.planner.outcome import RunRefused as PlanRefused
from src.studio.planner.outcome import RunRejected as PlanRejected
from src.studio.run_failures import RunFailure, failure_of_error, failure_of_outcome
from tests.studio_llm import make_draft

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LlmUnavailable("x"), "llm_unavailable"),
        (LlmRejected(400), "llm_rejected"),
        (LlmOutputTruncated("x"), "llm_output_truncated"),
        (LlmOutputInvalid("x"), "llm_output_invalid"),
        (VersionLimitReached(), "version_limit_reached"),
        (RunInProgress(), "run_in_progress"),
        (RuntimeError("boom"), "internal_error"),
        (ValueError("anything"), "internal_error"),
        (PlannerStateError("validate ran before draft"), "internal_error"),
        # the run's own deadline is recognised where it is applied, not by its type
        (TimeoutError(), "internal_error"),
    ],
)
def test_errors_map_to_codes(error: Exception, code: str) -> None:
    assert failure_of_error(error).code == code


def test_a_rejection_carries_only_the_status_code() -> None:
    failure = failure_of_error(LlmRejected(422))
    assert failure.detail == {"status_code": 422}


def test_an_unexpected_error_carries_no_detail() -> None:
    assert failure_of_error(RuntimeError("the brief says SECRET")) == RunFailure("internal_error")


def test_a_planner_state_error_is_internal_and_carries_no_detail() -> None:
    assert failure_of_error(PlannerStateError("revise ran before draft")) == RunFailure(
        "internal_error"
    )
    assert issubclass(PlannerStateError, RuntimeError)


def test_a_paid_for_reply_error_is_still_the_same_code() -> None:
    """Carrying usage changes nothing about how the error is reported."""
    from src.llm.port import LlmUsage

    usage = LlmUsage(prompt_tokens=5, completion_tokens=6)
    assert failure_of_error(LlmOutputInvalid("x", usage=usage)).code == "llm_output_invalid"
    assert failure_of_error(LlmOutputTruncated("x", usage=usage)).code == "llm_output_truncated"


# -- failure_of_outcome over the three variants ---------------------------------------------------


def test_a_ready_outcome_is_not_a_failure() -> None:
    plan_ready = PlanReady(plan=_a_plan())
    assert failure_of_outcome(plan_ready) is None


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (DraftRefusal.CONTENT_POLICY, "content_policy_refused"),
        (DraftRefusal.BRIEF_UNUSABLE, "brief_unusable"),
    ],
)
def test_refusals(reason: DraftRefusal, code: str) -> None:
    assert failure_of_outcome(PlanRefused(reason)) == RunFailure(code)


def test_a_rejected_plan_lists_paths_and_codes_only() -> None:
    issues = (DocumentIssue("sections[1].delivery[0]", "reserved_descriptor"),)
    failure = failure_of_outcome(PlanRejected(issues))
    assert failure == RunFailure(
        "plan_invalid",
        {"issues": [{"path": "sections[1].delivery[0]", "code": "reserved_descriptor"}]},
    )


def test_every_outcome_variant_is_covered() -> None:
    """The union has exactly the three variants the mapping handles."""
    from src.studio.planner import outcome

    assert set(outcome.PlanOutcome.__value__.__args__) == {PlanReady, PlanRefused, PlanRejected}


def _a_plan():  # type: ignore[no-untyped-def]
    from src.studio.planner.mapping import plan_from_draft
    from src.studio.planner.validate import validate_plan

    plan, issues = validate_plan(plan_from_draft(make_draft(), brief="a brief"))
    assert plan is not None
    assert not issues
    return plan


# -- S1-P2a-B: lyrics and revise_lyrics -----------------------------------------------------------


def test_a_lyrics_outcome_that_is_ready_is_not_a_failure() -> None:
    from src.studio.planner.outcome import LyricsReady
    from tests.studio_lyrics_support import doc

    assert failure_of_outcome(LyricsReady(doc())) is None


@pytest.mark.parametrize("kind", ["lyrics", "revise_lyrics"])
def test_a_rejected_lyrics_run_lists_paths_and_codes_only(kind: str) -> None:
    from src.core.enums import StudioRunKind

    issues = (DocumentIssue("sections[2].lines[0].text", "reserved_characters"),)
    failure = failure_of_outcome(PlanRejected(issues, StudioRunKind(kind)))
    assert failure == RunFailure(
        "lyrics_invalid",
        {"issues": [{"path": "sections[2].lines[0].text", "code": "reserved_characters"}]},
    )


def test_a_refusal_is_not_tied_to_the_kind_that_asked() -> None:
    assert failure_of_outcome(PlanRefused(DraftRefusal.CONTENT_POLICY)) == RunFailure(
        "content_policy_refused"
    )
    assert failure_of_outcome(PlanRefused(DraftRefusal.BRIEF_UNUSABLE)) == RunFailure(
        "brief_unusable"
    )


def test_the_run_outcome_union_is_the_four_variants_the_mapping_handles() -> None:
    from src.studio.planner import outcome
    from src.studio.planner.outcome import LyricsReady

    assert set(outcome.RunOutcome.__value__.__args__) == {
        PlanReady,
        LyricsReady,
        PlanRefused,
        PlanRejected,
    }
    assert set(outcome.LyricsOutcome.__value__.__args__) == {LyricsReady, PlanRefused, PlanRejected}


def test_a_rejection_defaults_to_a_plan_so_existing_callers_are_unchanged() -> None:
    from src.core.enums import StudioRunKind

    assert PlanRejected(()).kind is StudioRunKind.PLAN
