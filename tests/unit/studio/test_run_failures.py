"""The run failure table: errors and graph outcomes -> a run's ``error_code``."""

from __future__ import annotations

import pytest

from src.studio.errors import DocumentIssue, RunInProgress, VersionLimitReached
from src.studio.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.graphs.plan import PlanOutcome
from src.studio.run_failures import RunFailure, failure_of_error, failure_of_outcome

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LlmUnavailable("x"), "llm_unavailable"),
        (LlmRejected(400), "llm_rejected"),
        (LlmOutputTruncated("x"), "llm_output_truncated"),
        (LlmOutputInvalid("x"), "llm_output_invalid"),
        (TimeoutError(), "run_timeout"),
        (VersionLimitReached(), "version_limit_reached"),
        (RunInProgress(), "run_in_progress"),
        (RuntimeError("boom"), "internal_error"),
        (ValueError("anything"), "internal_error"),
    ],
)
def test_errors_map_to_codes(error: Exception, code: str) -> None:
    assert failure_of_error(error).code == code


def test_a_rejection_carries_only_the_status_code() -> None:
    failure = failure_of_error(LlmRejected(422))
    assert failure.detail == {"status_code": 422}


def test_an_unexpected_error_carries_no_detail() -> None:
    assert failure_of_error(RuntimeError("the brief says SECRET")) == RunFailure("internal_error")


def test_a_valid_outcome_is_not_a_failure() -> None:
    assert failure_of_outcome(PlanOutcome("ok"), ()) is None


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (DraftRefusal.CONTENT_POLICY, "content_policy_refused"),
        (DraftRefusal.BRIEF_UNUSABLE, "brief_unusable"),
    ],
)
def test_refusals(reason: DraftRefusal, code: str) -> None:
    assert failure_of_outcome(PlanOutcome("refused", reason), ()) == RunFailure(code)


def test_an_invalid_plan_lists_paths_and_codes_only() -> None:
    issues = (DocumentIssue("sections[1].delivery[0]", "reserved_descriptor"),)
    failure = failure_of_outcome(PlanOutcome("invalid"), issues)
    assert failure == RunFailure(
        "plan_invalid",
        {"issues": [{"path": "sections[1].delivery[0]", "code": "reserved_descriptor"}]},
    )


def test_a_missing_outcome_is_an_internal_error() -> None:
    assert failure_of_outcome(None, ()) == RunFailure("internal_error")
