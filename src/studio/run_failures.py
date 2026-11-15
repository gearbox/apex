"""Why a run failed: the one place errors and outcomes become a run's ``error_code``.

Pure. A failure carries a stable code and, at most, structured detail made of **paths, codes and
numbers** - never model output or user text (the contract promises it, ``RunOut`` serves it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, overload

from src.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.errors import StudioError
from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.outcome import PlanReady, PlanRefused, PlanRejected

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.studio.errors import DocumentIssue
    from src.studio.planner.outcome import PlanOutcome

__all__ = [
    "ERR_BRIEF_UNUSABLE",
    "ERR_CONTENT_POLICY",
    "ERR_INTERNAL",
    "ERR_LLM_INVALID",
    "ERR_LLM_REJECTED",
    "ERR_LLM_TRUNCATED",
    "ERR_LLM_UNAVAILABLE",
    "ERR_PLAN_INVALID",
    "ERR_TIMEOUT",
    "RunFailure",
    "failure_of_error",
    "failure_of_outcome",
]

ERR_LLM_UNAVAILABLE: Final = "llm_unavailable"
ERR_LLM_REJECTED: Final = "llm_rejected"
ERR_LLM_TRUNCATED: Final = "llm_output_truncated"
ERR_LLM_INVALID: Final = "llm_output_invalid"
ERR_CONTENT_POLICY: Final = "content_policy_refused"
ERR_BRIEF_UNUSABLE: Final = "brief_unusable"
ERR_PLAN_INVALID: Final = "plan_invalid"
ERR_TIMEOUT: Final = "run_timeout"
ERR_INTERNAL: Final = "internal_error"


@dataclass(frozen=True, slots=True)
class RunFailure:
    """A run's terminal failure."""

    code: str
    detail: dict[str, object] | None = None


def failure_of_error(exc: BaseException) -> RunFailure:
    """Map an exception raised while running to a failure (``internal_error`` for the rest).

    ``TimeoutError`` is deliberately **not** mapped: the run's own deadline is recognised where it
    is applied (around the graph only), so a timeout raised by anything else - the save step's
    database, say - is an ``internal_error`` and not a ``run_timeout`` that hides it.
    """
    if isinstance(exc, LlmUnavailable):
        return RunFailure(ERR_LLM_UNAVAILABLE)
    if isinstance(exc, LlmRejected):
        return RunFailure(ERR_LLM_REJECTED, {"status_code": exc.status_code})
    if isinstance(exc, LlmOutputTruncated):
        return RunFailure(ERR_LLM_TRUNCATED)
    if isinstance(exc, LlmOutputInvalid):
        return RunFailure(ERR_LLM_INVALID)
    if isinstance(exc, StudioError):
        return RunFailure(exc.code)
    return RunFailure(ERR_INTERNAL)


def _issues_detail(issues: Sequence[DocumentIssue]) -> dict[str, object]:
    return {"issues": [{"path": issue.path, "code": issue.code} for issue in issues]}


@overload
def failure_of_outcome(outcome: PlanReady) -> None: ...
@overload
def failure_of_outcome(outcome: PlanRefused | PlanRejected) -> RunFailure: ...
@overload
def failure_of_outcome(outcome: PlanOutcome) -> RunFailure | None: ...
def failure_of_outcome(outcome: PlanOutcome) -> RunFailure | None:
    """The failure a finished graph stands for; ``None`` when it produced a valid plan."""
    match outcome:
        case PlanReady():
            return None
        case PlanRefused(reason=DraftRefusal.BRIEF_UNUSABLE):
            return RunFailure(ERR_BRIEF_UNUSABLE)
        case PlanRefused():
            return RunFailure(ERR_CONTENT_POLICY)
        case PlanRejected(issues=issues):
            return RunFailure(ERR_PLAN_INVALID, _issues_detail(issues))
