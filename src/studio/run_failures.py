"""Why a run failed: the one place errors and outcomes become a run's ``error_code``.

Pure. A failure carries a stable code and, at most, structured detail made of **paths, codes and
numbers** - never model output or user text (the contract promises it, ``RunOut`` serves it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.studio.errors import StudioError
from src.studio.llm.port import (
    LlmOutputInvalid,
    LlmOutputTruncated,
    LlmRejected,
    LlmUnavailable,
)
from src.studio.planner.drafts import DraftRefusal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.studio.errors import DocumentIssue
    from src.studio.planner.graphs.plan import PlanOutcome

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
    """Map an exception raised while running to a failure (``internal_error`` for the rest)."""
    if isinstance(exc, TimeoutError):
        return RunFailure(ERR_TIMEOUT)
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


def failure_of_outcome(
    outcome: PlanOutcome | None, issues: Sequence[DocumentIssue]
) -> RunFailure | None:
    """The failure a finished graph stands for; ``None`` when it produced a valid plan."""
    if outcome is None:
        return RunFailure(ERR_INTERNAL)
    if outcome.status == "ok":
        return None
    if outcome.status == "refused":
        if outcome.refusal is DraftRefusal.BRIEF_UNUSABLE:
            return RunFailure(ERR_BRIEF_UNUSABLE)
        return RunFailure(ERR_CONTENT_POLICY)
    return RunFailure(ERR_PLAN_INVALID, _issues_detail(issues))
