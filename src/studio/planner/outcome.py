"""How a plan graph ended, as a tagged union - so "ok" cannot exist without a plan.

Plain frozen dataclasses with no ``langgraph`` import: the run failure mapping (``run_failures.py``)
matches on them, and a process that never runs the planner must not have to load the graph library.
``graphs/plan.py`` re-exports them for the code that drives the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.studio.documents import PlanDocument
    from src.studio.errors import DocumentIssue
    from src.studio.planner.drafts import DraftRefusal

__all__ = ["PlanOutcome", "PlanReady", "PlanRefused", "PlanRejected", "PlanRun"]


@dataclass(frozen=True, slots=True)
class PlanReady:
    """The model's draft became a valid plan."""

    plan: PlanDocument


@dataclass(frozen=True, slots=True)
class PlanRefused:
    """The model declined to plan, with its own reason."""

    reason: DraftRefusal


@dataclass(frozen=True, slots=True)
class PlanRejected:
    """No draft validated within the revision budget; these are the last draft's issues."""

    issues: tuple[DocumentIssue, ...]


type PlanOutcome = PlanReady | PlanRefused | PlanRejected


@dataclass(frozen=True, slots=True)
class PlanRun:
    """What a finished plan graph hands back: the outcome and what the model cost."""

    outcome: PlanOutcome
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
