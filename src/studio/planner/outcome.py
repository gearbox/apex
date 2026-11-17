"""How a planner graph ended, as a tagged union - so "ok" cannot exist without a document.

Plain frozen dataclasses with no ``langgraph`` import: the run failure mapping (``run_failures.py``)
matches on them, and a process that never runs the planner must not have to load the graph library.
The graphs re-export the ones they return.

One refusal type and one rejection type serve every run kind (P2a-48); only the success carries the
kind's own document (:class:`PlanReady`, :class:`LyricsReady`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.core.enums import StudioRunKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.errors import DocumentIssue
    from src.studio.planner.drafts import DraftRefusal

__all__ = [
    "MAX_WARNINGS",
    "WARNINGS_TRUNCATED",
    "LyricsOutcome",
    "LyricsReady",
    "LyricsRun",
    "PlanOutcome",
    "PlanReady",
    "PlanRun",
    "RunOutcome",
    "RunRefused",
    "RunRejected",
    "RunWarning",
    "cap_warnings",
]

MAX_WARNINGS: Final = 100
"""Most warnings stored on one run (the marker below included)."""
WARNINGS_TRUNCATED: Final = "warnings_truncated"
"""Last warning of a list that had to be cut at :data:`MAX_WARNINGS`."""


@dataclass(frozen=True, slots=True)
class RunWarning:
    """A non-fatal note about a succeeded run: a stable code, where it applies, and a number.

    Codes, paths and numbers only - never text from a document (the avoid word that matched is not
    recorded).
    """

    code: str
    path: str | None = None
    count: int | None = None

    def to_json(self) -> dict[str, object]:
        """The stored form: only the fields that are set."""
        stored: dict[str, object] = {"code": self.code}
        if self.path is not None:
            stored["path"] = self.path
        if self.count is not None:
            stored["count"] = self.count
        return stored


def cap_warnings(warnings: Sequence[RunWarning]) -> list[RunWarning]:
    """At most :data:`MAX_WARNINGS` warnings; a longer list ends with ``warnings_truncated``."""
    if len(warnings) <= MAX_WARNINGS:
        return list(warnings)
    return [*warnings[: MAX_WARNINGS - 1], RunWarning(WARNINGS_TRUNCATED)]


@dataclass(frozen=True, slots=True)
class PlanReady:
    """The model's draft became a valid plan."""

    plan: PlanDocument


@dataclass(frozen=True, slots=True)
class LyricsReady:
    """The model's draft became valid lyrics; ``warnings`` are the soft misses that remain.

    Warning paths index **this** document (``sections[i].lines[j]``), which is exactly the document
    that is saved.
    """

    lyrics: LyricsDocument
    warnings: tuple[RunWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class RunRefused:
    """The model declined, with its own reason."""

    reason: DraftRefusal


@dataclass(frozen=True, slots=True)
class RunRejected:
    """No draft validated within the budget; these are the last draft's issues.

    ``kind`` says whose rule failed, so the run is failed with that kind's own error code
    (``plan_invalid`` / ``lyrics_invalid``). A plan is the default.
    """

    issues: tuple[DocumentIssue, ...]
    kind: StudioRunKind = StudioRunKind.PLAN


type PlanOutcome = PlanReady | RunRefused | RunRejected
type LyricsOutcome = LyricsReady | RunRefused | RunRejected
type RunOutcome = PlanReady | LyricsReady | RunRefused | RunRejected


@dataclass(frozen=True, slots=True)
class PlanRun:
    """What a finished plan graph hands back: the outcome and what the model cost."""

    outcome: PlanOutcome
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class LyricsRun:
    """What a finished lyrics or revise-lyrics graph hands back: the outcome and what it cost."""

    outcome: LyricsOutcome
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
