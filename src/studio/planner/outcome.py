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

from src.studio.planner.drafts import DraftRefusal
from src.studio.planner.memories import memory_ids_for_keys

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from src.core.enums import StudioRunKind
    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.errors import DocumentIssue
    from src.studio.planner.context import MemorySnapshot

__all__ = [
    "MAX_WARNINGS",
    "PINNED_MEMORY_UNAVAILABLE",
    "REVISION_INCOMPLETE",
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
    "kept_after_failed_round",
    "lead_with",
    "refused",
]

MAX_WARNINGS: Final = 100
"""Most warnings stored on one run (the marker below included)."""
WARNINGS_TRUNCATED: Final = "warnings_truncated"
"""Last warning of a list that had to be cut at :data:`MAX_WARNINGS`."""
REVISION_INCOMPLETE: Final = "revision_incomplete"
"""Run warning: a later fix round failed, so the run kept its best valid result."""
PINNED_MEMORY_UNAVAILABLE: Final = "pinned_memory_unavailable"
"""Run warning: pinned artist memories were archived, deleted or too large by the time the run
started, so they were left out (``count`` = how many)."""


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


def cap_warnings(warnings: Sequence[RunWarning], *, limit: int = MAX_WARNINGS) -> list[RunWarning]:
    """At most ``limit`` warnings; a longer list ends with ``warnings_truncated``."""
    if len(warnings) <= limit:
        return list(warnings)
    return [*warnings[: limit - 1], RunWarning(WARNINGS_TRUNCATED)]


def lead_with(warning: RunWarning, warnings: Sequence[RunWarning]) -> list[RunWarning]:
    """``warning`` first, then ``warnings``, still within :data:`MAX_WARNINGS`.

    A list that is already full gives up an entry to make room - never ``warning`` (it is first),
    and never the marker a full list ends with (``warnings_truncated`` / ``revision_incomplete``:
    the closing entry stays last). Where nothing marks the cut, ``warnings_truncated`` does.
    """
    if len(warnings) < MAX_WARNINGS:
        return [warning, *warnings]
    *body, last = warnings
    if last.code in (WARNINGS_TRUNCATED, REVISION_INCOMPLETE):
        return [warning, *body[:-1], last]
    return cap_warnings([warning, *warnings])


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
    """The model declined, with its own reason.

    ``memory_ids`` are the artist memories the model named as the cause of a ``content_policy``
    refusal, in prompt order: best-effort (it may be empty, and an id may be gone by the time the
    run is read). Built only for ``content_policy``; ``()`` for every other reason.
    """

    reason: DraftRefusal
    memory_ids: tuple[UUID, ...] = ()


def refused(
    reason: DraftRefusal, keys: Sequence[str], memories: Sequence[MemorySnapshot]
) -> RunRefused:
    """The refusal a draft stands for.

    Args:
        reason: The draft's ``refusal``.
        keys: The draft's ``refusal_memories``: the ``m<n>`` keys the model named.
        memories: The memories the run was shown, in prompt order (``context.memories``).

    Returns:
        A :class:`RunRefused`; for ``content_policy`` it carries the ids of the named memories
        (unknown or malformed keys and repeats dropped), for any other reason it carries none, so
        keys a model offers next to ``brief_unusable`` are ignored.
    """
    if reason is DraftRefusal.CONTENT_POLICY:
        return RunRefused(reason, memory_ids_for_keys(keys, memories))
    return RunRefused(reason)


@dataclass(frozen=True, slots=True)
class RunRejected:
    """No draft validated within the budget; these are the last draft's issues.

    ``kind`` says whose rule failed, so the run is failed with that kind's own error code
    (``plan_invalid`` / ``lyrics_invalid``, see ``run_failures``). It is required: a forgotten
    argument must not silently report a lyrics run as ``plan_invalid``.
    """

    issues: tuple[DocumentIssue, ...]
    kind: StudioRunKind


def kept_after_failed_round(lyrics: LyricsDocument, warnings: Sequence[RunWarning]) -> LyricsReady:
    """The result of a run whose last fix round failed at the port, with a valid document in hand.

    ``warnings`` are the soft misses of ``lyrics``. ``revision_incomplete`` is added **last** and is
    never the one a long list cuts: the others are capped one short of :data:`MAX_WARNINGS`.
    """
    return LyricsReady(
        lyrics,
        (*cap_warnings(warnings, limit=MAX_WARNINGS - 1), RunWarning(REVISION_INCOMPLETE)),
    )


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
