"""The ``revise_lyrics`` graph: revise -> check -> (fix -> check)* -> done.

An in-memory LangGraph state machine over the planner's ports; **it never touches the database**.
The model rewrites only what was targeted (whole sections and/or single lines); everything else in
the input lyrics is carried over untouched, and only the lines the model *changed* are checked
(syllables, avoid-words): the user's own lines are not the model's to judge.

Budget: at most :data:`MAX_LLM_CALLS` model calls - the revision, then at most :data:`MAX_FIXES`
fix rounds. A fix is a *patch* (replacements for exactly the changed lines that were rejected, one
for one, everything else frozen) or, when the reply broke the targets (something unexpected, or a
target missing), one more full revision. After the budget a hard issue fails the run
(``lyrics_invalid``); soft ones become warnings (P2a-44).

A fix round that fails at the port never discards a validated revision: when only soft problems
were left, the run ends with that revision and the warning ``revision_incomplete``; with a hard
problem there is nothing usable and the error propagates (R1). A run that the 180 s deadline cancels
mid-round still ends ``run_timeout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import msgspec
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime  # noqa: TC002 - runtime: langgraph inspects node signatures

from src.core.enums import StudioRunKind, StudioRunStage
from src.llm.port import LlmError, LlmMessage
from src.studio.documents import (  # noqa: TC001 - runtime: LangGraph reads the state hints
    LyricsDocument,
    PlanDocument,
    SyllableRange,
)
from src.studio.errors import DocumentIssue  # noqa: TC001
from src.studio.planner.context import RunContextSnapshot  # noqa: TC001
from src.studio.planner.drafts import LyricsPatchDraft, LyricsRevisionDraft
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.ask import ask_structured
from src.studio.planner.lyric_checks import LineRef, warnings_of
from src.studio.planner.lyrics_mapping import (
    apply_patch,
    apply_revision,
    plan_positions,
    revision_draft_of,
)
from src.studio.planner.lyrics_review import Findings, review
from src.studio.planner.outcome import (
    LyricsOutcome,
    LyricsReady,
    LyricsRun,
    RunRejected,
    cap_warnings,
    kept_after_failed_round,
    refused,
)
from src.studio.planner.prompts.revise_lyrics_v2 import (
    build_revise_fix_messages,
    build_revise_messages,
    build_revise_redraft_messages,
)
from src.studio.planner.syllables import counter_for
from src.studio.run_requests import ReviseLyricsRunRequest  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from langgraph.graph.state import CompiledStateGraph

    from src.llm.port import StructuredLlmPort

__all__ = [
    "MAX_FIXES",
    "MAX_LLM_CALLS",
    "PATCH_SCHEMA_NAME",
    "REVISION_SCHEMA_NAME",
    "ReviseDeps",
    "ReviseState",
    "build_revise_graph",
    "compile_revise_graph",
    "initial_state",
    "run_revise_graph",
]

MAX_LLM_CALLS: Final = 3
"""Model calls per run, counting re-asks (a transport retry inside the adapter is not counted)."""
MAX_FIXES: Final = 2
REVISION_SCHEMA_NAME: Final = "lyrics_revision"
PATCH_SCHEMA_NAME: Final = "lyrics_patch"
_RECURSION_LIMIT: Final = 12
"""A full run is six supersteps (revise, check, then twice fix and check)."""


@dataclass(frozen=True, slots=True)
class ReviseDeps:
    """What the nodes may use besides their state: the model port and a stage callback."""

    llm: StructuredLlmPort
    on_stage: Callable[[StudioRunStage], Awaitable[None]]
    """Awaited when a stage starts; the run records it (and may stop the run)."""
    max_fixes: int = MAX_FIXES


class ReviseState(TypedDict):
    """The graph state. ``messages`` is the opening conversation, kept for the follow-up turns."""

    context: RunContextSnapshot
    request: ReviseLyricsRunRequest
    plan: PlanDocument
    lyrics: LyricsDocument
    """The input lyrics - valid, and never modified."""
    line_syllables: SyllableRange | None
    messages: tuple[LlmMessage, ...]
    echo: LyricsRevisionDraft | None
    """The model's work as it stands (accepted lines frozen) - what a follow-up shows it."""
    document: LyricsDocument | None
    """``lyrics`` with the model's work merged in; built, not yet validated."""
    touched: frozenset[LineRef]
    """The lines the model wrote."""
    mapping_issues: tuple[DocumentIssue, ...]
    findings: Findings | None
    fixes: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    outcome: LyricsOutcome | None


def initial_state(
    context: RunContextSnapshot,
    request: ReviseLyricsRunRequest,
    plan: PlanDocument,
    lyrics: LyricsDocument,
    *,
    line_syllables: SyllableRange | None,
) -> ReviseState:
    """The state a run starts from."""
    return ReviseState(
        context=context,
        request=request,
        plan=plan,
        lyrics=lyrics,
        line_syllables=line_syllables,
        messages=(),
        echo=None,
        document=None,
        touched=frozenset(),
        mapping_issues=(),
        findings=None,
        fixes=0,
        llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        outcome=None,
    )


def _as_is(reply: LyricsRevisionDraft) -> LyricsRevisionDraft:
    return reply


def _merged(state: ReviseState, reply: LyricsRevisionDraft) -> dict[str, Any]:
    """The state a full reply leaves: the reply and the input lyrics with it merged in."""
    document, touched, issues = apply_revision(
        state["lyrics"], reply, state["request"].targets, state["plan"]
    )
    return {
        "echo": reply,
        # The saved document carries the range this run used (request, else the input's own).
        "document": msgspec.structs.replace(document, line_syllables=state["line_syllables"]),
        "touched": touched,
        "mapping_issues": issues,
    }


async def _revise(state: ReviseState, runtime: Runtime[ReviseDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.DRAFTING)
    messages = tuple(
        build_revise_messages(
            state["context"],
            state["request"],
            state["plan"],
            state["lyrics"],
            line_syllables=state["line_syllables"],
        )
    )
    asked = await ask_structured(
        deps.llm,
        messages,
        schema=LyricsRevisionDraft,
        schema_name=REVISION_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=_as_is,
    )
    return {
        "messages": messages,
        **_merged(state, asked.value),
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }


async def _check(state: ReviseState, runtime: Runtime[ReviseDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.VALIDATING)
    echo = state["echo"]
    document = state["document"]
    if echo is None or document is None:
        raise PlannerStateError("check ran before the revision")
    if echo.refusal is not None:
        refusal = refused(echo.refusal, echo.refusal_memories, state["context"].memories)
        return {"findings": None, "outcome": refusal}
    findings = review(
        document,
        state["plan"],
        structural=state["mapping_issues"],
        counter=counter_for(state["lyrics"].language),
        syllables=state["line_syllables"],
        avoid=state["request"].avoid_words,
        only=state["touched"],  # the user's own lines are not the model's to judge
    )
    return {"findings": findings, "outcome": _outcome(state, findings, deps)}


def _outcome(state: ReviseState, findings: Findings, deps: ReviseDeps) -> LyricsOutcome | None:
    """``None`` when another round is both needed and affordable."""
    if findings.clean:
        return _ready(findings)
    if state["fixes"] < deps.max_fixes and state["llm_calls"] < MAX_LLM_CALLS:
        return None
    if findings.hard:
        return RunRejected(findings.hard, StudioRunKind.REVISE_LYRICS)
    return _ready(findings)  # only soft misses are left: a paid, usable revision is kept (P2a-44)


def _ready(findings: Findings) -> LyricsReady:
    if findings.lyrics is None or findings.report is None:
        raise PlannerStateError("the revision was ready without a validated document")
    return LyricsReady(
        findings.lyrics, tuple(cap_warnings(warnings_of(findings.report, findings.lyrics)))
    )


async def _fix(state: ReviseState, runtime: Runtime[ReviseDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.REVISING)
    findings = state["findings"]
    if findings is None or state["echo"] is None or state["document"] is None:
        raise PlannerStateError("fix ran before check")
    try:
        if findings.structural:
            update = await _again(state, deps, findings)  # the reply broke the targets
        else:
            update = await _patch(state, deps, findings)
    except LlmError:
        if findings.lyrics is None or findings.report is None:
            raise  # a hard problem is open: there is no valid result to keep
        # The round was only trying to improve a revision that is already usable.
        return {
            "outcome": kept_after_failed_round(
                findings.lyrics, warnings_of(findings.report, findings.lyrics)
            )
        }
    return {"fixes": state["fixes"] + 1, **update}


async def _again(state: ReviseState, deps: ReviseDeps, findings: Findings) -> dict[str, Any]:
    """Ask for the whole revision again, with every hard problem named (merged from the input again)."""
    echo = state["echo"]
    if echo is None:
        raise PlannerStateError("re-ask before a revision")
    asked = await ask_structured(
        deps.llm,
        (*state["messages"], *build_revise_redraft_messages(echo, findings.hard)),
        schema=LyricsRevisionDraft,
        schema_name=REVISION_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=_as_is,
    )
    return {
        **_merged(state, asked.value),
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }


async def _patch(state: ReviseState, deps: ReviseDeps, findings: Findings) -> dict[str, Any]:
    """Ask for replacements of exactly the changed lines with a problem; everything else frozen."""
    plan = state["plan"]
    echo = state["echo"]
    document = state["document"]
    if echo is None or document is None:
        raise PlannerStateError("patch before a revision")
    section_at = {position: section_id for section_id, position in plan_positions(plan).items()}
    requested = {LineRef(section_at[p.plan_index], p.line) for p in findings.problems}
    problems = tuple(sorted(findings.problems, key=lambda p: (p.plan_index, p.line)))

    def accept(patch: LyricsPatchDraft) -> tuple[LyricsPatchDraft | None, LyricsDocument | None]:
        if patch.refusal is not None:
            return patch, None  # the declining patch itself: its reason and its memory keys
        return None, apply_patch(document, patch, requested, plan)

    asked = await ask_structured(
        deps.llm,
        (
            *state["messages"],
            *build_revise_fix_messages(echo, problems, state["line_syllables"]),
        ),
        schema=LyricsPatchDraft,
        schema_name=PATCH_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=accept,
    )
    declined, patched = asked.value
    spent = {
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }
    if declined is not None:
        # Declining mid-run ends the run like declining at the start: the check reads it off the echo.
        declining = LyricsRevisionDraft(
            refusal=declined.refusal,
            refusal_memories=declined.refusal_memories,
            sections=[],
            lines=[],
        )
        return {"echo": declining, **spent}
    if patched is None:
        raise PlannerStateError("a patch was accepted without a document")
    return {
        "echo": revision_draft_of(patched, state["request"].targets, plan),
        "document": patched,
        "mapping_issues": (),
        **spent,
    }


def _route(state: ReviseState) -> Literal["fix", "done"]:
    return "done" if state["outcome"] is not None else "fix"


def _after_fix(state: ReviseState) -> Literal["check", "done"]:
    """A round that failed at the port has already set the outcome (R1); otherwise check again."""
    return "done" if state["outcome"] is not None else "check"


def build_revise_graph() -> StateGraph[ReviseState, ReviseDeps, ReviseState, ReviseState]:
    """The (uncompiled) revise-lyrics graph."""
    graph: StateGraph[ReviseState, ReviseDeps, ReviseState, ReviseState] = StateGraph(
        ReviseState, context_schema=ReviseDeps
    )
    graph.add_node("revise", _revise)
    graph.add_node("check", _check)
    graph.add_node("fix", _fix)
    graph.add_edge(START, "revise")
    graph.add_edge("revise", "check")
    graph.add_conditional_edges("check", _route, {"fix": "fix", "done": END})
    graph.add_conditional_edges("fix", _after_fix, {"check": "check", "done": END})
    return graph


def compile_revise_graph() -> CompiledStateGraph[ReviseState, ReviseDeps, ReviseState, ReviseState]:
    """Compile the graph (do this once, at wiring time; the result is reusable and stateless)."""
    return build_revise_graph().compile()


async def run_revise_graph(
    compiled: CompiledStateGraph[ReviseState, ReviseDeps, ReviseState, ReviseState],
    state: ReviseState,
    deps: ReviseDeps,
) -> LyricsRun:
    """Run a compiled revise-lyrics graph to its end.

    Raises:
        LlmError: Any port error (see ``src/llm/port.py``); nothing is swallowed here.
        PlannerStateError: The graph ended without an outcome (its wiring should prevent it).
    """
    final = cast(
        "ReviseState",
        await compiled.ainvoke(state, context=deps, config={"recursion_limit": _RECURSION_LIMIT}),
    )
    outcome = final["outcome"]
    if outcome is None:
        raise PlannerStateError("the revise-lyrics graph ended without an outcome")
    return LyricsRun(
        outcome=outcome,
        llm_calls=final["llm_calls"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
    )
