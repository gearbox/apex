"""The ``lyrics`` graph: draft -> check -> (revise -> check)* -> done.

An in-memory LangGraph state machine over the planner's ports; **it never touches the database**.
It returns a :class:`LyricsRun` whose outcome is a tagged union - valid lyrics (with the soft
misses that remain as warnings), a refusal, or the last reply's hard issues.

Budget: at most :data:`MAX_LLM_CALLS` model calls per run - the first draft, then at most
:data:`MAX_REVISIONS` revision rounds; a reply that cannot be used is re-asked within the same
budget and consumes no round. A **round** is one of:

* a *patch*: the model returns replacements for exactly the lines that were rejected, one for one,
  and every accepted line stays frozen (P2a-43) - so the loop converges;
* a *full re-draft*, when something is wrong that no single line can fix (a missing section, an
  unknown key, a section over the line limit).

After the budget: a **hard** issue fails the run (``lyrics_invalid``); only **soft** ones (a missed
syllable range, an avoid-word) succeed with warnings (P2a-44).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime  # noqa: TC002 - runtime: langgraph inspects node signatures

from src.core.enums import StudioRunKind, StudioRunStage
from src.llm.port import LlmMessage
from src.studio.documents import (  # noqa: TC001 - runtime: LangGraph reads the state hints
    LyricsDocument,
    PlanDocument,
    SyllableRange,
)
from src.studio.errors import DocumentIssue  # noqa: TC001
from src.studio.planner.context import RunContextSnapshot  # noqa: TC001
from src.studio.planner.drafts import DraftRefusal, LyricsDraft, LyricsPatchDraft
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.ask import ask_structured
from src.studio.planner.lyric_checks import LineRef, warnings_of
from src.studio.planner.lyrics_mapping import (
    apply_patch,
    draft_of,
    lyrics_from_draft,
    plan_positions,
)
from src.studio.planner.lyrics_review import Findings, review
from src.studio.planner.outcome import (
    LyricsOutcome,
    LyricsReady,
    LyricsRun,
    RunRefused,
    RunRejected,
    cap_warnings,
)
from src.studio.planner.prompts.lyrics_v1 import (
    build_lyrics_messages,
    build_lyrics_patch_messages,
    build_lyrics_redraft_messages,
)
from src.studio.planner.syllables import counter_for
from src.studio.run_requests import LyricsRunRequest  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from langgraph.graph.state import CompiledStateGraph

    from src.llm.port import LlmMessage, StructuredLlmPort

__all__ = [
    "LYRICS_SCHEMA_NAME",
    "MAX_LLM_CALLS",
    "MAX_REVISIONS",
    "PATCH_SCHEMA_NAME",
    "LyricsDeps",
    "LyricsState",
    "build_lyrics_graph",
    "compile_lyrics_graph",
    "initial_state",
    "run_lyrics_graph",
]

MAX_LLM_CALLS: Final = 5
"""Model calls per run, counting re-asks (a transport retry inside the adapter is not counted)."""
MAX_REVISIONS: Final = 3
LYRICS_SCHEMA_NAME: Final = "lyrics_draft"
PATCH_SCHEMA_NAME: Final = "lyrics_patch"
_RECURSION_LIMIT: Final = 16
"""A full run is eight supersteps (draft, check, then three times revise and check)."""


@dataclass(frozen=True, slots=True)
class LyricsDeps:
    """What the nodes may use besides their state: the model port and a stage callback."""

    llm: StructuredLlmPort
    on_stage: Callable[[StudioRunStage], Awaitable[None]]
    """Awaited when a stage starts; the run records it (and may stop the run)."""
    max_revisions: int = MAX_REVISIONS


class LyricsState(TypedDict):
    """The graph state. ``messages`` is the opening conversation, kept for the follow-up turns."""

    context: RunContextSnapshot
    request: LyricsRunRequest
    plan: PlanDocument
    language: str
    line_syllables: SyllableRange | None
    messages: tuple[LlmMessage, ...]
    draft: LyricsDraft | None
    """The model's work as it stands (accepted lines frozen) - what a follow-up shows it."""
    document: LyricsDocument | None
    """The same work as a built, not yet validated, document."""
    mapping_issues: tuple[DocumentIssue, ...]
    """Structural problems found while building ``document`` (plan-space paths)."""
    findings: Findings | None
    revisions: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    outcome: LyricsOutcome | None


def initial_state(
    context: RunContextSnapshot,
    request: LyricsRunRequest,
    plan: PlanDocument,
    *,
    language: str,
    line_syllables: SyllableRange | None,
) -> LyricsState:
    """The state a run starts from."""
    return LyricsState(
        context=context,
        request=request,
        plan=plan,
        language=language,
        line_syllables=line_syllables,
        messages=(),
        draft=None,
        document=None,
        mapping_issues=(),
        findings=None,
        revisions=0,
        llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        outcome=None,
    )


def _as_is(draft: LyricsDraft) -> LyricsDraft:
    return draft


def _built(state: LyricsState, draft: LyricsDraft) -> dict[str, Any]:
    """The state a full reply leaves: the draft and the document built from it."""
    document, issues = lyrics_from_draft(
        draft,
        state["plan"],
        language=state["language"],
        line_syllables=state["line_syllables"],
    )
    return {"draft": draft, "document": document, "mapping_issues": issues}


async def _draft(state: LyricsState, runtime: Runtime[LyricsDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.DRAFTING)
    messages = tuple(
        build_lyrics_messages(
            state["context"],
            state["request"],
            state["plan"],
            language=state["language"],
            line_syllables=state["line_syllables"],
        )
    )
    asked = await ask_structured(
        deps.llm,
        messages,
        schema=LyricsDraft,
        schema_name=LYRICS_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=_as_is,
    )
    return {
        "messages": messages,
        **_built(state, asked.value),
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }


async def _check(state: LyricsState, runtime: Runtime[LyricsDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.VALIDATING)
    draft = state["draft"]
    document = state["document"]
    if draft is None or document is None:
        raise PlannerStateError("check ran before draft")
    if draft.refusal is not None:
        return {"findings": None, "outcome": RunRefused(draft.refusal)}
    findings = review(
        document,
        state["plan"],
        structural=state["mapping_issues"],
        counter=counter_for(state["language"]),
        syllables=state["line_syllables"],
        avoid=state["request"].avoid_words,
    )
    return {"findings": findings, "outcome": _outcome(state, findings, deps)}


def _outcome(state: LyricsState, findings: Findings, deps: LyricsDeps) -> LyricsOutcome | None:
    """``None`` when another round is both needed and affordable."""
    if findings.clean:
        return _ready(findings)
    if state["revisions"] < deps.max_revisions and state["llm_calls"] < MAX_LLM_CALLS:
        return None
    if findings.hard:
        return RunRejected(findings.hard, StudioRunKind.LYRICS)
    return _ready(findings)  # only soft misses are left: a paid, usable draft is kept (P2a-44)


def _ready(findings: Findings) -> LyricsReady:
    if findings.lyrics is None or findings.report is None:
        raise PlannerStateError("lyrics were ready without a validated document")
    return LyricsReady(
        findings.lyrics, tuple(cap_warnings(warnings_of(findings.report, findings.lyrics)))
    )


async def _revise(state: LyricsState, runtime: Runtime[LyricsDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.REVISING)
    findings = state["findings"]
    if findings is None or state["draft"] is None or state["document"] is None:
        raise PlannerStateError("revise ran before check")
    if findings.structural:
        update = await _redraft(state, deps, findings)  # no single line can fix this
    else:
        update = await _patch(state, deps, findings)
    return {"revisions": state["revisions"] + 1, **update}


async def _redraft(state: LyricsState, deps: LyricsDeps, findings: Findings) -> dict[str, Any]:
    """Ask for the whole reply again, with every hard problem named (not only the structural one:
    a full reply is the model's one chance to fix them all)."""
    draft = state["draft"]
    if draft is None:
        raise PlannerStateError("re-draft before a draft")
    asked = await ask_structured(
        deps.llm,
        (*state["messages"], *build_lyrics_redraft_messages(draft, findings.hard)),
        schema=LyricsDraft,
        schema_name=LYRICS_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=_as_is,
    )
    return {
        **_built(state, asked.value),
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }


async def _patch(state: LyricsState, deps: LyricsDeps, findings: Findings) -> dict[str, Any]:
    """Ask for replacements of exactly the lines with a problem; every other line is frozen."""
    plan = state["plan"]
    draft = state["draft"]
    document = state["document"]
    if draft is None or document is None:
        raise PlannerStateError("patch before a draft")
    section_at = {position: section_id for section_id, position in plan_positions(plan).items()}
    requested = {LineRef(section_at[p.plan_index], p.line) for p in findings.problems}
    problems = tuple(sorted(findings.problems, key=lambda p: (p.plan_index, p.line)))

    def accept(patch: LyricsPatchDraft) -> tuple[DraftRefusal | None, LyricsDocument | None]:
        if patch.refusal is not None:
            return patch.refusal, None
        return None, apply_patch(document, patch, requested, plan)

    asked = await ask_structured(
        deps.llm,
        (
            *state["messages"],
            *build_lyrics_patch_messages(draft, problems, state["line_syllables"]),
        ),
        schema=LyricsPatchDraft,
        schema_name=PATCH_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=accept,
    )
    refusal, patched = asked.value
    spent = {
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }
    if refusal is not None:
        # Declining mid-run ends the run like declining at the start: the check reads it off the draft.
        return {"draft": LyricsDraft(refusal=refusal, sections=[]), **spent}
    if patched is None:
        raise PlannerStateError("a patch was accepted without a document")
    return {"draft": draft_of(patched, plan), "document": patched, "mapping_issues": (), **spent}


def _route(state: LyricsState) -> Literal["revise", "done"]:
    return "done" if state["outcome"] is not None else "revise"


def build_lyrics_graph() -> StateGraph[LyricsState, LyricsDeps, LyricsState, LyricsState]:
    """The (uncompiled) lyrics graph."""
    graph: StateGraph[LyricsState, LyricsDeps, LyricsState, LyricsState] = StateGraph(
        LyricsState, context_schema=LyricsDeps
    )
    graph.add_node("draft", _draft)
    graph.add_node("check", _check)
    graph.add_node("revise", _revise)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "check")
    graph.add_conditional_edges("check", _route, {"revise": "revise", "done": END})
    graph.add_edge("revise", "check")
    return graph


def compile_lyrics_graph() -> CompiledStateGraph[LyricsState, LyricsDeps, LyricsState, LyricsState]:
    """Compile the graph (do this once, at wiring time; the result is reusable and stateless)."""
    return build_lyrics_graph().compile()


async def run_lyrics_graph(
    compiled: CompiledStateGraph[LyricsState, LyricsDeps, LyricsState, LyricsState],
    state: LyricsState,
    deps: LyricsDeps,
) -> LyricsRun:
    """Run a compiled lyrics graph to its end.

    Raises:
        LlmError: Any port error (see ``src/llm/port.py``); nothing is swallowed here.
        PlannerStateError: The graph ended without an outcome (its wiring should prevent it).
    """
    final = cast(
        "LyricsState",
        await compiled.ainvoke(state, context=deps, config={"recursion_limit": _RECURSION_LIMIT}),
    )
    outcome = final["outcome"]
    if outcome is None:
        raise PlannerStateError("the lyrics graph ended without an outcome")
    return LyricsRun(
        outcome=outcome,
        llm_calls=final["llm_calls"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
    )
