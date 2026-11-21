"""The ``plan`` graph: draft -> validate -> (revise -> validate)* -> done.

An in-memory LangGraph state machine over the planner's ports; **it never touches the database**.
It returns a :class:`PlanRun` whose outcome is a tagged union - a valid plan, a refusal, or the
last draft's issues - and lets the run service save (or fail) it. Port errors (``LlmUnavailable``
...) propagate out of ``ainvoke`` for the run service to map; a state the wiring should make
impossible raises :class:`PlannerStateError` instead of asserting.

Budget: at most :data:`MAX_LLM_CALLS` model calls per run - the first draft, then at most two
revisions. A reply that does not decode is re-asked within the same budget (and consumes no
revision). The explicit counters are the first guard; LangGraph's ``recursion_limit`` is the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime  # noqa: TC002 - runtime: langgraph inspects node signatures

from src.core.enums import StudioRunKind, StudioRunStage
from src.llm.port import LlmMessage  # noqa: TC001 - runtime: LangGraph reads the state hints
from src.studio.errors import DocumentIssue  # noqa: TC001 - runtime: LangGraph reads the state hints
from src.studio.planner.context import RunContextSnapshot  # noqa: TC001
from src.studio.planner.drafts import PlanDraft
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.ask import ask_structured
from src.studio.planner.mapping import draft_key_issues, plan_from_draft
from src.studio.planner.outcome import (
    PlanOutcome,
    PlanReady,
    PlanRun,
    RunRefused,
    RunRejected,
)
from src.studio.planner.prompts.plan_v2 import (
    build_plan_messages,
    build_plan_revision_messages,
)
from src.studio.planner.validate import validate_plan
from src.studio.run_requests import PlanRunRequest  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from langgraph.graph.state import CompiledStateGraph

    from src.llm.port import StructuredLlmPort

__all__ = [
    "MAX_LLM_CALLS",
    "MAX_REVISIONS",
    "PLAN_SCHEMA_NAME",
    "PlanDeps",
    "PlanOutcome",
    "PlanReady",
    "PlanRun",
    "PlanState",
    "RunRefused",
    "RunRejected",
    "build_plan_graph",
    "compile_plan_graph",
    "initial_state",
    "run_plan_graph",
]

MAX_LLM_CALLS: Final = 3
"""Model calls per run, counting re-asks (a transport retry inside the adapter is not counted)."""
MAX_REVISIONS: Final = 2
PLAN_SCHEMA_NAME: Final = "plan_draft"
_RECURSION_LIMIT: Final = 12
"""A full run is six supersteps (draft, validate, revise, validate, revise, validate)."""


@dataclass(frozen=True, slots=True)
class PlanDeps:
    """What the nodes may use besides their state: the model port and a stage callback."""

    llm: StructuredLlmPort
    on_stage: Callable[[StudioRunStage], Awaitable[None]]
    """Awaited when a stage starts; the run service records it (and may stop the run)."""
    max_revisions: int = MAX_REVISIONS


class PlanState(TypedDict):
    """The graph state. ``messages`` is the opening conversation, kept for the revision turns."""

    context: RunContextSnapshot
    request: PlanRunRequest
    messages: tuple[LlmMessage, ...]
    draft: PlanDraft | None
    issues: tuple[DocumentIssue, ...]
    revisions: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    outcome: PlanOutcome | None


def initial_state(context: RunContextSnapshot, request: PlanRunRequest) -> PlanState:
    """The state a run starts from."""
    return PlanState(
        context=context,
        request=request,
        messages=(),
        draft=None,
        issues=(),
        revisions=0,
        llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        outcome=None,
    )


def _as_is(draft: PlanDraft) -> PlanDraft:
    return draft


async def _ask(
    deps: PlanDeps, messages: tuple[LlmMessage, ...], state: PlanState
) -> dict[str, Any]:
    """Ask for a draft, re-asking (within the call budget) when the reply does not decode."""
    asked = await ask_structured(
        deps.llm,
        messages,
        schema=PlanDraft,
        schema_name=PLAN_SCHEMA_NAME,
        max_calls=MAX_LLM_CALLS,
        llm_calls=state["llm_calls"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        accept=_as_is,
    )
    return {
        "draft": asked.value,
        "llm_calls": asked.llm_calls,
        "prompt_tokens": asked.prompt_tokens,
        "completion_tokens": asked.completion_tokens,
    }


async def _draft(state: PlanState, runtime: Runtime[PlanDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.DRAFTING)
    messages = tuple(build_plan_messages(state["context"], state["request"]))
    return {"messages": messages, **await _ask(deps, messages, state)}


async def _validate(state: PlanState, runtime: Runtime[PlanDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.VALIDATING)
    draft = state["draft"]
    if draft is None:
        raise PlannerStateError("validate ran before draft")
    if draft.refusal is not None:
        return {"issues": (), "outcome": RunRefused(draft.refusal)}
    plan, plan_issues = validate_plan(plan_from_draft(draft, brief=state["request"].brief))
    issues = (*draft_key_issues(draft), *plan_issues)
    if plan is not None and not issues:
        return {"issues": (), "outcome": PlanReady(plan)}
    can_revise = state["revisions"] < deps.max_revisions and state["llm_calls"] < MAX_LLM_CALLS
    return {
        "issues": issues,
        "outcome": None if can_revise else RunRejected(issues, StudioRunKind.PLAN),
    }


async def _revise(state: PlanState, runtime: Runtime[PlanDeps]) -> dict[str, Any]:
    deps = runtime.context
    await deps.on_stage(StudioRunStage.REVISING)
    draft = state["draft"]
    if draft is None:
        raise PlannerStateError("revise ran before draft")
    messages = (*state["messages"], *build_plan_revision_messages(draft, state["issues"]))
    return {"revisions": state["revisions"] + 1, **await _ask(deps, messages, state)}


def _route(state: PlanState) -> Literal["revise", "done"]:
    return "done" if state["outcome"] is not None else "revise"


def build_plan_graph() -> StateGraph[PlanState, PlanDeps, PlanState, PlanState]:
    """The (uncompiled) plan graph."""
    graph: StateGraph[PlanState, PlanDeps, PlanState, PlanState] = StateGraph(
        PlanState, context_schema=PlanDeps
    )
    graph.add_node("draft", _draft)
    graph.add_node("validate", _validate)
    graph.add_node("revise", _revise)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "validate")
    graph.add_conditional_edges("validate", _route, {"revise": "revise", "done": END})
    graph.add_edge("revise", "validate")
    return graph


def compile_plan_graph() -> CompiledStateGraph[PlanState, PlanDeps, PlanState, PlanState]:
    """Compile the graph (do this once, at wiring time; the result is reusable and stateless)."""
    return build_plan_graph().compile()


async def run_plan_graph(
    compiled: CompiledStateGraph[PlanState, PlanDeps, PlanState, PlanState],
    state: PlanState,
    deps: PlanDeps,
) -> PlanRun:
    """Run a compiled plan graph to its end.

    Raises:
        LlmError: Any port error (see ``src/llm/port.py``); nothing is swallowed here.
        PlannerStateError: The graph ended without an outcome (its wiring should prevent it).
    """
    final = cast(
        "PlanState",
        await compiled.ainvoke(state, context=deps, config={"recursion_limit": _RECURSION_LIMIT}),
    )
    outcome = final["outcome"]
    if outcome is None:
        raise PlannerStateError("the plan graph ended without an outcome")
    return PlanRun(
        outcome=outcome,
        llm_calls=final["llm_calls"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
    )
