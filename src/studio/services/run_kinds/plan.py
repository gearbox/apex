"""``plan``: a brief becomes a new plan version (the worker side).

Behaves exactly as the planner runs of S1-P2a-A did: the result is a child of the plan that was
current at claim (or, if that plan was deleted meanwhile, of the current one - ``parent_version_
replaced``), ``origin = llm``, never made current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.schemas.studio import PlanVersionCreate
from src.core.enums import StudioDocumentKind, StudioRunKind
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.plan import (
    PlanDeps,
    PlanReady,
    compile_plan_graph,
    initial_state,
    run_plan_graph,
)
from src.studio.planner.prompts.plan_v1 import PROMPT_VERSION
from src.studio.run_requests import PlanRunRequest, RunRequest
from src.studio.services.run_kinds.base import (
    LoadedRun,
    SavedRun,
    decode_request_as,
    save_under_current,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioRun
    from src.llm.port import StructuredLlmPort
    from src.studio.planner.context import RunContextSnapshot
    from src.studio.planner.outcome import RunOutcome
    from src.studio.services.run_kinds.base import ClaimedRun, StageCallback

__all__ = ["PlanHandler"]


class PlanHandler:
    """The worker side of ``plan`` runs."""

    kind = StudioRunKind.PLAN
    prompt_version = PROMPT_VERSION

    def __init__(self) -> None:
        self._graph = compile_plan_graph()

    def decode_request(self, stored: Mapping[str, object]) -> RunRequest:
        return decode_request_as(stored, PlanRunRequest)

    async def load(
        self,
        session: AsyncSession,  # noqa: ARG002
        run: StudioRun,  # noqa: ARG002
        request: RunRequest,  # noqa: ARG002
        context: RunContextSnapshot,
    ) -> LoadedRun:
        return LoadedRun(context)

    async def run(
        self, claimed: ClaimedRun, *, llm: StructuredLlmPort, on_stage: StageCallback
    ) -> RunOutcome:
        request = claimed.request
        if not isinstance(request, PlanRunRequest):
            raise PlannerStateError("the plan handler was given another kind's run")
        result = await run_plan_graph(
            self._graph,
            initial_state(claimed.context, request),
            PlanDeps(llm=llm, on_stage=on_stage),
        )
        return result.outcome

    async def save(
        self, session: AsyncSession, claimed: ClaimedRun, outcome: RunOutcome
    ) -> SavedRun:
        """Save the plan as a child of the plan that was current at claim."""
        if not isinstance(outcome, PlanReady):
            raise PlannerStateError("the plan handler was asked to save another outcome")
        plan = outcome.plan
        saved = await save_under_current(
            session,
            claimed,
            kind=StudioDocumentKind.PLAN,
            parent_id=claimed.context.plan_version_id,
            build=lambda parent: PlanVersionCreate(
                content=plan, parent_version_id=parent, make_current=False
            ),
        )
        return SavedRun(result_plan_version_id=saved.version_id, warnings=saved.warnings)
