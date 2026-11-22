"""``lyrics``: a plan version becomes a new lyrics version (the worker side).

Lineage (P2a-42): ``based_on`` is the input plan; the parent is the lyrics that were current at
claim - and if those were deleted while the model worked, the track's current lyrics at save (a
root when there are none) with the warning ``parent_version_replaced`` (P2a-29). ``origin = llm``,
never made current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from src.api.schemas.studio import LyricsVersionCreate
from src.core.enums import StudioDocumentKind, StudioRunKind
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.lyrics import (
    LyricsDeps,
    compile_lyrics_graph,
    initial_state,
    run_lyrics_graph,
)
from src.studio.planner.outcome import LyricsReady, cap_warnings
from src.studio.planner.prompts.lyrics_v2 import PROMPT_VERSION
from src.studio.run_requests import LyricsRunRequest, RunRequest
from src.studio.services.mappers import decode_plan
from src.studio.services.run_kinds.base import (
    LoadedRun,
    SavedRun,
    WorkingDocuments,
    decode_request_as,
    load_input_version,
    require_inputs,
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

__all__ = ["LyricsHandler"]


class LyricsHandler:
    """The worker side of ``lyrics`` runs."""

    kind = StudioRunKind.LYRICS
    prompt_version = PROMPT_VERSION

    def __init__(self) -> None:
        self._graph = compile_lyrics_graph()

    def decode_request(self, stored: Mapping[str, object]) -> RunRequest:
        return decode_request_as(stored, LyricsRunRequest)

    async def load(
        self,
        session: AsyncSession,
        run: StudioRun,
        request: RunRequest,  # noqa: ARG002
        context: RunContextSnapshot,
    ) -> LoadedRun:
        """Load the input plan named in the run's inputs (never the track's current plan)."""
        inputs = require_inputs(run)
        plan_row = await load_input_version(
            session, run, inputs.plan_version_id, StudioDocumentKind.PLAN
        )
        snapshot = msgspec.structs.replace(
            context,
            input_plan_version_id=plan_row.id,
            input_plan_content_hash=plan_row.content_hash,
            language=inputs.language,
        )
        return LoadedRun(snapshot, inputs, WorkingDocuments(plan=decode_plan(plan_row)))

    async def run(
        self, claimed: ClaimedRun, *, llm: StructuredLlmPort, on_stage: StageCallback
    ) -> RunOutcome:
        request = claimed.request
        plan = claimed.documents.plan
        if not isinstance(request, LyricsRunRequest) or plan is None or claimed.inputs is None:
            raise PlannerStateError("the lyrics handler was given a run it cannot work on")
        state = initial_state(
            claimed.context,
            request,
            plan,
            language=claimed.inputs.language,
            line_syllables=request.line_syllables,
        )
        result = await run_lyrics_graph(self._graph, state, LyricsDeps(llm=llm, on_stage=on_stage))
        return result.outcome

    async def save(
        self, session: AsyncSession, claimed: ClaimedRun, outcome: RunOutcome
    ) -> SavedRun:
        """Save the lyrics based on the input plan, as a child of the lyrics current at claim."""
        if not isinstance(outcome, LyricsReady) or claimed.inputs is None:
            raise PlannerStateError("the lyrics handler was asked to save something else")
        lyrics = outcome.lyrics
        plan_version_id = claimed.inputs.plan_version_id
        saved = await save_under_current(
            session,
            claimed,
            kind=StudioDocumentKind.LYRICS,
            parent_id=claimed.context.lyrics_version_id,
            build=lambda parent: LyricsVersionCreate(
                content=lyrics,
                based_on_plan_version_id=plan_version_id,
                parent_version_id=parent,
                make_current=False,
            ),
        )
        return SavedRun(
            result_lyrics_version_id=saved.version_id,
            warnings=tuple(cap_warnings([*saved.warnings, *outcome.warnings])),
        )
