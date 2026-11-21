"""``revise_lyrics``: part of a lyrics version is rewritten into a new lyrics version.

Lineage (P2a-42): the parent is the input lyrics and ``based_on`` is **their** plan (which need not
be the track's current one). The parent is an input, so it cannot vanish while the run is active
(P2a-41): there is no fallback here. A revision that changes nothing succeeds with the input
version as its result and the warning ``revision_unchanged``; no version is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from src.api.schemas.studio import LyricsVersionCreate
from src.core.enums import StudioDocumentKind, StudioRunKind
from src.studio.planner.errors import PlannerStateError
from src.studio.planner.graphs.revise_lyrics import (
    ReviseDeps,
    compile_revise_graph,
    initial_state,
    run_revise_graph,
)
from src.studio.planner.outcome import LyricsReady, RunWarning, cap_warnings
from src.studio.planner.prompts.revise_lyrics_v2 import PROMPT_VERSION
from src.studio.run_requests import ReviseLyricsRunRequest, RunRequest
from src.studio.services.mappers import decode_lyrics, decode_plan
from src.studio.services.run_kinds.base import (
    WARNING_REVISION_UNCHANGED,
    LoadedRun,
    SavedRun,
    WorkingDocuments,
    create_llm_version,
    decode_request_as,
    load_input_version,
    require_inputs,
)
from src.studio.services.version import DocumentVersionService

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.studio import StudioRun
    from src.llm.port import StructuredLlmPort
    from src.studio.planner.context import RunContextSnapshot
    from src.studio.planner.outcome import RunOutcome
    from src.studio.services.run_kinds.base import ClaimedRun, StageCallback

__all__ = ["ReviseLyricsHandler"]


class ReviseLyricsHandler:
    """The worker side of ``revise_lyrics`` runs."""

    kind = StudioRunKind.REVISE_LYRICS
    prompt_version = PROMPT_VERSION

    def __init__(self) -> None:
        self._graph = compile_revise_graph()

    def decode_request(self, stored: Mapping[str, object]) -> RunRequest:
        return decode_request_as(stored, ReviseLyricsRunRequest)

    async def load(
        self,
        session: AsyncSession,
        run: StudioRun,
        request: RunRequest,  # noqa: ARG002
        context: RunContextSnapshot,
    ) -> LoadedRun:
        """Load the input lyrics and their plan named in the run's inputs."""
        inputs = require_inputs(run)
        if inputs.lyrics_version_id is None:
            raise RuntimeError("studio revision run without lyrics")
        lyrics_row = await load_input_version(
            session, run, inputs.lyrics_version_id, StudioDocumentKind.LYRICS
        )
        plan_row = await load_input_version(
            session, run, inputs.plan_version_id, StudioDocumentKind.PLAN
        )
        snapshot = msgspec.structs.replace(
            context,
            input_plan_version_id=plan_row.id,
            input_plan_content_hash=plan_row.content_hash,
            input_lyrics_version_id=lyrics_row.id,
            input_lyrics_content_hash=lyrics_row.content_hash,
            language=inputs.language,
        )
        documents = WorkingDocuments(plan=decode_plan(plan_row), lyrics=decode_lyrics(lyrics_row))
        return LoadedRun(snapshot, inputs, documents)

    async def run(
        self, claimed: ClaimedRun, *, llm: StructuredLlmPort, on_stage: StageCallback
    ) -> RunOutcome:
        request = claimed.request
        plan = claimed.documents.plan
        lyrics = claimed.documents.lyrics
        if not isinstance(request, ReviseLyricsRunRequest) or plan is None or lyrics is None:
            raise PlannerStateError("the revise handler was given a run it cannot work on")
        state = initial_state(
            claimed.context,
            request,
            plan,
            lyrics,
            line_syllables=request.line_syllables or lyrics.line_syllables,
        )
        result = await run_revise_graph(self._graph, state, ReviseDeps(llm=llm, on_stage=on_stage))
        return result.outcome

    async def save(
        self, session: AsyncSession, claimed: ClaimedRun, outcome: RunOutcome
    ) -> SavedRun:
        """Save the revision as a child of the input lyrics (or report that nothing changed)."""
        if (
            not isinstance(outcome, LyricsReady)
            or claimed.inputs is None
            or claimed.inputs.lyrics_version_id is None
        ):
            raise PlannerStateError("the revise handler was asked to save something else")
        saved = await create_llm_version(
            DocumentVersionService(session),
            claimed,
            LyricsVersionCreate(
                content=outcome.lyrics,
                based_on_plan_version_id=claimed.inputs.plan_version_id,
                parent_version_id=claimed.inputs.lyrics_version_id,
                make_current=False,
            ),
        )
        notes = [RunWarning(WARNING_REVISION_UNCHANGED)] if saved.unchanged else []
        return SavedRun(
            result_lyrics_version_id=saved.version_id,
            warnings=tuple(cap_warnings([*notes, *outcome.warnings])),
        )
