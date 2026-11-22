"""Studio planner runs - submit, read, list and cancel an LLM job.

Endpoints:
  POST /v1/studio/tracks/{id}/runs       — submit a run: 202 queued, 200 replay of the same key
  GET  /v1/studio/runs/{id}              — one run (the polling fallback of ``studio.run.updated``)
  GET  /v1/studio/tracks/{id}/runs       — paginated, newest first
  POST /v1/studio/runs/{id}/cancel       — cancel a queued run (refunds the reservation)

Guarded by the studio feature, then by ``STUDIO_RUNS_ENABLED`` (a separate kill switch for LLM
spend), then by authentication; all answer 404 while either switch is off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_200_OK, HTTP_202_ACCEPTED
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio_runs import RunOut, RunRequest
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard, studio_runs_guard
from src.studio.services.mappers import run_out
from src.studio.services.run import StudioRunService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    400: ResponseSpec(
        data_container=ErrorEnvelope, description="idempotency_key_required (missing or invalid)."
    ),
    402: ResponseSpec(data_container=ErrorEnvelope, description="Insufficient token balance."),
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "Not found (or studio runs disabled): the track or run, plan_version_not_found or "
            "lyrics_version_not_found."
        ),
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "run_in_progress, run_limit_reached, run_not_cancellable, idempotency_conflict, "
            "plan_required (lyrics) or lyrics_required (revise_lyrics)."
        ),
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "document_invalid (the body, or an input that cannot be used: no vocal section, no "
            "language, lyrics that are not valid today, a target that does not fit, or a pinned "
            "memory in memory_ids that is unknown, archived, repeated, over the budget, or the "
            "project has no artist)."
        ),
    ),
    429: ResponseSpec(data_container=ErrorEnvelope, description="run_rate_limited."),
}


class StudioRunController(Controller):
    """Planner runs."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    guards = [studio_feature_guard, studio_runs_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @post("/tracks/{track_id:uuid}/runs", status_code=HTTP_202_ACCEPTED, responses=_ERRORS)
    async def submit_run(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: RunRequest,
        session: AsyncSession,
        studio_run_service: StudioRunService,
        idempotency_key_header: Annotated[
            str | None,
            Parameter(
                header="Idempotency-Key",
                description=(
                    "Required (1-128 characters). The same key with the same body returns the "
                    "same run (200); the same key with a different body is 409."
                ),
            ),
        ] = None,
    ) -> Response[RunOut]:
        """Queue a run (``plan``, ``lyrics`` or ``revise_lyrics``).

        What the run works on is resolved now and fixed: a missing or wrong input is a ``4xx``
        with no charge. The brief, plan, lyrics and instruction are sent to the configured LLM
        provider. Answers ``202`` with the queued run, or ``200`` when the key replays an earlier
        submit. The result is an ordinary version that is **never** made current automatically.
        """
        write = await studio_run_service.submit(
            track_id,
            data,
            idempotency_key=idempotency_key_header,
            user_id=current_user_id,
            product_id=product_id,
        )
        # Durable first: the event tells the client to fetch a row that must already exist.
        await session.commit()
        await studio_run_service.publish(write)
        return Response(
            content=run_out(write.run),
            status_code=HTTP_202_ACCEPTED if write.changed else HTTP_200_OK,
        )

    @get("/runs/{run_id:uuid}", responses=_ERRORS)
    async def get_run(
        self,
        current_user_id: UUID,
        product_id: str,
        run_id: UUID,
        studio_run_service: StudioRunService,
    ) -> RunOut:
        """One run: status, stage, revision, result version ids and error code."""
        return await studio_run_service.get(run_id, user_id=current_user_id, product_id=product_id)

    @get("/tracks/{track_id:uuid}/runs", responses=_ERRORS)
    async def list_runs(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_run_service: StudioRunService,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[RunOut]:
        """A track's runs, newest first."""
        return await studio_run_service.list_for_track(
            track_id, user_id=current_user_id, product_id=product_id, limit=limit, cursor=cursor
        )

    @post("/runs/{run_id:uuid}/cancel", status_code=HTTP_200_OK, responses=_ERRORS)
    async def cancel_run(
        self,
        current_user_id: UUID,
        product_id: str,
        run_id: UUID,
        session: AsyncSession,
        studio_run_service: StudioRunService,
    ) -> RunOut:
        """Cancel a ``queued`` run and refund its reservation. A running run cannot be cancelled."""
        write = await studio_run_service.cancel(
            run_id, user_id=current_user_id, product_id=product_id
        )
        await session.commit()
        await studio_run_service.publish(write)
        return run_out(write.run)


__all__ = ["StudioRunController"]
