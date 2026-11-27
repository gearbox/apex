"""Studio timelines — realized timelines of a track's audio versions, and lyric-file exports (S3a).

Endpoints:
  POST   /v1/studio/audio/{id}/timelines          — save a version (201) or replay it (200)
  GET    /v1/studio/audio/{id}/timelines          — keyset page of summaries, newest first
  GET    /v1/studio/timelines/{id}                — one timeline with its content
  DELETE /v1/studio/timelines/{id}                — delete (204) one that is not current or a parent
  PUT    /v1/studio/audio/{id}/current-timeline   — choose or clear the current timeline
  GET    /v1/studio/timelines/{id}/exports/{format} — LRC or SRT (pure GET)

A timeline places the plan sections (and optionally the lyric lines) of an audio version in time;
the server never invents a time. Everything here is free. Mutations are legal-enforced like every
other mutating route (none opts out). Errors are :class:`~src.studio.errors.StudioError`
subclasses rendered by one app-level handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, Response, delete, get, post, put
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio import (
    CurrentTimelinePut,
    TimelineCreate,
    TimelineExportOut,
    TimelineOut,
    TimelineSummaryOut,
    TrackAudioOut,
)
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.studio.services.timeline import TimelineService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    400: ResponseSpec(data_container=ErrorEnvelope, description="invalid_cursor."),
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "audio_version_not_found, timeline_not_found or not_found (an unknown export format, "
            "or studio disabled)."
        ),
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "audio_has_no_plan, timeline_unchanged, timeline_limit_reached, timeline_is_current "
            "or timeline_has_children."
        ),
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description="document_invalid — every issue is listed in detail.issues.",
    ),
}


class StudioTimelineController(Controller):
    """Realized timelines of audio versions: save, list, read, choose, delete, export."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    # The feature guard runs first: a disabled studio answers 404 even when unauthenticated.
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @post("/audio/{audio_id:uuid}/timelines", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_timeline(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        data: TimelineCreate,
        studio_timeline_service: TimelineService,
    ) -> Response[TimelineOut]:
        """Save a timeline as a new version of the audio version's timeline.

        Idempotent: an identical sibling (same parent and content) is returned with ``200``
        instead of duplicated, and ``make_current`` is still applied.
        """
        out, created = await studio_timeline_service.create(
            audio_id, current_user_id, product_id, data
        )
        return Response(content=out, status_code=HTTP_201_CREATED if created else HTTP_200_OK)

    @get("/audio/{audio_id:uuid}/timelines", responses=_ERRORS)
    async def list_timelines(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        studio_timeline_service: TimelineService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[TimelineSummaryOut]:
        """The audio version's timelines, newest (highest version number) first.

        Summaries only: read one timeline to get its content.
        """
        return await studio_timeline_service.list_page(
            audio_id, current_user_id, product_id, limit=limit, cursor=cursor
        )

    @get("/timelines/{timeline_id:uuid}", responses=_ERRORS)
    async def get_timeline(
        self,
        current_user_id: UUID,
        product_id: str,
        timeline_id: UUID,
        studio_timeline_service: TimelineService,
    ) -> TimelineOut:
        """One timeline with its content."""
        return await studio_timeline_service.get(timeline_id, current_user_id, product_id)

    @delete("/timelines/{timeline_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_timeline(
        self,
        current_user_id: UUID,
        product_id: str,
        timeline_id: UUID,
        studio_timeline_service: TimelineService,
    ) -> None:
        """Delete a timeline that is not current and has no children."""
        await studio_timeline_service.delete(timeline_id, current_user_id, product_id)

    @put("/audio/{audio_id:uuid}/current-timeline", status_code=HTTP_200_OK, responses=_ERRORS)
    async def put_current_timeline(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        data: CurrentTimelinePut,
        studio_timeline_service: TimelineService,
    ) -> TrackAudioOut:
        """Choose the audio version's current timeline, or clear it with ``null``."""
        return await studio_timeline_service.set_current(
            audio_id, current_user_id, product_id, data.timeline_id
        )

    @get("/timelines/{timeline_id:uuid}/exports/{format:str}", responses=_ERRORS)
    async def export_timeline(
        self,
        current_user_id: UUID,
        product_id: str,
        timeline_id: UUID,
        format: str,  # the path parameter's name; "lrc" or "srt"
        studio_timeline_service: TimelineService,
    ) -> TimelineExportOut:
        """Render the timeline as an LRC or SRT file. Always renders; never invents a time."""
        return await studio_timeline_service.export(
            timeline_id, format, current_user_id, product_id
        )


__all__ = ["StudioTimelineController"]
