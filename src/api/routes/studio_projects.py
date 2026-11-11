"""Studio projects — the studio profile of a library project, and its tracks.

Endpoints:
  GET    /v1/studio/projects                       — paginated list of studio-enabled projects
  GET    /v1/studio/projects/{id}                  — profile
  PUT    /v1/studio/projects/{id}                  — enable / replace the profile (201 / 200)
  DELETE /v1/studio/projects/{id}                  — disable (204; 409 project_has_tracks)
  GET    /v1/studio/projects/{id}/tracks           — ordered tracks (capped, not paginated)
  POST   /v1/studio/projects/{id}/tracks           — add a track (201)
  PUT    /v1/studio/projects/{id}/tracks/order     — reorder (exact id set, else 422)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, Response, delete, get, post, put
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
)

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio import (
    ProjectProfileOut,
    ProjectProfilePut,
    TrackCreate,
    TrackOrderPut,
    TrackOut,
)
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.studio.services.project import StudioProjectService
from src.studio.services.track import TrackService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    404: ResponseSpec(data_container=ErrorEnvelope, description="Not found (or studio disabled)."),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="project_has_tracks or track_limit_reached.",
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope, description="document_invalid or track_order_mismatch."
    ),
}


class StudioProjectController(Controller):
    """Project profiles and their tracks."""

    path = "/v1/studio/projects"
    tags: Sequence[str] | None = ("Studio",)
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/", responses=_ERRORS)
    async def list_projects(
        self,
        current_user_id: UUID,
        product_id: str,
        studio_project_service: StudioProjectService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[ProjectProfileOut]:
        """Paginated list of the caller's studio projects, newest first."""
        return await studio_project_service.list_profiles(
            current_user_id, product_id, limit=limit, cursor=cursor
        )

    @get("/{project_id:uuid}", responses=_ERRORS)
    async def get_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        studio_project_service: StudioProjectService,
    ) -> ProjectProfileOut:
        """The studio profile of a library project."""
        return await studio_project_service.get_profile(project_id, current_user_id, product_id)

    @put("/{project_id:uuid}", status_code=HTTP_200_OK, responses=_ERRORS)
    async def put_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        data: ProjectProfilePut,
        studio_project_service: StudioProjectService,
    ) -> Response[ProjectProfileOut]:
        """Enable studio on an owned library project (201) or replace its profile (200)."""
        profile, created = await studio_project_service.upsert_profile(
            project_id, current_user_id, product_id, data
        )
        return Response(content=profile, status_code=HTTP_201_CREATED if created else HTTP_200_OK)

    @delete("/{project_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        studio_project_service: StudioProjectService,
    ) -> None:
        """Disable studio on a project (the library project itself is untouched)."""
        await studio_project_service.disable(project_id, current_user_id, product_id)

    @get("/{project_id:uuid}/tracks", responses=_ERRORS)
    async def list_tracks(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        studio_track_service: TrackService,
    ) -> list[TrackOut]:
        """The project's tracks in order."""
        return await studio_track_service.list_for_project(project_id, current_user_id, product_id)

    @post("/{project_id:uuid}/tracks", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_track(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        data: TrackCreate,
        studio_track_service: TrackService,
    ) -> TrackOut:
        """Append a track to the project."""
        return await studio_track_service.create(
            project_id, current_user_id, product_id, data.title
        )

    @put("/{project_id:uuid}/tracks/order", status_code=HTTP_200_OK, responses=_ERRORS)
    async def reorder_tracks(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        data: TrackOrderPut,
        studio_track_service: TrackService,
    ) -> list[TrackOut]:
        """Reorder: ``track_ids`` must list every track of the project exactly once."""
        return await studio_track_service.reorder(
            project_id, current_user_id, product_id, data.track_ids
        )


__all__ = ["StudioProjectController"]
