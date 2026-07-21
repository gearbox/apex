"""Library project endpoints — user-created groupings for library assets.

Endpoints:
  GET    /v1/library/projects          — paginated project list
  POST   /v1/library/projects          — create a project
  GET    /v1/library/projects/{id}     — project detail
  PATCH  /v1/library/projects/{id}     — rename/redescribe
  DELETE /v1/library/projects/{id}     — delete (204; assets are unassigned via ON DELETE SET NULL, P2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get, patch, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.library import (
    LibraryProject,
    LibraryProjectCreate,
    LibraryProjectListItem,
    LibraryProjectPatch,
)
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.library_project import LibraryProjectService, LibraryProjectValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_PROJECT_NOT_FOUND = "Library project not found"

_CONFLICT_RESPONSES = {
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="A project with this name already exists for the caller (case-insensitive).",
    )
}


class LibraryProjectController(Controller):
    """Project CRUD endpoints."""

    path = "/v1/library/projects"
    tags: Sequence[str] | None = ("Library",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/")
    async def list_projects(
        self,
        current_user_id: UUID,
        product_id: str,
        session: AsyncSession,
        library_project_service: LibraryProjectService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> Response[CursorPage[LibraryProjectListItem] | ErrorEnvelope]:
        """Paginated list of the caller's projects, newest first."""
        try:
            page = await library_project_service.list_projects(
                current_user_id, product_id, session=session, limit=limit, cursor=cursor
            )
        except ValueError:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_cursor",
                    message="Invalid pagination cursor",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        return Response(content=page)

    @post("/", status_code=HTTP_201_CREATED, responses=_CONFLICT_RESPONSES)
    async def create_project(
        self,
        current_user_id: UUID,
        product_id: str,
        data: LibraryProjectCreate,
        session: AsyncSession,
        library_project_service: LibraryProjectService,
    ) -> Response[LibraryProject | ErrorEnvelope]:
        """Create a new project. Name uniqueness is case-insensitive per owner (409 on conflict)."""
        try:
            project = await library_project_service.create(
                current_user_id, product_id, data.name, data.description, session=session
            )
        except LibraryProjectValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        return Response(content=project, status_code=HTTP_201_CREATED)

    @get("/{project_id:uuid}", raises=[NotFoundException])
    async def get_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        session: AsyncSession,
        library_project_service: LibraryProjectService,
    ) -> LibraryProject:
        """Fetch a single project."""
        project = await library_project_service.get(
            project_id, current_user_id, product_id, session=session
        )
        if project is None:
            raise NotFoundException(detail=_PROJECT_NOT_FOUND)
        return project

    @patch("/{project_id:uuid}", raises=[NotFoundException], responses=_CONFLICT_RESPONSES)
    async def patch_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        data: LibraryProjectPatch,
        session: AsyncSession,
        library_project_service: LibraryProjectService,
    ) -> Response[LibraryProject | ErrorEnvelope]:
        """Rename/redescribe a project (tri-state fields). Name conflicts → 409."""
        try:
            project = await library_project_service.patch(
                project_id, data, current_user_id, product_id, session=session
            )
        except LibraryProjectValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        if project is None:
            raise NotFoundException(detail=_PROJECT_NOT_FOUND)
        return Response(content=project)

    @delete("/{project_id:uuid}", status_code=HTTP_204_NO_CONTENT, raises=[NotFoundException])
    async def delete_project(
        self,
        current_user_id: UUID,
        product_id: str,
        project_id: UUID,
        session: AsyncSession,
        library_project_service: LibraryProjectService,
    ) -> None:
        """Delete a project. Assigned assets survive, unassigned via ON DELETE SET NULL (P2)."""
        deleted = await library_project_service.delete(
            project_id, current_user_id, product_id, session=session
        )
        if not deleted:
            raise NotFoundException(detail=_PROJECT_NOT_FOUND)
