"""Library tag endpoints — user-created, per-user/per-product tags for library assets.

Endpoints:
  GET    /v1/library/tags          — paginated tag list
  POST   /v1/library/tags          — create a tag
  GET    /v1/library/tags/{id}     — tag detail
  PATCH  /v1/library/tags/{id}     — rename
  DELETE /v1/library/tags/{id}     — delete (204; asset assignments cascade via ON DELETE CASCADE)
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
    LibraryTag,
    LibraryTagCreate,
    LibraryTagListItem,
    LibraryTagPatch,
)
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.library_tag import LibraryTagService, LibraryTagValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_TAG_NOT_FOUND = "Library tag not found"

_CONFLICT_RESPONSES = {
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="A tag with this name already exists for the caller (case-insensitive).",
    )
}


class LibraryTagController(Controller):
    """Tag CRUD endpoints."""

    path = "/v1/library/tags"
    tags: Sequence[str] | None = ("Library",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/")
    async def list_tags(
        self,
        current_user_id: UUID,
        product_id: str,
        session: AsyncSession,
        library_tag_service: LibraryTagService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> Response[CursorPage[LibraryTagListItem] | ErrorEnvelope]:
        """Paginated list of the caller's tags, newest first."""
        try:
            page = await library_tag_service.list_tags(
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
    async def create_tag(
        self,
        current_user_id: UUID,
        product_id: str,
        data: LibraryTagCreate,
        session: AsyncSession,
        library_tag_service: LibraryTagService,
    ) -> Response[LibraryTag | ErrorEnvelope]:
        """Create a new tag. Name uniqueness is case-insensitive per owner (409 on conflict)."""
        try:
            tag = await library_tag_service.create(
                current_user_id, product_id, data.name, session=session
            )
        except LibraryTagValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        return Response(content=tag, status_code=HTTP_201_CREATED)

    @get("/{tag_id:uuid}", raises=[NotFoundException])
    async def get_tag(
        self,
        current_user_id: UUID,
        product_id: str,
        tag_id: UUID,
        session: AsyncSession,
        library_tag_service: LibraryTagService,
    ) -> LibraryTag:
        """Fetch a single tag."""
        tag = await library_tag_service.get(tag_id, current_user_id, product_id, session=session)
        if tag is None:
            raise NotFoundException(detail=_TAG_NOT_FOUND)
        return tag

    @patch("/{tag_id:uuid}", raises=[NotFoundException], responses=_CONFLICT_RESPONSES)
    async def patch_tag(
        self,
        current_user_id: UUID,
        product_id: str,
        tag_id: UUID,
        data: LibraryTagPatch,
        session: AsyncSession,
        library_tag_service: LibraryTagService,
    ) -> Response[LibraryTag | ErrorEnvelope]:
        """Rename a tag. Name conflicts → 409."""
        try:
            tag = await library_tag_service.patch(
                tag_id, data, current_user_id, product_id, session=session
            )
        except LibraryTagValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        if tag is None:
            raise NotFoundException(detail=_TAG_NOT_FOUND)
        return Response(content=tag)

    @delete("/{tag_id:uuid}", status_code=HTTP_204_NO_CONTENT, raises=[NotFoundException])
    async def delete_tag(
        self,
        current_user_id: UUID,
        product_id: str,
        tag_id: UUID,
        session: AsyncSession,
        library_tag_service: LibraryTagService,
    ) -> None:
        """Delete a tag. Asset assignments cascade via ON DELETE CASCADE."""
        deleted = await library_tag_service.delete(
            tag_id, current_user_id, product_id, session=session
        )
        if not deleted:
            raise NotFoundException(detail=_TAG_NOT_FOUND)
