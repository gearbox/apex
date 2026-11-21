"""Studio artist memories - episodes of an artist's fictional life (``/v1/studio``).

Endpoints:
  GET    /v1/studio/artists/{artist_id}/memories   - paginated list (diary order; ``?status=``)
  POST   /v1/studio/artists/{artist_id}/memories   - create (201)
  GET    /v1/studio/memories/{memory_id}           - detail
  PATCH  /v1/studio/memories/{memory_id}           - tri-state update (also archive / un-archive)
  DELETE /v1/studio/memories/{memory_id}           - delete (204; always allowed)

Memory text goes to the configured LLM provider when a run uses the memory, and nowhere else.
Errors are :class:`~src.studio.errors.StudioError` subclasses rendered by one app-level handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, delete, get, patch, post
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio import MemoryCreate, MemoryOut, MemoryPatch
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.core.enums import StudioMemoryStatus
from src.studio.services.memory import MemoryService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description="artist_not_found, memory_not_found (or studio disabled).",
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="memory_limit_reached (200 active memories per artist).",
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope, description="document_invalid (the memory's fields)."
    ),
}


class StudioMemoryController(Controller):
    """Artist memory CRUD."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    # The feature guard runs first: a disabled studio answers 404 even when unauthenticated.
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/artists/{artist_id:uuid}/memories", responses=_ERRORS)
    async def list_memories(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        studio_memory_service: MemoryService,
        status: StudioMemoryStatus = StudioMemoryStatus.ACTIVE,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[MemoryOut]:
        """Paginated memories of one artist in diary order; ``status`` defaults to ``active``.

        Diary order: by ``occurred_on`` (undated last), then newest written first.
        """
        return await studio_memory_service.list_page(
            artist_id, current_user_id, product_id, status=status, limit=limit, cursor=cursor
        )

    @post("/artists/{artist_id:uuid}/memories", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_memory(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        data: MemoryCreate,
        studio_memory_service: MemoryService,
    ) -> MemoryOut:
        """Create a memory. An artist holds at most 200 active ones (409 on the 201st)."""
        return await studio_memory_service.create(artist_id, current_user_id, product_id, data)

    @get("/memories/{memory_id:uuid}", responses=_ERRORS)
    async def get_memory(
        self,
        current_user_id: UUID,
        product_id: str,
        memory_id: UUID,
        studio_memory_service: MemoryService,
    ) -> MemoryOut:
        """Fetch one memory."""
        return await studio_memory_service.get(memory_id, current_user_id, product_id)

    @patch("/memories/{memory_id:uuid}", responses=_ERRORS)
    async def patch_memory(
        self,
        current_user_id: UUID,
        product_id: str,
        memory_id: UUID,
        data: MemoryPatch,
        studio_memory_service: MemoryService,
    ) -> MemoryOut:
        """Tri-state update: omitted fields are kept, ``occurred_on: null`` clears the date.

        ``status`` archives (``archived``) or restores (``active``) the memory.
        """
        return await studio_memory_service.patch(memory_id, current_user_id, product_id, data)

    @delete("/memories/{memory_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_memory(
        self,
        current_user_id: UUID,
        product_id: str,
        memory_id: UUID,
        studio_memory_service: MemoryService,
    ) -> None:
        """Delete a memory at any time; a run that already used it keeps its own copy."""
        await studio_memory_service.delete(memory_id, current_user_id, product_id)


__all__ = ["StudioMemoryController"]
