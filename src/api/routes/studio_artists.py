"""Studio artists — reusable artist personas (``/v1/studio/artists``).

Endpoints:
  GET    /v1/studio/artists        — paginated list (newest first)
  POST   /v1/studio/artists        — create (201)
  GET    /v1/studio/artists/{id}   — detail
  PATCH  /v1/studio/artists/{id}   — tri-state update
  DELETE /v1/studio/artists/{id}   — delete (204; 409 artist_in_use while a project uses it)

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
from src.api.schemas.studio import ArtistCreate, ArtistOut, ArtistPatch
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.studio.services.artist import ArtistService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    404: ResponseSpec(data_container=ErrorEnvelope, description="Not found (or studio disabled)."),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="artist_name_conflict, artist_limit_reached or artist_in_use.",
    ),
    422: ResponseSpec(data_container=ErrorEnvelope, description="document_invalid (style tags)."),
}


class StudioArtistController(Controller):
    """Artist CRUD."""

    path = "/v1/studio/artists"
    tags: Sequence[str] | None = ("Studio",)
    # The feature guard runs first: a disabled studio answers 404 even when unauthenticated.
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/", responses=_ERRORS)
    async def list_artists(
        self,
        current_user_id: UUID,
        product_id: str,
        studio_artist_service: ArtistService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[ArtistOut]:
        """Paginated list of the caller's artists, newest first."""
        return await studio_artist_service.list_page(
            current_user_id, product_id, limit=limit, cursor=cursor
        )

    @post("/", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_artist(
        self,
        current_user_id: UUID,
        product_id: str,
        data: ArtistCreate,
        studio_artist_service: ArtistService,
    ) -> ArtistOut:
        """Create an artist. Names are unique per owner, case-insensitively (409 on conflict)."""
        return await studio_artist_service.create(current_user_id, product_id, data)

    @get("/{artist_id:uuid}", responses=_ERRORS)
    async def get_artist(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        studio_artist_service: ArtistService,
    ) -> ArtistOut:
        """Fetch one artist."""
        return await studio_artist_service.get(artist_id, current_user_id, product_id)

    @patch("/{artist_id:uuid}", responses=_ERRORS)
    async def patch_artist(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        data: ArtistPatch,
        studio_artist_service: ArtistService,
    ) -> ArtistOut:
        """Tri-state update: omitted fields are kept, ``null`` clears nullable ones."""
        return await studio_artist_service.patch(artist_id, current_user_id, product_id, data)

    @delete("/{artist_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_artist(
        self,
        current_user_id: UUID,
        product_id: str,
        artist_id: UUID,
        studio_artist_service: ArtistService,
    ) -> None:
        """Delete an artist that no project uses (409 ``artist_in_use`` otherwise)."""
        await studio_artist_service.delete(artist_id, current_user_id, product_id)


__all__ = ["StudioArtistController"]
