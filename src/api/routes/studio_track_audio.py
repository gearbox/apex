"""Studio track audio — a track's numbered audio versions (S2a).

Endpoints:
  POST   /v1/studio/tracks/{id}/audio     — attach an uploaded library audio file (201), pinning it
  GET    /v1/studio/tracks/{id}/audio     — keyset page, newest version first
  GET    /v1/studio/audio/{id}            — one audio version
  PATCH  /v1/studio/audio/{id}            — label and realized-version links (the asset is immutable)
  DELETE /v1/studio/audio/{id}            — delete (204); 409 audio_version_is_current for the current one

The current audio is switched with ``PUT /v1/studio/tracks/{id}/current`` (``audio_version_id``).
The file itself is uploaded first (``POST /v1/storage/upload``) and attached here by its ``asset``
ref: the studio never touches ingest or storage. Mutations are legal-enforced like every other
mutating route (none opts out). Errors are :class:`~src.studio.errors.StudioError` subclasses
rendered by one app-level handler.
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
from src.api.schemas.studio import TrackAudioCreate, TrackAudioOut, TrackAudioPatch
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.studio.services.track_audio import TrackAudioService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    400: ResponseSpec(data_container=ErrorEnvelope, description="invalid_cursor."),
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "not_found (the track), audio_version_not_found, asset_not_found or "
            "version_not_found (or studio disabled)."
        ),
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description="audio_version_limit_reached or audio_version_is_current.",
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description="document_invalid, asset_not_eligible or version_kind_mismatch.",
    ),
}


class StudioTrackAudioController(Controller):
    """A track's audio versions: attach, list, read, edit, delete."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    # The feature guard runs first: a disabled studio answers 404 even when unauthenticated.
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @post("/tracks/{track_id:uuid}/audio", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_audio(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: TrackAudioCreate,
        studio_track_audio_service: TrackAudioService,
    ) -> TrackAudioOut:
        """Attach an uploaded audio asset to the track as its next audio version (pins it)."""
        return await studio_track_audio_service.create(track_id, current_user_id, product_id, data)

    @get("/tracks/{track_id:uuid}/audio", responses=_ERRORS)
    async def list_audio(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_track_audio_service: TrackAudioService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[TrackAudioOut]:
        """The track's audio versions, newest (highest version number) first."""
        return await studio_track_audio_service.list_page(
            track_id, current_user_id, product_id, limit=limit, cursor=cursor
        )

    @get("/audio/{audio_id:uuid}", responses=_ERRORS)
    async def get_audio(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        studio_track_audio_service: TrackAudioService,
    ) -> TrackAudioOut:
        """One audio version."""
        return await studio_track_audio_service.get(audio_id, current_user_id, product_id)

    @patch("/audio/{audio_id:uuid}", responses=_ERRORS)
    async def patch_audio(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        data: TrackAudioPatch,
        studio_track_audio_service: TrackAudioService,
    ) -> TrackAudioOut:
        """Relabel an audio version or change the plan / lyrics versions it realizes."""
        return await studio_track_audio_service.patch(audio_id, current_user_id, product_id, data)

    @delete("/audio/{audio_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_audio(
        self,
        current_user_id: UUID,
        product_id: str,
        audio_id: UUID,
        studio_track_audio_service: TrackAudioService,
    ) -> None:
        """Delete an audio version that is not the track's current audio; releases its hold."""
        await studio_track_audio_service.delete(audio_id, current_user_id, product_id)
