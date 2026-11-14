"""Studio tracks — track detail, versioned documents, current pair and Suno export.

Endpoints:
  GET    /v1/studio/tracks/{id}                    — track + current plan / lyrics
  PATCH  /v1/studio/tracks/{id}                    — rename
  DELETE /v1/studio/tracks/{id}                    — delete (versions go with it)
  GET    /v1/studio/tracks/{id}/versions           — paginated summaries (?kind=&cursor=&limit=)
  POST   /v1/studio/tracks/{id}/versions           — create a version (201) or replay it (200)
  GET    /v1/studio/versions/{id}                  — one version with its content
  DELETE /v1/studio/versions/{id}                  — delete a non-current, childless version
  PUT    /v1/studio/tracks/{id}/current            — switch the current plan / lyrics
  GET    /v1/studio/tracks/{id}/exports/suno       — deterministic Suno rendering (pure GET)
  POST   /v1/studio/tracks/{id}/imports/suno       — paste Suno's boxes back: 201 written, 200 not
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, Response, delete, get, patch, post, put
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio import (
    CurrentPut,
    LyricsVersionDetail,
    PlanVersionDetail,
    SunoExportOut,
    SunoImportOut,
    SunoImportRequest,
    TrackDetail,
    TrackOut,
    TrackPatch,
    VersionCreate,
    VersionCreateOut,
    VersionSummary,
)
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.core.enums import StudioDocumentKind, StudioDocumentOrigin
from src.studio.services.export import ExportService
from src.studio.services.importer import SunoImportService
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    404: ResponseSpec(data_container=ErrorEnvelope, description="Not found (or studio disabled)."),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "version_unchanged, plan_required, lyrics_incompatible_with_plan, "
            "version_limit_reached, version_is_current, version_has_children or "
            "version_is_lyrics_base."
        ),
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "document_invalid or import_invalid — every issue is listed in detail.issues "
            "(import_invalid also maps them to pasted lines in detail.source_lines)."
        ),
    ),
}


class StudioTrackController(Controller):
    """Tracks, their document versions and exports."""

    path = "/v1/studio"
    tags: Sequence[str] | None = ("Studio",)
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    # -- track -------------------------------------------------------------------------------

    @get("/tracks/{track_id:uuid}", responses=_ERRORS)
    async def get_track(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_track_service: TrackService,
    ) -> TrackDetail:
        """A track with its current plan and lyrics — one call opens a track."""
        return await studio_track_service.get_detail(track_id, current_user_id, product_id)

    @patch("/tracks/{track_id:uuid}", responses=_ERRORS)
    async def patch_track(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: TrackPatch,
        studio_track_service: TrackService,
    ) -> TrackOut:
        """Rename a track."""
        return await studio_track_service.patch(track_id, current_user_id, product_id, data)

    @delete("/tracks/{track_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_track(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_track_service: TrackService,
    ) -> None:
        """Delete a track together with all its versions."""
        await studio_track_service.delete(track_id, current_user_id, product_id)

    # -- versions ----------------------------------------------------------------------------

    @get("/tracks/{track_id:uuid}/versions", responses=_ERRORS)
    async def list_versions(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_version_service: DocumentVersionService,
        kind: StudioDocumentKind | None = None,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 30,
        cursor: str | None = None,
    ) -> CursorPage[VersionSummary]:
        """Version summaries (no content), newest first; filter with ``?kind=plan|lyrics``."""
        return await studio_version_service.list_page(
            track_id, kind, current_user_id, product_id, limit=limit, cursor=cursor
        )

    @post("/tracks/{track_id:uuid}/versions", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_version(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: VersionCreate,
        studio_version_service: DocumentVersionService,
    ) -> Response[VersionCreateOut]:
        """Create a plan or lyrics version.

        Idempotent: an identical sibling (same parent and normalized content) is returned with
        ``200`` instead of duplicated, and ``make_current`` is still applied.
        """
        out, created = await studio_version_service.create(
            track_id, data, current_user_id, product_id, origin=StudioDocumentOrigin.USER
        )
        return Response(content=out, status_code=HTTP_201_CREATED if created else HTTP_200_OK)

    @get("/versions/{version_id:uuid}", responses=_ERRORS)
    async def get_version(
        self,
        current_user_id: UUID,
        product_id: str,
        version_id: UUID,
        studio_version_service: DocumentVersionService,
    ) -> PlanVersionDetail | LyricsVersionDetail:
        """One version with its full content."""
        return await studio_version_service.get(version_id, current_user_id, product_id)

    @delete("/versions/{version_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_version(
        self,
        current_user_id: UUID,
        product_id: str,
        version_id: UUID,
        studio_version_service: DocumentVersionService,
    ) -> None:
        """Delete a version that is not current, has no children and is not a lyrics base."""
        await studio_version_service.delete(version_id, current_user_id, product_id)

    @put("/tracks/{track_id:uuid}/current", status_code=HTTP_200_OK, responses=_ERRORS)
    async def put_current(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: CurrentPut,
        studio_version_service: DocumentVersionService,
    ) -> TrackOut:
        """Switch the current plan and/or lyrics; the resulting pair must be compatible."""
        return await studio_version_service.set_current(
            track_id,
            current_user_id,
            product_id,
            plan_version_id=data.plan_version_id,
            lyrics_version_id=data.lyrics_version_id,
        )

    # -- export ------------------------------------------------------------------------------

    @get("/tracks/{track_id:uuid}/exports/suno", responses=_ERRORS)
    async def export_suno(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        studio_export_service: ExportService,
        plan_version_id: UUID | None = None,
        lyrics_version_id: UUID | None = None,
    ) -> SunoExportOut:
        """Render Suno's Title / Style / Exclude / Lyrics. Always renders; never truncates."""
        return await studio_export_service.render_suno(
            track_id,
            current_user_id,
            product_id,
            plan_version_id=plan_version_id,
            lyrics_version_id=lyrics_version_id,
        )

    # -- import ------------------------------------------------------------------------------

    @post("/tracks/{track_id:uuid}/imports/suno", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def import_suno(
        self,
        current_user_id: UUID,
        product_id: str,
        track_id: UUID,
        data: SunoImportRequest,
        studio_suno_import_service: SunoImportService,
    ) -> Response[SunoImportOut]:
        """Paste Suno's Style / Exclude / Lyrics back as the track's current plan and lyrics.

        Imported versions are children of the current ones; content that equals the current
        version is reported as ``unchanged`` instead of failing. ``dry_run`` parses, validates
        and reports without writing. Answers ``201`` when a version row was written, ``200``
        otherwise (a dry run, or nothing changed).
        """
        out = await studio_suno_import_service.import_suno(
            track_id, data, current_user_id, product_id
        )
        return Response(content=out, status_code=HTTP_201_CREATED if out.created else HTTP_200_OK)


__all__ = ["StudioTrackController"]
