"""Library endpoints — unified grid/detail over uploads and generation outputs.

Endpoints:
  GET    /v1/library                          — paginated library grid
  GET    /v1/library/assets/{asset_ref}        — asset detail
  GET    /v1/library/groups/{job_id}           — generation group detail
  PATCH  /v1/library/assets/{asset_ref}        — update mutable metadata
  PUT    /v1/library/assets/{asset_ref}/favorite    — mark favorite
  DELETE /v1/library/assets/{asset_ref}/favorite    — unmark favorite
  DELETE /v1/library/assets/{asset_ref}        — delete asset
  POST   /v1/library/assets/bulk               — bulk favorite/project/delete
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get, patch, post, put
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.params import Parameter
from litestar.status_codes import (
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.library import (
    BulkOperation,
    BulkOperationResult,
    LibraryAssetDetail,
    LibraryAssetItem,
    LibraryAssetPatch,
    LibraryGroupDetail,
)
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import (
    LibraryBulkValidationError,
    LibraryProjectNotFoundError,
    LibraryService,
    LibraryValidationError,
)
from src.core.enums import LibrarySort, OutputMediaType
from src.core.library_ref import LibraryAssetSource
from src.core.product import ProductConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_ASSET_NOT_FOUND = "Library asset not found"
_GROUP_NOT_FOUND = "Generation group not found"
_PROJECT_NOT_FOUND = "Library project not found"


class LibraryController(Controller):
    """Library grid, detail, and mutation endpoints."""

    path = "/v1/library"
    tags: Sequence[str] | None = ("Library",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/")
    async def list_assets(
        self,
        current_user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        session: AsyncSession,
        library_service: LibraryService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
        source: LibraryAssetSource | None = None,
        media_type: OutputMediaType | None = None,
        model: str | None = None,
        favorite: bool | None = None,
        project_id: UUID | None = None,
        expiring: bool | None = None,
        query: Annotated[str | None, Parameter(max_length=200)] = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        sort: LibrarySort = LibrarySort.NEWEST,
    ) -> Response[CursorPage[LibraryAssetItem] | ErrorEnvelope]:
        """Paginated library grid mixing uploads and generation outputs."""
        try:
            page = await library_service.list_assets(
                current_user_id,
                product_id,
                product_config,
                session=session,
                limit=limit,
                cursor=cursor,
                source=source,
                media_type=media_type,
                model=model,
                favorite=favorite,
                project_id=project_id,
                expiring=expiring,
                query=query,
                created_from=created_from,
                created_to=created_to,
                sort=sort,
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

    @get("/assets/{asset_ref:str}")
    async def get_asset_detail(
        self,
        current_user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        asset_ref: str,
        session: AsyncSession,
        library_service: LibraryService,
    ) -> Response[LibraryAssetDetail | ErrorEnvelope]:
        """Full detail view of a single library asset."""
        detail = await library_service.get_asset_detail(
            asset_ref,
            current_user_id,
            product_id,
            product_config,
            session=session,
        )
        if detail is None:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message=_ASSET_NOT_FOUND,
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        return Response(content=detail)

    @get("/groups/{job_id:uuid}")
    async def get_group_detail(
        self,
        current_user_id: UUID,
        product_id: str,
        job_id: UUID,
        session: AsyncSession,
        library_service: LibraryService,
    ) -> Response[LibraryGroupDetail | ErrorEnvelope]:
        """Full detail view of a generation group."""
        detail = await library_service.get_group_detail(
            job_id,
            current_user_id,
            product_id,
            session=session,
        )
        if detail is None:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message=_GROUP_NOT_FOUND,
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        return Response(content=detail)

    @patch("/assets/{asset_ref:str}")
    async def patch_asset(
        self,
        current_user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        asset_ref: str,
        data: LibraryAssetPatch,
        session: AsyncSession,
        library_service: LibraryService,
    ) -> Response[LibraryAssetDetail | ErrorEnvelope]:
        """Update mutable metadata (display_title, project_id)."""
        try:
            detail = await library_service.patch_asset(
                asset_ref,
                data,
                current_user_id,
                product_id,
                product_config,
                session=session,
            )
        except LibraryValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        except LibraryProjectNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message=_PROJECT_NOT_FOUND,
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        if detail is None:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message=_ASSET_NOT_FOUND,
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        return Response(content=detail)

    @put("/assets/{asset_ref:str}/favorite", status_code=HTTP_204_NO_CONTENT)
    async def add_favorite(
        self,
        current_user_id: UUID,
        product_id: str,
        asset_ref: str,
        session: AsyncSession,
        library_service: LibraryService,
    ) -> None:
        """Mark an asset as favorite. Idempotent."""
        found = await library_service.set_favorite(
            asset_ref, value=True, user_id=current_user_id, product_id=product_id, session=session
        )
        if not found:
            raise NotFoundException(detail=_ASSET_NOT_FOUND)

    @delete("/assets/{asset_ref:str}/favorite", status_code=HTTP_204_NO_CONTENT)
    async def remove_favorite(
        self,
        current_user_id: UUID,
        product_id: str,
        asset_ref: str,
        session: AsyncSession,
        library_service: LibraryService,
    ) -> None:
        """Unmark an asset as favorite. Idempotent."""
        found = await library_service.set_favorite(
            asset_ref, value=False, user_id=current_user_id, product_id=product_id, session=session
        )
        if not found:
            raise NotFoundException(detail=_ASSET_NOT_FOUND)

    @post("/assets/bulk")
    async def bulk_apply(
        self,
        current_user_id: UUID,
        product_id: str,
        data: BulkOperation,
        session: AsyncSession,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
    ) -> Response[BulkOperationResult | ErrorEnvelope]:
        """Bulk favorite/project-assign/delete up to 100 assets in one request.

        Validates every asset_ref before executing anything — a single bad
        ref fails the whole request with a 400 listing every offender
        (never a silent partial skip).
        """
        try:
            result = await library_service.bulk_apply(
                data,
                current_user_id,
                product_id,
                session=session,
                content_proxy=content_proxy,
            )
        except LibraryBulkValidationError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_asset_refs",
                    message=str(exc),
                    status_code=HTTP_400_BAD_REQUEST,
                    detail={"invalid_refs": exc.invalid_refs},
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        except LibraryProjectNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message=_PROJECT_NOT_FOUND,
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        return Response(content=result)

    @delete("/assets/{asset_ref:str}", status_code=HTTP_204_NO_CONTENT)
    async def delete_asset(
        self,
        current_user_id: UUID,
        product_id: str,
        asset_ref: str,
        session: AsyncSession,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
    ) -> None:
        """Delete a library asset (output or upload). Cannot be undone."""
        found = await library_service.delete_asset(
            asset_ref,
            current_user_id,
            product_id,
            session=session,
            content_proxy=content_proxy,
        )
        if not found:
            raise NotFoundException(detail=_ASSET_NOT_FOUND)
