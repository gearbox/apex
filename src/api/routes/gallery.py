"""Gallery endpoints — generation group grid and detail views.

Endpoints:
  GET /v1/gallery           — paginated gallery grid
  GET /v1/gallery/{job_id}  — full detail for a generation group
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, get
from litestar.di import Provide
from litestar.params import Parameter
from litestar.status_codes import HTTP_404_NOT_FOUND
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.gallery import GalleryGridItem, GalleryGroupDetail
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.gallery import GalleryService
from src.core.enums import GenerationType, OutputMediaType

logger = structlog.get_logger(__name__)


class GalleryController(Controller):
    """Gallery grid and detail endpoints."""

    path = "/v1/gallery"
    tags: Sequence[str] | None = ["Gallery"]
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/")
    async def list_gallery(
        self,
        current_user_id: UUID,
        product_id: str,
        session: AsyncSession,
        gallery_service: GalleryService,
        limit: Annotated[int, Parameter(ge=1, le=25)] = 20,
        cursor: str | None = None,
        media_type: OutputMediaType | None = None,
        generation_type: GenerationType | None = None,
        model: str | None = None,
    ) -> CursorPage[GalleryGridItem]:
        """Paginated gallery grid of generation groups."""
        return await gallery_service.list_gallery(
            current_user_id,
            product_id,
            session=session,
            limit=limit,
            cursor=cursor,
            media_type=media_type,
            generation_type=generation_type,
            model=model,
        )

    @get("/{job_id:uuid}")
    async def get_gallery_detail(
        self,
        current_user_id: UUID,
        product_id: str,
        job_id: UUID,
        session: AsyncSession,
        gallery_service: GalleryService,
    ) -> Response[GalleryGroupDetail | ErrorEnvelope]:
        """Full detail view of a generation group."""
        detail = await gallery_service.get_gallery_detail(
            job_id,
            current_user_id,
            product_id,
            session=session,
        )
        if detail is None:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Gallery item not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )
        return Response(content=detail)
