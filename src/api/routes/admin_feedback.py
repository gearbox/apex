"""Admin triage of in-product problem reports.

All endpoints require ADMIN or SUPERADMIN and are scoped to the request's
product — a report of another product is indistinguishable from a missing one.

Endpoints:
  GET   /v1/admin/feedback              — cursor-paginated list (filter: status, category)
  GET   /v1/admin/feedback/{report_id}  — detail
  PATCH /v1/admin/feedback/{report_id}  — change status and/or admin_note
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import msgspec
from litestar import Controller, Response, get, patch
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.feedback import FeedbackAdminPatch, FeedbackReportAdmin
from src.api.schemas.pagination import CursorPage
from src.api.security import auth_guard
from src.api.services.feedback import (
    FeedbackNotFoundError,
    FeedbackService,
    InvalidFeedbackTransitionError,
)
from src.core.enums import FeedbackCategory, FeedbackStatus
from src.db.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

_NOT_FOUND_RESPONSES = {
    HTTP_404_NOT_FOUND: ResponseSpec(
        data_container=ErrorEnvelope, description="No such report in this product."
    )
}
_PATCH_RESPONSES = {
    **_NOT_FOUND_RESPONSES,
    HTTP_409_CONFLICT: ResponseSpec(
        data_container=ErrorEnvelope,
        description="Status transition not allowed (terminal status, or same as current).",
    ),
}


def _error(error: str, message: str, status_code: int) -> ErrorEnvelope:
    return ErrorEnvelope(error=error, message=message, status_code=status_code)


_NOT_FOUND = ErrorEnvelope(
    error="feedback_not_found", message="Feedback report not found", status_code=HTTP_404_NOT_FOUND
)


class AdminFeedbackController(Controller):
    """Admin list/read/triage of problem reports."""

    path = "/v1/admin/feedback"
    tags: Sequence[str] | None = ("Admin Feedback",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"admin": Provide(get_current_admin_user)}  # noqa: RUF012

    @get("/")
    async def list_reports(
        self,
        admin: User,  # noqa: ARG002 — dependency enforces the admin role
        product_id: str,
        feedback_service: FeedbackService,
        status: FeedbackStatus | None = None,
        category: FeedbackCategory | None = None,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 30,
        cursor: str | None = None,
    ) -> Response[CursorPage[FeedbackReportAdmin] | ErrorEnvelope]:
        """Reports of the request's product, newest first."""
        try:
            page = await feedback_service.list_for_admin(
                product_id=product_id,
                status=status,
                category=category,
                limit=limit,
                cursor=cursor,
            )
        except ValueError:
            return Response(
                content=_error("invalid_cursor", "Invalid pagination cursor", HTTP_400_BAD_REQUEST),
                status_code=HTTP_400_BAD_REQUEST,
            )
        return Response(content=page)

    @get("/{report_id:uuid}", responses=_NOT_FOUND_RESPONSES)
    async def get_report(
        self,
        admin: User,  # noqa: ARG002 — dependency enforces the admin role
        product_id: str,
        report_id: UUID,
        feedback_service: FeedbackService,
    ) -> Response[FeedbackReportAdmin | ErrorEnvelope]:
        """One report of the request's product."""
        try:
            report = await feedback_service.get_for_admin(report_id, product_id=product_id)
        except FeedbackNotFoundError:
            return Response(content=_NOT_FOUND, status_code=HTTP_404_NOT_FOUND)
        return Response(content=report)

    @patch("/{report_id:uuid}", responses=_PATCH_RESPONSES)
    async def update_report(
        self,
        admin: User,
        product_id: str,
        report_id: UUID,
        data: FeedbackAdminPatch,
        session: AsyncSession,
        feedback_service: FeedbackService,
    ) -> Response[FeedbackReportAdmin | ErrorEnvelope]:
        """Change status (``open → in_progress → resolved | dismissed``) and/or the note."""
        if data.status is msgspec.UNSET and data.admin_note is msgspec.UNSET:
            return Response(
                content=_error(
                    "validation_error",
                    "Provide at least one of status, admin_note",
                    HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        try:
            await feedback_service.update_by_admin(
                report_id, product_id=product_id, admin_id=admin.id, patch=data
            )
        except FeedbackNotFoundError:
            return Response(content=_NOT_FOUND, status_code=HTTP_404_NOT_FOUND)
        except InvalidFeedbackTransitionError as exc:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_status_transition",
                    message=str(exc),
                    status_code=HTTP_409_CONFLICT,
                    detail={"current": exc.current.value, "target": exc.target.value},
                ),
                status_code=HTTP_409_CONFLICT,
            )
        await session.commit()
        return Response(
            content=await feedback_service.get_for_admin(report_id, product_id=product_id)
        )
