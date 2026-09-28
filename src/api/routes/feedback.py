"""In-product problem report submission.

Endpoints:
  POST /v1/feedback — submit a report (201; legal-exempt, IP rate-limited)

Wire contract: ``docs/contracts/feedback-contract.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from litestar import Controller, Request, Response, post
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.feedback import FeedbackCreate, FeedbackCreated
from src.api.security import auth_guard
from src.api.services.feedback import (
    FeedbackContextNotFoundError,
    FeedbackService,
    InvalidFeedbackContextError,
    InvalidFeedbackMessageError,
)
from src.core.enums import FeedbackStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

# Transport bound for this endpoint only (the app default is sized for uploads).
# 4000 chars fully \uXXXX-escaped ≈ 24 KB + other fields; 64 KiB leaves headroom.
FEEDBACK_MAX_BODY_BYTES: Final = 64 * 1024

_ERROR_RESPONSES = {
    HTTP_400_BAD_REQUEST: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "Invalid message (too short or too long after trimming, or NUL) or malformed asset_ref."
        ),
    ),
    HTTP_404_NOT_FOUND: ResponseSpec(
        data_container=ErrorEnvelope,
        description="job_id / asset_ref is missing or not owned by the caller.",
    ),
    HTTP_413_REQUEST_ENTITY_TOO_LARGE: ResponseSpec(
        data_container=ErrorEnvelope,
        description="Request body larger than 64 KiB.",
    ),
}


def _error(error: str, message: str, status_code: int) -> ErrorEnvelope:
    return ErrorEnvelope(error=error, message=message, status_code=status_code)


class FeedbackController(Controller):
    """Authenticated problem-report submission."""

    path = "/v1/feedback"
    tags: Sequence[str] | None = ("Feedback",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    # Legal-exempt: a user blocked by a pending re-acceptance must still be
    # able to report a problem (Terms §11.1 in-product reporting function).
    @post(
        "/",
        status_code=HTTP_201_CREATED,
        opt={"legal_exempt": True},
        responses=_ERROR_RESPONSES,
        request_max_body_size=FEEDBACK_MAX_BODY_BYTES,
    )
    async def submit(
        self,
        request: Request[Any, Any, Any],
        current_user_id: UUID,
        product_id: str,
        data: FeedbackCreate,
        session: AsyncSession,
        feedback_service: FeedbackService,
    ) -> Response[FeedbackCreated | ErrorEnvelope]:
        """Submit a problem report. Operators are pinged with IDs only, never the text."""
        try:
            report = await feedback_service.submit(
                user_id=current_user_id,
                product_id=product_id,
                data=data,
                user_agent=request.headers.get("user-agent"),
            )
        except (InvalidFeedbackMessageError, InvalidFeedbackContextError) as exc:
            return Response(
                content=_error("validation_error", str(exc), HTTP_400_BAD_REQUEST),
                status_code=HTTP_400_BAD_REQUEST,
            )
        except FeedbackContextNotFoundError as exc:
            return Response(
                content=_error(f"{exc.kind}_not_found", str(exc), HTTP_404_NOT_FOUND),
                status_code=HTTP_404_NOT_FOUND,
            )

        # Durable first: the ping carries a report_id an admin looks up at once.
        await session.commit()
        await feedback_service.publish_submitted(report)
        return Response(
            content=FeedbackCreated(
                id=report.id,
                status=FeedbackStatus(report.status),
                created_at=report.created_at,
            ),
            status_code=HTTP_201_CREATED,
        )
