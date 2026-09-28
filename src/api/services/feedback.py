"""Feedback service — in-product problem reports (submit + admin triage).

Request-scoped and commit-free: the route commits, and only then publishes
the ops event (``publish_submitted``) — the Telegram ping carries a
``report_id`` an admin will immediately look up, so it must never precede
durability.

Data-exposure rule: user-authored text (``message``, ``admin_note``) and
client context (``client_path``, ``user_agent``) never leave apex — not in
the ops payload, not in any log event.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import msgspec
import structlog

from src.api.schemas.feedback import FeedbackReportAdmin
from src.api.schemas.ops_events import FeedbackSubmittedOpsPayload, OpsEventType
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.core.enums import FeedbackCategory, FeedbackStatus
from src.core.library_ref import LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.core.uid import new_id
from src.db.models.feedback import FeedbackReport
from src.db.repositories.feedback import FeedbackReportRepository
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.feedback import FeedbackAdminPatch, FeedbackCreate
    from src.api.services.ops_event_bus import OpsEventBus

logger = structlog.get_logger(__name__)

MESSAGE_MIN_LENGTH = 10
MESSAGE_MAX_LENGTH = 4000
USER_AGENT_MAX_LENGTH = 512


class FeedbackError(Exception):
    """Base error for feedback operations."""


class InvalidFeedbackMessageError(FeedbackError):
    """The message is too short or too long after ``strip()``, or contains NUL. → 400

    Length is counted in Unicode code points (``len()``), not UTF-16 units.
    """


class InvalidFeedbackContextError(FeedbackError):
    """``asset_ref`` is not a well-formed asset reference. → 400"""


class FeedbackContextNotFoundError(FeedbackError):
    """A referenced job/asset is missing or not owned by the caller. → 404

    Deliberately one error for both cases so the response never reveals
    whether someone else's id exists.
    """

    def __init__(self, kind: Literal["job", "asset"]) -> None:
        self.kind = kind
        super().__init__(f"{kind.capitalize()} not found")


class FeedbackNotFoundError(FeedbackError):
    """No report with this id in the request's product. → 404"""


class InvalidFeedbackTransitionError(FeedbackError):
    """The requested status transition is not allowed. → 409"""

    def __init__(self, current: FeedbackStatus, target: FeedbackStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot change status from {current.value!r} to {target.value!r}")


class FeedbackService:
    """Business logic for ``/v1/feedback`` and ``/v1/admin/feedback``."""

    def __init__(self, *, session: AsyncSession, ops_event_bus: OpsEventBus) -> None:
        self._session = session
        self._ops_event_bus = ops_event_bus
        self._repo = FeedbackReportRepository(session)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(
        self,
        *,
        user_id: UUID,
        product_id: str,
        data: FeedbackCreate,
        user_agent: str | None,
    ) -> FeedbackReport:
        """Validate context ownership and stage the row. Does NOT commit or publish.

        Raises:
            InvalidFeedbackMessageError: Message shorter than the minimum or
                longer than the maximum after ``strip()``, or containing NUL.
            InvalidFeedbackContextError: ``asset_ref`` is malformed.
            FeedbackContextNotFoundError: ``job_id``/``asset_ref`` is missing
                or not owned by ``user_id`` (soft-deleted jobs count as missing).
        """
        message = data.message.strip()
        if not MESSAGE_MIN_LENGTH <= len(message) <= MESSAGE_MAX_LENGTH:
            raise InvalidFeedbackMessageError(
                f"Message must be {MESSAGE_MIN_LENGTH}-{MESSAGE_MAX_LENGTH} characters "
                "after trimming"
            )
        if "\x00" in message:
            raise InvalidFeedbackMessageError("Message must not contain NUL characters")

        if (
            data.job_id is not None
            and await JobRepository(self._session).get(data.job_id, user_id=user_id) is None
        ):
            raise FeedbackContextNotFoundError("job")

        asset_source: LibraryAssetSource | None = None
        asset_id: UUID | None = None
        if data.asset_ref is not None:
            try:
                ref = parse_asset_ref(data.asset_ref)
            except ValueError as exc:
                raise InvalidFeedbackContextError("Invalid asset reference") from exc
            if not await self._asset_is_owned(ref.source, ref.asset_id, user_id=user_id):
                raise FeedbackContextNotFoundError("asset")
            asset_source, asset_id = ref.source, ref.asset_id

        report = FeedbackReport(
            id=new_id(),
            product_id=product_id,
            user_id=user_id,
            category=data.category,
            status=FeedbackStatus.OPEN,
            message=message,
            job_id=data.job_id,
            asset_source=asset_source,
            asset_id=asset_id,
            client_path=data.client_path,
            app_version=data.app_version,
            user_agent=_sanitize_user_agent(user_agent),
        )
        self._repo.add(report)
        logger.info(
            "feedback.submitted",
            report_id=str(report.id),
            category=data.category.value,
            product_id=product_id,
        )
        return report

    async def publish_submitted(self, report: FeedbackReport) -> None:
        """Post-commit only. Best-effort (``OpsEventBus.publish`` never raises)."""
        if report.user_id is None:  # pragma: no cover - always set on insert
            return
        await self._ops_event_bus.publish(
            event_type=OpsEventType.FEEDBACK_SUBMITTED,
            product_id=report.product_id,
            payload=FeedbackSubmittedOpsPayload(
                report_id=report.id,
                user_id=report.user_id,
                category=report.category,
                job_id=report.job_id,
            ),
        )

    async def _asset_is_owned(
        self, source: LibraryAssetSource, asset_id: UUID, *, user_id: UUID
    ) -> bool:
        if source is LibraryAssetSource.OUTPUT:
            return await OutputRepository(self._session).get(asset_id, user_id=user_id) is not None
        return await UserImageRepository(self._session).get(asset_id, user_id=user_id) is not None

    # ------------------------------------------------------------------
    # Admin triage
    # ------------------------------------------------------------------

    async def list_for_admin(
        self,
        *,
        product_id: str,
        status: FeedbackStatus | None = None,
        category: FeedbackCategory | None = None,
        limit: int = 30,
        cursor: str | None = None,
    ) -> CursorPage[FeedbackReportAdmin]:
        """Paginated reports of one product, newest first.

        Raises:
            ValueError: If ``cursor`` is malformed.
        """
        cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        rows = await self._repo.list_page(
            product_id=product_id,
            status=status,
            category=category,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        has_more = len(rows) > limit
        page_rows = rows[:limit]

        next_cursor: str | None = None
        if has_more and page_rows:
            last, _email = page_rows[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(
            items=[to_admin_view(report, email) for report, email in page_rows],
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    async def get_for_admin(self, report_id: UUID, *, product_id: str) -> FeedbackReportAdmin:
        """Fetch one report of this product.

        Raises:
            FeedbackNotFoundError: No such report in ``product_id``.
        """
        row = await self._repo.get_with_email(report_id, product_id=product_id)
        if row is None:
            raise FeedbackNotFoundError
        report, email = row
        return to_admin_view(report, email)

    async def update_by_admin(
        self,
        report_id: UUID,
        *,
        product_id: str,
        admin_id: UUID,
        patch: FeedbackAdminPatch,
    ) -> FeedbackReport:
        """Lock the row, validate the transition, stage the change. Does NOT commit.

        ``resolved_at``/``resolved_by`` are written exactly once: only on the
        transition into a terminal status, and terminal statuses have no
        outgoing transitions. The row lock serialises concurrent PATCHes, so
        the loser re-reads the terminal status and gets a 409.

        Returns the staged row; the caller commits and then re-reads it via
        ``get_for_admin`` (``updated_at`` is server-generated at flush time).

        Raises:
            FeedbackNotFoundError: No such report in ``product_id``.
            InvalidFeedbackTransitionError: ``patch.status`` is not reachable
                from the current status (including the current status itself).
        """
        report = await self._repo.get_for_update(report_id, product_id=product_id)
        if report is None:
            raise FeedbackNotFoundError

        if patch.status is not msgspec.UNSET:
            current = FeedbackStatus(report.status)
            if not current.can_transition_to(patch.status):
                raise InvalidFeedbackTransitionError(current, patch.status)
            report.status = patch.status
            if patch.status.is_terminal:
                report.resolved_at = datetime.now(UTC)
                report.resolved_by = admin_id

        if patch.admin_note is not msgspec.UNSET:
            report.admin_note = patch.admin_note

        logger.info(
            "feedback.updated",
            report_id=str(report_id),
            status=report.status,
            admin_id=str(admin_id),
            product_id=product_id,
        )
        return report


def to_admin_view(report: FeedbackReport, user_email: str | None) -> FeedbackReportAdmin:
    """Map a report row (+ joined reporter email) to the admin DTO."""
    asset_ref: str | None = None
    if report.asset_source is not None and report.asset_id is not None:
        asset_ref = format_asset_ref(LibraryAssetSource(report.asset_source), report.asset_id)
    return FeedbackReportAdmin(
        id=report.id,
        category=FeedbackCategory(report.category),
        status=FeedbackStatus(report.status),
        message=report.message,
        user_id=report.user_id,
        user_email=user_email,
        job_id=report.job_id,
        asset_ref=asset_ref,
        client_path=report.client_path,
        app_version=report.app_version,
        user_agent=report.user_agent,
        admin_note=report.admin_note,
        resolved_at=report.resolved_at,
        resolved_by=report.resolved_by,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


def _sanitize_user_agent(user_agent: str | None) -> str | None:
    """Truncate to the column width; drop NUL (asyncpg rejects it in text)."""
    if not user_agent:
        return None
    return user_agent.replace("\x00", "")[:USER_AGENT_MAX_LENGTH] or None
