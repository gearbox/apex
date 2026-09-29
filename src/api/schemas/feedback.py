"""Request/response DTOs for in-product problem reports.

Wire contract: ``docs/contracts/feedback-contract.md``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec

from src.core.enums import FeedbackCategory, FeedbackStatus

# location.pathname only — no query string or fragment (those can carry tokens).
_ClientPath = Annotated[str, msgspec.Meta(max_length=512, pattern=r"\A/[^?#\x00-\x1f\x7f]*\Z")]
# No control characters (asyncpg rejects NUL in text columns). \A…\Z, not ^…$:
# ``$`` also matches before a trailing newline.
_AppVersion = Annotated[
    str, msgspec.Meta(min_length=1, max_length=64, pattern=r"\A[^\x00-\x1f\x7f]+\Z")
]
_AdminNote = Annotated[str, msgspec.Meta(max_length=4000, pattern=r"\A[^\x00]*\Z")]


class FeedbackCreate(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """``POST /v1/feedback`` body."""

    category: FeedbackCategory
    # Length (10-4000 after strip()) and NUL are enforced in FeedbackService.submit()
    # — never on the raw value, or a trailing textarea newline would reject a valid
    # 4000-char report with a framework 400 instead of validation_error.
    message: str
    job_id: UUID | None = None
    asset_ref: str | None = None  # "upload:<uuid>" / "output:<uuid>"
    client_path: _ClientPath | None = None
    app_version: _AppVersion | None = None


class FeedbackCreated(msgspec.Struct, kw_only=True):
    """``POST /v1/feedback`` 201 response."""

    id: UUID
    status: FeedbackStatus
    created_at: datetime


class FeedbackReportAdmin(msgspec.Struct, kw_only=True):
    """Admin view of a report — used for both list items and detail."""

    id: UUID
    category: FeedbackCategory
    status: FeedbackStatus
    message: str
    user_id: UUID | None
    user_email: str | None
    job_id: UUID | None
    asset_ref: str | None
    # Relative path (like MediaVariant.url); set iff asset_ref is. Admin-only proxy route.
    asset_url: str | None
    client_path: str | None
    app_version: str | None
    user_agent: str | None
    admin_note: str | None
    resolved_at: datetime | None
    resolved_by: UUID | None
    created_at: datetime
    updated_at: datetime


class FeedbackAdminPatch(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """``PATCH /v1/admin/feedback/{id}`` body. Omitted = unchanged; ``admin_note: null`` clears."""

    status: FeedbackStatus | msgspec.UnsetType = msgspec.UNSET
    admin_note: _AdminNote | msgspec.UnsetType | None = msgspec.UNSET
