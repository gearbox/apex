"""User profile schemas using msgspec."""

from __future__ import annotations

from datetime import datetime

import msgspec

# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------


class UpdateProfileRequest(msgspec.Struct, kw_only=True):
    """Update user profile request.

    All fields are optional - only provided fields are updated.
    """

    display_name: str | None = None
    email: str | None = None


class ChangePasswordRequest(msgspec.Struct, kw_only=True):
    """Change password request."""

    current_password: str
    new_password: str


# -----------------------------------------------------------------------------
# Response schemas
# -----------------------------------------------------------------------------


class UserProfileResponse(msgspec.Struct, kw_only=True):
    """User profile response."""

    id: str
    email: str
    display_name: str | None
    subscription_tier: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserStatsResponse(msgspec.Struct, kw_only=True):
    """User statistics response."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_outputs: int
    total_uploads: int
    storage_used_bytes: int


class JobSummaryResponse(msgspec.Struct, kw_only=True):
    """Generation job summary for user's job list."""

    id: str
    name: str
    status: str
    generation_type: str
    prompt: str
    output_count: int
    created_at: datetime
    completed_at: datetime | None


class UserJobsResponse(msgspec.Struct, kw_only=True):
    """User's generation jobs list response."""

    items: list[JobSummaryResponse]
    total: int


class DeleteAccountResponse(msgspec.Struct, kw_only=True):
    """Account deletion response."""

    message: str
    deactivated_at: datetime
