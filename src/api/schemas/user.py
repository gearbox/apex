"""User profile schemas using msgspec."""

from __future__ import annotations

from datetime import date, datetime

import msgspec

from src.core.enums import SupportedLocale

# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------


class UpdateProfileRequest(msgspec.Struct, kw_only=True):
    """Update user profile request.

    All fields are optional - only provided fields are updated.
    """

    display_name: str | None = None
    email: str | None = None
    locale: SupportedLocale | None = None
    age_confirmed: bool | None = None
    date_of_birth: date | None = None


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
    locale: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    age_verified: bool
    age_verified_at: datetime | None = None
    date_of_birth: date | None = None


class UserStatsResponse(msgspec.Struct, kw_only=True):
    """User statistics response."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_outputs: int
    total_uploads: int
    storage_used_bytes: int


class DeleteAccountResponse(msgspec.Struct, kw_only=True):
    """Account deletion response."""

    message: str
    deactivated_at: datetime
