"""Admin API schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.core.enums import SubscriptionTier, SupportedLocale, UserRole

# --- Request structs ---


class AdminPatchUserRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Partial update for a user. Only provided fields are updated."""

    role: UserRole | None = None
    subscription_tier: SubscriptionTier | None = None
    is_active: bool | None = None
    locale: SupportedLocale | None = None


# --- Response structs ---


class AdminUserResponse(msgspec.Struct, kw_only=True):
    """Admin view of a single user."""

    id: UUID
    email: str
    display_name: str | None
    role: str
    subscription_tier: str
    is_active: bool
    email_verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class AdminUserListResponse(msgspec.Struct, kw_only=True):
    """Paginated list of users."""

    items: list[AdminUserResponse]
    total: int


class AdminOrgResponse(msgspec.Struct, kw_only=True):
    """Admin view of a single organisation."""

    id: UUID
    name: str
    slug: str
    owner_id: UUID
    is_active: bool
    member_count: int
    token_balance: int
    created_at: datetime


class AdminOrgListResponse(msgspec.Struct, kw_only=True):
    """Paginated list of organisations."""

    items: list[AdminOrgResponse]
    total: int
