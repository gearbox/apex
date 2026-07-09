"""Admin API schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.core.enums import AdminPermission, SubscriptionTier, SupportedLocale, UserRole

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


# --- Admin Management Request Structs ---


class GrantRoleRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Grant admin or superadmin role. Role must be ADMIN or SUPERADMIN."""

    role: UserRole


class GrantPermissionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Grant or revoke a permission for an admin user."""

    permission: AdminPermission


class PaymentProviderPatchRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Partial runtime update for a product payment provider."""

    is_enabled: bool | None = None
    display_order: int | None = None


# --- Admin Management Response Structs ---


class AdminRoleResponse(msgspec.Struct, kw_only=True):
    """Admin listing entry with role and permissions."""

    id: UUID
    email: str
    display_name: str | None
    role: str
    permissions: list[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AuditLogEntry(msgspec.Struct, kw_only=True):
    """Single admin audit log entry."""

    id: UUID
    actor_id: UUID
    target_user_id: UUID | None
    action: str
    detail: str
    source: str
    created_at: datetime


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
