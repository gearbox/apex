"""Organization API schemas using msgspec."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

# --- Response structs ---


class OrgResponse(msgspec.Struct, kw_only=True):
    """Organization details."""

    id: UUID
    name: str
    slug: str
    owner_id: UUID
    is_active: bool
    created_at: datetime


class AccountSummary(msgspec.Struct, kw_only=True):
    """Summary of a token account."""

    account_id: UUID
    account_type: str
    balance: int


class MemberResponse(msgspec.Struct, kw_only=True):
    """Organization member."""

    id: UUID
    user_id: UUID
    role: str
    joined_at: datetime


class OrgCreateResponse(msgspec.Struct, kw_only=True):
    """Response for organization creation."""

    organization: OrgResponse
    account: AccountSummary
    membership: MemberResponse


class OrgDetailResponse(msgspec.Struct, kw_only=True):
    """Organization detail with user's role and balance."""

    organization: OrgResponse
    role: str
    balance: int


# --- Request structs ---


class CreateOrganizationRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create an organization."""

    name: str


class AddMemberRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to add a member to an organization."""

    user_id: UUID
    role: str  # 'admin' or 'member'


class ChangeMemberRoleRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to change a member's role."""

    role: str  # 'admin' or 'member'
