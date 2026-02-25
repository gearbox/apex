"""Organization service for managing organizations and memberships."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.api.services.billing_errors import OrganizationPermissionError
from src.core.enums import OrgRole
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.billing import Organization, OrganizationMember, TokenAccount

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Convert a string to a URL-safe slug.

    Transliterates unicode, lowercases, replaces non-alphanumeric with hyphens,
    and deduplicates hyphens.
    """
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "-", value)
    return value.strip("-")


class OrganizationService:
    """Service for organization management."""

    async def create_organization(
        self,
        name: str,
        owner_id: UUID,
        *,
        session: AsyncSession,
    ) -> tuple[Organization, TokenAccount]:
        """Create Organization + enterprise TokenAccount + owner membership.

        All in the same transaction. Generates slug from name.
        """
        repo = BillingRepository(session)

        # Generate unique slug
        base_slug = slugify(name) or "org"

        slug = base_slug
        suffix = 2
        while await repo.get_organization_by_slug(slug) is not None:
            slug = f"{base_slug}-{suffix}"
            suffix += 1

        # Create organization
        org = await repo.create_organization(
            id=uuid4(),
            name=name,
            slug=slug,
            owner_id=owner_id,
        )

        # Create enterprise token account
        account = await repo.create_enterprise_account(
            id=uuid4(),
            organization_id=org.id,
        )

        # Create owner membership
        await repo.create_membership(
            id=uuid4(),
            organization_id=org.id,
            user_id=owner_id,
            role=OrgRole.OWNER.value,
        )

        logger.info(
            "organization_created org_id=%s owner_id=%s slug=%s",
            org.id,
            owner_id,
            slug,
        )

        return org, account

    async def get_user_organization(
        self,
        user_id: UUID,
        *,
        session: AsyncSession,
    ) -> Organization | None:
        repo = BillingRepository(session)
        membership = await repo.get_active_membership(user_id)
        if membership is None:
            return None
        return await repo.get_organization(membership.organization_id)

    async def get_organization(
        self,
        org_id: UUID,
        *,
        session: AsyncSession,
    ) -> Organization | None:
        repo = BillingRepository(session)
        return await repo.get_organization(org_id)

    async def list_members(
        self,
        org_id: UUID,
        *,
        session: AsyncSession,
    ) -> Sequence[OrganizationMember]:
        repo = BillingRepository(session)
        return await repo.list_members(org_id)

    async def add_member(
        self,
        org_id: UUID,
        user_id: UUID,
        role: str,
        *,
        actor_id: UUID,
        session: AsyncSession,
    ) -> OrganizationMember:
        """Add a member to an organization.

        Raises:
            OrganizationPermissionError: If actor lacks permission.
            ValueError: If user is already a member.
        """
        repo = BillingRepository(session)

        # Verify actor has admin/owner role
        await self._require_admin_or_owner(repo, org_id, actor_id)

        # Check existing membership
        existing = await repo.get_membership(org_id, user_id)
        if existing is not None:
            raise ValueError(f"User {user_id} is already a member of organization {org_id}")

        return await repo.create_membership(
            id=uuid4(),
            organization_id=org_id,
            user_id=user_id,
            role=role,
        )

    async def remove_member(
        self,
        org_id: UUID,
        user_id: UUID,
        *,
        actor_id: UUID,
        session: AsyncSession,
    ) -> None:
        """Remove a member from an organization.

        Raises:
            OrganizationPermissionError: If actor lacks permission or trying to remove owner.
        """
        repo = BillingRepository(session)

        await self._require_admin_or_owner(repo, org_id, actor_id)

        # Cannot remove owner
        member = await repo.get_membership(org_id, user_id)
        if member is None:
            raise ValueError(f"User {user_id} is not a member of organization {org_id}")
        if member.role == OrgRole.OWNER.value:
            raise OrganizationPermissionError("Cannot remove the organization owner")

        await repo.delete_membership(org_id, user_id)

    async def change_role(
        self,
        org_id: UUID,
        user_id: UUID,
        new_role: str,
        *,
        actor_id: UUID,
        session: AsyncSession,
    ) -> OrganizationMember:
        """Change a member's role. Only owner can change roles.

        Raises:
            OrganizationPermissionError: If actor is not owner.
            ValueError: If attempting to demote own owner role.
        """
        repo = BillingRepository(session)

        # Only owner can change roles
        actor_membership = await repo.get_membership(org_id, actor_id)
        if actor_membership is None or actor_membership.role != OrgRole.OWNER.value:
            raise OrganizationPermissionError("Only the owner can change roles")

        if actor_id == user_id and new_role != OrgRole.OWNER.value:
            raise ValueError("Cannot demote yourself from owner. Transfer ownership first.")

        member = await repo.get_membership(org_id, user_id)
        if member is None:
            raise ValueError(f"User {user_id} is not a member of organization {org_id}")

        member.role = new_role
        await session.flush()
        return member

    async def get_membership(
        self,
        org_id: UUID,
        user_id: UUID,
        *,
        session: AsyncSession,
    ) -> OrganizationMember | None:
        repo = BillingRepository(session)
        return await repo.get_membership(org_id, user_id)

    async def _require_admin_or_owner(
        self,
        repo: BillingRepository,
        org_id: UUID,
        actor_id: UUID,
    ) -> OrganizationMember:
        """Verify actor has admin or owner role.

        Raises:
            OrganizationPermissionError: If actor lacks permission.
        """
        membership = await repo.get_membership(org_id, actor_id)
        if membership is None or membership.role not in (
            OrgRole.OWNER.value,
            OrgRole.ADMIN.value,
        ):
            raise OrganizationPermissionError("Insufficient permissions for this action")
        return membership
