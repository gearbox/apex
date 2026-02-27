"""Organization service for managing organizations and memberships."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

from src.api.services.billing_errors import OrganizationBalanceError, OrganizationPermissionError
from src.core.enums import OrgRole, TransactionType, UserRole
from src.db.repositories.billing import BillingRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.billing import Organization, OrganizationMember, TokenAccount

logger = structlog.get_logger(__name__)


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
            "org.created",
            org_id=str(org.id),
            owner_id=str(owner_id),
            slug=slug,
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

    async def delete_organization(
        self,
        org_id: UUID,
        actor_id: UUID,
        *,
        force_delete: bool = False,
        session: AsyncSession,
    ) -> None:
        """Soft-delete an organization. Only owner or system admin may do this.

        If the organization's token account has a positive balance and
        force_delete is False, raises OrganizationBalanceError. When
        force_delete is True a negative ADMIN_ADJUSTMENT transaction is
        created to zero out the balance before the soft-delete.

        Args:
            org_id: Organization to delete.
            actor_id: User performing the action.
            force_delete: When True, zeroes the balance and proceeds.
            session: Database session.

        Raises:
            OrganizationPermissionError: If actor is not the owner or a system admin.
            OrganizationBalanceError: If balance > 0 and force_delete is False.
        """
        repo = BillingRepository(session)
        user_repo = UserRepository(session)

        # System admin bypasses the membership/owner requirement
        actor = await user_repo.get_active_user(actor_id)
        is_admin = actor is not None and actor.role == UserRole.ADMIN

        if not is_admin:
            membership = await repo.get_membership(org_id, actor_id)
            if membership is None or membership.role != OrgRole.OWNER.value:
                raise OrganizationPermissionError(
                    "Only the organization owner or system admin can delete the organization"
                )

        # Check balance on the enterprise token account
        account = await repo.get_account_by_organization(org_id)
        if account is not None:
            balance = await repo.get_balance(account.id)
            if balance > 0:
                if not force_delete:
                    raise OrganizationBalanceError(balance)
                # Zero out the balance with an adjustment transaction
                await repo.create_transaction(
                    id=uuid4(),
                    account_id=account.id,
                    transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
                    amount=-balance,
                    balance_after=0,
                    description="Organization is force deleted by owner",
                    created_by=actor_id,
                )

        # Soft-delete the organization
        org = await repo.get_organization(org_id)
        if org is not None:
            org.is_active = False
            await session.flush()

        logger.info(
            "org.updated",
            org_id=str(org_id),
            actor_id=str(actor_id),
            force_delete=force_delete,
        )

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
