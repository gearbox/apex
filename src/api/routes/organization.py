"""Organization API routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get, patch, post
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_404_NOT_FOUND,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_token_payload, get_current_user_id
from src.api.schemas.organization import (
    AccountSummary,
    AddMemberRequest,
    ChangeMemberRoleRequest,
    CreateOrganizationRequest,
    MemberResponse,
    OrgCreateResponse,
    OrgDetailResponse,
    OrgResponse,
)
from src.api.security import auth_guard, recheck_revocation_or_raise
from src.api.security.jwt import TokenPayload
from src.api.services.billing import BillingService
from src.api.services.organization import OrganizationService
from src.api.services.token_revocation import TokenRevocationService

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class OrganizationController(Controller):
    """Organization management endpoints."""

    path = "/v1/organizations"
    tags: Sequence[str] | None = ("Organizations",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "current_user_id": Provide(get_current_user_id),
        "token_payload": Provide(get_current_token_payload),
    }

    @post("/")
    async def create_organization(
        self,
        current_user_id: UUID,
        data: CreateOrganizationRequest,
        session: AsyncSession,
        organization_service: OrganizationService,
        billing_service: BillingService,
        product_id: str,
    ) -> Response[OrgCreateResponse]:
        """Create a new organization."""
        org, account = await organization_service.create_organization(
            data.name, current_user_id, session=session, product_id=product_id
        )
        balance = await billing_service.get_balance(account.id, session=session)

        # Get the owner membership
        members = await organization_service.list_members(org.id, session=session)
        owner_member = next(m for m in members if m.user_id == current_user_id)

        await session.commit()

        return Response(
            content=OrgCreateResponse(
                organization=OrgResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    owner_id=org.owner_id,
                    is_active=org.is_active,
                    created_at=org.created_at,
                ),
                account=AccountSummary(
                    account_id=account.id,
                    account_type=account.account_type,
                    balance=balance,
                ),
                membership=MemberResponse(
                    id=owner_member.id,
                    user_id=owner_member.user_id,
                    role=owner_member.role,
                    joined_at=owner_member.joined_at,
                ),
            ),
            status_code=HTTP_201_CREATED,
        )

    @get("/me")
    async def get_my_organization(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        organization_service: OrganizationService,
        billing_service: BillingService,
    ) -> Response[OrgDetailResponse]:
        """Get current user's organization, role, and balance."""
        org = await organization_service.get_user_organization(current_user_id, session=session)
        if org is None:
            return Response(
                content=OrgDetailResponse(
                    organization=OrgResponse(
                        id=UUID("00000000-0000-0000-0000-000000000000"),
                        name="",
                        slug="",
                        owner_id=UUID("00000000-0000-0000-0000-000000000000"),
                        is_active=False,
                        created_at=org.created_at if org else None,  # type: ignore[arg-type]
                    ),
                    role="",
                    balance=0,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        membership = await organization_service.get_membership(
            org.id, current_user_id, session=session
        )
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        balance = await billing_service.get_balance(account.id, session=session)

        return Response(
            content=OrgDetailResponse(
                organization=OrgResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    owner_id=org.owner_id,
                    is_active=org.is_active,
                    created_at=org.created_at,
                ),
                role=membership.role if membership else "",
                balance=balance,
            ),
            status_code=HTTP_200_OK,
        )

    @get("/{org_id:uuid}")
    async def get_organization(
        self,
        current_user_id: UUID,
        org_id: UUID,
        session: AsyncSession,
        organization_service: OrganizationService,
    ) -> Response[OrgResponse]:
        """Get organization by ID. Requires membership."""
        membership = await organization_service.get_membership(
            org_id, current_user_id, session=session
        )
        if membership is None:
            return Response(content=None, status_code=HTTP_404_NOT_FOUND)  # type: ignore[arg-type]

        org = await organization_service.get_organization(org_id, session=session)
        if org is None:
            return Response(content=None, status_code=HTTP_404_NOT_FOUND)  # type: ignore[arg-type]

        return Response(
            content=OrgResponse(
                id=org.id,
                name=org.name,
                slug=org.slug,
                owner_id=org.owner_id,
                is_active=org.is_active,
                created_at=org.created_at,
            ),
            status_code=HTTP_200_OK,
        )

    @get("/{org_id:uuid}/members")
    async def list_members(
        self,
        current_user_id: UUID,
        org_id: UUID,
        session: AsyncSession,
        organization_service: OrganizationService,
    ) -> Response[list[MemberResponse]]:
        """List organization members. Requires membership."""
        membership = await organization_service.get_membership(
            org_id, current_user_id, session=session
        )
        if membership is None:
            return Response(content=None, status_code=HTTP_404_NOT_FOUND)  # type: ignore[arg-type]

        members = await organization_service.list_members(org_id, session=session)
        return Response(
            content=[
                MemberResponse(
                    id=m.id,
                    user_id=m.user_id,
                    role=m.role,
                    joined_at=m.joined_at,
                )
                for m in members
            ],
            status_code=HTTP_200_OK,
        )

    @post("/{org_id:uuid}/members")
    async def add_member(
        self,
        current_user_id: UUID,
        org_id: UUID,
        data: AddMemberRequest,
        session: AsyncSession,
        organization_service: OrganizationService,
        product_id: str,
        token_payload: TokenPayload,
        token_revocation_service: TokenRevocationService,
    ) -> Response[MemberResponse]:
        """Add a member. Requires admin or owner role.

        Re-checks revocation of the acting member's own session first
        (src/api/security/revocation_recheck.py) — membership grants
        persistent access to a shared, org-billed token account, the same
        durable-side-effect concern that first motivated this pattern for
        push subscriptions.
        """
        await recheck_revocation_or_raise(
            session=session,
            actor_id=current_user_id,
            token_payload=token_payload,
            token_revocation_service=token_revocation_service,
        )
        member = await organization_service.add_member(
            org_id,
            data.user_id,
            data.role,
            actor_id=current_user_id,
            session=session,
            product_id=product_id,
        )
        await session.commit()
        return Response(
            content=MemberResponse(
                id=member.id,
                user_id=member.user_id,
                role=member.role,
                joined_at=member.joined_at,
            ),
            status_code=HTTP_201_CREATED,
        )

    @delete("/{org_id:uuid}", status_code=HTTP_200_OK)
    async def delete_organization(
        self,
        current_user_id: UUID,
        org_id: UUID,
        session: AsyncSession,
        organization_service: OrganizationService,
        force_delete: bool = False,
    ) -> dict[str, Any]:
        """Delete an organization (soft-delete). Only the owner or a system admin may call this.

        Returns 409 if the org has a positive token balance. Pass
        ``force_delete=true`` to zero out the balance and proceed.
        """
        org = await organization_service.get_organization(org_id, session=session)
        if org is None:
            return Response(  # type: ignore[return-value]
                content={"detail": "Organization not found"},
                status_code=HTTP_404_NOT_FOUND,
            )

        await organization_service.delete_organization(
            org_id,
            current_user_id,
            force_delete=force_delete,
            session=session,
        )
        await session.commit()
        return {"message": "Organization deleted"}

    @delete("/{org_id:uuid}/members/{user_id:uuid}", status_code=HTTP_200_OK)
    async def remove_member(
        self,
        current_user_id: UUID,
        org_id: UUID,
        user_id: UUID,
        session: AsyncSession,
        organization_service: OrganizationService,
    ) -> dict[str, Any]:
        """Remove a member. Requires admin or owner role. Cannot remove owner."""
        await organization_service.remove_member(
            org_id, user_id, actor_id=current_user_id, session=session
        )
        await session.commit()
        return {"message": "Member removed"}

    @patch("/{org_id:uuid}/members/{user_id:uuid}")
    async def change_member_role(
        self,
        current_user_id: UUID,
        org_id: UUID,
        user_id: UUID,
        data: ChangeMemberRoleRequest,
        session: AsyncSession,
        organization_service: OrganizationService,
        token_payload: TokenPayload,
        token_revocation_service: TokenRevocationService,
    ) -> Response[MemberResponse]:
        """Change member role. Requires owner role only.

        Re-checks revocation of the acting owner's own session first — see
        ``add_member`` above and ``src/api/security/revocation_recheck.py``
        for why: a role change persistently escalates or demotes access to
        the shared org-billed account.
        """
        await recheck_revocation_or_raise(
            session=session,
            actor_id=current_user_id,
            token_payload=token_payload,
            token_revocation_service=token_revocation_service,
        )
        member = await organization_service.change_role(
            org_id,
            user_id,
            data.role,
            actor_id=current_user_id,
            session=session,
        )
        await session.commit()
        return Response(
            content=MemberResponse(
                id=member.id,
                user_id=member.user_id,
                role=member.role,
                joined_at=member.joined_at,
            ),
            status_code=HTTP_200_OK,
        )
