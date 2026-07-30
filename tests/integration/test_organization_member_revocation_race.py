"""Integration test for review r1 remediation (09r) finding C1.

E2 (in test_admin_grant_role_revocation_race.py) showed that a concurrent
mutual role grant between two superadmins deadlocks unless
``recheck_revocation_or_raise`` locks the acting user's row *and* the
target's row together. C1 observed that this isn't specific to
``grant_role``'s explicit ``UPDATE users`` — ``OrganizationService.add_member``
reaches the same target-row lock *implicitly*, via the ``FOR KEY SHARE``
Postgres takes on the referenced ``users`` row when inserting an
``organization_members`` row with a foreign key to it. This module proves
that FK-lock path deadlocks the same way, and that ``also_lock`` on
``POST /v1/organizations/{org_id}/members`` (src/api/routes/organization.py)
closes it exactly like the explicit-``UPDATE`` case.

Two organizations, each owned by one of two users, who add each other as a
member of their own organization concurrently (mirrored actor/target
pairs) — this fails on the pre-remediation SHA with ``deadlock detected``
(SQLSTATE 40P01).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from litestar.exceptions import NotAuthorizedException
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.organization import OrganizationController
from src.api.schemas.organization import AddMemberRequest
from src.api.security.jwt import JWTConfig, JWTService
from src.api.services.organization import OrganizationService
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import OrgRole
from src.core.uid import new_id
from src.db.models.billing import Organization, OrganizationMember
from src.db.models.user import RefreshToken, User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
PRODUCT_ID = "vex"


class _NullRedis:
    """No-op stand-in — this test only needs is_revoked() to return False,
    never a live revocation write."""

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [None for _ in keys]

    async def get(self, key: str) -> str | None:  # noqa: ARG002
        return None


async def _seed_two_owners_with_orgs(
    engine: AsyncEngine,
    *,
    user_a_id: UUID,
    user_b_id: UUID,
    org1_id: UUID,
    org2_id: UUID,
) -> None:
    """user_a owns org1, user_b owns org2 — set up so each can add the
    *other* as a member of their own organization (mirrored actor/target)."""
    user_a = User(
        id=user_a_id,
        email=f"race-org-a-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
    )
    user_b = User(
        id=user_b_id,
        email=f"race-org-b-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id=PRODUCT_ID,
        is_active=True,
    )
    org1 = Organization(
        id=org1_id,
        name=f"Race Org 1 {uuid4().hex[:8]}",
        slug=f"race-org1-{uuid4().hex[:8]}",
        owner_id=user_a_id,
        product_id=PRODUCT_ID,
    )
    org2 = Organization(
        id=org2_id,
        name=f"Race Org 2 {uuid4().hex[:8]}",
        slug=f"race-org2-{uuid4().hex[:8]}",
        owner_id=user_b_id,
        product_id=PRODUCT_ID,
    )
    owner_membership_a = OrganizationMember(
        id=new_id(),
        organization_id=org1_id,
        user_id=user_a_id,
        role=OrgRole.OWNER.value,
        product_id=PRODUCT_ID,
    )
    owner_membership_b = OrganizationMember(
        id=new_id(),
        organization_id=org2_id,
        user_id=user_b_id,
        role=OrgRole.OWNER.value,
        product_id=PRODUCT_ID,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user_a, user_b, org1, org2, owner_membership_a, owner_membership_b])
        await session.commit()


async def _cleanup(
    engine: AsyncEngine,
    *,
    user_a_id: UUID,
    user_b_id: UUID,
    org1_id: UUID,
    org2_id: UUID,
) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(
            delete(OrganizationMember).where(
                OrganizationMember.organization_id.in_([org1_id, org2_id])
            )
        )
        await session.execute(delete(Organization).where(Organization.id.in_([org1_id, org2_id])))
        await session.execute(
            delete(RefreshToken).where(RefreshToken.user_id.in_([user_a_id, user_b_id]))
        )
        await session.execute(delete(User).where(User.id.in_([user_a_id, user_b_id])))
        await session.commit()


async def _run_add_member(
    *,
    engine: AsyncEngine,
    org_id: UUID,
    actor_id: UUID,
    new_member_id: UUID,
    jwt_service: JWTService,
    token_revocation: TokenRevocationService,
) -> dict[str, object]:
    token, _ = jwt_service.create_access_token(actor_id, product_id=PRODUCT_ID)
    token_payload = jwt_service.decode_access_token(token)
    assert token_payload is not None

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        organization_service = OrganizationService()
        try:
            await OrganizationController.add_member.fn(
                MagicMock(),
                current_user_id=actor_id,
                org_id=org_id,
                data=AddMemberRequest(user_id=new_member_id, role=OrgRole.MEMBER.value),
                session=session,
                organization_service=organization_service,
                product_id=PRODUCT_ID,
                token_payload=token_payload,
                token_revocation_service=token_revocation,
            )
        except NotAuthorizedException:
            return {"status": "rejected"}
        return {"status": "added"}


class TestMutualAddMemberDeadlock:
    """C1 — two org owners adding each other as a member of their own
    organization concurrently must never deadlock. Mirrored actor/target
    pairs (A adds B to org1, B adds A to org2) would otherwise take the
    acting user's row lock first and the target's row lock second — here
    implicitly, via the ``FOR KEY SHARE`` Postgres takes on the
    ``organization_members.user_id`` foreign key during INSERT — forming a
    lock cycle Postgres aborts with SQLSTATE 40P01. ``also_lock`` closes
    this the same way it does for ``grant_role``. This test fails on the
    pre-remediation SHA.
    """

    async def test_mutual_add_member_never_deadlocks(self, db_engine: AsyncEngine) -> None:
        jwt_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        iterations = 50

        for _ in range(iterations):
            user_a_id, user_b_id = new_id(), new_id()
            org1_id, org2_id = new_id(), new_id()
            await _seed_two_owners_with_orgs(
                db_engine,
                user_a_id=user_a_id,
                user_b_id=user_b_id,
                org1_id=org1_id,
                org2_id=org2_id,
            )
            token_revocation = TokenRevocationService(_NullRedis(), max_token_ttl_seconds=3600)  # type: ignore[arg-type]

            try:
                results = await asyncio.gather(
                    _run_add_member(
                        engine=db_engine,
                        org_id=org1_id,
                        actor_id=user_a_id,
                        new_member_id=user_b_id,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                    ),
                    _run_add_member(
                        engine=db_engine,
                        org_id=org2_id,
                        actor_id=user_b_id,
                        new_member_id=user_a_id,
                        jwt_service=jwt_service,
                        token_revocation=token_revocation,
                    ),
                    return_exceptions=True,
                )

                for result in results:
                    assert not isinstance(result, BaseException), (
                        f"mutual add_member must never raise (deadlock or otherwise): {result!r}"
                    )

                statuses = {result["status"] for result in results}  # type: ignore[index]
                assert statuses == {"added"}, (
                    f"both mutual add_member calls should commit without contention "
                    f"rejection: {results}"
                )
            finally:
                await _cleanup(
                    db_engine,
                    user_a_id=user_a_id,
                    user_b_id=user_b_id,
                    org1_id=org1_id,
                    org2_id=org2_id,
                )
