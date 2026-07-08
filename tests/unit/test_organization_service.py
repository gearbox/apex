"""Unit tests for OrganizationService (slug generation + membership management)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.billing_errors import OrganizationPermissionError
from src.api.services.organization import OrganizationService, slugify
from src.core.enums import OrgRole

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# slugify helper
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic_lowercase_hyphen(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_unicode_transliteration(self) -> None:
        result = slugify("Héllo Wörld")
        assert result == "hello-world"

    def test_strips_special_chars(self) -> None:
        assert slugify("foo!@#bar") == "foobar"

    def test_deduplicates_hyphens(self) -> None:
        assert slugify("foo  bar") == "foo-bar"

    def test_empty_string_returns_empty(self) -> None:
        assert slugify("") == ""

    def test_strips_leading_trailing_hyphens(self) -> None:
        assert slugify("-foo-") == "foo"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_org(**kwargs: object) -> MagicMock:
    org = MagicMock()
    org.id = kwargs.get("id", uuid4())
    org.name = kwargs.get("name", "Test Org")
    org.slug = kwargs.get("slug", "test-org")
    org.product_id = kwargs.get("product_id", "vex")
    return org


def _make_membership(role: str = OrgRole.OWNER.value, **kwargs: object) -> MagicMock:
    m = MagicMock()
    m.id = uuid4()
    m.organization_id = kwargs.get("organization_id", uuid4())
    m.user_id = kwargs.get("user_id", uuid4())
    m.role = role
    return m


def _make_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.get_organization_by_slug = AsyncMock(return_value=None)
    repo.create_organization = AsyncMock(return_value=_make_org())
    repo.create_enterprise_account = AsyncMock(return_value=MagicMock())
    repo.create_membership = AsyncMock(return_value=_make_membership())
    repo.get_active_membership = AsyncMock(return_value=None)
    repo.get_organization = AsyncMock(return_value=None)
    repo.list_members = AsyncMock(return_value=[])
    repo.get_membership = AsyncMock(return_value=None)
    repo.delete_membership = AsyncMock()
    return repo


# ---------------------------------------------------------------------------
# create_organization
# ---------------------------------------------------------------------------


class TestCreateOrganization:
    async def test_creates_org_with_slug_and_account(self) -> None:
        repo = _make_repo()
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.api.services.organization.BillingRepository",
                lambda _: repo,
            )
            svc = OrganizationService()
            _org, _account = await svc.create_organization(
                "My Org", uuid4(), session=session, product_id="vex"
            )

        repo.create_organization.assert_awaited_once()
        repo.create_enterprise_account.assert_awaited_once()
        repo.create_membership.assert_awaited_once()

    async def test_slug_collision_adds_suffix(self) -> None:
        repo = _make_repo()
        # First call returns existing org (slug taken), second returns None (available)
        repo.get_organization_by_slug = AsyncMock(side_effect=[_make_org(slug="my-org"), None])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.api.services.organization.BillingRepository",
                lambda _: repo,
            )
            svc = OrganizationService()
            await svc.create_organization("my-org", uuid4(), session=session, product_id="vex")

        # Should have tried "my-org" (taken) then "my-org-2" (free)
        calls = [c.args[0] for c in repo.get_organization_by_slug.call_args_list]
        assert calls[0] == "my-org"
        assert calls[1] == "my-org-2"


# ---------------------------------------------------------------------------
# get_user_organization
# ---------------------------------------------------------------------------


class TestGetUserOrganization:
    async def test_returns_none_when_no_membership(self) -> None:
        repo = _make_repo()
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.get_user_organization(uuid4(), session=session)

        assert result is None

    async def test_returns_org_when_membership_found(self) -> None:
        org = _make_org()
        membership = _make_membership()
        repo = _make_repo()
        repo.get_active_membership = AsyncMock(return_value=membership)
        repo.get_organization = AsyncMock(return_value=org)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.get_user_organization(uuid4(), session=session)

        assert result is org


# ---------------------------------------------------------------------------
# get_organization / list_members / get_membership
# ---------------------------------------------------------------------------


class TestSimpleGetters:
    async def test_get_organization_delegates_to_repo(self) -> None:
        org = _make_org()
        repo = _make_repo()
        repo.get_organization = AsyncMock(return_value=org)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.get_organization(org.id, session=session)

        assert result is org

    async def test_list_members_delegates_to_repo(self) -> None:
        members = [_make_membership(), _make_membership()]
        repo = _make_repo()
        repo.list_members = AsyncMock(return_value=members)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.list_members(uuid4(), session=session)

        assert list(result) == members

    async def test_get_membership_delegates_to_repo(self) -> None:
        membership = _make_membership()
        repo = _make_repo()
        repo.get_membership = AsyncMock(return_value=membership)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.get_membership(uuid4(), uuid4(), session=session)

        assert result is membership


# ---------------------------------------------------------------------------
# add_member
# ---------------------------------------------------------------------------


class TestAddMember:
    async def test_adds_member_when_actor_is_owner(self) -> None:
        actor_id = uuid4()
        org_id = uuid4()
        new_user_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value, user_id=actor_id)
        new_membership = _make_membership(role=OrgRole.MEMBER.value, user_id=new_user_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, None, new_membership])
        repo.create_membership = AsyncMock(return_value=new_membership)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.add_member(
                org_id,
                new_user_id,
                OrgRole.MEMBER.value,
                actor_id=actor_id,
                session=session,
                product_id="vex",
            )

        assert result is new_membership

    async def test_raises_when_actor_lacks_permission(self) -> None:
        actor_id = uuid4()
        repo = _make_repo()
        # Actor has no membership
        repo.get_membership = AsyncMock(return_value=None)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(OrganizationPermissionError):
                await svc.add_member(
                    uuid4(),
                    uuid4(),
                    "member",
                    actor_id=actor_id,
                    session=session,
                    product_id="vex",
                )

    async def test_raises_when_user_already_member(self) -> None:
        actor_id = uuid4()
        user_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value)
        existing = _make_membership(role=OrgRole.MEMBER.value)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, existing])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(ValueError, match="already a member"):
                await svc.add_member(
                    uuid4(),
                    user_id,
                    "member",
                    actor_id=actor_id,
                    session=session,
                    product_id="vex",
                )


# ---------------------------------------------------------------------------
# remove_member
# ---------------------------------------------------------------------------


class TestRemoveMember:
    async def test_removes_non_owner_member(self) -> None:
        actor_id = uuid4()
        user_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value)
        target_membership = _make_membership(role=OrgRole.MEMBER.value, user_id=user_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, target_membership])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            await svc.remove_member(uuid4(), user_id, actor_id=actor_id, session=session)

        repo.delete_membership.assert_awaited_once()

    async def test_raises_when_removing_owner(self) -> None:
        actor_id = uuid4()
        user_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value)
        owner_membership = _make_membership(role=OrgRole.OWNER.value, user_id=user_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, owner_membership])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(OrganizationPermissionError, match="owner"):
                await svc.remove_member(uuid4(), user_id, actor_id=actor_id, session=session)

    async def test_raises_when_member_not_found(self) -> None:
        actor_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, None])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(ValueError, match="not a member"):
                await svc.remove_member(uuid4(), uuid4(), actor_id=actor_id, session=session)


# ---------------------------------------------------------------------------
# change_role
# ---------------------------------------------------------------------------


class TestChangeRole:
    async def test_changes_role_when_actor_is_owner(self) -> None:
        actor_id = uuid4()
        user_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value, user_id=actor_id)
        target = _make_membership(role=OrgRole.MEMBER.value, user_id=user_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, target])
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            result = await svc.change_role(
                uuid4(),
                user_id,
                OrgRole.ADMIN.value,
                actor_id=actor_id,
                session=session,
            )

        assert result.role == OrgRole.ADMIN.value

    async def test_raises_when_actor_not_owner(self) -> None:
        actor_id = uuid4()
        # Actor is an admin (not owner)
        actor_membership = _make_membership(role=OrgRole.ADMIN.value, user_id=actor_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(return_value=actor_membership)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(OrganizationPermissionError, match="owner"):
                await svc.change_role(
                    uuid4(), uuid4(), "member", actor_id=actor_id, session=session
                )

    async def test_raises_when_owner_tries_to_self_demote(self) -> None:
        actor_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value, user_id=actor_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(return_value=actor_membership)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(ValueError, match="demote yourself"):
                await svc.change_role(
                    uuid4(),
                    actor_id,
                    OrgRole.ADMIN.value,
                    actor_id=actor_id,
                    session=session,
                )

    async def test_raises_when_target_member_not_found(self) -> None:
        actor_id = uuid4()
        actor_membership = _make_membership(role=OrgRole.OWNER.value, user_id=actor_id)

        repo = _make_repo()
        repo.get_membership = AsyncMock(side_effect=[actor_membership, None])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.organization.BillingRepository", lambda _: repo)
            svc = OrganizationService()
            with pytest.raises(ValueError, match="not a member"):
                await svc.change_role(
                    uuid4(), uuid4(), "member", actor_id=actor_id, session=session
                )
