"""Integration tests for BillingRepository against a real PostgreSQL database."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import AccountType, TransactionType
from src.db.repositories.billing import BillingRepository

# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------


async def test_get_account_returns_existing(
    billing_repo: BillingRepository, make_token_account
) -> None:
    """get_account returns the account by primary key."""
    account = await make_token_account()
    found = await billing_repo.get_account(account.id)
    assert found is not None
    assert found.id == account.id


async def test_get_account_returns_none_for_unknown(
    billing_repo: BillingRepository,
) -> None:
    """get_account returns None for an unknown UUID."""
    assert await billing_repo.get_account(uuid4()) is None


# ---------------------------------------------------------------------------
# get_account_with_organization
# ---------------------------------------------------------------------------


async def test_get_account_with_organization_enterprise(
    billing_repo: BillingRepository, make_token_account, make_org
) -> None:
    """get_account_with_organization eagerly loads the organization relationship."""
    org = await make_org()
    account = await make_token_account(account_type="enterprise", org=org)
    found = await billing_repo.get_account_with_organization(account.id)
    assert found is not None
    assert found.organization is not None
    assert found.organization.id == org.id


async def test_get_account_with_organization_personal_no_org(
    billing_repo: BillingRepository, make_token_account
) -> None:
    """get_account_with_organization on a personal account has no organization."""
    account = await make_token_account(account_type="personal")
    found = await billing_repo.get_account_with_organization(account.id)
    assert found is not None
    assert found.organization is None


# ---------------------------------------------------------------------------
# get_account_for_update
# ---------------------------------------------------------------------------


async def test_get_account_for_update_returns_account(
    billing_repo: BillingRepository, make_token_account
) -> None:
    """get_account_for_update acquires a FOR UPDATE lock and returns the account."""
    account = await make_token_account()
    found = await billing_repo.get_account_for_update(account.id)
    assert found is not None
    assert found.id == account.id


async def test_get_account_for_update_returns_none_for_unknown(
    billing_repo: BillingRepository,
) -> None:
    """get_account_for_update returns None when account does not exist."""
    assert await billing_repo.get_account_for_update(uuid4()) is None


# ---------------------------------------------------------------------------
# get_account_by_user
# ---------------------------------------------------------------------------


async def test_get_account_by_user_returns_personal_account(
    billing_repo: BillingRepository, make_user, make_token_account
) -> None:
    """get_account_by_user returns the personal account for a user."""
    user = await make_user(email=f"personal-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    found = await billing_repo.get_account_by_user(user.id)
    assert found is not None
    assert found.id == account.id


async def test_get_account_by_user_no_account_returns_none(
    billing_repo: BillingRepository, make_user
) -> None:
    """get_account_by_user returns None when user has no personal account."""
    user = await make_user(email=f"noaccount-{uuid4().hex[:6]}@example.com")
    assert await billing_repo.get_account_by_user(user.id) is None


# ---------------------------------------------------------------------------
# get_account_by_organization
# ---------------------------------------------------------------------------


async def test_get_account_by_organization_returns_enterprise_account(
    billing_repo: BillingRepository, make_token_account, make_org
) -> None:
    """get_account_by_organization returns the enterprise account for an org."""
    org = await make_org()
    account = await make_token_account(account_type="enterprise", org=org)
    found = await billing_repo.get_account_by_organization(org.id)
    assert found is not None
    assert found.id == account.id


async def test_get_account_by_organization_no_account_returns_none(
    billing_repo: BillingRepository, make_org
) -> None:
    """get_account_by_organization returns None when org has no account."""
    org = await make_org()
    assert await billing_repo.get_account_by_organization(org.id) is None


# ---------------------------------------------------------------------------
# create_personal_account
# ---------------------------------------------------------------------------


async def test_create_personal_account(billing_repo: BillingRepository, make_user) -> None:
    """create_personal_account creates an account with type=personal."""
    user = await make_user(email=f"newpersonal-{uuid4().hex[:6]}@example.com")
    account = await billing_repo.create_personal_account(id=uuid4(), user_id=user.id)
    assert account.account_type == AccountType.PERSONAL.value
    assert account.user_id == user.id


async def test_create_personal_account_duplicate_user_raises(
    billing_repo: BillingRepository, make_user, make_token_account
) -> None:
    """Creating a second personal account for the same user raises IntegrityError."""
    user = await make_user(email=f"dupaccount-{uuid4().hex[:6]}@example.com")
    await make_token_account(account_type="personal", user=user)
    with pytest.raises(IntegrityError):
        await billing_repo.create_personal_account(id=uuid4(), user_id=user.id)


# ---------------------------------------------------------------------------
# create_enterprise_account
# ---------------------------------------------------------------------------


async def test_create_enterprise_account(billing_repo: BillingRepository, make_org) -> None:
    """create_enterprise_account creates an account with type=enterprise."""
    org = await make_org()
    account = await billing_repo.create_enterprise_account(id=uuid4(), organization_id=org.id)
    assert account.account_type == AccountType.ENTERPRISE.value
    assert account.organization_id == org.id


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------


async def test_get_balance_new_account_is_zero(
    billing_repo: BillingRepository, make_token_account
) -> None:
    """get_balance returns 0 for a new account with no transactions."""
    account = await make_token_account()
    balance = await billing_repo.get_balance(account.id)
    assert balance == 0


async def test_get_balance_after_credit(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """get_balance reflects credited tokens correctly."""
    user = await make_user(email=f"balcredit-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.CREDIT.value,
        amount=500,
        balance_after=500,
        created_by=user.id,
    )
    balance = await billing_repo.get_balance(account.id)
    assert balance == 500


async def test_get_balance_debit_then_credit(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """get_balance reflects both credits and debits consistently."""
    user = await make_user(email=f"baldebit-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.CREDIT.value,
        amount=1000,
        balance_after=1000,
        created_by=user.id,
    )
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.DEBIT.value,
        amount=-300,
        balance_after=700,
        created_by=user.id,
    )
    balance = await billing_repo.get_balance(account.id)
    assert balance == 700


# ---------------------------------------------------------------------------
# create_transaction
# ---------------------------------------------------------------------------


async def test_create_transaction_persists(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """create_transaction inserts a TokenTransaction row."""
    user = await make_user(email=f"txn-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    txn = await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.CREDIT.value,
        amount=100,
        balance_after=100,
        description="Test credit",
        created_by=user.id,
    )
    assert txn.amount == 100
    assert txn.transaction_type == TransactionType.CREDIT.value


async def test_create_transaction_amount_nonzero_constraint(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """Creating a transaction with amount=0 violates the chk_amount_nonzero constraint."""
    user = await make_user(email=f"txnzero-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    with pytest.raises(IntegrityError):
        await billing_repo.create_transaction(
            id=uuid4(),
            account_id=account.id,
            transaction_type=TransactionType.CREDIT.value,
            amount=0,
            balance_after=0,
            created_by=user.id,
        )


# ---------------------------------------------------------------------------
# list_transactions
# ---------------------------------------------------------------------------


async def test_list_transactions_returns_ordered_by_created_at_desc(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """list_transactions returns transactions ordered by created_at DESC."""
    user = await make_user(email=f"listtxn-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    for i in range(3):
        await billing_repo.create_transaction(
            id=uuid4(),
            account_id=account.id,
            transaction_type=TransactionType.CREDIT.value,
            amount=100 + i,
            balance_after=100 + i,
            created_by=user.id,
        )
    txns, total = await billing_repo.list_transactions(account.id)
    assert total == 3
    assert len(txns) == 3
    # Most recent should be first (amounts 102, 101, 100)
    amounts = [t.amount for t in txns]
    assert amounts == sorted(amounts, reverse=True)


async def test_list_transactions_offset_beyond_end_returns_empty(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """list_transactions with offset beyond total returns empty list."""
    user = await make_user(email=f"txnoff-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    txns, total = await billing_repo.list_transactions(account.id, offset=1000)
    assert txns == [] or not list(txns)


async def test_list_transactions_filter_by_type(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """list_transactions filters by transaction_type when specified."""
    user = await make_user(email=f"txnfilter-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.CREDIT.value,
        amount=100,
        balance_after=100,
        created_by=user.id,
    )
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.DEBIT.value,
        amount=-50,
        balance_after=50,
        created_by=user.id,
    )
    credits, total = await billing_repo.list_transactions(
        account.id, transaction_type=TransactionType.CREDIT.value
    )
    assert total == 1
    assert all(t.transaction_type == TransactionType.CREDIT.value for t in credits)


# ---------------------------------------------------------------------------
# get_debit_for_job / has_refund_for_job
# ---------------------------------------------------------------------------


async def test_get_debit_for_job_found(
    billing_repo: BillingRepository, make_token_account, make_user, make_job
) -> None:
    """get_debit_for_job returns the debit transaction for a job."""
    user = await make_user(email=f"debitjob-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    job = await make_job(user=user)
    txn = await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.DEBIT.value,
        amount=-100,
        balance_after=400,
        job_id=job.id,
        created_by=user.id,
    )
    found = await billing_repo.get_debit_for_job(job.id)
    assert found is not None
    assert found.id == txn.id


async def test_get_debit_for_job_not_found_returns_none(
    billing_repo: BillingRepository,
) -> None:
    """get_debit_for_job returns None for a job with no debit."""
    assert await billing_repo.get_debit_for_job(uuid4()) is None


async def test_has_refund_for_job_false_no_refund(
    billing_repo: BillingRepository,
) -> None:
    """has_refund_for_job returns False when no refund exists."""
    assert await billing_repo.has_refund_for_job(uuid4()) is False


async def test_has_refund_for_job_true_with_refund(
    billing_repo: BillingRepository, make_token_account, make_user, make_job
) -> None:
    """has_refund_for_job returns True when a refund transaction exists."""
    user = await make_user(email=f"refundjob-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    job = await make_job(user=user)
    await billing_repo.create_transaction(
        id=uuid4(),
        account_id=account.id,
        transaction_type=TransactionType.REFUND.value,
        amount=100,
        balance_after=600,
        job_id=job.id,
        created_by=user.id,
    )
    assert await billing_repo.has_refund_for_job(job.id) is True


# ---------------------------------------------------------------------------
# Organization operations
# ---------------------------------------------------------------------------


async def test_get_organization_returns_existing(billing_repo: BillingRepository, make_org) -> None:
    """get_organization returns the org by primary key."""
    org = await make_org()
    found = await billing_repo.get_organization(org.id)
    assert found is not None
    assert found.id == org.id


async def test_get_organization_returns_none_for_unknown(
    billing_repo: BillingRepository,
) -> None:
    """get_organization returns None for an unknown UUID."""
    assert await billing_repo.get_organization(uuid4()) is None


async def test_get_organization_by_slug(billing_repo: BillingRepository, make_org) -> None:
    """get_organization_by_slug finds org by slug."""
    org = await make_org(slug="unique-slug-xyz")
    found = await billing_repo.get_organization_by_slug("unique-slug-xyz")
    assert found is not None
    assert found.id == org.id


async def test_get_organization_by_slug_not_found(
    billing_repo: BillingRepository,
) -> None:
    """get_organization_by_slug returns None for an unknown slug."""
    assert await billing_repo.get_organization_by_slug("no-such-slug") is None


async def test_create_organization(billing_repo: BillingRepository, make_user) -> None:
    """create_organization persists a new Organization."""
    user = await make_user(email=f"orgowner-{uuid4().hex[:6]}@example.com")
    org = await billing_repo.create_organization(
        id=uuid4(),
        name="My Org",
        slug="my-org-slug",
        owner_id=user.id,
    )
    assert org.name == "My Org"
    assert org.owner_id == user.id


# ---------------------------------------------------------------------------
# Membership operations
# ---------------------------------------------------------------------------


async def test_create_and_get_membership(
    billing_repo: BillingRepository, make_user, make_org
) -> None:
    """create_membership and get_membership round-trip."""
    user = await make_user(email=f"member-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    member = await billing_repo.create_membership(
        id=uuid4(),
        organization_id=org.id,
        user_id=user.id,
        role="member",
    )
    assert member.role == "member"

    found = await billing_repo.get_membership(org.id, user.id)
    assert found is not None
    assert found.id == member.id


async def test_get_membership_not_found_returns_none(
    billing_repo: BillingRepository, make_org, make_user
) -> None:
    """get_membership returns None when user is not a member."""
    user = await make_user(email=f"nonmember-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    assert await billing_repo.get_membership(org.id, user.id) is None


async def test_delete_membership(billing_repo: BillingRepository, make_user, make_org) -> None:
    """delete_membership removes the membership row."""
    user = await make_user(email=f"delmember-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    await billing_repo.create_membership(
        id=uuid4(), organization_id=org.id, user_id=user.id, role="member"
    )
    result = await billing_repo.delete_membership(org.id, user.id)
    assert result is True
    assert await billing_repo.get_membership(org.id, user.id) is None


async def test_delete_membership_not_found_returns_false(
    billing_repo: BillingRepository, make_org, make_user
) -> None:
    """delete_membership returns False when membership does not exist."""
    user = await make_user(email=f"delmiss-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    assert await billing_repo.delete_membership(org.id, user.id) is False


async def test_list_members(billing_repo: BillingRepository, make_user, make_org) -> None:
    """list_members returns all members of an organization."""
    org = await make_org()
    for i in range(3):
        user = await make_user(email=f"listmember{i}-{uuid4().hex[:6]}@example.com")
        await billing_repo.create_membership(
            id=uuid4(), organization_id=org.id, user_id=user.id, role="member"
        )
    members = await billing_repo.list_members(org.id)
    assert len(members) >= 3


# ---------------------------------------------------------------------------
# get_active_membership
# ---------------------------------------------------------------------------


async def test_get_active_membership_returns_membership(
    billing_repo: BillingRepository, make_user, make_org
) -> None:
    """get_active_membership returns the active membership for a user."""
    user = await make_user(email=f"activemember-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    await billing_repo.create_membership(
        id=uuid4(), organization_id=org.id, user_id=user.id, role="member"
    )
    found = await billing_repo.get_active_membership(user.id)
    assert found is not None


async def test_get_active_membership_no_membership_returns_none(
    billing_repo: BillingRepository, make_user
) -> None:
    """get_active_membership returns None for a user with no org membership."""
    user = await make_user(email=f"nomembership-{uuid4().hex[:6]}@example.com")
    assert await billing_repo.get_active_membership(user.id) is None


# ---------------------------------------------------------------------------
# PricingRule operations
# ---------------------------------------------------------------------------


async def test_create_and_get_pricing_rule(billing_repo: BillingRepository, make_user) -> None:
    """create_pricing_rule and get_pricing_rule round-trip."""
    admin = await make_user(email=f"adminrule-{uuid4().hex[:6]}@example.com")
    rule = await billing_repo.create_pricing_rule(
        id=uuid4(),
        provider="comfyui",
        generation_type="t2i",
        model=None,
        token_cost=50,
        notes="Base cost",
        created_by=admin.id,
    )
    found = await billing_repo.get_pricing_rule(rule.id)
    assert found is not None
    assert found.token_cost == 50


async def test_get_active_price_exact_match(billing_repo: BillingRepository, make_user) -> None:
    """get_active_price returns the most specific active rule (exact model match)."""
    admin = await make_user(email=f"adminprice-{uuid4().hex[:6]}@example.com")
    await billing_repo.create_pricing_rule(
        id=uuid4(),
        provider="grok",
        generation_type="t2i",
        model="grok-imagine-image",
        token_cost=75,
        notes=None,
        created_by=admin.id,
    )
    rule = await billing_repo.get_active_price("grok", "t2i", "grok-imagine-image")
    assert rule is not None
    assert rule.token_cost == 75


async def test_get_active_price_not_found_returns_none(
    billing_repo: BillingRepository,
) -> None:
    """get_active_price returns None when no matching rule exists."""
    rule = await billing_repo.get_active_price("unknown_provider", "t2i", None)
    assert rule is None


# ---------------------------------------------------------------------------
# Payment operations
# ---------------------------------------------------------------------------


async def test_create_and_get_payment(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """create_payment and get_payment round-trip."""
    user = await make_user(email=f"payyuser-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    payment = await billing_repo.create_payment(
        id=uuid4(),
        account_id=account.id,
        payment_provider="stripe",
        external_id=f"pi_{uuid4().hex}",
        status="completed",
        amount_usd=Decimal("9.99"),
        tokens_granted=1000,
        created_by=user.id,
    )
    found = await billing_repo.get_payment(payment.id)
    assert found is not None
    assert found.tokens_granted == 1000


async def test_get_payment_by_external_id(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """get_payment_by_external_id finds payment by Stripe payment intent ID."""
    user = await make_user(email=f"extpay-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    ext_id = f"pi_{uuid4().hex}"
    await billing_repo.create_payment(
        id=uuid4(),
        account_id=account.id,
        payment_provider="stripe",
        external_id=ext_id,
        status="pending",
        amount_usd=Decimal("4.99"),
        tokens_granted=500,
        created_by=user.id,
    )
    found = await billing_repo.get_payment_by_external_id(ext_id)
    assert found is not None
    assert found.external_id == ext_id


async def test_list_payments_filter_by_status(
    billing_repo: BillingRepository, make_token_account, make_user
) -> None:
    """list_payments filters by status correctly."""
    user = await make_user(email=f"listpay-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    await billing_repo.create_payment(
        id=uuid4(),
        account_id=account.id,
        payment_provider="stripe",
        external_id=f"pi_completed_{uuid4().hex}",
        status="completed",
        amount_usd=Decimal("9.99"),
        tokens_granted=1000,
        created_by=user.id,
    )
    await billing_repo.create_payment(
        id=uuid4(),
        account_id=account.id,
        payment_provider="stripe",
        external_id=f"pi_pending_{uuid4().hex}",
        status="pending",
        amount_usd=Decimal("4.99"),
        tokens_granted=500,
        created_by=user.id,
    )
    completed, total = await billing_repo.list_payments(status="completed")
    assert total >= 1
    assert all(p.status == "completed" for p in completed)


# ---------------------------------------------------------------------------
# list_organizations
# ---------------------------------------------------------------------------


class TestListOrganizations:
    async def test_list_organizations_returns_all_by_default(
        self, billing_repo: BillingRepository, make_org
    ) -> None:
        """list_organizations returns all organisations when no filter applied."""
        org = await make_org()
        rows, total = await billing_repo.list_organizations()
        assert total >= 1
        assert any(row[0].id == org.id for row in rows)

    async def test_list_organizations_filters_by_is_active_false(
        self, billing_repo: BillingRepository, make_user, db_session: AsyncSession
    ) -> None:
        """list_organizations returns only inactive orgs when is_active=False."""
        owner = await make_user(email=f"org-inactive-owner-{uuid4().hex[:6]}@example.com")
        from src.db.models.billing import Organization

        inactive_org = Organization(
            id=uuid4(),
            name="Inactive Org",
            slug=f"inactive-{uuid4().hex[:8]}",
            owner_id=owner.id,
            is_active=False,
        )
        db_session.add(inactive_org)
        await db_session.flush()

        rows, total = await billing_repo.list_organizations(is_active=False)
        assert total >= 1
        assert all(not row[0].is_active for row in rows)
        assert any(row[0].id == inactive_org.id for row in rows)

    async def test_list_organizations_includes_member_count(
        self, billing_repo: BillingRepository, make_org, make_user
    ) -> None:
        """Each row includes the correct member count for the organisation."""
        org = await make_org()
        for i in range(2):
            member = await make_user(email=f"memcount-{uuid4().hex[:6]}-{i}@example.com")
            await billing_repo.create_membership(
                id=uuid4(), organization_id=org.id, user_id=member.id, role="member"
            )

        rows, _ = await billing_repo.list_organizations()
        org_row = next((r for r in rows if r[0].id == org.id), None)
        assert org_row is not None
        assert org_row[1] >= 2  # member_count

    async def test_list_organizations_includes_token_balance(
        self, billing_repo: BillingRepository, make_org, make_user, make_token_account
    ) -> None:
        """Each row includes the correct token balance for the organisation's account."""
        org = await make_org()
        user = await make_user(email=f"orgbal-{uuid4().hex[:6]}@example.com")
        account = await make_token_account(account_type="enterprise", org=org)
        await billing_repo.create_transaction(
            id=uuid4(),
            account_id=account.id,
            transaction_type=TransactionType.CREDIT.value,
            amount=750,
            balance_after=750,
            created_by=user.id,
        )

        rows, _ = await billing_repo.list_organizations()
        org_row = next((r for r in rows if r[0].id == org.id), None)
        assert org_row is not None
        assert org_row[2] == 750  # token_balance

    async def test_list_organizations_pagination_limit_offset(
        self, billing_repo: BillingRepository, make_org
    ) -> None:
        """list_organizations respects limit and offset."""
        for _ in range(4):
            await make_org()

        page1, total = await billing_repo.list_organizations(limit=2, offset=0)
        page2, _ = await billing_repo.list_organizations(limit=2, offset=2)
        assert total >= 4
        assert len(page1) <= 2
        assert len(page2) <= 2
        page1_ids = {row[0].id for row in page1}
        page2_ids = {row[0].id for row in page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_organizations_total_reflects_filtered_count(
        self, billing_repo: BillingRepository, make_user, db_session: AsyncSession
    ) -> None:
        """total returned matches only orgs matching the filter."""
        from src.db.models.billing import Organization

        owner = await make_user(email=f"total-filter-owner-{uuid4().hex[:6]}@example.com")
        for _ in range(2):
            inactive_org = Organization(
                id=uuid4(),
                name="Inactive",
                slug=f"inactive-total-{uuid4().hex[:8]}",
                owner_id=owner.id,
                is_active=False,
            )
            db_session.add(inactive_org)
        await db_session.flush()

        _rows, total = await billing_repo.list_organizations(is_active=False)
        assert total >= 2

    async def test_list_organizations_member_count_zero_when_no_members(
        self, billing_repo: BillingRepository, make_org
    ) -> None:
        """An org with no members has member_count=0."""
        org = await make_org()
        rows, _ = await billing_repo.list_organizations()
        org_row = next((r for r in rows if r[0].id == org.id), None)
        assert org_row is not None
        assert org_row[1] == 0  # member_count

    async def test_list_organizations_balance_zero_when_no_account(
        self, billing_repo: BillingRepository, make_org
    ) -> None:
        """An org with no token account has token_balance=0."""
        org = await make_org()
        rows, _ = await billing_repo.list_organizations()
        org_row = next((r for r in rows if r[0].id == org.id), None)
        assert org_row is not None
        assert org_row[2] == 0  # token_balance
