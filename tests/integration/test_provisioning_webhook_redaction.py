"""Integration coverage for U2 (round-4 remediation): a malformed URL in the
provisioner's failure webhook payload must not 500 out of the trust boundary.

`urlsplit` raises `ValueError` on shapes like `http://[` (unbalanced IPv6
brackets), which ordinary log text — a wrapped/truncated URL, bracketed IPv6
in a container/Cloudflare context — can produce without anyone trying. Before
this was guarded, that shape 500'd the webhook route entirely: the node had
already given up, so this is precisely the one signal apex has that the
session needs to fail and its base reservation refunded, and a crash here
left the session stuck in 'provisioning' with the reservation still debited.
Mirrors test_gpu_session_refund_reconciliation.py's fixtures (real Postgres,
a real GpuSessionService + BillingService) but drives the request through
ProvisioningWebhookService.handle_failure via the actual route handler, since
redact_secrets is the code under test here, not fail_pre_active_session itself.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.status_codes import HTTP_200_OK
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.provisioning import ProvisioningController
from src.api.schemas.provisioning import ProvisionerFailureWebhookBody
from src.api.services.billing import BillingService
from src.api.services.gpu_session import GpuSessionService, NullNodeCooldownStore
from src.api.services.provisioning_webhook import ProvisioningWebhookService
from src.core.enums import GpuSessionStatus
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.billing import BillingRepository
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

pytestmark = pytest.mark.asyncio

_CALLBACK_TOKEN = "webhook-redaction-token"


def _make_webhook_service(
    session_factory: async_sessionmaker[AsyncSession], *, billing_service: object
) -> ProvisioningWebhookService:
    gpu_session_service = GpuSessionService(
        vastai_client=AsyncMock(),
        cf_client=AsyncMock(),
        bundle_index=MagicMock(),
        session_factory=session_factory,
        settings=MagicMock(),
        billing_service=billing_service,  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=AsyncMock(),
    )
    settings = MagicMock()
    settings.github_content_token = ""
    settings.hf_token = ""
    settings.civitai_api_token = ""
    return ProvisioningWebhookService(
        gpu_session_service=gpu_session_service,
        session_factory=session_factory,
        settings=settings,
    )


async def test_malformed_url_in_error_is_200_fails_session_and_refunds(
    db_engine: AsyncEngine,
) -> None:
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=new_id(),
        email=f"webhook-redaction-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    account = TokenAccount(id=new_id(), account_type="personal", user_id=user.id, product_id="vex")
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        account_id=account.id,
        callback_token_hash=hashlib.sha256(_CALLBACK_TOKEN.encode()).hexdigest(),
    )

    try:
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(account)
            await db.flush()
            # A balance to reserve the base reservation debit against.
            db.add(
                TokenTransaction(
                    id=new_id(),
                    account_id=account.id,
                    transaction_type="credit",
                    amount=10_000,
                    balance_after=10_000,
                    product_id="vex",
                )
            )
            await db.flush()
            db.add(gpu_session)
            await db.flush()
            await BillingService().check_and_reserve(
                account.id,
                500,
                gpu_session.id,
                metadata={"type": "gpu_session_reservation"},
                description="GPU session base reservation",
                session=db,
                product_id="vex",
                user_id=user.id,
            )

        webhook_service = _make_webhook_service(session_factory, billing_service=BillingService())
        controller = object.__new__(ProvisioningController)

        response = await ProvisioningController.webhook.fn(
            controller,
            session_id=gpu_session.id,
            data=ProvisionerFailureWebhookBody(
                action="continue",
                manifest="see http://[ for details",
                error="script fetch failed: http://[",
                container_id="123",
                timestamp="2026-09-13T12:36:41",
            ),
            provisioning_webhook_service=webhook_service,
            token=_CALLBACK_TOKEN,
        )

        assert response.status_code == HTTP_200_OK

        async with session_factory() as db:
            refreshed = await GpuSessionRepository(db).get_by_id(gpu_session.id)
            assert refreshed is not None
            assert refreshed.status == GpuSessionStatus.failed
            has_refund = await BillingRepository(db).has_refund_for_job(gpu_session.id)
            assert has_refund is True
    finally:
        async with session_factory() as db:
            # token_transactions carries a BEFORE UPDATE OR DELETE trigger
            # (enforce_token_transactions_immutable) blocking all mutation, by
            # name per project_token_transactions_immutable.
            await db.execute(
                text(
                    "ALTER TABLE token_transactions DISABLE TRIGGER "
                    "enforce_token_transactions_immutable"
                )
            )
            await db.execute(
                delete(TokenTransaction).where(TokenTransaction.account_id == account.id)
            )
            await db.execute(
                text(
                    "ALTER TABLE token_transactions ENABLE TRIGGER "
                    "enforce_token_transactions_immutable"
                )
            )
            await db.execute(delete(GpuSession).where(GpuSession.id == gpu_session.id))
            await db.execute(delete(TokenAccount).where(TokenAccount.id == account.id))
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
