"""V1-c/d/e — the claim's revocation epoch travels callback → handoff → exchange.

``get_current_epoch`` cannot distinguish "no epoch" from "breaker open / read failed",
so the OAuth claim hands the epoch it *wrote* to ``/exchange`` via the handoff instead
of letting ``issue_session`` re-read it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest

from src.api.security.oauth_tx_cookie import binding_hash
from src.api.services.legal.acceptance import RequestContext
from src.api.services.oauth.models import LoginOutcome, OAuthFlow, OAuthHandoff
from src.api.services.oauth.service import OAuthService
from src.api.services.session_termination import SessionTerminationResult
from src.core.product import OAuthProvider
from src.core.product_registry import VEX_CONFIG
from tests.oauth_support import (
    FakeProviderClient,
    InMemoryOAuthFlowStore,
    fake_registry,
    identity,
    oauth_settings,
)

pytestmark = pytest.mark.unit

_EPOCH = 1_800_000_000
_BINDING = "binding-value"
_CONTEXT = RequestContext(ip_address="203.0.113.9", user_agent="ua")


class _Harness:
    def __init__(self, *, verified: bool = False) -> None:
        self.settings = oauth_settings()
        self.store = InMemoryOAuthFlowStore()
        self.user = MagicMock(
            id=uuid4(),
            is_active=True,
            password_hash="hash",
            email_verified_at=datetime.now(UTC) if verified else None,
        )
        self.identity_repo = MagicMock()
        self.identity_repo.get_by_subject = AsyncMock(return_value=None)
        self.identity_repo.get_for_user = AsyncMock(return_value=None)
        self.identity_repo.touch_last_login = AsyncMock()
        self.identity_repo.add = MagicMock()
        self.user_repo = MagicMock()
        self.user_repo.get_active_user_by_email = AsyncMock(return_value=self.user)
        self.user_repo.get_user = AsyncMock(return_value=self.user)
        self.user_repo.get_active_user = AsyncMock(return_value=self.user)
        self.user_repo.claim_unverified_email = AsyncMock(return_value=True)
        self.sessions = MagicMock()
        self.sessions.terminate_all = AsyncMock(
            return_value=SessionTerminationResult(revoked_refresh_tokens=1, epoch=_EPOCH)
        )
        self.auth = MagicMock()
        self.auth.issue_session = AsyncMock()
        nested = MagicMock()
        nested.__aenter__ = AsyncMock(return_value=None)
        nested.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.begin_nested = MagicMock(return_value=nested)
        self.service = OAuthService(
            registry=fake_registry(self.settings, FakeProviderClient(identity())),
            store=self.store,
            identity_repo=self.identity_repo,
            user_repo=self.user_repo,
            auth_service=self.auth,
            sessions=self.sessions,
            session=session,
            settings=self.settings,
        )

    async def callback(self) -> LoginOutcome:
        await self.store.put_flow(
            "st",
            OAuthFlow(
                product_id="vex",
                provider=OAuthProvider.GOOGLE,
                code_verifier="v",
                nonce="n",
                binding_hash=binding_hash(_BINDING),
                return_to=None,
            ),
            600,
        )
        outcome, _return_to = await self.service.resolve_callback(
            product=VEX_CONFIG,
            provider=OAuthProvider.GOOGLE,
            state="st",
            code="c",
            binding=_BINDING,
        )
        assert isinstance(outcome, LoginOutcome)
        return outcome


class TestClaimReturnsEpoch:
    async def test_v1_c_claim_returns_the_epoch_terminate_all_wrote(self) -> None:
        h = _Harness()
        outcome = await h.callback()
        assert outcome.user_id == h.user.id
        assert outcome.not_before_epoch == _EPOCH

    async def test_v1_c_lost_race_returns_none(self) -> None:
        h = _Harness()
        h.user_repo.claim_unverified_email = AsyncMock(return_value=False)
        outcome = await h.callback()
        assert outcome.not_before_epoch is None
        h.sessions.terminate_all.assert_not_awaited()

    async def test_v1_c_failed_epoch_write_returns_none(self) -> None:
        h = _Harness()
        h.sessions.terminate_all = AsyncMock(
            return_value=SessionTerminationResult(revoked_refresh_tokens=1, epoch=None)
        )
        outcome = await h.callback()
        assert outcome.not_before_epoch is None

    async def test_v1_c_verified_link_has_no_epoch(self) -> None:
        h = _Harness(verified=True)
        outcome = await h.callback()
        assert outcome.not_before_epoch is None
        h.sessions.terminate_all.assert_not_awaited()

    async def test_v1_c_existing_identity_login_has_no_epoch(self) -> None:
        h = _Harness()
        h.identity_repo.get_by_subject = AsyncMock(
            return_value=MagicMock(id=uuid4(), user_id=h.user.id)
        )
        outcome = await h.callback()
        assert outcome.not_before_epoch is None
        h.sessions.terminate_all.assert_not_awaited()


class TestHandoffCarriesEpoch:
    async def test_v1_d_issue_redirect_stores_epoch_and_exchange_passes_it_on(self) -> None:
        h = _Harness()
        redirect = await h.service.issue_redirect(
            product=VEX_CONFIG,
            outcome=LoginOutcome(user_id=h.user.id, not_before_epoch=_EPOCH),
            return_to=None,
            binding=_BINDING,
        )
        assert redirect.url
        ((code, handoff),) = h.store.handoffs.items()
        assert handoff.not_before_epoch == _EPOCH

        await h.service.exchange(product=VEX_CONFIG, code=code, binding=_BINDING, context=_CONTEXT)

        h.auth.issue_session.assert_awaited_once_with(
            h.user.id, product_id="vex", context=_CONTEXT, not_before_epoch=_EPOCH
        )

    async def test_v1_d_login_without_claim_passes_none(self) -> None:
        h = _Harness()
        await h.service.issue_redirect(
            product=VEX_CONFIG,
            outcome=LoginOutcome(user_id=h.user.id),
            return_to=None,
            binding=_BINDING,
        )
        ((code, handoff),) = h.store.handoffs.items()
        assert handoff.not_before_epoch is None

        await h.service.exchange(product=VEX_CONFIG, code=code, binding=_BINDING, context=_CONTEXT)

        assert h.auth.issue_session.await_args.kwargs["not_before_epoch"] is None


class TestHandoffDecodeCompat:
    def test_v1_e_handoff_json_without_the_field_decodes_to_none(self) -> None:
        """A handoff written by the previous build (60 s TTL) must still decode after deploy."""
        user_id = uuid4()
        old = f'{{"product_id":"vex","user_id":"{user_id}","binding_hash":"abc"}}'.encode()
        handoff = msgspec.json.decode(old, type=OAuthHandoff)
        assert handoff.not_before_epoch is None
        assert handoff.user_id == user_id

    def test_v1_e_round_trip_keeps_the_epoch(self) -> None:
        original = OAuthHandoff(
            product_id="vex", user_id=uuid4(), binding_hash="abc", not_before_epoch=_EPOCH
        )
        decoded = msgspec.json.decode(msgspec.json.encode(original), type=OAuthHandoff)
        assert decoded == original
