"""W1 — every revoke-all path goes through ``SessionTerminationService``.

Covers logout-all, change-password, deactivate-account and refresh-token reuse
detection: each calls ``terminate_all`` exactly once with the right ``op`` /
``source`` (W1-b), reports a failed epoch write once under its own event name
(W1-e), and the service constructors refuse to run without one (W1-f). The
reuse-detection wording is pinned here too (W1-d).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import structlog.testing

from src.api.schemas.ops_events import OpsEventType
from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.age_verification import AgeVerificationService
from src.api.services.auth import (
    AuthService,
    TokenReuseDetectedError,
    _reuse_detected_message,
)
from src.api.services.session_termination import (
    SessionTerminationResult,
    SessionTerminationService,
)
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.product_registry import VEX_CONFIG
from src.db.models import RefreshToken
from src.db.repositories.user_identity import UserIdentityRepository
from tests.legal_support import TEST_REQUEST_CONTEXT, make_legal_acceptance_service
from tests.revocation_support import make_session_termination

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

pytestmark = pytest.mark.unit

_EPOCH = 1_800_000_000
_RESULT = SessionTerminationResult(revoked_refresh_tokens=4, epoch=_EPOCH)

_PUSH_CLEANUP = "src.api.services.session_termination.delete_user_push_subscriptions"


def _jwt() -> JWTService:
    return JWTService(
        JWTConfig(
            secret_key="test_secret_key_for_testing_only_256bits",
            access_token_expire_minutes=15,
            refresh_token_expire_days=7,
        )
    )


def _noop_revocation() -> TokenRevocationService:
    return TokenRevocationService(None, max_token_ttl_seconds=0)


def _fake_terminator(result: SessionTerminationResult = _RESULT) -> MagicMock:
    fake = MagicMock(spec=SessionTerminationService)
    fake.terminate_all = AsyncMock(return_value=result)
    return fake


def _auth_service(repo: AsyncMock, terminator: Any) -> AuthService:
    return AuthService(
        legal_acceptance_service=make_legal_acceptance_service(),
        repository=repo,
        jwt_service=_jwt(),
        password_service=PasswordService(),
        token_revocation_service=_noop_revocation(),
        session_termination=terminator,
    )


def _user_row() -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    user.email = "u@example.com"
    user.password_hash = "hashed_pw"
    user.product_id = "vex"
    return user


def _user_service(repo: AsyncMock, terminator: Any, identities: Any = None) -> UserService:
    password = MagicMock()
    password.averify = AsyncMock(return_value=True)
    password.ahash = AsyncMock(return_value="new_hash")
    return UserService(
        legal_acceptance_service=make_legal_acceptance_service(),
        repository=repo,
        password_service=password,
        age_verification_service=AgeVerificationService(),
        identity_repository=identities or AsyncMock(spec=UserIdentityRepository),
        session_termination=terminator,
    )


def _revoked_token(user_id: Any, family_id: Any) -> MagicMock:
    token = MagicMock(spec=RefreshToken)
    token.user_id = user_id
    token.family_id = family_id
    token.is_revoked = True
    token.revoked_reason = None  # legacy / theft path, not a benign bulk revocation
    return token


class TestW1bEachPathCallsTerminateAllOnce:
    async def test_w1_b_logout_all(self) -> None:
        repo, terminator, user_id = AsyncMock(), _fake_terminator(), uuid4()

        count = await _auth_service(repo, terminator).logout_all(user_id)

        assert count == 4
        terminator.terminate_all.assert_awaited_once_with(user_id, op="logout_all", source="auth")
        repo.revoke_all_refresh_tokens.assert_not_awaited()

    async def test_w1_b_change_password_terminates_after_storing_the_new_hash(self) -> None:
        repo, terminator, user = AsyncMock(), _fake_terminator(), _user_row()
        repo.get_user = AsyncMock(return_value=user)
        order = MagicMock()
        order.attach_mock(repo.update_user, "store_hash")
        order.attach_mock(terminator.terminate_all, "terminate")

        await _user_service(repo, terminator).change_password(
            user.id, current_password="old", new_password="new"
        )

        terminator.terminate_all.assert_awaited_once_with(
            user.id, op="change_password", source="user"
        )
        assert [c[0] for c in order.mock_calls] == ["store_hash", "terminate"]
        repo.revoke_all_refresh_tokens.assert_not_awaited()

    async def test_w1_b_deactivate_account_order_is_soft_delete_unlink_terminate(self) -> None:
        repo, terminator, user = AsyncMock(), _fake_terminator(), _user_row()
        repo.soft_delete_user = AsyncMock(return_value=user)
        identities = AsyncMock(spec=UserIdentityRepository)
        order = MagicMock()
        order.attach_mock(repo.soft_delete_user, "soft_delete")
        order.attach_mock(identities.delete_for_user, "unlink")
        order.attach_mock(terminator.terminate_all, "terminate")

        await _user_service(repo, terminator, identities).deactivate_account(
            user.id, product=VEX_CONFIG, context=TEST_REQUEST_CONTEXT
        )

        terminator.terminate_all.assert_awaited_once_with(
            user.id, op="deactivate_account", source="user"
        )
        steps = {"soft_delete", "unlink", "terminate"}
        assert [c[0] for c in order.mock_calls if c[0] in steps] == [
            "soft_delete",
            "unlink",
            "terminate",
        ]
        repo.revoke_all_refresh_tokens.assert_not_awaited()

    async def test_w1_b_reuse_detection_stamps_the_family_first_then_terminates_all(
        self,
    ) -> None:
        user_id, family_id = uuid4(), uuid4()
        repo, terminator = AsyncMock(), _fake_terminator()
        repo.get_refresh_token_owner.return_value = user_id
        repo.get_refresh_token_by_hash_for_update.return_value = _revoked_token(user_id, family_id)
        repo.revoke_token_family.return_value = 1
        order = MagicMock()
        order.attach_mock(repo.revoke_token_family, "stamp_family")
        order.attach_mock(terminator.terminate_all, "terminate")

        with pytest.raises(TokenReuseDetectedError) as excinfo:
            await _auth_service(repo, terminator).refresh_tokens("stolen")

        # Pitfall: the stolen family must be stamped reuse_detected BEFORE the bulk
        # revocation, or its replays would be classified as a benign lost race.
        assert [c[0] for c in order.mock_calls] == ["stamp_family", "terminate"]
        repo.revoke_token_family.assert_awaited_once_with(family_id)
        terminator.terminate_all.assert_awaited_once_with(
            user_id, op="token_reuse_detected", source="auth"
        )
        repo.revoke_all_refresh_tokens.assert_not_awaited()
        assert str(excinfo.value) == _reuse_detected_message(bulk_access_revoked=True)


class TestW1dReuseDetectedMessage:
    def test_w1_d_epoch_landed_says_signed_out_on_all_devices(self) -> None:
        message = _reuse_detected_message(bulk_access_revoked=True)

        assert "signed out on all devices" in message
        assert "All sessions have been invalidated" not in message

    def test_w1_d_epoch_failed_keeps_the_caveat_and_says_every_device_must_sign_in(self) -> None:
        message = _reuse_detected_message(bulk_access_revoked=False)

        assert "Every device must sign in again" in message
        assert "may stay active until their access tokens expire" in message
        assert "signed out on all devices" not in message


def _failing_epoch_stack() -> tuple[SessionTerminationService, MagicMock, AsyncMock]:
    """A real service over a configured-but-failing Redis epoch write and a recording ops bus."""
    repo = AsyncMock()
    repo.revoke_all_refresh_tokens.return_value = 2
    revocation = MagicMock()
    revocation.enabled = True
    revocation.revoke_user_sessions = AsyncMock(return_value=None)
    ops = MagicMock()
    ops.publish = AsyncMock()
    service = make_session_termination(
        user_repo=repo, token_revocation=revocation, ops_event_bus=ops
    )
    return service, ops, repo


async def _run_logout_all(service: SessionTerminationService, repo: AsyncMock) -> str:
    await _auth_service(repo, service).logout_all(uuid4())
    return "logout_all"


async def _run_change_password(service: SessionTerminationService, repo: AsyncMock) -> str:
    user = _user_row()
    repo.get_user = AsyncMock(return_value=user)
    await _user_service(repo, service).change_password(
        user.id, current_password="old", new_password="new"
    )
    return "change_password"


async def _run_deactivate(service: SessionTerminationService, repo: AsyncMock) -> str:
    user = _user_row()
    repo.soft_delete_user = AsyncMock(return_value=user)
    await _user_service(repo, service).deactivate_account(
        user.id, product=VEX_CONFIG, context=TEST_REQUEST_CONTEXT
    )
    return "deactivate_account"


async def _run_reuse(service: SessionTerminationService, repo: AsyncMock) -> str:
    user_id = uuid4()
    repo.get_refresh_token_owner.return_value = user_id
    repo.get_refresh_token_by_hash_for_update.return_value = _revoked_token(user_id, uuid4())
    repo.revoke_token_family.return_value = 1
    with pytest.raises(TokenReuseDetectedError):
        await _auth_service(repo, service).refresh_tokens("stolen")
    return "token_reuse_detected"


@pytest.mark.parametrize(
    ("run", "source"),
    [
        (_run_logout_all, "auth"),
        (_run_change_password, "user"),
        (_run_deactivate, "user"),
        (_run_reuse, "auth"),
    ],
    ids=["logout_all", "change_password", "deactivate_account", "token_reuse_detected"],
)
async def test_w1_e_failed_epoch_write_is_reported_exactly_once(
    run: Callable[[SessionTerminationService, AsyncMock], Awaitable[str]], source: str
) -> None:
    service, ops, repo = _failing_epoch_stack()

    with patch(_PUSH_CLEANUP, new=AsyncMock()), structlog.testing.capture_logs() as logs:
        op = await run(service, repo)

    failed = [entry for entry in logs if str(entry["event"]).endswith(".bulk_revocation_failed")]
    assert [entry["event"] for entry in failed] == [f"{source}.bulk_revocation_failed"]
    assert failed[0]["op"] == op
    ops.publish.assert_awaited_once()
    kwargs = ops.publish.await_args.kwargs
    assert kwargs["event_type"] == OpsEventType.TOKEN_REVOCATION_FAILED
    assert kwargs["payload"].op == op


class TestW1fSessionTerminationIsRequired:
    def test_w1_f_auth_service_without_session_termination_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="session_termination"):
            AuthService(  # type: ignore[call-arg]
                legal_acceptance_service=make_legal_acceptance_service(),
                repository=AsyncMock(),
                jwt_service=_jwt(),
                password_service=PasswordService(),
                token_revocation_service=_noop_revocation(),
            )

    def test_w1_f_user_service_without_session_termination_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="session_termination"):
            UserService(  # type: ignore[call-arg]
                legal_acceptance_service=make_legal_acceptance_service(),
                repository=AsyncMock(),
                password_service=MagicMock(),
                age_verification_service=AgeVerificationService(),
                identity_repository=AsyncMock(spec=UserIdentityRepository),
            )
