"""Tests for authentication service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
import structlog.testing

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.security.jwt import TokenPayload
from src.api.services.auth import (
    AuthService,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    TokenReuseDetectedError,
)
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import RefreshTokenRevocationReason
from src.db.models import RefreshToken, User


def _noop_token_revocation() -> TokenRevocationService:
    """A TokenRevocationService with no Redis client — a visible no-op choice
    for tests that don't exercise revocation (issue #142 A1)."""
    return TokenRevocationService(None, max_token_ttl_seconds=0)


@pytest.fixture
def password_service() -> PasswordService:
    """Create password service."""
    return PasswordService()


@pytest.fixture
def jwt_service() -> JWTService:
    """Create JWT service."""
    config = JWTConfig(
        secret_key="test_secret_key_for_testing_only_256bits",
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
    )
    return JWTService(config)


@pytest.fixture
def mock_repository() -> AsyncMock:
    """Create mock user repository."""
    return AsyncMock()


@pytest.fixture
def auth_service(
    mock_repository: AsyncMock,
    jwt_service: JWTService,
    password_service: PasswordService,
) -> AuthService:
    """Create auth service with mocked repository."""
    return AuthService(
        repository=mock_repository,
        jwt_service=jwt_service,
        password_service=password_service,
        token_revocation_service=_noop_token_revocation(),
    )


class TestPasswordService:
    """Tests for password hashing."""

    def test_hash_password(self, password_service: PasswordService) -> None:
        """Test password hashing."""
        password = "my_secure_password"
        hashed = password_service.hash(password)

        assert hashed != password
        assert hashed.startswith("$argon2")

    def test_verify_correct_password(self, password_service: PasswordService) -> None:
        """Test password verification with correct password."""
        password = "my_secure_password"
        hashed = password_service.hash(password)

        assert password_service.verify(hashed, password) is True

    def test_verify_incorrect_password(self, password_service: PasswordService) -> None:
        """Test password verification with wrong password."""
        password = "my_secure_password"
        hashed = password_service.hash(password)

        assert password_service.verify(hashed, "wrong_password") is False

    def test_different_hashes_for_same_password(self, password_service: PasswordService) -> None:
        """Test that same password produces different hashes."""
        password = "my_secure_password"
        hash1 = password_service.hash(password)
        hash2 = password_service.hash(password)

        assert hash1 != hash2
        # But both should verify
        assert password_service.verify(hash1, password) is True
        assert password_service.verify(hash2, password) is True


class TestJWTService:
    """Tests for JWT token handling."""

    def test_create_access_token(self, jwt_service: JWTService) -> None:
        """Test access token creation."""
        user_id = uuid4()
        token, expires_at = jwt_service.create_access_token(user_id)

        assert token is not None
        assert len(token) > 50  # JWT tokens are long
        assert expires_at > datetime.now(UTC)

    def test_decode_valid_token(self, jwt_service: JWTService) -> None:
        """Test decoding a valid token."""
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id)

        payload = jwt_service.decode_access_token(token)

        assert payload is not None
        assert payload.sub == str(user_id)
        assert payload.type == "access"

    def test_decode_invalid_token(self, jwt_service: JWTService) -> None:
        """Test decoding an invalid token."""
        payload = jwt_service.decode_access_token("invalid.token.here")
        assert payload is None

    def test_decode_expired_token(self) -> None:
        """Test that expired tokens are rejected."""
        # Create a service with very short expiration
        config = JWTConfig(
            secret_key="test_secret_key_for_testing_only_256bits",
            access_token_expire_minutes=-1,  # Already expired
        )
        short_jwt = JWTService(config)

        user_id = uuid4()
        token, _ = short_jwt.create_access_token(user_id)

        payload = short_jwt.decode_access_token(token)
        assert payload is None

    def test_get_user_id_from_token(self, jwt_service: JWTService) -> None:
        """Test extracting user ID from token."""
        user_id = uuid4()
        token, _ = jwt_service.create_access_token(user_id)

        extracted_id = jwt_service.get_user_id_from_token(token)

        assert extracted_id == user_id


class TestTokenPayloadUserId:
    """E1 — TokenPayload.user_id is the single safe sub->UUID conversion
    every caller (guards.py, AuthService.logout) must route through."""

    def test_valid_uuid_sub(self) -> None:
        user_id = uuid4()
        payload = TokenPayload(sub=str(user_id), exp=0, iat=0, jti="jti")
        assert payload.user_id == user_id

    @pytest.mark.parametrize("bad_sub", ["not-a-uuid", "", "f" * 35])
    def test_malformed_sub_returns_none(self, bad_sub: str) -> None:
        payload = TokenPayload(sub=bad_sub, exp=0, iat=0, jti="jti")
        assert payload.user_id is None


class TestAuthServiceRegister:
    """Tests for user registration."""

    @pytest.mark.asyncio
    async def test_register_success(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test successful registration."""
        user_id = uuid4()
        email = "test@example.com"

        # Mock: email doesn't exist
        mock_repository.email_exists.return_value = False

        # Mock: user creation
        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_user.email = email
        mock_user.is_active = True
        mock_repository.create_user.return_value = mock_user

        # Mock: refresh token creation
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        user, tokens = await auth_service.register(
            email=email,
            password="secure_password",
            display_name="Test User",
            product_id="vex",
        )

        assert user.email == email
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        mock_repository.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_email_exists(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test registration fails if email exists."""
        mock_repository.email_exists.return_value = True

        with pytest.raises(EmailAlreadyExistsError):
            await auth_service.register(
                email="existing@example.com",
                password="password123",
                product_id="vex",
            )


class TestAuthServiceLogin:
    """Tests for user login."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
        password_service: PasswordService,
    ) -> None:
        """Test successful login."""
        user_id = uuid4()
        email = "test@example.com"
        password = "correct_password"
        password_hash = password_service.hash(password)

        # Mock: user exists
        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_user.email = email
        mock_user.password_hash = password_hash
        mock_user.is_active = True
        mock_repository.get_active_user_by_email.return_value = mock_user

        # Mock: refresh token creation
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        user, tokens = await auth_service.login(
            email=email,
            password=password,
            product_id="vex",
        )

        assert user.email == email
        assert tokens.access_token is not None

    @pytest.mark.asyncio
    async def test_login_wrong_password(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
        password_service: PasswordService,
    ) -> None:
        """Test login fails with wrong password."""
        password_hash = password_service.hash("correct_password")

        mock_user = MagicMock(spec=User)
        mock_user.id = uuid4()
        mock_user.password_hash = password_hash
        mock_user.is_active = True
        mock_repository.get_active_user_by_email.return_value = mock_user

        with pytest.raises(InvalidCredentialsError):
            await auth_service.login(
                email="test@example.com",
                password="wrong_password",
                product_id="vex",
            )

    @pytest.mark.asyncio
    async def test_login_user_not_found(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test login fails if user doesn't exist."""
        mock_repository.get_active_user_by_email.return_value = None

        with pytest.raises(InvalidCredentialsError):
            await auth_service.login(
                email="nonexistent@example.com",
                password="password",
                product_id="vex",
            )

    @pytest.mark.asyncio
    async def test_login_inactive_user(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
        password_service: PasswordService,  # noqa: ARG002
    ) -> None:
        """Test login fails for inactive user (treated as not found via active-only lookup)."""
        mock_repository.get_active_user_by_email.return_value = None

        with pytest.raises(InvalidCredentialsError):
            await auth_service.login(
                email="inactive@example.com",
                password="password",
                product_id="vex",
            )


class TestAuthServiceRefresh:
    """Tests for token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test successful token refresh."""
        user_id = uuid4()
        family_id = uuid4()
        refresh_token = "valid_refresh_token"

        # Mock: valid token exists
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = user_id
        mock_token.family_id = family_id
        mock_token.product_id = "vex"
        mock_token.is_revoked = False
        mock_token.expires_at = datetime.now(UTC) + timedelta(days=1)
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token

        # Mock: user is active
        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_user.is_active = True
        mock_repository.get_active_user.return_value = mock_user

        # Mock: token revocation
        mock_repository.revoke_refresh_token.return_value = True

        # Mock: new token creation
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        tokens, returned_user_id = await auth_service.refresh_tokens(refresh_token)

        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert returned_user_id == user_id
        mock_repository.revoke_refresh_token.assert_called_once_with(
            mock_token.id, reason=RefreshTokenRevocationReason.ROTATED
        )

    @pytest.mark.asyncio
    async def test_refresh_revoked_token_triggers_family_revoke(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test that reusing a revoked token revokes entire family."""
        family_id = uuid4()

        # Mock: token exists but is revoked
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.user_id = uuid4()
        mock_token.family_id = family_id
        mock_token.is_revoked = True  # Already revoked!
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token

        # Mock: family revocation
        mock_repository.revoke_token_family.return_value = 3

        with pytest.raises(TokenReuseDetectedError):
            await auth_service.refresh_tokens("reused_token")

        mock_repository.revoke_token_family.assert_called_once_with(family_id)

    @pytest.mark.asyncio
    async def test_refresh_revoked_token_bulk_revokes_access_tokens(
        self,
        jwt_service: JWTService,
        password_service: PasswordService,
        mock_repository: AsyncMock,
    ) -> None:
        """R2 (issue #142) — reuse detection must also bulk-revoke live
        access tokens/content cookies, not just the refresh-token family.
        Otherwise the "All sessions have been invalidated" message it
        raises is false: a stolen access token/content cookie the attacker
        already holds would keep working."""
        user_id = uuid4()
        family_id = uuid4()

        mock_token = MagicMock(spec=RefreshToken)
        mock_token.user_id = user_id
        mock_token.family_id = family_id
        mock_token.is_revoked = True
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token
        mock_repository.revoke_token_family.return_value = 3

        mock_token_revocation = AsyncMock()
        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=mock_token_revocation,
        )

        with pytest.raises(TokenReuseDetectedError):
            await service.refresh_tokens("reused_token")

        mock_token_revocation.revoke_user_sessions.assert_awaited_once_with(user_id)

    @pytest.mark.asyncio
    async def test_refresh_lost_race_to_bulk_revocation_yields_benign_error(
        self,
        jwt_service: JWTService,
        password_service: PasswordService,
        mock_repository: AsyncMock,
    ) -> None:
        """B2 (issue #142) — a refresh that loses the user-row-lock race to
        a bulk revocation (logout-all/password-change/deactivation/
        password-reset) is a benign, ordinary outcome, not theft: it must
        raise InvalidRefreshTokenError, never revoke a token family, and
        never fire an ops alert."""
        user_id = uuid4()
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.user_id = user_id
        mock_token.family_id = uuid4()
        mock_token.is_revoked = True
        mock_token.revoked_reason = RefreshTokenRevocationReason.BULK_REVOCATION.value
        mock_repository.get_refresh_token_owner.return_value = user_id
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token

        mock_ops_event_bus = AsyncMock()
        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
            ops_event_bus=mock_ops_event_bus,
        )

        with pytest.raises(InvalidRefreshTokenError):
            await service.refresh_tokens("lost-race-token")

        mock_repository.revoke_token_family.assert_not_called()
        mock_ops_event_bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_null_revoked_reason_still_triggers_theft_path(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """B2 — legacy rows revoked before this column existed have
        revoked_reason IS NULL, which is *not* bulk_revocation and must
        still raise TokenReuseDetectedError (no weakening of existing
        detection for pre-migration data)."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.user_id = uuid4()
        mock_token.family_id = uuid4()
        mock_token.is_revoked = True
        mock_token.revoked_reason = None
        mock_repository.get_refresh_token_owner.return_value = mock_token.user_id
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token
        mock_repository.revoke_token_family.return_value = 1

        with pytest.raises(TokenReuseDetectedError):
            await auth_service.refresh_tokens("legacy-reused-token")

        mock_repository.revoke_token_family.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_single_logout_reason_still_triggers_theft_path(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """B2 — a token revoked by single-device logout and then replayed
        is more plausibly theft than a race, so it still raises
        TokenReuseDetectedError rather than the benign B2 path."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.user_id = uuid4()
        mock_token.family_id = uuid4()
        mock_token.is_revoked = True
        mock_token.revoked_reason = RefreshTokenRevocationReason.SINGLE_LOGOUT.value
        mock_repository.get_refresh_token_owner.return_value = mock_token.user_id
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token
        mock_repository.revoke_token_family.return_value = 1

        with pytest.raises(TokenReuseDetectedError):
            await auth_service.refresh_tokens("logged-out-then-replayed-token")

        mock_repository.revoke_token_family.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_expired_token(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test that expired token is rejected."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.is_revoked = False
        mock_token.expires_at = datetime.now(UTC) - timedelta(days=1)  # Expired
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token

        with pytest.raises(InvalidRefreshTokenError):
            await auth_service.refresh_tokens("expired_token")


class TestAuthServiceLogout:
    """Tests for logout."""

    @pytest.mark.asyncio
    async def test_logout_success(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test successful logout."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = uuid4()
        mock_repository.get_refresh_token_by_hash.return_value = mock_token
        mock_repository.revoke_refresh_token.return_value = True

        result = await auth_service.logout("valid_token")

        assert result is True
        mock_repository.revoke_refresh_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_logout_invalid_token(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test logout with invalid token."""
        mock_repository.get_refresh_token_by_hash.return_value = None

        result = await auth_service.logout("invalid_token")

        assert result is False

    @pytest.mark.asyncio
    async def test_logout_with_payload_locks_user_row_before_denylisting_jti(
        self,
        mock_repository: AsyncMock,
        jwt_service: JWTService,
        password_service: PasswordService,
    ) -> None:
        """C1/E1 — the user-row lock must be the *first* thing logout does
        when a payload is present, strictly before both
        revoke_refresh_token (an UPDATE that locks the refresh-token row —
        acquiring the user lock after it inverts G1's documented
        user-row-before-refresh-token-row order and deadlocks against
        AuthService.refresh_tokens/revoke_all_user_tokens/
        revoke_all_refresh_tokens) and the jti denylist write. Pinning the
        full three-call order on one shared mock — not just the last pair —
        is what actually catches a regression that puts the lock between
        revoke_refresh_token and revoke_token."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = uuid4()
        mock_repository.get_refresh_token_by_hash.return_value = mock_token

        token_revocation = AsyncMock()
        # A concrete bool, not an auto-generated MagicMock — `not <result>`
        # in AuthService.logout would otherwise call __bool__() on the
        # mock's return value, adding a phantom entry to tracker.mock_calls.
        token_revocation.revoke_token.return_value = True

        tracker = Mock()
        tracker.attach_mock(
            mock_repository.lock_user_for_session_change, "lock_user_for_session_change"
        )
        tracker.attach_mock(mock_repository.revoke_refresh_token, "revoke_refresh_token")
        tracker.attach_mock(token_revocation.revoke_token, "revoke_token")

        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=token_revocation,
        )

        user_id = uuid4()
        payload = TokenPayload(sub=str(user_id), exp=9999999999, iat=0, jti="jti-x")

        result = await service.logout("valid_token", token_payload=payload)

        assert result is True
        assert [call[0] for call in tracker.mock_calls] == [
            "lock_user_for_session_change",
            "revoke_refresh_token",
            "revoke_token",
        ]
        mock_repository.lock_user_for_session_change.assert_awaited_once_with(user_id)
        token_revocation.revoke_token.assert_awaited_once_with("jti-x", payload.exp)

    @pytest.mark.asyncio
    async def test_logout_with_malformed_sub_skips_lock_and_denylist(
        self,
        mock_repository: AsyncMock,
        jwt_service: JWTService,
        password_service: PasswordService,
    ) -> None:
        """E1 — a payload whose sub isn't a valid UUID has no user row to
        lock and no authenticated actor to denylist a jti for. It must not
        crash the uniform-200 logout contract; the refresh token is still
        revoked."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = uuid4()
        mock_repository.get_refresh_token_by_hash.return_value = mock_token

        token_revocation = AsyncMock()

        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=token_revocation,
        )

        payload = TokenPayload(sub="not-a-uuid", exp=9999999999, iat=0, jti="jti-x")

        result = await service.logout("valid_token", token_payload=payload)

        assert result is True
        mock_repository.revoke_refresh_token.assert_awaited_once()
        mock_repository.lock_user_for_session_change.assert_not_called()
        token_revocation.revoke_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_logout_without_payload_takes_no_lock(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Logout is reachable with an expired or absent bearer — only lock
        when a payload was decoded (there's no jti/sub to act on
        otherwise), and an unauthenticated logout must not pay for a row
        lock it has no use for."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = uuid4()
        mock_repository.get_refresh_token_by_hash.return_value = mock_token
        mock_repository.revoke_refresh_token.return_value = True

        result = await auth_service.logout("valid_token")

        assert result is True
        mock_repository.revoke_refresh_token.assert_called_once()
        mock_repository.lock_user_for_session_change.assert_not_called()

    @pytest.mark.asyncio
    async def test_logout_jti_denylist_failure_still_logs_and_returns_success(
        self,
        mock_repository: AsyncMock,
        jwt_service: JWTService,
        password_service: PasswordService,
    ) -> None:
        """A Redis write failure (as opposed to Redis being unconfigured)
        is a genuine degradation worth an error log, but must not fail the
        logout — the refresh-token revocation is the primary guarantee and
        already succeeded."""
        mock_token = MagicMock(spec=RefreshToken)
        mock_token.id = uuid4()
        mock_token.user_id = uuid4()
        mock_repository.get_refresh_token_by_hash.return_value = mock_token
        mock_repository.revoke_refresh_token.return_value = True

        token_revocation = AsyncMock()
        token_revocation.revoke_token.return_value = False  # write failed

        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=token_revocation,
        )

        user_id = uuid4()
        payload = TokenPayload(sub=str(user_id), exp=9999999999, iat=0, jti="jti-y")

        with structlog.testing.capture_logs() as logs:
            result = await service.logout("valid_token", token_payload=payload)

        assert result is True
        mock_repository.lock_user_for_session_change.assert_awaited_once_with(user_id)
        events = [log["event"] for log in logs]
        assert "auth.jti_denylist_failed" in events

    @pytest.mark.asyncio
    async def test_logout_all(
        self,
        auth_service: AuthService,
        mock_repository: AsyncMock,
    ) -> None:
        """Test logout from all devices."""
        user_id = uuid4()
        mock_repository.revoke_all_user_tokens.return_value = 5

        count = await auth_service.logout_all(user_id)

        assert count == 5
        mock_repository.revoke_all_user_tokens.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_logout_all_bulk_revokes_access_tokens(
        self,
        mock_repository: AsyncMock,
        jwt_service: JWTService,
        password_service: PasswordService,
    ) -> None:
        """logout_all (issue #142) must also write the user's revocation
        epoch, not just revoke refresh tokens — otherwise a stolen access
        token survives 'log out everywhere' for its full remaining lifetime."""
        mock_token_revocation = AsyncMock()
        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=mock_token_revocation,
        )
        user_id = uuid4()
        mock_repository.revoke_all_user_tokens.return_value = 2

        await service.logout_all(user_id)

        mock_token_revocation.revoke_user_sessions.assert_awaited_once_with(user_id)


class TestAuthServiceMissingBranches:
    """Cover remaining uncovered branches."""

    @pytest.mark.asyncio
    async def test_register_with_session_creates_billing_account(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        from unittest.mock import patch

        from src.api.services.auth import AuthService

        session = AsyncMock()
        mock_repository.email_exists.return_value = False
        mock_user = MagicMock(spec=User)
        mock_user.id = uuid4()
        mock_repository.create_user.return_value = mock_user
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
            session=session,
        )

        with patch("src.api.services.auth.BillingRepository") as billing_repo_cls:
            billing_repo = AsyncMock()
            billing_repo_cls.return_value = billing_repo

            await svc.register(email="new@example.com", password="pw", product_id="vex")

        billing_repo.create_personal_account.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_with_email_verification_sends_email(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        from src.api.services.auth import AuthService
        from src.api.services.email_verification import EmailVerificationService

        session = AsyncMock()
        session.add = MagicMock()
        mock_repository.email_exists.return_value = False
        mock_user = MagicMock(spec=User)
        mock_user.id = uuid4()
        mock_repository.create_user.return_value = mock_user
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        email_verification = AsyncMock(spec=EmailVerificationService)
        email_verification.send_verification_email = AsyncMock()

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
            session=session,
            email_verification_service=email_verification,
        )

        await svc.register(email="new@example.com", password="pw", product_id="vex")

        email_verification.send_verification_email.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_email_verification_failure_is_non_blocking(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        """Email verification failure during register should not raise."""
        from src.api.services.auth import AuthService
        from src.api.services.email_verification import EmailVerificationService

        session = AsyncMock()
        session.add = MagicMock()
        mock_repository.email_exists.return_value = False
        mock_user = MagicMock(spec=User)
        mock_user.id = uuid4()
        mock_repository.create_user.return_value = mock_user
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        email_verification = AsyncMock(spec=EmailVerificationService)
        email_verification.send_verification_email = AsyncMock(side_effect=Exception("smtp down"))

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
            session=session,
            email_verification_service=email_verification,
        )

        # Should not raise even though email fails
        user, _tokens = await svc.register(email="new@example.com", password="pw", product_id="vex")
        assert user is mock_user

    @pytest.mark.asyncio
    async def test_login_triggers_password_rehash(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        from src.api.services.auth import AuthService

        password = "correct"
        password_hash = password_service.hash(password)

        mock_user = MagicMock(spec=User)
        mock_user.id = uuid4()
        mock_user.password_hash = password_hash
        mock_repository.get_active_user_by_email.return_value = mock_user
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
        )

        # Make needs_rehash return True
        svc._password.needs_rehash = MagicMock(return_value=True)

        await svc.login(email="user@example.com", password=password, product_id="vex")

        mock_repository.update_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_raises_when_token_not_found(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        from src.api.services.auth import AuthService, InvalidRefreshTokenError

        mock_repository.get_refresh_token_by_hash_for_update.return_value = None

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
        )

        with pytest.raises(InvalidRefreshTokenError, match="Invalid refresh token"):
            await svc.refresh_tokens("nonexistent_token")

    @pytest.mark.asyncio
    async def test_refresh_raises_user_inactive(
        self, mock_repository: AsyncMock, jwt_service: JWTService, password_service: PasswordService
    ) -> None:
        from src.api.services.auth import AuthService, UserInactiveError

        mock_token = MagicMock(spec=RefreshToken)
        mock_token.is_revoked = False
        mock_token.expires_at = datetime.now(UTC) + timedelta(days=1)
        mock_token.user_id = uuid4()
        mock_token.family_id = uuid4()
        mock_token.product_id = "vex"
        mock_repository.get_refresh_token_by_hash_for_update.return_value = mock_token
        mock_repository.get_active_user.return_value = None  # user not found/inactive

        svc = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=password_service,
            token_revocation_service=_noop_token_revocation(),
        )

        with pytest.raises(UserInactiveError):
            await svc.refresh_tokens("valid_token_inactive_user")
