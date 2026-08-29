"""Pytest configuration and shared fixtures."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.age_verification import AgeVerificationService
from src.api.services.auth import AuthService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.config import Settings
from src.db.models import User
from src.db.repositories import UserRepository

_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=_DEFAULT_COMFYUI_PORT,
        debug=True,
    )


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture
def password_service() -> PasswordService:
    """Create password service for testing."""
    return PasswordService()


@pytest.fixture
def jwt_config() -> JWTConfig:
    """Create JWT config for testing."""
    return JWTConfig(
        secret_key="test_secret_key_for_testing_only_256bits",
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
    )


@pytest.fixture
def jwt_service(jwt_config: JWTConfig) -> JWTService:
    """Create JWT service for testing."""
    return JWTService(jwt_config)


@pytest.fixture
def mock_user_repository() -> AsyncMock:
    """Create mock user repository."""
    return AsyncMock(spec=UserRepository)


@pytest.fixture
def auth_service(
    mock_user_repository: AsyncMock,
    jwt_service: JWTService,
    password_service: PasswordService,
) -> AuthService:
    """Create auth service with mocked repository."""
    return AuthService(
        repository=mock_user_repository,
        jwt_service=jwt_service,
        password_service=password_service,
        token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
    )


@pytest.fixture
def user_service(
    mock_user_repository: AsyncMock,
    password_service: PasswordService,
) -> UserService:
    """Create user service with mocked repository."""
    return UserService(
        repository=mock_user_repository,
        password_service=password_service,
        age_verification_service=AgeVerificationService(),
        token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
    )


@pytest.fixture
def mock_user() -> MagicMock:
    """Create a mock user for testing."""
    user = MagicMock(spec=User)
    user.id = uuid4()
    user.email = "test@example.com"
    user.display_name = "Test User"
    user.subscription_tier = "free"
    user.is_active = True
    user.password_hash = PasswordService().hash("test_password")
    return user
