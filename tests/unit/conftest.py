"""Pytest configuration and shared fixtures."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.auth import AuthService
from src.api.services.user import UserService
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings
from src.db.models import User
from src.db.repositories import UserRepository


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=18188,
        debug=True,
    )


@pytest.fixture
def workflow_service(tmp_path: Path) -> WorkflowService:
    """Provide workflow service with test bundle."""
    # Create test bundle structure
    bundle_dir = tmp_path / "config" / "bundles" / "qwen_rapid_aio" / "260103-18"
    bundle_dir.mkdir(parents=True)

    # Copy actual workflow for testing
    source_workflow = (
        Path(__file__).parent.parent
        / "config"
        / "bundles"
        / "qwen_rapid_aio"
        / "260103-18"
        / "workflow.json"
    )

    if source_workflow.exists():
        shutil.copy(source_workflow, bundle_dir / "workflow.json")
    else:
        # Create minimal test workflow with all required nodes
        test_workflow = {
            "nodes": [
                {
                    "id": 9,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "widgets_values": [1024, 1024, 1],
                },
                {
                    "id": 3,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [],
                    "widgets_values": ["test prompt"],
                },
                {
                    "id": 4,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [],
                    "widgets_values": ["negative"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [],
                    "widgets_values": [12345, "fixed", 12, 1.1, "euler", "beta", 1],
                },
                {
                    "id": 7,
                    "type": "LoadImage",
                    "inputs": [],
                    "widgets_values": ["input_image_1.png", "image"],
                },
                {
                    "id": 8,
                    "type": "LoadImage",
                    "inputs": [],
                    "widgets_values": ["input_image_2.png", "image"],
                },
                {
                    "id": 11,
                    "type": "SaveImage",
                    "inputs": [],
                    "widgets_values": ["output"],
                },
            ],
            "links": [],
        }
        (bundle_dir / "workflow.json").write_text(json.dumps(test_workflow))

    return WorkflowService(base_path=tmp_path)


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
