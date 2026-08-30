"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from pathlib import Path

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


@pytest.fixture
def wan_workflow_bundle(tmp_path: Path) -> Path:
    """A minimal WAN-shaped video bundle used by workflow contract tests.

    It deliberately follows the bundle-on-disk shape: ``bundle.yaml`` holds
    the v2 workflow map and ``workflow.api.json`` holds the matching ComfyUI
    API graph.  Keeping this fixture shared makes the video contract
    executable before an actual WAN bundle is checked into this repository.
    """
    bundle_dir = tmp_path / "wan-synthetic" / "260101-01"
    bundle_dir.mkdir(parents=True)
    workflow = {
        "contract_version": 2,
        "media": "video",
        "nodes": {
            "latent": {
                "id": "1",
                "class": "EmptyLatentImage",
                "inputs": {"width": "width", "height": "height", "length": "length"},
            },
            "positive_prompt": {
                "id": "2",
                "class": "CLIPTextEncode",
                "inputs": {"text": "text"},
            },
            "sampler": {
                "id": "3",
                "class": "KSampler",
                "inputs": {"seed": "seed", "steps": "steps"},
            },
            "save": {
                "id": "4",
                "class": "SaveWEBM",
                "inputs": {"filename_prefix": "filename_prefix"},
            },
        },
        "media_inputs": [
            {
                "id": "5",
                "class": "WanImageToVideo",
                "input": "image",
                "kind": "image",
                "slot": "first_frame",
                "target_role": "positive_prompt",
                "target_input": "first_frame",
            },
            {
                "id": "6",
                "class": "WanImageToVideo",
                "input": "image",
                "kind": "image",
                "slot": "last_frame",
                "target_role": "positive_prompt",
                "target_input": "last_frame",
            },
        ],
        "model_inputs": [
            {
                "id": "7",
                "class": "UNETLoader",
                "input": "unet_name",
                "model_type": "unet",
                "filename": "wan.safetensors",
            }
        ],
    }
    graph = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 0, "height": 0, "length": 0}},
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "first_frame": ["5", 0], "last_frame": ["6", 0]},
        },
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 0}},
        "4": {"class_type": "SaveWEBM", "inputs": {"filename_prefix": ""}},
        "5": {"class_type": "WanImageToVideo", "inputs": {"image": ""}},
        "6": {"class_type": "WanImageToVideo", "inputs": {"image": ""}},
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": ""}},
    }
    (bundle_dir / "bundle.yaml").write_text(json.dumps({"workflow": workflow}))
    (bundle_dir / "workflow.api.json").write_text(json.dumps(graph))
    return bundle_dir
