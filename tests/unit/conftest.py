"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.age_verification import AgeVerificationService
from src.api.services.auth import AuthService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings
from src.db.models import User
from src.db.repositories import UserRepository

_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default

# Minimal GUI-format workflow with all required nodes for qwen_rapid_aio v19.
# Node 11 (SaveImage) is wired to VAEDecode (node 5) via link 19.
# Node 10 (orphan SaveImage) is intentionally absent — it was removed in ai-bundles.
_MINIMAL_GUI_WORKFLOW = {
    "nodes": [
        {
            "id": 9,
            "type": "EmptyLatentImage",
            "inputs": [],
            "widgets_values": [1024, 1024, 1],
        },
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "inputs": [],
            "widgets_values": ["STALE.safetensors"],
        },
        {
            "id": 3,
            "type": "TextEncodeQwenImageEditPlus",
            "inputs": [
                {"name": "clip", "link": 2},
                {"name": "vae", "link": 8},
                {"name": "image1", "link": None},
                {"name": "image2", "link": None},
            ],
            "widgets_values": ["test prompt"],
        },
        {
            "id": 4,
            "type": "TextEncodeQwenImageEditPlus",
            "inputs": [
                {"name": "clip", "link": 7},
                {"name": "vae", "link": 9},
            ],
            "widgets_values": ["negative"],
        },
        {
            "id": 2,
            "type": "KSampler",
            "inputs": [
                {"name": "model", "link": 1},
                {"name": "positive", "link": 3},
                {"name": "negative", "link": 4},
                {"name": "latent_image", "link": 13},
            ],
            "widgets_values": [12345, "fixed", 12, 1.1, "euler", "beta", 1],
        },
        {
            "id": 5,
            "type": "VAEDecode",
            "inputs": [
                {"name": "samples", "link": 5},
                {"name": "vae", "link": 6},
            ],
            "widgets_values": [],
        },
        {
            "id": 7,
            "type": "LoadImage",
            "inputs": [],
            "widgets_values": ["example.png", "image"],
        },
        {
            "id": 8,
            "type": "LoadImage",
            "inputs": [],
            "widgets_values": ["example.png", "image"],
        },
        {
            "id": 11,
            "type": "SaveImage",
            "inputs": [{"name": "images", "link": 19}],
            "widgets_values": ["generated"],
        },
    ],
    "links": [
        [1, 1, 0, 2, 0, "MODEL"],
        [2, 1, 1, 3, 0, "CLIP"],
        [3, 3, 0, 2, 1, "CONDITIONING"],
        [4, 4, 0, 2, 2, "CONDITIONING"],
        [5, 2, 0, 5, 0, "LATENT"],
        [6, 1, 2, 5, 1, "VAE"],
        [7, 1, 1, 4, 0, "CLIP"],
        [8, 1, 2, 3, 1, "VAE"],
        [9, 1, 2, 4, 1, "VAE"],
        [13, 9, 0, 2, 3, "LATENT"],
        [19, 5, 0, 11, 0, "IMAGE"],
    ],
}


def make_bundle_dir(
    tmp_path: Path,
    *,
    version: str = "260103-19",
    ckpt_name: str = "Qwen-Rapid-AIO-NSFW-v19.safetensors",
    gui_workflow: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Create a temp bundle directory with bundle.yaml and workflow.json.

    Returns:
        (bundle_root, version_dir) — bundle_root is the dir passed to
        WorkflowService methods; version_dir is bundle_root / version.
    """
    bundle_root = tmp_path / "bundles" / "qwen_rapid_aio"
    version_dir = bundle_root / version
    version_dir.mkdir(parents=True)

    bundle_yaml = {
        "hardware": {
            "gpu_whitelist": [],
            "min_disk_gb": 100,
            "min_network_upload_mbps": 100,
            "min_network_download_mbps": 100,
            "cuda_min_version": "12.1",
            "num_gpus": 1,
        },
        "models": [
            {
                "model_type": "checkpoints",
                "files": [{"filename": ckpt_name}],
            }
        ],
    }
    (version_dir / "bundle.yaml").write_text(yaml.dump(bundle_yaml))

    wf = gui_workflow if gui_workflow is not None else _MINIMAL_GUI_WORKFLOW
    (version_dir / "workflow.json").write_text(json.dumps(wf))

    # Create "current" symlink pointing to the version directory
    current_link = bundle_root / "current"
    current_link.symlink_to(version)

    return bundle_root, version_dir


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Provide a temp bundle root directory for the default test bundle."""
    root, _ = make_bundle_dir(tmp_path)
    return root


@pytest.fixture
def workflow_service() -> WorkflowService:
    """Provide a fresh WorkflowService instance."""
    return WorkflowService()


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
