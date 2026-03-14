"""Tests for R2 storage service."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.api.services.storage import (
    MediaFormat,
    R2StorageService,
    R2StorageSettings,
    StorageType,
    StorageValidationError,
)


@pytest.fixture
def r2_settings() -> R2StorageSettings:
    """Create R2 settings for testing."""
    return R2StorageSettings(
        account_id="test_account",
        access_key_id="test_key",
        secret_access_key="test_secret",
        bucket_name="test-bucket",
        retention_days=7,
    )


@pytest.fixture
def r2_service(r2_settings: R2StorageSettings) -> R2StorageService:
    """Create R2 service for testing."""
    return R2StorageService(r2_settings)


class TestR2StorageKeyBuilding:
    """Tests for storage key building."""

    def test_build_upload_key(self, r2_service: R2StorageService) -> None:
        """Test building storage key for uploads."""
        user_id = uuid4()
        file_id = uuid4()

        key = r2_service.build_storage_key(
            user_id=user_id,
            file_id=file_id,
            storage_type=StorageType.UPLOAD,
            format=MediaFormat.PNG,
        )

        assert key == f"users/{user_id}/uploads/{file_id}.png"

    def test_build_upload_key_jpeg(self, r2_service: R2StorageService) -> None:
        """Test building storage key for JPEG uploads."""
        user_id = uuid4()
        file_id = uuid4()

        key = r2_service.build_storage_key(
            user_id=user_id,
            file_id=file_id,
            storage_type=StorageType.UPLOAD,
            format=MediaFormat.JPEG,
        )

        assert key == f"users/{user_id}/uploads/{file_id}.jpeg"

    def test_build_output_key(self, r2_service: R2StorageService) -> None:
        """Test building storage key for outputs."""
        user_id = uuid4()
        file_id = uuid4()
        job_id = uuid4()

        key = r2_service.build_storage_key(
            user_id=user_id,
            file_id=file_id,
            storage_type=StorageType.OUTPUT,
            format=MediaFormat.PNG,
            job_id=job_id,
        )

        assert key == f"users/{user_id}/outputs/{job_id}/{file_id}.png"

    def test_build_output_key_requires_job_id(self, r2_service: R2StorageService) -> None:
        """Test that output storage type requires job_id."""
        with pytest.raises(ValueError, match="job_id is required"):
            r2_service.build_storage_key(
                user_id=uuid4(),
                file_id=uuid4(),
                storage_type=StorageType.OUTPUT,
                format=MediaFormat.PNG,
                job_id=None,
            )


class TestR2ValidationRules:
    """Tests for upload validation rules."""

    def test_validate_empty_file(self, r2_service: R2StorageService) -> None:
        """Test that empty files are rejected."""
        with pytest.raises(StorageValidationError, match="empty"):
            r2_service._validate_upload(b"", "image/png", "test.png")

    def test_validate_file_too_large(self, r2_service: R2StorageService) -> None:
        """Test that files over 20MB are rejected."""
        large_data = b"x" * (21 * 1024 * 1024)  # 21MB
        with pytest.raises(StorageValidationError, match="exceeds maximum"):
            r2_service._validate_upload(large_data, "image/png", "test.png")

    def test_validate_invalid_content_type(self, r2_service: R2StorageService) -> None:
        """Test that invalid content types are rejected."""
        with pytest.raises(StorageValidationError, match="not allowed"):
            r2_service._validate_upload(b"test", "image/gif", "test.gif")

        with pytest.raises(StorageValidationError, match="not allowed"):
            r2_service._validate_upload(b"test", "text/plain", "test.txt")

    def test_validate_valid_png(self, r2_service: R2StorageService) -> None:
        """Test that valid PNG is accepted."""
        result = r2_service._validate_upload(b"test", "image/png", "test.png")
        assert result == MediaFormat.PNG

    def test_validate_valid_jpeg(self, r2_service: R2StorageService) -> None:
        """Test that valid JPEG is accepted."""
        result = r2_service._validate_upload(b"test", "image/jpeg", "test.jpg")
        assert result == MediaFormat.JPEG

    def test_validate_valid_webp(self, r2_service: R2StorageService) -> None:
        """Test that valid WebP is accepted."""
        result = r2_service._validate_upload(b"test", "image/webp", "test.webp")
        assert result == MediaFormat.WEBP

    def test_validate_max_size_boundary(self, r2_service: R2StorageService) -> None:
        """Test file at exactly max size is accepted."""
        max_data = b"x" * (20 * 1024 * 1024)  # Exactly 20MB
        result = r2_service._validate_upload(max_data, "image/png", "test.png")
        assert result == MediaFormat.PNG


class TestStorageKeyParsing:
    """Tests for storage key parsing."""

    def test_parse_upload_key(self, r2_service: R2StorageService) -> None:
        """Test parsing upload storage key."""
        user_id = uuid4()
        file_id = uuid4()
        key = f"users/{user_id}/uploads/{file_id}.png"

        result = r2_service._parse_storage_key(
            storage_key=key,
            size_bytes=1024,
            last_modified=None,
        )

        assert result is not None
        assert result.user_id == user_id
        assert result.id == file_id
        assert result.storage_type == StorageType.UPLOAD
        assert result.format == MediaFormat.PNG
        assert result.size_bytes == 1024

    def test_parse_output_key(self, r2_service: R2StorageService) -> None:
        """Test parsing output storage key."""
        user_id = uuid4()
        job_id = uuid4()
        file_id = uuid4()
        key = f"users/{user_id}/outputs/{job_id}/{file_id}.jpeg"

        result = r2_service._parse_storage_key(
            storage_key=key,
            size_bytes=2048,
            last_modified=None,
        )

        assert result is not None
        assert result.user_id == user_id
        assert result.job_id == job_id
        assert result.id == file_id
        assert result.storage_type == StorageType.OUTPUT
        assert result.format == MediaFormat.JPEG

    def test_parse_invalid_key(self, r2_service: R2StorageService) -> None:
        """Test parsing invalid storage key returns None."""
        result = r2_service._parse_storage_key(
            storage_key="invalid/key",
            size_bytes=1024,
            last_modified=None,
        )
        assert result is None

    def test_parse_key_with_invalid_uuid(self, r2_service: R2StorageService) -> None:
        """Test parsing key with invalid UUID returns None."""
        result = r2_service._parse_storage_key(
            storage_key="users/not-a-uuid/uploads/also-not-uuid.png",
            size_bytes=1024,
            last_modified=None,
        )
        assert result is None


class TestSignKey:
    """Tests for R2StorageService.sign_key (no head_object round-trip)."""

    async def test_sign_key_returns_presigned_url(self, r2_service: R2StorageService) -> None:
        """sign_key calls generate_presigned_url and returns the URL string."""
        expected_url = "https://test-account.r2.cloudflarestorage.com/test-key?sig=abc"
        storage_key = "users/uid/outputs/job/img.jpg"

        mock_client = AsyncMock()
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            url = await r2_service.sign_key(storage_key)

        assert url == expected_url
        mock_client.generate_presigned_url.assert_awaited_once_with(
            "get_object",
            Params={"Bucket": r2_service._settings.bucket_name, "Key": storage_key},
            ExpiresIn=3600,
        )

    async def test_sign_key_custom_expiry(self, r2_service: R2StorageService) -> None:
        """sign_key passes custom expires_in to generate_presigned_url."""
        mock_client = AsyncMock()
        mock_client.generate_presigned_url = AsyncMock(return_value="https://url")

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            await r2_service.sign_key("some/key", expires_in=1800)

        _, kwargs = mock_client.generate_presigned_url.call_args
        assert kwargs["ExpiresIn"] == 1800

    async def test_sign_key_does_not_call_head_object(self, r2_service: R2StorageService) -> None:
        """sign_key must not issue a head_object request."""
        mock_client = AsyncMock()
        mock_client.generate_presigned_url = AsyncMock(return_value="https://url")

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            await r2_service.sign_key("any/key")

        mock_client.head_object.assert_not_awaited()
