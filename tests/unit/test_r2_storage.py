"""Tests for R2 storage service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError

from src.api.services.storage import (
    MediaFormat,
    R2StorageService,
    R2StorageSettings,
    StorageType,
    StorageValidationError,
)
from src.api.services.storage.exceptions import (
    StorageDownloadError,
    StorageNotFoundError,
    StorageRangeNotSatisfiableError,
    StorageUploadError,
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


class TestPutRaw:
    """Tests for R2StorageService.put_raw (caller-chosen key, no validation)."""

    async def test_put_raw_stores_bytes_at_exact_key(self, r2_service: R2StorageService) -> None:
        mock_client = AsyncMock()

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            await r2_service.put_raw(
                "frame-previews/u/j/000.webp", b"webpbytes", content_type="image/webp"
            )

        mock_client.put_object.assert_awaited_once_with(
            Bucket=r2_service._settings.bucket_name,
            Key="frame-previews/u/j/000.webp",
            Body=b"webpbytes",
            ContentType="image/webp",
        )

    async def test_put_raw_passes_cache_control_when_given(
        self, r2_service: R2StorageService
    ) -> None:
        mock_client = AsyncMock()

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            await r2_service.put_raw(
                "payment-currency-logos/abc.svg",
                b"<svg/>",
                content_type="image/svg+xml",
                cache_control="public, max-age=31536000, immutable",
            )

        mock_client.put_object.assert_awaited_once_with(
            Bucket=r2_service._settings.bucket_name,
            Key="payment-currency-logos/abc.svg",
            Body=b"<svg/>",
            ContentType="image/svg+xml",
            CacheControl="public, max-age=31536000, immutable",
        )

    async def test_put_raw_raises_storage_upload_error_on_client_error(
        self, r2_service: R2StorageService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "500", "Message": "boom"}}, "PutObject")
        )

        @asynccontextmanager
        async def _fake_get_client():
            yield mock_client

        with (
            patch.object(r2_service, "_get_client", _fake_get_client),
            pytest.raises(StorageUploadError),
        ):
            await r2_service.put_raw("some/key", b"data", content_type="image/webp")


class TestStreamObject:
    """Tests for R2StorageService.stream_object (D2: single round-trip, D1: Range).

    Response metadata (content_type/content_length/content_range) comes
    entirely from the stubbed GetObject response — no separate head_object
    call is made or expected.
    """

    @staticmethod
    def _make_get_object_response(
        *,
        content_type: str = "image/png",
        content_length: int = 1234,
        content_range: str | None = None,
        body_chunks: tuple[bytes, ...] = (b"hello", b"world"),
    ) -> dict[str, object]:
        body_mock = MagicMock()

        async def _iter_chunks(chunk_size: int = 65536) -> object:
            del chunk_size
            for chunk in body_chunks:
                yield chunk

        body_mock.iter_chunks = _iter_chunks

        response: dict[str, object] = {
            "ContentType": content_type,
            "ContentLength": content_length,
            "Body": body_mock,
        }
        if content_range is not None:
            response["ContentRange"] = content_range
        return response

    async def test_full_object_single_get_object_call_no_head(
        self, r2_service: R2StorageService
    ) -> None:
        """A full-body stream issues exactly one get_object call and no head_object."""
        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(return_value=self._make_get_object_response())

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            async with r2_service.stream_object("users/u/outputs/j/f.png") as obj:
                chunks = [c async for c in obj.chunks]

        assert chunks == [b"hello", b"world"]
        assert obj.content_type == "image/png"
        assert obj.content_length == 1234
        assert obj.content_range is None
        mock_client.get_object.assert_awaited_once_with(
            Bucket=r2_service._settings.bucket_name,
            Key="users/u/outputs/j/f.png",
        )
        mock_client.head_object.assert_not_awaited()

    async def test_ranged_request_forwards_range_param(self, r2_service: R2StorageService) -> None:
        """A range_header is forwarded verbatim as the GetObject Range param."""
        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(
            return_value=self._make_get_object_response(
                content_type="video/mp4",
                content_length=500,
                content_range="bytes 0-499/1234",
                body_chunks=(b"partial",),
            )
        )

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            async with r2_service.stream_object(
                "users/u/outputs/j/v.mp4", range_header="bytes=0-499"
            ) as obj:
                chunks = [c async for c in obj.chunks]

        assert chunks == [b"partial"]
        assert obj.content_length == 500
        assert obj.content_range == "bytes 0-499/1234"
        mock_client.get_object.assert_awaited_once_with(
            Bucket=r2_service._settings.bucket_name,
            Key="users/u/outputs/j/v.mp4",
            Range="bytes=0-499",
        )

    async def test_no_such_key_raises_storage_not_found_error(
        self, r2_service: R2StorageService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        )

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with (
            patch.object(r2_service, "_get_client", _fake_get_client),
            pytest.raises(StorageNotFoundError),
        ):
            async with r2_service.stream_object("missing/key"):
                pass

    async def test_invalid_range_raises_storage_range_not_satisfiable_error(
        self, r2_service: R2StorageService
    ) -> None:
        """R2 rejecting a forwarded range surfaces as a dedicated exception (416 path)."""
        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "InvalidRange"}}, "GetObject")
        )

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with (
            patch.object(r2_service, "_get_client", _fake_get_client),
            pytest.raises(StorageRangeNotSatisfiableError),
        ):
            async with r2_service.stream_object("some/key", range_header="bytes=9999-19999"):
                pass

    async def test_other_client_error_raises_storage_download_error(
        self, r2_service: R2StorageService
    ) -> None:
        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(
            side_effect=ClientError({"Error": {"Code": "500", "Message": "boom"}}, "GetObject")
        )

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with (
            patch.object(r2_service, "_get_client", _fake_get_client),
            pytest.raises(StorageDownloadError),
        ):
            async with r2_service.stream_object("some/key"):
                pass

    async def test_missing_content_type_defaults_to_octet_stream(
        self, r2_service: R2StorageService
    ) -> None:
        mock_client = AsyncMock()
        response = self._make_get_object_response()
        del response["ContentType"]
        mock_client.get_object = AsyncMock(return_value=response)

        @asynccontextmanager
        async def _fake_get_client() -> AsyncIterator[AsyncMock]:
            yield mock_client

        with patch.object(r2_service, "_get_client", _fake_get_client):
            async with r2_service.stream_object("some/key") as obj:
                assert obj.content_type == "application/octet-stream"
