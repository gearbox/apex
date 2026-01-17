"""Cloudflare R2 storage service implementation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .exceptions import (
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from .schemas import (
    DownloadResult,
    ImageFormat,
    StorageType,
    StoredFile,
    UploadResult,
)

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
DEFAULT_RETENTION_DAYS = 7


class R2StorageSettings:
    """R2-specific configuration."""

    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        public_url_base: str | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.account_id = account_id
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name
        self.public_url_base = public_url_base
        self.retention_days = retention_days

    @property
    def endpoint_url(self) -> str:
        """Cloudflare R2 endpoint URL."""
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class R2StorageService:
    """Cloudflare R2 storage service implementation.

    Provides S3-compatible storage operations using Cloudflare R2.
    All operations are async and use connection pooling.
    """

    def __init__(self, settings: R2StorageSettings) -> None:
        """Initialize R2 storage service.

        Args:
            settings: R2 configuration settings.
        """
        self._settings = settings
        self._session = aioboto3.Session()
        self._client_config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        )

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[S3Client]:
        """Get S3 client with context management.

        Yields:
            Configured S3 client.
        """
        async with self._session.client(  # type: ignore[reportGeneralTypeIssues]
            "s3",
            endpoint_url=self._settings.endpoint_url,
            aws_access_key_id=self._settings.access_key_id,
            aws_secret_access_key=self._settings.secret_access_key,
            config=self._client_config,
        ) as client:
            yield client

    def _validate_upload(
        self,
        data: bytes,
        content_type: str,
        _filename: str,
    ) -> ImageFormat:
        """Validate upload parameters.

        Args:
            data: File content.
            content_type: MIME type.
            filename: Original filename.

        Returns:
            Validated image format.

        Raises:
            StorageValidationError: If validation fails.
        """
        # Check file size
        size = len(data)
        if size > MAX_FILE_SIZE:
            raise StorageValidationError(
                f"File size {size} bytes exceeds maximum {MAX_FILE_SIZE} bytes"
            )

        if size == 0:
            raise StorageValidationError("File is empty")

        # Check content type
        if content_type.lower() not in ALLOWED_CONTENT_TYPES:
            raise StorageValidationError(
                f"Content type '{content_type}' not allowed. "
                f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )

        # Determine format
        try:
            return ImageFormat.from_content_type(content_type)
        except ValueError as e:
            raise StorageValidationError(str(e)) from e

    def build_storage_key(
        self,
        *,
        user_id: UUID,
        file_id: UUID,
        storage_type: StorageType,
        format: ImageFormat,
        job_id: UUID | None = None,
    ) -> str:
        """Build the storage key for a file.

        Key format:
        - Uploads: users/{user_id}/uploads/{file_id}.{ext}
        - Outputs: users/{user_id}/outputs/{job_id}/{file_id}.{ext}
        """
        user_str = str(user_id)
        file_str = str(file_id)
        ext = format.extension

        if storage_type == StorageType.UPLOAD:
            return f"users/{user_str}/uploads/{file_str}.{ext}"
        if job_id is None:
            raise ValueError("job_id is required for output storage type")
        job_str = str(job_id)
        return f"users/{user_str}/outputs/{job_str}/{file_str}.{ext}"

    async def upload(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
        storage_type: StorageType,
        job_id: UUID | None = None,
    ) -> UploadResult:
        """Upload a file to R2 storage."""
        # Validate and get format
        image_format = self._validate_upload(data, content_type, filename)

        # Generate unique file ID
        file_id = uuid4()

        # Build storage key
        storage_key = self.build_storage_key(
            user_id=user_id,
            file_id=file_id,
            storage_type=storage_type,
            format=image_format,
            job_id=job_id,
        )

        # Calculate expiration
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._settings.retention_days)

        # Build metadata
        metadata = {
            "user-id": str(user_id),
            "original-filename": filename,
            "storage-type": storage_type.value,
            "uploaded-at": now.isoformat(),
        }
        if job_id:
            metadata["job-id"] = str(job_id)

        try:
            async with self._get_client() as client:
                await client.put_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                    Body=data,
                    ContentType=content_type,
                    Metadata=metadata,
                )

            logger.info(
                f"Uploaded file to R2: {storage_key} " f"({len(data)} bytes, user={user_id})"
            )

            return UploadResult(
                id=file_id,
                storage_key=storage_key,
                expires_at=expires_at,
            )

        except ClientError as e:
            logger.error(f"R2 upload failed for {storage_key}: {e}")
            raise StorageUploadError(
                f"Failed to upload file: {e.response['Error']['Message']}",
                cause=e,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error uploading to R2: {e}")
            raise StorageUploadError(f"Upload failed: {e}", cause=e) from e

    async def download(self, storage_key: str) -> bytes:
        """Download file content from R2."""
        try:
            async with self._get_client() as client:
                response = await client.get_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )
                data = await response["Body"].read()
                logger.debug(f"Downloaded {len(data)} bytes from R2: {storage_key}")
                return data

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("NoSuchKey", "404"):
                raise StorageNotFoundError(f"File not found: {storage_key}") from e
            logger.error(f"R2 download failed for {storage_key}: {e}")
            raise StorageDownloadError(
                f"Failed to download file: {e.response['Error']['Message']}",
                cause=e,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error downloading from R2: {e}")
            raise StorageDownloadError(f"Download failed: {e}", cause=e) from e

    async def get_presigned_url(
        self,
        storage_key: str,
        *,
        expires_in: int = 3600,
    ) -> DownloadResult:
        """Generate a presigned URL for temporary access."""
        try:
            async with self._get_client() as client:
                # First, get object metadata to verify existence and get content info
                head = await client.head_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )

                # Generate presigned URL
                url = await client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self._settings.bucket_name,
                        "Key": storage_key,
                    },
                    ExpiresIn=expires_in,
                )

                return DownloadResult(
                    storage_key=storage_key,
                    presigned_url=url,
                    content_type=head.get("ContentType", "application/octet-stream"),
                    size_bytes=head.get("ContentLength", 0),
                    expires_in_seconds=expires_in,
                )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("NoSuchKey", "404"):
                raise StorageNotFoundError(f"File not found: {storage_key}") from e
            logger.error(f"Failed to generate presigned URL for {storage_key}: {e}")
            raise StorageDownloadError(
                f"Failed to generate URL: {e.response['Error']['Message']}",
                cause=e,
            ) from e

    async def delete(self, storage_key: str) -> bool:
        """Delete a file from R2 storage."""
        try:
            async with self._get_client() as client:
                # Check if exists first
                try:
                    await client.head_object(
                        Bucket=self._settings.bucket_name,
                        Key=storage_key,
                    )
                except ClientError as e:
                    if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                        return False
                    raise

                # Delete the object
                await client.delete_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )
                logger.info(f"Deleted file from R2: {storage_key}")
                return True

        except ClientError as e:
            logger.error(f"R2 delete failed for {storage_key}: {e}")
            raise StorageDeleteError(
                f"Failed to delete file: {e.response['Error']['Message']}",
                cause=e,
            ) from e

    async def delete_many(self, storage_keys: list[str]) -> int:
        """Delete multiple files from R2 storage."""
        if not storage_keys:
            return 0

        try:
            async with self._get_client() as client:
                # R2/S3 delete_objects supports up to 1000 keys per request
                deleted_count = 0
                for i in range(0, len(storage_keys), 1000):
                    batch = storage_keys[i : i + 1000]
                    response = await client.delete_objects(
                        Bucket=self._settings.bucket_name,
                        Delete={
                            "Objects": [{"Key": key} for key in batch],
                            "Quiet": False,
                        },
                    )
                    deleted_count += len(response.get("Deleted", []))

                logger.info(f"Deleted {deleted_count} files from R2")
                return deleted_count

        except ClientError as e:
            logger.error(f"R2 batch delete failed: {e}")
            raise StorageDeleteError(
                f"Failed to delete files: {e.response['Error']['Message']}",
                cause=e,
            ) from e

    async def exists(self, storage_key: str) -> bool:
        """Check if a file exists in R2 storage."""
        try:
            async with self._get_client() as client:
                await client.head_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return False
            logger.error(f"R2 exists check failed for {storage_key}: {e}")
            raise StorageConnectionError(
                f"Failed to check file existence: {e}",
                cause=e,
            ) from e

    async def list_user_files(
        self,
        user_id: UUID,
        *,
        storage_type: StorageType | None = None,
        limit: int = 100,
    ) -> list[StoredFile]:
        """List files for a user in R2 storage."""
        # Build prefix based on storage type
        user_str = str(user_id)
        if storage_type:
            prefix = f"users/{user_str}/{storage_type.value}s/"
        else:
            prefix = f"users/{user_str}/"

        try:
            async with self._get_client() as client:
                files: list[StoredFile] = []
                paginator = client.get_paginator("list_objects_v2")

                async for page in paginator.paginate(
                    Bucket=self._settings.bucket_name,
                    Prefix=prefix,
                    MaxKeys=limit,
                ):
                    for obj in page.get("Contents", []):
                        # Parse the storage key to extract metadata
                        storage_key = obj["Key"]
                        if stored_file := self._parse_storage_key(
                            storage_key=storage_key,
                            size_bytes=obj.get("Size", 0),
                            last_modified=obj.get("LastModified"),
                        ):
                            files.append(stored_file)

                        if len(files) >= limit:
                            return files

                return files

        except ClientError as e:
            logger.error(f"R2 list failed for user {user_id}: {e}")
            return []

    def _parse_storage_key(
        self,
        storage_key: str,
        size_bytes: int,
        last_modified: datetime | None,
    ) -> StoredFile | None:
        """Parse a storage key into StoredFile metadata.

        Key format: users/{user_id}/{type}s/{file_id}.{ext}
        or: users/{user_id}/outputs/{job_id}/{file_id}.{ext}
        """
        try:
            parts = storage_key.split("/")
            if len(parts) < 4:
                return None

            user_id = UUID(parts[1])
            type_str = parts[2].rstrip("s")  # "uploads" -> "upload"
            storage_type = StorageType(type_str)

            # Parse filename and extract file_id
            if storage_type == StorageType.OUTPUT and len(parts) >= 5:
                job_id = UUID(parts[3])
                filename_with_ext = parts[4]
            else:
                job_id = None
                filename_with_ext = parts[3]

            file_part, ext = filename_with_ext.rsplit(".", 1)
            file_id = UUID(file_part)
            image_format = ImageFormat.from_extension(ext)

            created_at = last_modified or datetime.now(timezone.utc)

            return StoredFile(
                id=file_id,
                user_id=user_id,
                storage_type=storage_type,
                storage_key=storage_key,
                filename=filename_with_ext,
                format=image_format,
                size_bytes=size_bytes,
                content_type=image_format.content_type,
                created_at=created_at,
                expires_at=created_at + timedelta(days=self._settings.retention_days),
                job_id=job_id,
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse storage key {storage_key}: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if R2 storage is accessible."""
        try:
            async with self._get_client() as client:
                await client.head_bucket(Bucket=self._settings.bucket_name)
                return True
        except Exception as e:
            logger.warning(f"R2 health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close any open connections.

        Note: aioboto3 manages connections per-context, so this is a no-op.
        Kept for protocol compliance.
        """
        pass
