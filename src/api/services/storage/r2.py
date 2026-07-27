"""Cloudflare R2 storage service implementation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import aioboto3
import structlog
from botocore.config import Config
from botocore.exceptions import ClientError

from src.core.uid import new_id

from .exceptions import (
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageNotFoundError,
    StorageRangeNotSatisfiableError,
    StorageUploadError,
    StorageValidationError,
)
from .schemas import (
    ALLOWED_UPLOAD_CONTENT_TYPES,
    DownloadResult,
    MediaFormat,
    StorageType,
    StoredFile,
    UploadResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from types_aiobotocore_s3.client import S3Client

logger = structlog.get_logger(__name__)


def _get_error_code(e: ClientError) -> str:
    """Safely extract error code from ClientError."""
    return e.response.get("Error", {}).get("Code", "Unknown")


def _get_error_message(e: ClientError) -> str:
    """Safely extract error message from ClientError."""
    return e.response.get("Error", {}).get("Message", str(e))


# Constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
# Re-exported for backward compatibility — routes/content.py imports this name.
ALLOWED_CONTENT_TYPES = ALLOWED_UPLOAD_CONTENT_TYPES
DEFAULT_RETENTION_DAYS = 7


@dataclass(frozen=True, slots=True)
class ObjectStream:
    """Result of `R2StorageService.stream_object` — chunks plus response metadata.

    All fields come from the single GetObject response (see stream_object's
    docstring) — no separate head_object call is made.
    """

    chunks: AsyncIterator[bytes]
    content_type: str
    content_length: int
    """Bytes in this response body — the served range length, or full object size."""
    content_range: str | None
    """Raw `Content-Range` response header (e.g. "bytes 0-499/1234"), present iff
    a satisfiable range was served (206); None for a full-body (200) response."""


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
        client_ctx: Any = self._session.client(
            "s3",
            endpoint_url=self._settings.endpoint_url,
            aws_access_key_id=self._settings.access_key_id,
            aws_secret_access_key=self._settings.secret_access_key,
            config=self._client_config,
        )
        async with client_ctx as client:
            yield client

    def _validate_upload(
        self,
        data: bytes,
        content_type: str,
    ) -> MediaFormat:
        """Validate upload parameters.

        Args:
            data: File content.
            content_type: MIME type.

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
            return MediaFormat.from_content_type(content_type)
        except ValueError as e:
            raise StorageValidationError(str(e)) from e

    def build_storage_key(
        self,
        *,
        user_id: UUID,
        file_id: UUID,
        storage_type: StorageType,
        format: MediaFormat,
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
        content_type: str,
        storage_type: StorageType,
        job_id: UUID | None = None,
    ) -> UploadResult:
        """Upload a file to R2 storage.

        Stored objects are identified solely by their storage key
        (``users/{user_id}/uploads|outputs/.../{file_id}.{ext}``) — no
        client-supplied filename is ever persisted as R2 object metadata.
        """
        # Validate and get format
        image_format = self._validate_upload(data, content_type)

        # Generate unique file ID
        file_id = new_id()

        # Build storage key
        storage_key = self.build_storage_key(
            user_id=user_id,
            file_id=file_id,
            storage_type=storage_type,
            format=image_format,
            job_id=job_id,
        )

        # Calculate expiration
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=self._settings.retention_days)

        # Build metadata. Deliberately no client-controlled fields (e.g. an
        # original filename): HTTP header values are latin-1 constrained, and
        # a non-latin filename raised UnicodeEncodeError before the request
        # was even sent — nothing reads "original-filename" back, so it's
        # dropped rather than encoded.
        metadata = {
            "user-id": str(user_id),
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
                "r2.upload_completed", key=storage_key, bytes=len(data), user_id=str(user_id)
            )

            return UploadResult(
                id=file_id,
                storage_key=storage_key,
                expires_at=expires_at,
            )

        except ClientError as e:
            logger.exception("r2.upload_failed", key=storage_key, error=str(e))
            raise StorageUploadError(
                f"Failed to upload file: {_get_error_message(e)}",
                cause=e,
            ) from e
        except Exception as e:
            logger.exception("r2.upload_unexpected_error", error=str(e))
            raise StorageUploadError(f"Upload failed: {e}", cause=e) from e

    async def put_raw(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        """Store bytes at a caller-chosen key.

        Deliberately bypasses build_storage_key and upload validation — for
        internal derived artifacts (preview frames, video posters, cached
        currency logos) whose key layout is owned by the caller. Not for
        user-supplied content: use upload() for that.
        """
        try:
            async with self._get_client() as client:
                if cache_control is not None:
                    await client.put_object(
                        Bucket=self._settings.bucket_name,
                        Key=key,
                        Body=data,
                        ContentType=content_type,
                        CacheControl=cache_control,
                    )
                else:
                    await client.put_object(
                        Bucket=self._settings.bucket_name,
                        Key=key,
                        Body=data,
                        ContentType=content_type,
                    )
            logger.info("r2.put_raw_completed", key=key, bytes=len(data))
        except ClientError as e:
            logger.exception("r2.put_raw_failed", key=key, error=str(e))
            raise StorageUploadError(
                f"Failed to store object: {_get_error_message(e)}",
                cause=e,
            ) from e

    async def download(self, storage_key: str) -> bytes:
        """Download file content from R2."""
        try:
            async with self._get_client() as client:
                response = await client.get_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )
                data = await response["Body"].read()
                logger.debug("r2.download_completed", key=storage_key, bytes=len(data))
                return data

        except ClientError as e:
            error_code = _get_error_code(e)
            if error_code in ("NoSuchKey", "404"):
                raise StorageNotFoundError(f"File not found: {storage_key}") from e
            logger.exception("r2.download_failed", key=storage_key, error=str(e))
            raise StorageDownloadError(
                f"Failed to download file: {_get_error_message(e)}",
                cause=e,
            ) from e
        except Exception as e:
            logger.exception("r2.download_unexpected_error", error=str(e))
            raise StorageDownloadError(f"Download failed: {e}", cause=e) from e

    async def sign_key(self, storage_key: str, *, expires_in: int = 3600) -> str:
        """Generate a presigned URL for a storage key without fetching object metadata.

        Cheaper than get_presigned_url — skips the head_object round-trip.
        Use this when the key is known to exist (e.g. already stored in the DB).

        Args:
            storage_key: R2 object key.
            expires_in: URL validity in seconds (default 1 hour).

        Returns:
            Presigned URL string.
        """
        async with self._get_client() as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.bucket_name, "Key": storage_key},
                ExpiresIn=expires_in,
            )

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
            error_code = _get_error_code(e)
            if error_code in ("NoSuchKey", "404"):
                raise StorageNotFoundError(f"File not found: {storage_key}") from e
            logger.exception("r2.presigned_url_failed", key=storage_key, error=str(e))
            raise StorageDownloadError(
                f"Failed to generate URL: {_get_error_message(e)}",
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
                    if _get_error_code(e) in ("NoSuchKey", "404"):
                        return False
                    raise

                # Delete the object
                await client.delete_object(
                    Bucket=self._settings.bucket_name,
                    Key=storage_key,
                )
                logger.info("r2.delete_completed", key=storage_key)
                return True

        except ClientError as e:
            logger.exception("r2.delete_failed", key=storage_key, error=str(e))
            raise StorageDeleteError(
                f"Failed to delete file: {_get_error_message(e)}",
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

                logger.info("r2.batch_delete_completed", count=deleted_count)
                return deleted_count

        except ClientError as e:
            logger.exception("r2.batch_delete_failed", error=str(e))
            raise StorageDeleteError(
                f"Failed to delete files: {_get_error_message(e)}",
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
            if _get_error_code(e) in ("NoSuchKey", "404"):
                return False
            logger.exception("r2.exists_check_failed", key=storage_key, error=str(e))
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
                    PaginationConfig={"MaxItems": limit},
                ):
                    for obj in page.get("Contents", []):
                        # Parse the storage key to extract metadata
                        storage_key = obj.get("Key")
                        if not storage_key:
                            continue
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
            logger.exception("r2.list_failed", user_id=str(user_id), error=str(e))
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
            image_format = MediaFormat.from_extension(ext)

            created_at = last_modified or datetime.now(UTC)

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
            logger.warning("r2.key_parse_failed", key=storage_key, error=str(e))
            return None

    @asynccontextmanager
    async def stream_object(
        self,
        storage_key: str,
        *,
        range_header: str | None = None,
    ) -> AsyncIterator[ObjectStream]:
        """Context-managed R2 object stream — the only R2 round-trip per request.

        A single GetObject call carries everything the caller needs:
        Content-Type/Content-Length come back on the response regardless of
        whether a range was requested (no separate head_object call), and
        Content-Range comes back too when ``range_header`` is forwarded and
        satisfiable. The client connection stays open for the lifetime of
        the context so the body can be streamed lazily.

        Args:
            storage_key: R2 object key.
            range_header: Raw `bytes=start-end` value to forward as the
                GetObject `Range` parameter, or None for the full object.
                Callers are expected to have already validated this against
                a known size (see `src.api.utils.http_range.parse_range`) —
                this is a thin forwarding layer, not a validator.

        Yields:
            ObjectStream with the byte iterator plus response metadata.

        Raises:
            StorageNotFoundError: If the key doesn't exist.
            StorageRangeNotSatisfiableError: If R2 rejects the forwarded range.
            StorageDownloadError: If the stream fails for any other reason.
        """
        async with self._get_client() as client:
            params: dict[str, Any] = {
                "Bucket": self._settings.bucket_name,
                "Key": storage_key,
            }
            if range_header is not None:
                params["Range"] = range_header

            try:
                response = await client.get_object(**params)
            except ClientError as e:
                error_code = _get_error_code(e)
                if error_code in ("NoSuchKey", "404"):
                    raise StorageNotFoundError(f"File not found: {storage_key}") from e
                if error_code == "InvalidRange":
                    raise StorageRangeNotSatisfiableError(
                        f"Range not satisfiable: {storage_key}"
                    ) from e
                raise StorageDownloadError(
                    f"Stream failed: {_get_error_message(e)}",
                    cause=e,
                ) from e

            content_type = response.get("ContentType", "application/octet-stream")
            content_length = response.get("ContentLength", 0)
            content_range = response.get("ContentRange")

            async def _iter_chunks() -> AsyncIterator[bytes]:
                async for chunk in response["Body"].iter_chunks(chunk_size=65536):
                    yield chunk

            yield ObjectStream(
                chunks=_iter_chunks(),
                content_type=content_type,
                content_length=content_length,
                content_range=content_range,
            )

    async def health_check(self) -> bool:
        """Check if R2 storage is accessible."""
        try:
            async with self._get_client() as client:
                await client.head_bucket(Bucket=self._settings.bucket_name)
                return True
        except Exception as e:
            logger.warning("r2.health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close any open connections.

        Note: aioboto3 manages connections per-context, so this is a no-op.
        Kept for protocol compliance.
        """
