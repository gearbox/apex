"""Unit tests for UserContentService."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image

from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.schemas.user_content import GeneratedImage, ImageAccess, UploadedImage
from src.api.services.storage import StorageNotFoundError, StorageValidationError
from src.api.services.user_content import (
    UserContentNotFoundError,
    UserContentService,
    UserContentTooLargeError,
    UserContentValidationError,
)
from src.core.enums import MediaFormat, OutputMediaType

pytestmark = pytest.mark.unit


def _png_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _heic_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    # Importing user_content (above) transitively imports image_normalization,
    # which registers the HEIF opener/writer with Pillow at module import time.
    im = Image.new("RGB", size, (200, 100, 50))
    buf = io.BytesIO()
    im.save(buf, format="HEIF")
    return buf.getvalue()


def _make_media() -> MediaObject:
    return MediaObject(
        media_type=OutputMediaType.IMAGE,
        original=MediaOriginal(
            url="/v1/content/uploads/abc",
            width=800,
            height=600,
            content_type="image/png",
            size_bytes=1024,
        ),
        variants=[],
    )


def _make_upload_result() -> MagicMock:
    r = MagicMock()
    r.id = uuid4()
    r.storage_key = f"users/u/uploads/{r.id}.png"
    return r


def _make_db_image(**overrides: object) -> MagicMock:
    img = MagicMock()
    img.id = uuid4()
    img.storage_key = f"users/u/uploads/{img.id}.png"
    img.original_filename = "photo.png"
    img.content_type = "image/png"
    img.size_bytes = 1024
    img.created_at = datetime.now(UTC)
    img.expires_at = datetime.now(UTC) + timedelta(days=7)
    img.width = 800
    img.height = 600
    img.thumbnail_max_edge = None
    for k, v in overrides.items():
        setattr(img, k, v)
    return img


def _make_db_output(**overrides: object) -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.job_id = uuid4()
    out.storage_key = f"users/u/outputs/j/{out.id}.png"
    out.content_type = "image/png"
    out.size_bytes = 2048
    out.output_index = 0
    out.created_at = datetime.now(UTC)
    out.expires_at = datetime.now(UTC) + timedelta(days=7)
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def _make_service(*, max_input_megapixels: float = 100.0) -> tuple[UserContentService, AsyncMock]:
    storage = AsyncMock()
    session = AsyncMock()
    service = UserContentService(
        storage=storage,
        session=session,
        product_id="vex",
        max_input_megapixels=max_input_megapixels,
    )
    service._image_repo = AsyncMock()
    service._output_repo = AsyncMock()
    service._job_repo = AsyncMock()
    return service, storage


# ---------------------------------------------------------------------------
# upload_image
# ---------------------------------------------------------------------------


class TestUploadImage:
    async def test_happy_path_returns_uploaded_image(self) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
            patch("src.api.services.user_content.read_dimensions", return_value=None),
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo.png",
                content_type="image/png",
            )

        assert isinstance(result, UploadedImage)
        assert result.id == db_image.id
        assert result.filename == db_image.original_filename

    async def test_reads_dimensions_when_available(self) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image(width=1024, height=768)
        service._image_repo.create = AsyncMock(return_value=db_image)

        from src.api.services.image_thumbnail import ImageDimensions

        dims = ImageDimensions(width=1024, height=768)
        with (
            patch("src.api.services.user_content.read_dimensions", return_value=dims),
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="big.png",
                content_type="image/png",
            )

        assert isinstance(result, UploadedImage)
        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["width"] == 1024
        assert create_kwargs["height"] == 768

    async def test_creates_thumbnails_when_generated(self) -> None:
        service, storage = _make_service()

        main_result = _make_upload_result()
        thumb_result = _make_upload_result()
        storage.upload = AsyncMock(side_effect=[main_result, thumb_result])

        db_image = _make_db_image()
        thumb_db = _make_db_image()
        service._image_repo.create = AsyncMock(side_effect=[db_image, thumb_db])

        from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
        from src.core.thumbnails import ThumbnailSpec

        thumb = GeneratedThumbnail(
            spec=ThumbnailSpec("sm", 150),
            result=ThumbnailResult(data=b"webpdata", width=100, height=75),
        )

        with (
            patch("src.api.services.user_content.read_dimensions", return_value=None),
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[thumb]),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo.png",
                content_type="image/png",
            )

        assert storage.upload.call_count == 2
        assert service._image_repo.create.call_count == 2
        assert isinstance(result, UploadedImage)

    async def test_thumbnail_failure_does_not_abort_upload(self) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
            patch("src.api.services.user_content.read_dimensions", return_value=None),
            patch(
                "src.api.services.user_content.make_image_thumbnails",
                side_effect=Exception("thumbnail crash"),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo.png",
                content_type="image/png",
            )

        assert isinstance(result, UploadedImage)
        assert storage.upload.call_count == 1  # Only main upload

    async def test_storage_validation_error_raises_user_content_error(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(side_effect=StorageValidationError("too big"))

        with (
            patch("src.api.services.user_content.read_dimensions", return_value=None),
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
            pytest.raises(UserContentValidationError),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
                filename="photo.png",
                content_type="image/png",
            )


# ---------------------------------------------------------------------------
# upload_image — normalization (D1')
# ---------------------------------------------------------------------------


class TestUploadImageNormalization:
    async def test_upload_mislabeled_heic_stored_as_png(self) -> None:
        """A HEIC file mislabeled as image/webp is sniffed, converted to PNG,
        and stored as such — the declared content type is never trusted."""
        service, storage = _make_service()

        main_result = _make_upload_result()
        thumb_sm_result = _make_upload_result()
        thumb_md_result = _make_upload_result()
        storage.upload = AsyncMock(side_effect=[main_result, thumb_sm_result, thumb_md_result])

        db_image = _make_db_image(format="png", content_type="image/png")
        thumb_sm_db = _make_db_image()
        thumb_md_db = _make_db_image()
        service._image_repo.create = AsyncMock(side_effect=[db_image, thumb_sm_db, thumb_md_db])

        result = await service.upload_image(
            user_id=uuid4(),
            data=_heic_bytes(),
            filename="temp_image.webp",
            content_type="image/webp",
        )

        assert isinstance(result, UploadedImage)

        # Main upload received sniffed-and-converted PNG bytes, not the
        # original HEIC bytes or the declared webp content type.
        main_upload_kwargs = storage.upload.call_args_list[0].kwargs
        assert main_upload_kwargs["content_type"] == "image/png"
        assert main_upload_kwargs["data"][:8] == b"\x89PNG\r\n\x1a\n"

        create_kwargs = service._image_repo.create.call_args_list[0].kwargs
        assert create_kwargs["format"] == "png"
        assert create_kwargs["content_type"] == "image/png"

        # Thumbnails were generated — no longer silently skipped now that the
        # bytes handed to Pillow are real PNG, not mislabeled HEIC.
        assert storage.upload.call_count == 3
        assert service._image_repo.create.call_count == 3

    async def test_upload_undecodable_raises_validation_error(self) -> None:
        """Garbage bytes fail decode and raise before anything is persisted."""
        service, storage = _make_service()
        storage.upload = AsyncMock()
        service._image_repo.create = AsyncMock()

        with pytest.raises(UserContentValidationError):
            await service.upload_image(
                user_id=uuid4(),
                data=b"this is not an image, just plain text bytes",
                filename="temp_image.webp",
                content_type="image/webp",
            )

        storage.upload.assert_not_called()
        service._image_repo.create.assert_not_called()

    async def test_upload_png_unchanged(self) -> None:
        """Regression: a well-formed PNG upload passes through byte-for-byte."""
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        png_bytes = _png_bytes()

        with (
            patch("src.api.services.user_content.read_dimensions", return_value=None),
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=png_bytes,
                filename="photo.png",
                content_type="image/png",
            )

        assert isinstance(result, UploadedImage)
        upload_kwargs = storage.upload.call_args.kwargs
        assert upload_kwargs["data"] == png_bytes
        assert upload_kwargs["content_type"] == "image/png"

    async def test_upload_oversized_image_maps_to_413(self) -> None:
        """F4/D4: an image over the configured pixel cap raises
        UserContentTooLargeError — mapped to HTTP 413 at the route
        (see tests/unit/test_storage_routes.py::test_too_large_error_returns_413).

        Uses a real, tiny image against an artificially tiny cap (rather than
        a huge image) so the test stays fast.
        """
        service, storage = _make_service(max_input_megapixels=0.0001)  # 100 px cap
        png_bytes = _png_bytes()  # 16x12 = 192 px — over the 100 px cap

        with pytest.raises(UserContentTooLargeError):
            await service.upload_image(
                user_id=uuid4(),
                data=png_bytes,
                filename="photo.png",
                content_type="image/png",
            )

        storage.upload.assert_not_called()
        service._image_repo.create.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# get_upload / get_upload_by_key
# ---------------------------------------------------------------------------


class TestGetUpload:
    async def test_delegates_to_image_repo(self) -> None:
        service, _ = _make_service()
        img = _make_db_image()
        service._image_repo.get = AsyncMock(return_value=img)

        result = await service.get_upload(img.id, user_id=uuid4())
        assert result is img

    async def test_returns_none_when_not_found(self) -> None:
        service, _ = _make_service()
        service._image_repo.get = AsyncMock(return_value=None)

        result = await service.get_upload(uuid4(), user_id=uuid4())
        assert result is None

    async def test_get_by_key_delegates_to_repo(self) -> None:
        service, _ = _make_service()
        img = _make_db_image()
        service._image_repo.get_by_key = AsyncMock(return_value=img)

        result = await service.get_upload_by_key("some/key")
        assert result is img


# ---------------------------------------------------------------------------
# get_upload_access
# ---------------------------------------------------------------------------


class TestGetUploadAccess:
    async def test_returns_image_access_for_existing_image(self) -> None:
        service, storage = _make_service()
        img = _make_db_image()
        service._image_repo.get = AsyncMock(return_value=img)

        presigned = MagicMock()
        presigned.storage_key = img.storage_key
        presigned.presigned_url = "https://r2.example.com/key?sig=abc"
        presigned.content_type = "image/png"
        presigned.size_bytes = 1024
        presigned.expires_in_seconds = 3600
        storage.get_presigned_url = AsyncMock(return_value=presigned)

        result = await service.get_upload_access(img.id, user_id=uuid4())

        assert isinstance(result, ImageAccess)
        assert result.presigned_url == presigned.presigned_url

    async def test_raises_not_found_when_image_missing(self) -> None:
        service, _ = _make_service()
        service._image_repo.get = AsyncMock(return_value=None)

        with pytest.raises(UserContentNotFoundError):
            await service.get_upload_access(uuid4(), user_id=uuid4())


# ---------------------------------------------------------------------------
# download_upload
# ---------------------------------------------------------------------------


class TestDownloadUpload:
    async def test_returns_bytes_for_existing_image(self) -> None:
        service, storage = _make_service()
        img = _make_db_image()
        service._image_repo.get = AsyncMock(return_value=img)
        storage.download = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n")

        result = await service.download_upload(img.id, user_id=uuid4())
        assert result == b"\x89PNG\r\n\x1a\n"

    async def test_raises_not_found_when_db_record_missing(self) -> None:
        service, _ = _make_service()
        service._image_repo.get = AsyncMock(return_value=None)

        with pytest.raises(UserContentNotFoundError):
            await service.download_upload(uuid4(), user_id=uuid4())

    async def test_raises_not_found_when_r2_file_missing(self) -> None:
        service, storage = _make_service()
        img = _make_db_image()
        service._image_repo.get = AsyncMock(return_value=img)
        storage.download = AsyncMock(side_effect=StorageNotFoundError("r2 key missing"))

        with pytest.raises(UserContentNotFoundError):
            await service.download_upload(img.id, user_id=uuid4())


# ---------------------------------------------------------------------------
# upload derivatives
# ---------------------------------------------------------------------------


class TestListUploads:
    async def test_list_upload_derivatives_delegates(self) -> None:
        service, _ = _make_service()
        thumb = _make_db_image()
        service._image_repo.list_derivatives = AsyncMock(return_value=[thumb])

        result = await service.list_upload_derivatives(uuid4())
        assert result == [thumb]

    async def test_batch_output_derivatives_delegates(self) -> None:
        service, _ = _make_service()
        output_id = uuid4()
        service._output_repo.batch_derivatives = AsyncMock(return_value={output_id: []})

        result = await service.batch_output_derivatives([output_id])
        assert output_id in result


# ---------------------------------------------------------------------------
# delete_upload
# ---------------------------------------------------------------------------


class TestDeleteUpload:
    async def test_returns_true_and_deletes_from_r2_and_db(self) -> None:
        service, storage = _make_service()
        img = _make_db_image()
        thumb = _make_db_image()
        service._image_repo.get = AsyncMock(return_value=img)
        service._image_repo.list_derivatives = AsyncMock(return_value=[thumb])
        service._image_repo.delete = AsyncMock(return_value=True)
        storage.delete = AsyncMock()

        result = await service.delete_upload(img.id, user_id=uuid4())

        assert result is True
        assert storage.delete.call_count == 2  # thumb + main image

    async def test_returns_false_when_image_not_found(self) -> None:
        service, _ = _make_service()
        service._image_repo.get = AsyncMock(return_value=None)

        result = await service.delete_upload(uuid4(), user_id=uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# store_output
# ---------------------------------------------------------------------------


class TestStoreOutput:
    async def test_returns_generated_image(self) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_out = _make_db_output()
        service._output_repo.create = AsyncMock(return_value=db_out)

        result = await service.store_output(
            user_id=uuid4(),
            job_id=uuid4(),
            data=b"\x89PNG",
            content_type="image/png",
            output_index=0,
        )

        assert isinstance(result, GeneratedImage)
        assert result.id == db_out.id

    async def test_passes_input_image_id_to_repo(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result())

        db_out = _make_db_output()
        service._output_repo.create = AsyncMock(return_value=db_out)

        input_id = uuid4()
        await service.store_output(
            user_id=uuid4(),
            job_id=uuid4(),
            data=b"\x89PNG",
            content_type="image/png",
            output_index=1,
            input_image_id=input_id,
        )

        create_kwargs = service._output_repo.create.call_args.kwargs
        assert create_kwargs["input_image_id"] == input_id


# ---------------------------------------------------------------------------
# get_output / get_output_access / download_output
# ---------------------------------------------------------------------------


class TestGetOutput:
    async def test_delegates_to_output_repo(self) -> None:
        service, _ = _make_service()
        out = _make_db_output()
        service._output_repo.get = AsyncMock(return_value=out)

        result = await service.get_output(out.id, user_id=uuid4())
        assert result is out

    async def test_get_output_access_returns_image_access(self) -> None:
        service, storage = _make_service()
        out = _make_db_output()
        service._output_repo.get = AsyncMock(return_value=out)

        presigned = MagicMock()
        presigned.storage_key = out.storage_key
        presigned.presigned_url = "https://r2.example.com/out"
        presigned.content_type = "image/png"
        presigned.size_bytes = 2048
        presigned.expires_in_seconds = 3600
        storage.get_presigned_url = AsyncMock(return_value=presigned)

        result = await service.get_output_access(out.id, user_id=uuid4())
        assert isinstance(result, ImageAccess)

    async def test_get_output_access_raises_when_not_found(self) -> None:
        service, _ = _make_service()
        service._output_repo.get = AsyncMock(return_value=None)

        with pytest.raises(UserContentNotFoundError):
            await service.get_output_access(uuid4(), user_id=uuid4())

    async def test_download_output_returns_bytes(self) -> None:
        service, storage = _make_service()
        out = _make_db_output()
        service._output_repo.get = AsyncMock(return_value=out)
        storage.download = AsyncMock(return_value=b"\x89PNG")

        result = await service.download_output(out.id, user_id=uuid4())
        assert result == b"\x89PNG"

    async def test_download_output_raises_when_not_found_in_db(self) -> None:
        service, _ = _make_service()
        service._output_repo.get = AsyncMock(return_value=None)

        with pytest.raises(UserContentNotFoundError):
            await service.download_output(uuid4(), user_id=uuid4())

    async def test_download_output_raises_when_r2_file_missing(self) -> None:
        service, storage = _make_service()
        out = _make_db_output()
        service._output_repo.get = AsyncMock(return_value=out)
        storage.download = AsyncMock(side_effect=StorageNotFoundError("missing"))

        with pytest.raises(UserContentNotFoundError):
            await service.download_output(out.id, user_id=uuid4())


# ---------------------------------------------------------------------------
# list_job_outputs / list_user_outputs
# ---------------------------------------------------------------------------


class TestListOutputs:
    async def test_list_job_outputs_raises_when_job_not_found(self) -> None:
        service, _ = _make_service()
        service._job_repo.get = AsyncMock(return_value=None)

        with pytest.raises(UserContentNotFoundError):
            await service.list_job_outputs(uuid4(), user_id=uuid4())

    async def test_list_job_outputs_returns_outputs(self) -> None:
        service, _ = _make_service()
        job = MagicMock()
        service._job_repo.get = AsyncMock(return_value=job)
        outs = [_make_db_output()]
        service._output_repo.list_by_job = AsyncMock(return_value=outs)

        result = await service.list_job_outputs(uuid4(), user_id=uuid4())
        assert result == outs

    async def test_list_user_outputs_delegates(self) -> None:
        service, _ = _make_service()
        outs = [_make_db_output()]
        service._output_repo.list_by_user = AsyncMock(return_value=outs)

        result = await service.list_user_outputs(uuid4())
        assert result == outs


# ---------------------------------------------------------------------------
# Storage key utilities
# ---------------------------------------------------------------------------


class TestStorageKeyUtilities:
    def test_get_upload_storage_key(self) -> None:
        service, storage = _make_service()
        storage.build_storage_key = MagicMock(return_value="users/u/uploads/id.png")

        result = service.get_upload_storage_key(uuid4(), uuid4(), MediaFormat.PNG)
        assert result == "users/u/uploads/id.png"
        storage.build_storage_key.assert_called_once()

    def test_get_output_storage_key(self) -> None:
        service, storage = _make_service()
        storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/id.png")

        result = service.get_output_storage_key(uuid4(), uuid4(), uuid4(), MediaFormat.PNG)
        assert result == "users/u/outputs/j/id.png"
        storage.build_storage_key.assert_called_once()


# ---------------------------------------------------------------------------
# get_user_stats
# ---------------------------------------------------------------------------


class TestGetUserStats:
    async def test_aggregates_upload_and_output_stats(self) -> None:
        service, _ = _make_service()
        service._image_repo.count_and_sum_by_user = AsyncMock(return_value=(3, 3000))
        service._output_repo.count_and_sum_by_user = AsyncMock(return_value=(5, 5000))

        stats = await service.get_user_stats(uuid4())

        assert stats["upload_count"] == 3
        assert stats["output_count"] == 5
        assert stats["total_bytes"] == 8000
