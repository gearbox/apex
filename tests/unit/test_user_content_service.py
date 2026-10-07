"""Unit tests for UserContentService."""

from __future__ import annotations

import io
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image, PngImagePlugin

from src.api.schemas.media import MediaObject, MediaOriginal
from src.api.schemas.user_content import UploadedImage
from src.api.services.media_ingest import MediaProcessingError
from src.api.services.storage import StorageError, StorageValidationError
from src.api.services.user_content import (
    UserContentService,
    UserContentStorageError,
    UserContentTooLargeError,
    UserContentUnavailableError,
    UserContentValidationError,
    sanitize_display_filename,
)
from src.core.enums import OutputMediaType
from tests.media_ingest_support import make_media_ingestor

_CANONICAL_FILENAME_RE = re.compile(r"^[0-9a-f-]{36}\.(png|jpeg|webp)$")


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
    img.format = "png"
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
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def _make_service(*, max_input_megapixels: float = 100.0) -> tuple[UserContentService, AsyncMock]:
    storage = AsyncMock()
    session = AsyncMock()
    session.add_all = MagicMock()

    @asynccontextmanager
    async def savepoint():
        yield

    session.begin_nested = MagicMock(side_effect=savepoint)
    service = UserContentService(
        storage=storage,
        session=session,
        product_id="vex",
        max_input_megapixels=max_input_megapixels,
        media_ingestor=make_media_ingestor(max_image_megapixels=max_input_megapixels),
    )
    service._image_repo = AsyncMock()
    service._output_repo = AsyncMock()
    return service, storage


# ---------------------------------------------------------------------------
# upload_image
# ---------------------------------------------------------------------------


class TestUploadImage:
    async def test_storage_receives_only_sanitized_image_bytes(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result())
        service._image_repo.create = AsyncMock(return_value=_make_db_image())
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Comment", "private-metadata-canary")
        source = io.BytesIO()
        Image.new("RGB", (16, 12), (255, 0, 0)).save(source, format="PNG", pnginfo=pnginfo)
        with patch("src.api.services.user_content.make_image_thumbnails", return_value=[]):
            await service.upload_image(
                user_id=uuid4(),
                data=source.getvalue(),
                filename="photo.png",
                content_type="image/png",
            )
        stored = storage.upload.await_args.kwargs["data"]
        assert b"private-metadata-canary" not in stored

    async def test_ingest_operational_failure_is_unavailable_error(self) -> None:
        service, storage = _make_service()
        service._media_ingestor.prepare_image = AsyncMock(
            side_effect=MediaProcessingError("capacity exhausted")
        )
        with pytest.raises(UserContentUnavailableError, match="temporarily unavailable"):
            await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo.png",
                content_type="image/png",
            )
        storage.upload.assert_not_awaited()

    async def test_happy_path_returns_uploaded_image(self) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
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

        with (
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
        assert create_kwargs["width"] == 16
        assert create_kwargs["height"] == 12

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
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
            pytest.raises(UserContentValidationError),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
                filename="photo.png",
                content_type="image/png",
            )

    async def test_storage_error_raises_user_content_storage_error(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(side_effect=StorageError("R2 outage"))

        with (
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
            pytest.raises(UserContentStorageError),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo.png",
                content_type="image/png",
            )


# ---------------------------------------------------------------------------
# upload_image — filename safety (Issue B regression)
#
# Non-latin/injection/path-traversal filenames must never reach R2 (as an
# x-amz-meta-* header) or the filesystem. original_filename is always a
# canonical {uuid}.{ext} system name; the sanitized client filename (if any)
# only ever lands in display_filename, for display/search.
# ---------------------------------------------------------------------------


class TestUploadImageFilenameSafety:
    @pytest.mark.parametrize(
        "raw_filename",
        [
            "фотография тест.png",
            "写真.png",
            "photo\r\nX-Injected: 1.png",
            "../../etc/passwd",
            "emoji 🎨.png",
            "x" * 400 + ".png",
        ],
    )
    async def test_non_latin_and_malicious_filenames_succeed(self, raw_filename: str) -> None:
        service, storage = _make_service()

        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)

        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename=raw_filename,
                content_type="image/png",
            )

        assert isinstance(result, UploadedImage)

        # upload() must never receive a filename kwarg at all.
        assert "filename" not in storage.upload.call_args.kwargs

        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["original_filename"] == f"{upload_result.id}.png"
        assert _CANONICAL_FILENAME_RE.match(create_kwargs["original_filename"])

    async def test_original_filename_matches_uuid_ext_pattern(self) -> None:
        service, storage = _make_service()
        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)
        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="whatever.png",
                content_type="image/png",
            )

        create_kwargs = service._image_repo.create.call_args.kwargs
        assert _CANONICAL_FILENAME_RE.match(create_kwargs["original_filename"])

    async def test_thumbnail_rows_get_uuid_names_no_thumb_prefix(self) -> None:
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
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[thumb]),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="фото на русском.png",
                content_type="image/png",
            )

        thumb_create_kwargs = service._image_repo.create.call_args_list[1].kwargs
        thumb_filename = thumb_create_kwargs["original_filename"]
        assert thumb_filename == f"{thumb_result.id}.webp"
        assert not thumb_filename.startswith("thumb_")
        assert "фото" not in thumb_filename
        assert "русском" not in thumb_filename

    @pytest.mark.parametrize(
        ("raw_filename", "expected"),
        [
            ("photo\r\nX: 1.png", "photo X: 1.png"),
            ("x" * 400 + ".png", ("x" * 400 + ".png")[:255]),
        ],
    )
    async def test_display_filename_is_sanitized(self, raw_filename: str, expected: str) -> None:
        assert sanitize_display_filename(raw_filename) == expected

    async def test_display_filename_wired_into_repo_create(self) -> None:
        service, storage = _make_service()
        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)
        db_image = _make_db_image()
        service._image_repo.create = AsyncMock(return_value=db_image)

        with (
            patch("src.api.services.user_content.make_image_thumbnails", return_value=[]),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=_png_bytes(),
                filename="photo\r\nX-Injected: 1.png",
                content_type="image/png",
            )

        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["display_filename"] == "photo X-Injected: 1.png"


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
