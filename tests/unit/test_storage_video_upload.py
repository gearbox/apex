"""Unit tests for video upload support in UserContentService.upload_image."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.user_content import UploadedImage
from src.api.services.frames.ffmpeg import FfprobeError, VideoProbe
from src.api.services.user_content import UserContentService, UserContentValidationError

pytestmark = pytest.mark.unit


def _make_upload_result(*, ext: str = "mp4") -> MagicMock:
    r = MagicMock()
    r.id = uuid4()
    r.storage_key = f"users/u/uploads/{r.id}.{ext}"
    return r


def _make_db_video(**overrides: object) -> MagicMock:
    img = MagicMock()
    img.id = uuid4()
    img.storage_key = f"users/u/uploads/{img.id}.mp4"
    img.original_filename = "clip.mp4"
    img.content_type = "video/mp4"
    img.size_bytes = 4096
    img.created_at = datetime.now(UTC)
    img.expires_at = datetime.now(UTC) + timedelta(days=7)
    img.width = 1920
    img.height = 1080
    img.thumbnail_max_edge = None
    for k, v in overrides.items():
        setattr(img, k, v)
    return img


def _make_service(*, video_max_seconds: int = 300) -> tuple[UserContentService, AsyncMock]:
    storage = AsyncMock()
    session = AsyncMock()
    service = UserContentService(
        storage=storage,
        session=session,
        product_id="vex",
        video_max_seconds=video_max_seconds,
    )
    service._image_repo = AsyncMock()
    service._output_repo = AsyncMock()
    service._job_repo = AsyncMock()
    return service, storage


class TestUploadVideoAccepted:
    async def test_upload_video_mp4_accepted_with_probe_metadata(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result())
        db_video = _make_db_video(width=1280, height=720)
        service._image_repo.create = AsyncMock(return_value=db_video)

        probe_result = VideoProbe(duration_ms=8000, width=1280, height=720, codec="h264")
        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(return_value=None),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="clip.mp4",
                content_type="video/mp4",
            )

        assert isinstance(result, UploadedImage)
        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["duration_ms"] == 8000
        assert create_kwargs["width"] == 1280
        assert create_kwargs["height"] == 720
        assert create_kwargs["format"] == "mp4"

    async def test_upload_video_non_latin_filename_succeeds(self) -> None:
        """Issue B regression: a non-latin filename must not reach R2 metadata
        headers and must not raise — original_filename becomes {uuid}.mp4."""
        service, storage = _make_service()
        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)
        db_video = _make_db_video()
        service._image_repo.create = AsyncMock(return_value=db_video)

        probe_result = VideoProbe(duration_ms=8000, width=1280, height=720, codec="h264")
        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(return_value=None),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="видео тест.mp4",
                content_type="video/mp4",
            )

        assert isinstance(result, UploadedImage)
        assert "filename" not in storage.upload.call_args.kwargs
        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["original_filename"] == f"{upload_result.id}.mp4"
        assert create_kwargs["display_filename"] == "видео тест.mp4"

    async def test_upload_mov_container_accepted(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result(ext="mov"))
        db_video = _make_db_video(content_type="video/quicktime")
        service._image_repo.create = AsyncMock(return_value=db_video)

        probe_result = VideoProbe(duration_ms=3000, width=640, height=480, codec="hevc")
        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(return_value=None),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"fake mov bytes",
                filename="clip.mov",
                content_type="video/quicktime",
            )

        assert isinstance(result, UploadedImage)
        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["format"] == "mov"

    async def test_upload_video_creates_poster_derivatives(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(
            side_effect=[_make_upload_result(), _make_upload_result(ext="webp")]
        )
        db_video = _make_db_video()
        thumb_db = MagicMock()
        service._image_repo.create = AsyncMock(side_effect=[db_video, thumb_db])

        probe_result = VideoProbe(duration_ms=5000, width=800, height=600, codec="h264")
        generated = MagicMock()
        generated.spec.label = "sm"
        generated.spec.max_edge = 150
        generated.result.data = b"webpbytes"
        generated.result.content_type = "image/webp"
        generated.result.format = "webp"
        generated.result.width = 150
        generated.result.height = 112

        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(return_value=b"jpegposterbytes"),
            ),
            patch(
                "src.api.services.user_content.make_image_thumbnails",
                AsyncMock(return_value=[generated]),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="clip.mp4",
                content_type="video/mp4",
            )

        assert isinstance(result, UploadedImage)
        assert service._image_repo.create.call_count == 2
        thumb_call_kwargs = service._image_repo.create.call_args_list[1].kwargs
        assert thumb_call_kwargs["is_thumbnail"] is True
        assert thumb_call_kwargs["parent_image_id"] == db_video.id

    async def test_upload_video_poster_failure_is_non_fatal(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result())
        db_video = _make_db_video()
        service._image_repo.create = AsyncMock(return_value=db_video)

        probe_result = VideoProbe(duration_ms=5000, width=800, height=600, codec="h264")
        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(side_effect=RuntimeError("ffmpeg crashed")),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="clip.mp4",
                content_type="video/mp4",
            )

        # Poster generation failed but the upload itself still succeeds.
        assert isinstance(result, UploadedImage)
        assert service._image_repo.create.call_count == 1


class TestUploadVideoRejected:
    async def test_upload_video_over_duration_cap_rejected(self) -> None:
        service, _storage = _make_service(video_max_seconds=60)
        probe_result = VideoProbe(duration_ms=120_000, width=640, height=480, codec="h264")

        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(return_value=probe_result),
            ),
            pytest.raises(UserContentValidationError, match="exceeds maximum"),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="long.mp4",
                content_type="video/mp4",
            )

    async def test_upload_fake_video_mime_rejected_by_probe(self) -> None:
        service, _storage = _make_service()

        with (
            patch(
                "src.api.services.user_content.ffmpeg_probe",
                AsyncMock(side_effect=FfprobeError("Invalid data found when processing input")),
            ),
            pytest.raises(UserContentValidationError, match="not a decodable video"),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"totally not a video, just an .exe renamed",
                filename="fake.mp4",
                content_type="video/mp4",
            )
