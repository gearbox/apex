"""Unit tests for video upload support in UserContentService.upload_image."""

from __future__ import annotations

import dataclasses
import errno
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from litestar.status_codes import HTTP_503_SERVICE_UNAVAILABLE

from src.api.schemas.user_content import UploadedImage
from src.api.services.media_ingest import (
    InvalidMediaError,
    MediaIngestService,
    MediaProcessingError,
    PreparedVideo,
    VideoStreamProfile,
)
from src.api.services.user_content import (
    UserContentService,
    UserContentUnavailableError,
    UserContentValidationError,
)
from src.core.enums import MediaFormat
from src.core.media_hash import HashSample, HashSet, PdqHash

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
    img.format = "mp4"
    for k, v in overrides.items():
        setattr(img, k, v)
    return img


def _prepared_video(*, format: MediaFormat = MediaFormat.MP4) -> PreparedVideo:
    return PreparedVideo(
        data=b"prepared-video",
        format=format,
        width=1280,
        height=720,
        duration_ms=8000,
        hash_set=HashSet(
            profile_id="pdq-image-rgb-white-v1",
            sampling_profile="fps-1-v1",
            samples=(
                HashSample(
                    pdq=PdqHash(bits=b"\x00" * 32, quality=100),
                    sample_index=0,
                    frame_timestamp_ms=0,
                ),
            ),
        ),
        stream_profile=VideoStreamProfile(
            container=format,
            codec="h264",
            codec_profile="High",
            pix_fmt="yuv420p",
            color_transfer="bt709",
            color_primaries="bt709",
            rotation_degrees=0,
            sample_aspect_ratio="1:1",
            has_audio=False,
        ),
    )


def _make_service(*, video_max_seconds: int = 300) -> tuple[UserContentService, AsyncMock]:
    storage = AsyncMock()
    session = AsyncMock()
    session.add_all = MagicMock()
    session.begin_nested = MagicMock(side_effect=_async_context_manager)
    media_ingestor = MagicMock()
    media_ingestor.prepare_video = AsyncMock(return_value=_prepared_video())
    service = UserContentService(
        storage=storage,
        session=session,
        product_id="vex",
        video_max_seconds=video_max_seconds,
        media_ingestor=media_ingestor,
    )
    service._image_repo = AsyncMock()
    service._output_repo = AsyncMock()
    service._job_repo = AsyncMock()
    return service, storage


def _async_context_manager() -> MagicMock:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=None)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


class TestUploadVideoAccepted:
    async def test_upload_video_mp4_accepted_with_probe_metadata(self) -> None:
        service, storage = _make_service()
        storage.upload = AsyncMock(return_value=_make_upload_result())
        db_video = _make_db_video(width=1280, height=720)
        service._image_repo.create = AsyncMock(return_value=db_video)

        with (
            patch(
                "src.api.services.user_content.extract_video_thumbnail",
                AsyncMock(return_value=None),
            ),
        ):
            result = await service.upload_image(
                user_id=uuid4(),
                data=b"private-metadata-canary",
                filename="clip.mp4",
                content_type="video/mp4",
            )

        assert isinstance(result, UploadedImage)
        create_kwargs = service._image_repo.create.call_args.kwargs
        assert create_kwargs["duration_ms"] == 8000
        assert create_kwargs["width"] == 1280
        assert create_kwargs["height"] == 720
        assert create_kwargs["format"] == "mp4"
        assert storage.upload.await_args.kwargs["data"] == b"prepared-video"

    async def test_upload_video_non_latin_filename_succeeds(self) -> None:
        """Issue B regression: a non-latin filename must not reach R2 metadata
        headers and must not raise — original_filename becomes {uuid}.mp4."""
        service, storage = _make_service()
        upload_result = _make_upload_result()
        storage.upload = AsyncMock(return_value=upload_result)
        db_video = _make_db_video()
        service._image_repo.create = AsyncMock(return_value=db_video)

        with (
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
        service._media_ingestor.prepare_video = AsyncMock(
            return_value=_prepared_video(format=MediaFormat.MOV)
        )

        with (
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

        with (
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
        service._media_ingestor.prepare_video = AsyncMock(
            side_effect=InvalidMediaError("video duration exceeds the configured limit")
        )

        with (
            pytest.raises(UserContentValidationError, match="exceeds"),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="long.mp4",
                content_type="video/mp4",
            )

    async def test_upload_fake_video_mime_rejected_by_probe(self) -> None:
        service, _storage = _make_service()
        service._media_ingestor.prepare_video = AsyncMock(
            side_effect=InvalidMediaError("video is not decodable")
        )

        with (
            pytest.raises(UserContentValidationError, match="not decodable"),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"totally not a video, just an .exe renamed",
                filename="fake.mp4",
                content_type="video/mp4",
            )

    async def test_upload_video_capacity_exhausted_is_unavailable(self) -> None:
        service, storage = _make_service()
        service._media_ingestor.prepare_video = AsyncMock(
            side_effect=MediaProcessingError("video preparation capacity is exhausted")
        )

        with pytest.raises(UserContentUnavailableError, match="temporarily unavailable"):
            await service.upload_image(
                user_id=uuid4(),
                data=b"fake mp4 bytes",
                filename="clip.mp4",
                content_type="video/mp4",
            )
        storage.upload.assert_not_awaited()


class TestUploadVideoRoute:
    async def test_temp_dir_failure_returns_503_service_unavailable(self) -> None:
        """A full/unwritable temp dir is local unavailability (503), not a 500."""
        from src.api.routes.storage import StorageController

        service, storage = _make_service()
        service._media_ingestor = MediaIngestService(
            max_image_megapixels=10, max_input_bytes=2 * 1024 * 1024
        )
        upload_file = AsyncMock()
        upload_file.content_type = "video/mp4"
        upload_file.filename = "clip.mp4"
        upload_file.read = AsyncMock(return_value=b"fake mp4 bytes")
        form = MagicMock()
        form.data = upload_file
        form.source_asset_ref = None
        form.source_timestamp_ms = None

        with patch(
            "src.api.services.media_ingest.service.tempfile.mkdtemp",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            response = await StorageController.upload_image.fn(
                MagicMock(),
                current_user_id=uuid4(),
                user_content=service,
                data=form,
            )

        assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE
        assert response.content.error == "service_unavailable"
        storage.upload.assert_not_awaited()


class TestVideoProfileLog:
    """I13 — ``media.video_profile`` is logged once per stored video, from the probed facts."""

    @staticmethod
    async def _upload(service: UserContentService, storage: AsyncMock, **db: object) -> MagicMock:
        storage.upload = AsyncMock(return_value=_make_upload_result())
        db_video = _make_db_video(**db)
        service._image_repo.create = AsyncMock(return_value=db_video)
        with patch(
            "src.api.services.user_content.extract_video_thumbnail",
            AsyncMock(return_value=None),
        ):
            await service.upload_image(
                user_id=uuid4(),
                data=b"video",
                filename="clip.mp4",
                content_type="video/mp4",
            )
        return db_video

    async def test_video_upload_logs_one_profile_event(
        self, video_profile_events: list[dict[str, Any]]
    ) -> None:
        service, storage = _make_service()

        db_video = await self._upload(service, storage)

        events = [e for e in video_profile_events if e["event"] == "media.video_profile"]
        assert len(events) == 1
        event = events[0]
        assert event["origin"] == "upload"
        assert event["asset_ref"] == f"upload:{db_video.id}"
        assert event["container"] == "mp4"
        assert event["codec"] == "h264"
        assert event["codec_profile"] == "High"
        assert event["pix_fmt"] == "yuv420p"
        assert event["color_transfer"] == "bt709"
        assert event["color_primaries"] == "bt709"
        assert event["hdr"] is False
        assert event["rotation_degrees"] == 0
        assert event["sample_aspect_ratio"] == "1:1"
        assert event["has_audio"] is False
        assert (event["width"], event["height"], event["duration_ms"]) == (1280, 720, 8000)
        assert "job_id" not in event

    async def test_hdr_flag_is_derived_from_the_transfer_characteristic(
        self, video_profile_events: list[dict[str, Any]]
    ) -> None:
        service, storage = _make_service()
        hdr = dataclasses.replace(
            _prepared_video().stream_profile, color_transfer="smpte2084", color_primaries="bt2020"
        )
        service._media_ingestor.prepare_video = AsyncMock(
            return_value=dataclasses.replace(_prepared_video(), stream_profile=hdr)
        )

        await self._upload(service, storage)

        (event,) = [e for e in video_profile_events if e["event"] == "media.video_profile"]
        assert event["hdr"] is True

    async def test_failed_video_preparation_logs_no_profile(
        self, video_profile_events: list[dict[str, Any]]
    ) -> None:
        service, storage = _make_service()
        service._media_ingestor.prepare_video = AsyncMock(side_effect=InvalidMediaError("bad"))

        with pytest.raises(UserContentValidationError):
            await self._upload(service, storage)

        assert not [e for e in video_profile_events if e["event"] == "media.video_profile"]
