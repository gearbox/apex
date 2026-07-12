"""Unit tests for FrameExtractionService."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.schemas.frames import FrameJobResponse
from src.api.services.frames.service import (
    FrameExtractionService,
    FrameJobNotFoundError,
    FrameSourceNotFoundError,
    FrameSourceNotVideoError,
)
from src.core.enums import FrameExtractionKind, FrameExtractionStatus

pytestmark = pytest.mark.unit


def _make_output(**overrides: object) -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.product_id = "vex"
    out.content_type = "video/mp4"
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def _make_upload(**overrides: object) -> MagicMock:
    img = MagicMock()
    img.id = uuid4()
    img.product_id = "vex"
    img.content_type = "video/mp4"
    for k, v in overrides.items():
        setattr(img, k, v)
    return img


def _make_job(**overrides: object) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.kind = FrameExtractionKind.PREVIEW.value
    job.status = FrameExtractionStatus.QUEUED.value
    job.source_output_id = uuid4()
    job.source_upload_id = None
    job.result = None
    job.error = None
    job.created_at = datetime.now(UTC)
    job.started_at = None
    job.finished_at = None
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


def _make_service(
    *, product_id: str = "vex", retention_days: int = 7
) -> tuple[FrameExtractionService, AsyncMock]:
    storage = AsyncMock()
    session = AsyncMock()
    service = FrameExtractionService(
        session,
        storage,
        product_id=product_id,
        preview_url_ttl_seconds=3600,
        retention_days=retention_days,
    )
    service._job_repo = AsyncMock()
    service._image_repo = AsyncMock()
    service._output_repo = AsyncMock()
    service._image_repo.touch_expiry = AsyncMock(return_value=True)
    return service, storage


class TestCreatePreviewJob:
    async def test_rejects_image_source(self) -> None:
        service, _storage = _make_service()
        output = _make_output(content_type="image/png")
        service._output_repo.get = AsyncMock(return_value=output)

        with pytest.raises(FrameSourceNotVideoError):
            await service.create_preview_job(
                uuid4(),
                source_output_id=output.id,
                source_upload_id=None,
                frame_count=12,
            )

    async def test_creates_job_for_video_output(self) -> None:
        service, _storage = _make_service()
        output = _make_output()
        service._output_repo.get = AsyncMock(return_value=output)
        created = _make_job(source_output_id=output.id)
        service._job_repo.create = AsyncMock(return_value=created)

        job = await service.create_preview_job(
            uuid4(),
            source_output_id=output.id,
            source_upload_id=None,
            frame_count=12,
        )

        assert job is created
        service._job_repo.create.assert_awaited_once()
        kwargs = service._job_repo.create.call_args.kwargs
        assert kwargs["kind"] == FrameExtractionKind.PREVIEW.value
        assert kwargs["params"] == {"frame_count": 12}
        assert kwargs["source_output_id"] == output.id
        assert kwargs["source_upload_id"] is None


class TestTouchesSourceExpiry:
    async def test_upload_source_expiry_extended_on_preview(self) -> None:
        service, _storage = _make_service(retention_days=7)
        upload = _make_upload()
        service._image_repo.get = AsyncMock(return_value=upload)
        service._job_repo.create = AsyncMock(return_value=_make_job(source_upload_id=upload.id))

        await service.create_preview_job(
            uuid4(),
            source_output_id=None,
            source_upload_id=upload.id,
            frame_count=12,
        )

        service._image_repo.touch_expiry.assert_awaited_once()
        kwargs = service._image_repo.touch_expiry.call_args.kwargs
        assert kwargs["expires_at"] > datetime.now(UTC) + timedelta(days=6)

    async def test_upload_source_expiry_extended_on_extract(self) -> None:
        service, _storage = _make_service()
        upload = _make_upload()
        service._image_repo.get = AsyncMock(return_value=upload)
        service._job_repo.create = AsyncMock(return_value=_make_job(source_upload_id=upload.id))

        await service.create_extract_job(
            uuid4(),
            source_output_id=None,
            source_upload_id=upload.id,
            timestamps_ms=[0, 1000],
        )

        service._image_repo.touch_expiry.assert_awaited_once()

    async def test_output_source_does_not_touch_upload_expiry(self) -> None:
        service, _storage = _make_service()
        output = _make_output()
        service._output_repo.get = AsyncMock(return_value=output)
        service._job_repo.create = AsyncMock(return_value=_make_job(source_output_id=output.id))

        await service.create_preview_job(
            uuid4(),
            source_output_id=output.id,
            source_upload_id=None,
            frame_count=12,
        )

        service._image_repo.touch_expiry.assert_not_awaited()


class TestCreateJobRequiresExactlyOneSource:
    async def test_rejects_both_sources_set(self) -> None:
        service, _storage = _make_service()
        with pytest.raises(ValueError, match="Exactly one"):
            await service.create_preview_job(
                uuid4(),
                source_output_id=uuid4(),
                source_upload_id=uuid4(),
                frame_count=12,
            )

    async def test_rejects_neither_source_set(self) -> None:
        service, _storage = _make_service()
        with pytest.raises(ValueError, match="Exactly one"):
            await service.create_preview_job(
                uuid4(),
                source_output_id=None,
                source_upload_id=None,
                frame_count=12,
            )


class TestCreateJobEnforcesOwnershipAndProduct:
    async def test_foreign_user_output_not_found(self) -> None:
        service, _storage = _make_service()
        # Repo.get is called with user_id — a foreign user's row simply
        # never matches the ownership-scoped WHERE, so the repo returns None.
        service._output_repo.get = AsyncMock(return_value=None)

        with pytest.raises(FrameSourceNotFoundError):
            await service.create_extract_job(
                uuid4(),
                source_output_id=uuid4(),
                source_upload_id=None,
                timestamps_ms=[0, 1000],
            )

    async def test_wrong_product_upload_not_found(self) -> None:
        service, _storage = _make_service(product_id="synthara")
        upload = _make_upload(product_id="vex")  # belongs to a different product
        service._image_repo.get = AsyncMock(return_value=upload)

        with pytest.raises(FrameSourceNotFoundError):
            await service.create_extract_job(
                uuid4(),
                source_output_id=None,
                source_upload_id=upload.id,
                timestamps_ms=[0],
            )


class TestGetJob:
    async def test_presigns_preview_urls_at_read_time(self) -> None:
        service, storage = _make_service()
        job = _make_job(
            status=FrameExtractionStatus.COMPLETED.value,
            kind=FrameExtractionKind.PREVIEW.value,
            result={
                "frames": [
                    {"index": 0, "timestamp_ms": 0, "key": "frame-previews/u/j/000.webp"},
                    {"index": 1, "timestamp_ms": 500, "key": "frame-previews/u/j/001.webp"},
                ]
            },
        )
        service._job_repo.get = AsyncMock(return_value=job)
        storage.sign_key = AsyncMock(return_value="https://signed.example/frame")

        response = await service.get_job(uuid4(), job.id)

        assert isinstance(response, FrameJobResponse)
        assert response.preview is not None
        assert len(response.preview.frames) == 2
        assert response.preview.frames[0].url == "https://signed.example/frame"
        assert storage.sign_key.await_count == 2
        storage.sign_key.assert_any_await("frame-previews/u/j/000.webp", expires_in=3600)

    async def test_raises_not_found_for_missing_job(self) -> None:
        service, _storage = _make_service()
        service._job_repo.get = AsyncMock(return_value=None)

        with pytest.raises(FrameJobNotFoundError):
            await service.get_job(uuid4(), uuid4())

    async def test_extract_result_builds_media_objects(self) -> None:
        service, _storage = _make_service()
        upload_id = uuid4()
        job = _make_job(
            kind=FrameExtractionKind.EXTRACT.value,
            status=FrameExtractionStatus.COMPLETED.value,
            result={"frames": [{"timestamp_ms": 1000, "upload_id": str(upload_id)}]},
        )
        service._job_repo.get = AsyncMock(return_value=job)

        upload_row = MagicMock()
        upload_row.id = upload_id
        upload_row.content_type = "image/png"
        upload_row.width = 100
        upload_row.height = 100
        upload_row.size_bytes = 123
        service._image_repo.get = AsyncMock(return_value=upload_row)
        service._image_repo.batch_derivatives = AsyncMock(return_value={})

        response = await service.get_job(uuid4(), job.id)

        assert response.extracted is not None
        assert len(response.extracted.frames) == 1
        assert response.extracted.frames[0].upload_id == upload_id
        assert response.extracted.frames[0].timestamp_ms == 1000

    async def test_extract_result_skips_deleted_frames(self) -> None:
        service, _storage = _make_service()
        job = _make_job(
            kind=FrameExtractionKind.EXTRACT.value,
            status=FrameExtractionStatus.COMPLETED.value,
            result={"frames": [{"timestamp_ms": 1000, "upload_id": str(uuid4())}]},
        )
        service._job_repo.get = AsyncMock(return_value=job)
        service._image_repo.get = AsyncMock(return_value=None)  # deleted since completion
        service._image_repo.batch_derivatives = AsyncMock(return_value={})

        response = await service.get_job(uuid4(), job.id)

        assert response.extracted is not None
        assert response.extracted.frames == []
