"""Unit tests for FramesController route handlers.

Tests call ``Handler.fn(self, ...)`` directly to exercise handler logic
without spinning up Litestar's HTTP layer — mirrors tests/unit/test_storage_routes.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from src.api.routes.frames import FramesController
from src.api.schemas.frames import (
    FrameExtractRequest,
    FrameJobResponse,
    FrameJobSource,
    FramePreviewRequest,
)
from src.api.services.frames.service import (
    FrameJobNotFoundError,
    FrameSourceNotFoundError,
    FrameSourceNotVideoError,
)
from src.core.enums import FrameExtractionKind, FrameExtractionStatus

pytestmark = pytest.mark.unit


def _make_job(**overrides: object) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.status = FrameExtractionStatus.QUEUED.value
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


def _make_job_response(**overrides: object) -> FrameJobResponse:
    defaults: dict[str, object] = {
        "job_id": uuid4(),
        "kind": FrameExtractionKind.PREVIEW.value,
        "status": FrameExtractionStatus.QUEUED.value,
        "created_at": datetime.now(UTC),
        "source": FrameJobSource(type="output", id=uuid4()),
    } | overrides
    return FrameJobResponse(**defaults)  # type: ignore[arg-type]


class TestCreatePreviewHandler:
    async def test_rejects_neither_source(self) -> None:
        service = AsyncMock()
        request = FramePreviewRequest(source_output_id=None, source_upload_id=None)

        response = await FramesController.create_preview.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == HTTP_400_BAD_REQUEST
        service.create_preview_job.assert_not_awaited()

    async def test_rejects_both_sources(self) -> None:
        service = AsyncMock()
        request = FramePreviewRequest(source_output_id=uuid4(), source_upload_id=uuid4())

        response = await FramesController.create_preview.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == HTTP_400_BAD_REQUEST
        service.create_preview_job.assert_not_awaited()

    async def test_returns_202_with_job_id(self) -> None:
        job = _make_job()
        service = AsyncMock()
        service.create_preview_job = AsyncMock(return_value=job)
        request = FramePreviewRequest(source_output_id=uuid4(), source_upload_id=None)

        response = await FramesController.create_preview.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == 202
        assert response.content.job_id == job.id
        assert response.content.status == job.status

    async def test_source_not_found_returns_404(self) -> None:
        service = AsyncMock()
        service.create_preview_job = AsyncMock(
            side_effect=FrameSourceNotFoundError("Output not found")
        )
        request = FramePreviewRequest(source_output_id=uuid4(), source_upload_id=None)

        response = await FramesController.create_preview.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_source_not_video_returns_400(self) -> None:
        service = AsyncMock()
        service.create_preview_job = AsyncMock(side_effect=FrameSourceNotVideoError("not a video"))
        request = FramePreviewRequest(source_output_id=uuid4(), source_upload_id=None)

        response = await FramesController.create_preview.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "not_a_video"


class TestCreateExtractHandler:
    async def test_validates_timestamp_count_cap(self) -> None:
        # msgspec enforces max_length=50 on timestamps_ms at decode time.
        payload = msgspec.json.encode(
            {
                "source_output_id": str(uuid4()),
                "source_upload_id": None,
                "timestamps_ms": list(range(51)),
            }
        )
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(payload, type=FrameExtractRequest)

    async def test_returns_202_with_job_id(self) -> None:
        job = _make_job()
        service = AsyncMock()
        service.create_extract_job = AsyncMock(return_value=job)
        request = FrameExtractRequest(
            source_output_id=None, source_upload_id=uuid4(), timestamps_ms=[0, 1000]
        )

        response = await FramesController.create_extract.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            data=request,
        )

        assert response.status_code == 202
        assert response.content.job_id == job.id


class TestGetJobHandler:
    async def test_foreign_user_404(self) -> None:
        service = AsyncMock()
        service.get_job = AsyncMock(side_effect=FrameJobNotFoundError("not found"))

        response = await FramesController.get_job.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            job_id=uuid4(),
        )

        assert response.status_code == HTTP_404_NOT_FOUND

    async def test_extract_result_contains_media_objects(self) -> None:
        from src.api.schemas.frames import ExtractedFrame, FrameExtractResult
        from src.api.schemas.media import MediaObject, MediaOriginal
        from src.core.enums import OutputMediaType

        media = MediaObject(
            media_type=OutputMediaType.IMAGE,
            original=MediaOriginal(
                url="/v1/content/uploads/abc",
                width=100,
                height=100,
                content_type="image/png",
                size_bytes=10,
            ),
            variants=[],
        )
        job_response = _make_job_response(
            kind=FrameExtractionKind.EXTRACT.value,
            status=FrameExtractionStatus.COMPLETED.value,
            extracted=FrameExtractResult(
                frames=[ExtractedFrame(timestamp_ms=1000, upload_id=uuid4(), media=media)]
            ),
        )
        service = AsyncMock()
        service.get_job = AsyncMock(return_value=job_response)

        response = await FramesController.get_job.fn(  # type: ignore[attr-defined]
            MagicMock(),
            current_user_id=uuid4(),
            frame_extraction_service=service,
            job_id=job_response.job_id,
        )

        assert response.status_code == 200
        assert response.content.extracted is not None
        assert response.content.extracted.frames[0].media.original.content_type == "image/png"
