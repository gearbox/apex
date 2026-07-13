"""Tests for GrokJobService video poster frames using parent_output_id linking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.core.thumbnails import ThumbnailSpec

pytestmark = pytest.mark.unit

_FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
_SM_SPEC = ThumbnailSpec("sm", 150)
_MD_SPEC = ThumbnailSpec("md", 512)


def _make_thumbnails() -> list[GeneratedThumbnail]:
    return [
        GeneratedThumbnail(
            spec=_SM_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=100, height=56)
        ),
        GeneratedThumbnail(
            spec=_MD_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=400, height=225)
        ),
    ]


async def test_video_poster_frame_uses_parent_output_id_not_sentinel() -> None:
    """Both poster-frame rows must have parent_output_id set and output_index != -1."""
    from src.api.services.grok.job_service import GrokJobService

    storage = MagicMock()
    storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/f.mp4")
    storage._settings = MagicMock()
    storage._settings.bucket_name = "bucket"

    client_ctx = AsyncMock()
    client_mock = AsyncMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    storage._get_client = MagicMock(return_value=client_ctx)

    grok_client = MagicMock()
    svc = GrokJobService(grok_client=grok_client, storage=storage, retention_days=7)

    video_data = b"\x00\x01video"
    http_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.content = video_data
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock

    video_output_id = uuid4()
    sm_thumb_id = uuid4()
    md_thumb_id = uuid4()

    created_ids: list[dict[str, object]] = []

    async def capture_create(**kwargs: object) -> MagicMock:
        created_ids.append(dict(kwargs))
        m = MagicMock()
        m.id = uuid4()
        return m

    output_repo = MagicMock()
    output_repo.create = capture_create

    session = MagicMock()

    def _begin_nested() -> AsyncMock:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=None)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    session.begin_nested = MagicMock(side_effect=_begin_nested)
    jpeg_bytes = b"\xff\xd8\xff\xe0jpeg"

    with (
        patch(
            "src.api.services.grok.job_service.new_id",
            side_effect=[video_output_id, sm_thumb_id, md_thumb_id],
        ),
        patch(
            "src.api.services.grok.job_service.extract_video_thumbnail",
            new=AsyncMock(return_value=jpeg_bytes),
        ),
        patch(
            "src.api.services.grok.job_service.make_image_thumbnails",
            new=AsyncMock(return_value=_make_thumbnails()),
        ),
    ):
        result_mock = MagicMock()
        result_mock.url = "https://cdn.xai.com/video.mp4"

        await svc._store_video_result(
            session=session,
            output_repo=output_repo,  # type: ignore[arg-type]
            user_id=uuid4(),
            job_id=uuid4(),
            result=result_mock,  # type: ignore[arg-type]
            product_id="vex",
        )

    # Three creates: video + sm thumbnail + md thumbnail
    assert len(created_ids) == 3, f"Expected 3 creates, got {len(created_ids)}: {created_ids}"

    video_create = created_ids[0]
    sm_create = created_ids[1]
    md_create = created_ids[2]

    # Video output
    assert video_create["is_thumbnail"] is False
    assert video_create["output_index"] == 0

    # sm thumbnail
    assert sm_create["is_thumbnail"] is True
    assert sm_create["parent_output_id"] == video_output_id
    assert sm_create["output_index"] != -1
    assert sm_create["thumbnail_max_edge"] == 150
    assert sm_create["width"] == 100
    assert sm_create["height"] == 56

    # md thumbnail
    assert md_create["is_thumbnail"] is True
    assert md_create["parent_output_id"] == video_output_id
    assert md_create["output_index"] != -1
    assert md_create["thumbnail_max_edge"] == 512
    assert md_create["width"] == 400
    assert md_create["height"] == 225


async def test_no_poster_frames_when_extract_fails() -> None:
    """If extract_video_thumbnail returns None, only the video row is created."""
    from src.api.services.grok.job_service import GrokJobService

    storage = MagicMock()
    storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/f.mp4")
    storage._settings = MagicMock()
    storage._settings.bucket_name = "bucket"

    client_ctx = AsyncMock()
    client_mock = AsyncMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    storage._get_client = MagicMock(return_value=client_ctx)

    grok_client = MagicMock()
    svc = GrokJobService(grok_client=grok_client, storage=storage, retention_days=7)

    video_data = b"\x00\x01video"
    http_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.content = video_data
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock

    created_ids: list[dict[str, object]] = []

    async def capture_create(**kwargs: object) -> MagicMock:
        created_ids.append(dict(kwargs))
        m = MagicMock()
        m.id = uuid4()
        return m

    output_repo = MagicMock()
    output_repo.create = capture_create

    with (
        patch("src.api.services.grok.job_service.new_id", return_value=uuid4()),
        patch(
            "src.api.services.grok.job_service.extract_video_thumbnail",
            new=AsyncMock(return_value=None),  # extract fails
        ),
    ):
        result_mock = MagicMock()
        result_mock.url = "https://cdn.xai.com/video.mp4"

        await svc._store_video_result(
            session=AsyncMock(),
            output_repo=output_repo,  # type: ignore[arg-type]
            user_id=uuid4(),
            job_id=uuid4(),
            result=result_mock,  # type: ignore[arg-type]
            product_id="vex",
        )

    assert len(created_ids) == 1
    assert created_ids[0]["is_thumbnail"] is False
