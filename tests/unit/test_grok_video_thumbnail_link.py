"""Tests for GrokJobService video poster frame using parent_output_id (no -1 sentinel)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit


def _make_output_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    return repo


async def test_video_poster_frame_uses_parent_output_id_not_sentinel() -> None:
    """The video thumbnail row must have parent_output_id set and output_index != -1."""
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
    response_mock.raise_for_status = MagicMock()  # sync, as on a real httpx.Response
    response_mock.content = video_data
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock

    output_repo = _make_output_repo()
    session = AsyncMock()
    jpeg_bytes = b"\xff\xd8\xff\xe0jpeg"

    thumb_dims = MagicMock()
    thumb_dims.width = 320
    thumb_dims.height = 180

    video_output_id = uuid4()

    created_ids: list[object] = []

    async def capture_create(**kwargs: object) -> MagicMock:
        created_ids.append(kwargs)
        m = MagicMock()
        m.id = uuid4()
        return m

    output_repo.create = capture_create

    # Patch new_id so we get a predictable video output id for the first call
    with (
        patch("src.api.services.grok.job_service.new_id", return_value=video_output_id),
        patch(
            "src.api.services.grok.job_service.extract_video_thumbnail",
            new=AsyncMock(return_value=jpeg_bytes),
        ),
        patch(
            "src.api.services.grok.job_service.read_dimensions",
            new=AsyncMock(return_value=thumb_dims),
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

    # Two creates: video + thumbnail
    assert len(created_ids) == 2, f"Expected 2 creates, got {len(created_ids)}"
    video_create = created_ids[0]
    thumb_create = created_ids[1]

    # Video output
    assert video_create["is_thumbnail"] is False  # type: ignore[index]
    assert video_create["output_index"] == 0  # type: ignore[index]

    # Thumbnail: must link to video via parent_output_id, no -1 sentinel
    assert thumb_create["is_thumbnail"] is True  # type: ignore[index]
    assert thumb_create["parent_output_id"] == video_output_id  # type: ignore[index]
    assert thumb_create["output_index"] != -1  # type: ignore[index]
    assert thumb_create["width"] == 320  # type: ignore[index]
    assert thumb_create["height"] == 180  # type: ignore[index]
