"""Tests for GrokJobService video poster frames using parent_output_id linking."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.api.services.media_ingest import PreparedVideo
from src.core.enums import MediaFormat
from src.core.media_hash import HashSample, HashSet, PdqHash
from src.core.thumbnails import ThumbnailSpec

if TYPE_CHECKING:
    from pathlib import Path

    from src.api.services.media_ingest import MediaIngestService

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


def _prepared_video() -> PreparedVideo:
    return PreparedVideo(
        data=b"prepared-video",
        format=MediaFormat.MP4,
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
    )


async def test_video_poster_frame_uses_parent_output_id_not_sentinel() -> None:
    """Both poster-frame rows must have parent_output_id set and output_index != -1."""
    from src.api.services.grok.job_service import GrokJobService

    storage = MagicMock()
    storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/f.mp4")
    storage.put_raw = AsyncMock()

    grok_client = MagicMock()
    media_ingestor = MagicMock()
    media_ingestor.prepare_video = AsyncMock(return_value=_prepared_video())
    svc = GrokJobService(
        grok_client=grok_client,
        storage=storage,
        retention_days=7,
        media_ingestor=media_ingestor,
    )

    video_data = b"\x00\x01video-private-metadata-canary"
    http_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.content = video_data
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock

    video_output_id = uuid4()
    sm_thumb_id = uuid4()
    md_thumb_id = uuid4()

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

        materialized = await svc._materialize_video_result(
            user_id=uuid4(),
            job_id=uuid4(),
            result=result_mock,  # type: ignore[arg-type]
            product_id="vex",
        )

    # One original plus sm and md derivatives, later persisted atomically.
    assert len(materialized.outputs) == 3
    assert storage.put_raw.await_args_list[0].args[1] == b"prepared-video"
    video_create, sm_create, md_create = materialized.outputs

    # Video output
    assert video_create.is_thumbnail is False
    assert video_create.output_index == 0
    assert video_create.hash_set == _prepared_video().hash_set

    # sm thumbnail
    assert sm_create.is_thumbnail is True
    assert sm_create.parent_output_id == video_output_id
    assert sm_create.output_index != -1
    assert sm_create.thumbnail_max_edge == 150
    assert sm_create.width == 100
    assert sm_create.height == 56

    # md thumbnail
    assert md_create.is_thumbnail is True
    assert md_create.parent_output_id == video_output_id
    assert md_create.output_index != -1
    assert md_create.thumbnail_max_edge == 512
    assert md_create.width == 400
    assert md_create.height == 225


async def test_grok_video_storage_receives_remuxed_bytes_without_descriptive_tags(
    tmp_path: Path,
    media_ingestor: MediaIngestService,
) -> None:
    from src.api.services.grok.job_service import GrokJobService

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required for video ingest tests in CI")
        pytest.skip("ffmpeg is unavailable locally")
    source = tmp_path / "private.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-metadata",
            "title=private-metadata-canary",
            "-metadata",
            "comment=private-metadata-canary",
            "-metadata",
            "location=+12.34+56.78/",
            "-metadata:s:v:0",
            "handler_name=private-metadata-canary",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    storage = MagicMock()
    storage.build_storage_key.return_value = "test/grok/private.mp4"
    storage.put_raw = AsyncMock()
    service = GrokJobService(MagicMock(), storage, media_ingestor=media_ingestor)
    response = MagicMock(content=source.read_bytes())
    response.raise_for_status = MagicMock()
    service._http_client = MagicMock(get=AsyncMock(return_value=response))
    with patch(
        "src.api.services.grok.job_service.extract_video_thumbnail",
        new=AsyncMock(return_value=None),
    ):
        materialized = await service._materialize_video_result(
            user_id=uuid4(),
            job_id=uuid4(),
            result=MagicMock(url="https://provider.invalid/private.mp4"),
            product_id="vex",
        )
    stored = storage.put_raw.await_args.args[1]
    assert b"private-metadata-canary" not in stored
    assert len(materialized.outputs) == 1
    hash_set = materialized.outputs[0].hash_set
    assert hash_set is not None
    assert len(hash_set.samples) == 2


async def test_no_poster_frames_when_extract_fails() -> None:
    """If extract_video_thumbnail returns None, only the video row is created."""
    from src.api.services.grok.job_service import GrokJobService

    storage = MagicMock()
    storage.build_storage_key = MagicMock(return_value="users/u/outputs/j/f.mp4")
    storage.put_raw = AsyncMock()

    grok_client = MagicMock()
    media_ingestor = MagicMock()
    media_ingestor.prepare_video = AsyncMock(return_value=_prepared_video())
    svc = GrokJobService(
        grok_client=grok_client,
        storage=storage,
        retention_days=7,
        media_ingestor=media_ingestor,
    )

    video_data = b"\x00\x01video"
    http_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.content = video_data
    http_mock.get = AsyncMock(return_value=response_mock)
    svc._http_client = http_mock

    with (
        patch("src.api.services.grok.job_service.new_id", return_value=uuid4()),
        patch(
            "src.api.services.grok.job_service.extract_video_thumbnail",
            new=AsyncMock(return_value=None),  # extract fails
        ),
    ):
        result_mock = MagicMock()
        result_mock.url = "https://cdn.xai.com/video.mp4"

        materialized = await svc._materialize_video_result(
            user_id=uuid4(),
            job_id=uuid4(),
            result=result_mock,  # type: ignore[arg-type]
            product_id="vex",
        )

    assert len(materialized.outputs) == 1
    assert materialized.outputs[0].is_thumbnail is False
