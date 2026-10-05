"""``MediaOriginal.duration_ms`` on every surface that serializes a media original (I3).

Every ``_build_media_object`` call site in ``LibraryService`` plus the upload response
must carry the row's ``duration_ms``: a missed site silently yields ``null`` for that
surface only. Real PostgreSQL; the R2 boundary is stubbed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from src.api.services.generation.source_media import ResolvedSourceMedia
from src.api.services.library import LibraryService
from src.api.services.user_content import UserContentService
from src.core.enums import MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.repositories.generation_job_source import GenerationJobSourceRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.media import MediaObject
    from src.api.services.media_ingest import MediaIngestService
    from src.db.models.user import User

pytestmark = pytest.mark.asyncio

_VIDEO_UPLOAD_MS = 4321
_VIDEO_OUTPUT_MS = 7777
_FFMPEG = shutil.which("ffmpeg")


def _upload_ref(row: UserImage) -> str:
    return format_asset_ref(LibraryAssetSource.UPLOAD, row.id)


def _output_ref(row: GenerationOutput) -> str:
    return format_asset_ref(LibraryAssetSource.OUTPUT, row.id)


async def _upload(
    session: AsyncSession, user: User, *, video: bool, source_upload_id: UUID | None = None
) -> UserImage:
    image_id = uuid4()
    ext = "mp4" if video else "png"
    row = UserImage(
        id=image_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{image_id}.{ext}",
        original_filename=f"{image_id}.{ext}",
        content_type="video/mp4" if video else "image/png",
        size_bytes=100,
        format=ext,
        width=64,
        height=48,
        duration_ms=_VIDEO_UPLOAD_MS if video else None,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=user.product_id,
        source_upload_id=source_upload_id,
        source_timestamp_ms=1000 if source_upload_id else None,
    )
    session.add(row)
    await session.flush()
    return row


async def _job(
    session: AsyncSession,
    user: User,
    *,
    input_image_id: UUID | None = None,
    source_output_id: UUID | None = None,
) -> GenerationJob:
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        product_id=user.product_id,
        name="Duration Job",
        prompt="p",
        status="completed",
        generation_type="i2v",
        provider="grok",
        input_image_id=input_image_id,
        source_output_id=source_output_id,
    )
    session.add(job)
    await session.flush()
    return job


async def _video_output(session: AsyncSession, user: User, job: GenerationJob) -> GenerationOutput:
    output_id = uuid4()
    row = GenerationOutput(
        id=output_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{output_id}.mp4",
        content_type="video/mp4",
        size_bytes=1000,
        format="mp4",
        width=64,
        height=48,
        duration_ms=_VIDEO_OUTPUT_MS,
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        is_thumbnail=False,
        product_id=user.product_id,
    )
    session.add(row)
    await session.flush()
    return row


def _resolved(ref: str, source: LibraryAssetSource, asset_id: UUID, position: int) -> Any:
    return ResolvedSourceMedia(
        position=position,
        ref=AssetRef(source=source, asset_id=asset_id),
        asset_ref=ref,
        media_kind=MediaKind.VIDEO,
        content_type="video/mp4",
        storage_key="k",
        size_bytes=1,
        duration_ms=None,
        job_id=None,
    )


class _World:
    """One user's library: video/image uploads, a video output, a frame, and two jobs."""

    video: UserImage
    image: UserImage
    frame: UserImage
    output: GenerationOutput
    job: GenerationJob  # fallback source path: input_image_id only
    sourced_job: GenerationJob  # explicit generation_job_sources rows (upload + output)
    sourced_output: GenerationOutput
    user: User

    def expected(self) -> dict[str, int | None]:
        return {
            _upload_ref(self.video): _VIDEO_UPLOAD_MS,
            _upload_ref(self.image): None,
            _upload_ref(self.frame): None,
            _output_ref(self.output): _VIDEO_OUTPUT_MS,
            _output_ref(self.sourced_output): _VIDEO_OUTPUT_MS,
        }


@pytest.fixture
async def world(
    db_session: AsyncSession, make_user: Callable[..., Coroutine[Any, Any, User]]
) -> _World:
    w = _World()
    w.user = await make_user(email=f"duration-{uuid4().hex[:8]}@example.com")
    w.video = await _upload(db_session, w.user, video=True)
    w.image = await _upload(db_session, w.user, video=False)
    w.frame = await _upload(db_session, w.user, video=False, source_upload_id=w.video.id)
    w.job = await _job(db_session, w.user, input_image_id=w.video.id)
    w.output = await _video_output(db_session, w.user, w.job)
    w.sourced_job = await _job(db_session, w.user)
    w.sourced_output = await _video_output(db_session, w.user, w.sourced_job)
    await GenerationJobSourceRepository(db_session).create_many(
        job_id=w.sourced_job.id,
        product_id=w.user.product_id,
        sources=[
            _resolved(_upload_ref(w.video), LibraryAssetSource.UPLOAD, w.video.id, 0),
            _resolved(_output_ref(w.output), LibraryAssetSource.OUTPUT, w.output.id, 1),
        ],
    )
    return w


def _durations(media_by_ref: dict[str, MediaObject]) -> dict[str, int | None]:
    return {ref: media.original.duration_ms for ref, media in media_by_ref.items()}


async def test_library_list_carries_duration(db_session: AsyncSession, world: _World) -> None:
    page = await LibraryService(session=db_session).list_assets(
        world.user.id, "vex", VEX_CONFIG, session=db_session, limit=50
    )

    seen = {item.asset_ref: item.media.original.duration_ms for item in page.items}
    expected = world.expected()
    assert set(expected) <= set(seen)
    assert {ref: seen[ref] for ref in expected} == expected


@pytest.mark.parametrize("which", ["video", "image", "output"])
async def test_library_detail_carries_duration(
    db_session: AsyncSession, world: _World, which: str
) -> None:
    ref = {
        "video": _upload_ref(world.video),
        "image": _upload_ref(world.image),
        "output": _output_ref(world.output),
    }[which]

    detail = await LibraryService(session=db_session).get_asset_detail(
        ref, world.user.id, "vex", VEX_CONFIG, session=db_session
    )

    assert detail is not None
    assert detail.media.original.duration_ms == world.expected()[ref]


async def test_group_detail_outputs_and_fallback_source_carry_duration(
    db_session: AsyncSession, world: _World
) -> None:
    detail = await LibraryService(session=db_session).get_group_detail(
        world.job.id, world.user.id, "vex", session=db_session
    )

    assert detail is not None
    assert [o.media.original.duration_ms for o in detail.outputs] == [_VIDEO_OUTPUT_MS]
    # No generation_job_sources rows: the legacy input_image_id fallback builds the source.
    (source,) = detail.source_media
    assert source.media is not None
    assert source.media.original.duration_ms == _VIDEO_UPLOAD_MS


async def test_group_detail_legacy_output_source_fallback_carries_duration(
    db_session: AsyncSession, world: _World
) -> None:
    """No generation_job_sources rows, ``source_output_id`` only (pre-backfill job)."""
    legacy = await _job(db_session, world.user, source_output_id=world.output.id)

    detail = await LibraryService(session=db_session).get_group_detail(
        legacy.id, world.user.id, "vex", session=db_session
    )

    assert detail is not None
    (source,) = detail.source_media
    assert source.asset_ref == _output_ref(world.output)
    assert source.media is not None
    assert source.media.original.duration_ms == _VIDEO_OUTPUT_MS


async def test_group_detail_batched_upload_and_output_sources_carry_duration(
    db_session: AsyncSession, world: _World
) -> None:
    detail = await LibraryService(session=db_session).get_group_detail(
        world.sourced_job.id, world.user.id, "vex", session=db_session
    )

    assert detail is not None
    by_ref = {s.asset_ref: s for s in detail.source_media}
    upload_source = by_ref[_upload_ref(world.video)]
    output_source = by_ref[_output_ref(world.output)]
    assert upload_source.media is not None
    assert output_source.media is not None
    assert upload_source.media.original.duration_ms == _VIDEO_UPLOAD_MS
    assert output_source.media.original.duration_ms == _VIDEO_OUTPUT_MS


async def test_lineage_graph_nodes_carry_duration(db_session: AsyncSession, world: _World) -> None:
    service = LibraryService(session=db_session)
    expected = world.expected()
    nodes: dict[str, MediaObject] = {}

    for focus in (_upload_ref(world.video), _output_ref(world.output)):
        graph = await service.get_lineage_graph(focus, world.user.id, "vex", session=db_session)
        assert graph is not None
        nodes[graph.focus.asset_ref] = graph.focus.media
        for edge in (*graph.ancestors, *graph.descendants):
            nodes[edge.node.asset_ref] = edge.node.media

    # The walk must actually have reached video and image nodes, not just the focus.
    assert _upload_ref(world.video) in nodes
    assert _output_ref(world.output) in nodes
    assert len(nodes) >= 3
    assert _durations(nodes) == {ref: expected[ref] for ref in nodes}


def _require_ffmpeg() -> str:
    if _FFMPEG is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required for media ingest tests in CI")
        pytest.skip("ffmpeg is unavailable locally")
    return _FFMPEG


async def test_video_upload_response_and_row_carry_ingest_duration(
    db_session: AsyncSession,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    media_ingestor: MediaIngestService,
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            _require_ffmpeg(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    user = await make_user(email=f"duration-up-{uuid4().hex[:8]}@example.com")
    object_id = uuid4()
    storage = MagicMock()
    storage.upload = AsyncMock(
        return_value=MagicMock(id=object_id, storage_key=f"test/{object_id}.mp4")
    )
    service = UserContentService(
        storage, db_session, product_id=user.product_id, media_ingestor=media_ingestor
    )

    uploaded = await service.upload_image(
        user_id=user.id,
        data=await asyncio.to_thread(source.read_bytes),
        filename="clip.mp4",
        content_type="video/mp4",
    )

    row = await db_session.get(UserImage, object_id, populate_existing=True)
    assert row is not None
    assert row.duration_ms is not None
    assert row.duration_ms > 0
    assert uploaded.media.original.duration_ms == row.duration_ms
