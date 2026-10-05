"""Frame lineage on ``UserContentService.upload_image`` against a real database.

A browser captures a frame from a video it already has and uploads it with a
source ref + timestamp. Real PostgreSQL and real image ingest/thumbnailing; only
the R2 boundary is stubbed. Covers I4-I10 (lineage persistence, the single
no-oracle error, boundary timestamps, retention sliding, library lineage
visibility, UPLOAD policy + ledger + thumbnails).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import func, select

from src.api.services.frame_lineage import FrameLineage, FrameLineageReason
from src.api.services.media_ingest import ImageIngestPolicy
from src.api.services.user_content import UserContentFrameLineageError, UserContentService
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.db.models.media_hash import MediaHash
from src.db.models.storage import GenerationOutput, UserImage
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.media_ingest import MediaIngestService
    from src.db.models.user import User

pytestmark = pytest.mark.asyncio

_DURATION_MS = 10_000
_RETENTION_DAYS = 7

type UserFactory = Callable[..., Coroutine[Any, Any, User]]


def _png(width: int = 800, height: int = 600) -> bytes:
    """A frame big enough that both the sm (150) and md (512) thumbnails are produced."""
    image = Image.new("RGB", (width, height))
    for x in range(0, width, 7):
        for y in range(0, height, 11):
            image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _storage() -> MagicMock:
    storage = MagicMock()

    async def upload(**kwargs: Any) -> MagicMock:
        object_id = uuid4()
        ext = kwargs["content_type"].split("/")[-1]
        return MagicMock(id=object_id, storage_key=f"test/{object_id}.{ext}")

    storage.upload = AsyncMock(side_effect=upload)
    return storage


def _service(
    db_session: AsyncSession,
    media_ingestor: MediaIngestService,
    user: User,
    storage: MagicMock | None = None,
) -> UserContentService:
    return UserContentService(
        storage or _storage(),
        db_session,
        product_id=user.product_id,
        retention_days=_RETENTION_DAYS,
        media_ingestor=media_ingestor,
    )


async def _video_upload(
    session: AsyncSession,
    user: User,
    *,
    duration_ms: int | None = _DURATION_MS,
    product_id: str | None = None,
    expires_in: timedelta = timedelta(days=1),
) -> UserImage:
    image_id = uuid4()
    video = UserImage(
        id=image_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{image_id}.mp4",
        original_filename=f"{image_id}.mp4",
        content_type="video/mp4",
        size_bytes=1000,
        format="mp4",
        width=1280,
        height=720,
        duration_ms=duration_ms,
        expires_at=datetime.now(UTC) + expires_in,
        product_id=product_id or user.product_id,
    )
    session.add(video)
    await session.flush()
    return video


async def _upload_thumbnail(session: AsyncSession, parent: UserImage) -> UserImage:
    thumb_id = uuid4()
    thumb = UserImage(
        id=thumb_id,
        user_id=parent.user_id,
        storage_key=f"users/{parent.user_id}/uploads/{thumb_id}.webp",
        original_filename=f"{thumb_id}.webp",
        content_type="image/webp",
        size_bytes=10,
        format="webp",
        width=150,
        height=84,
        is_thumbnail=True,
        parent_image_id=parent.id,
        thumbnail_max_edge=150,
        expires_at=parent.expires_at,
        product_id=parent.product_id,
    )
    session.add(thumb)
    await session.flush()
    return thumb


async def _video_output(
    session: AsyncSession,
    user: User,
    make_job: Callable[..., Coroutine[Any, Any, Any]],
    *,
    duration_ms: int | None = _DURATION_MS,
    expires_in: timedelta = timedelta(days=1),
) -> GenerationOutput:
    job = await make_job(user=user, status="completed", generation_type="t2v")
    output_id = uuid4()
    output = GenerationOutput(
        id=output_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{output_id}.mp4",
        content_type="video/mp4",
        size_bytes=1000,
        format="mp4",
        width=1280,
        height=720,
        duration_ms=duration_ms,
        output_index=0,
        expires_at=datetime.now(UTC) + expires_in,
        is_thumbnail=False,
        product_id=user.product_id,
    )
    session.add(output)
    await session.flush()
    return output


def _lineage(source: LibraryAssetSource, asset_id: UUID, timestamp_ms: int) -> FrameLineage:
    return FrameLineage(
        source=AssetRef(source=source, asset_id=asset_id), timestamp_ms=timestamp_ms
    )


async def _count_uploads(session: AsyncSession, user: User) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(UserImage).where(UserImage.user_id == user.id)
        )
    ).scalar_one()


async def _reload[T](session: AsyncSession, model: type[T], row_id: UUID) -> T:
    """Re-read a row from the database, bypassing the identity map's stale attributes."""
    row = await session.get(model, row_id, populate_existing=True)
    assert row is not None
    return row


async def _frame_row(session: AsyncSession, upload_id: UUID) -> UserImage:
    return await _reload(session, UserImage, upload_id)


# ---------------------------------------------------------------------------
# I5 — lineage persists atomically with the row
# ---------------------------------------------------------------------------


async def test_upload_source_lineage_is_persisted_with_the_row(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-up-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user)

    uploaded = await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(LibraryAssetSource.UPLOAD, source.id, 2500),
    )

    row = await _frame_row(db_session, uploaded.id)
    assert row.source_upload_id == source.id
    assert row.source_output_id is None
    assert row.source_timestamp_ms == 2500
    assert row.is_thumbnail is False
    assert uploaded.media.original.duration_ms is None  # the frame itself is an image


async def test_output_source_lineage_is_persisted_with_the_row(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: Callable[..., Coroutine[Any, Any, Any]],
    media_ingestor: MediaIngestService,
) -> None:
    user = await make_user(email=f"frame-out-{uuid4().hex[:8]}@example.com")
    source = await _video_output(db_session, user, make_job)

    uploaded = await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(LibraryAssetSource.OUTPUT, source.id, 7000),
    )

    row = await _frame_row(db_session, uploaded.id)
    assert row.source_output_id == source.id
    assert row.source_upload_id is None
    assert row.source_timestamp_ms == 7000


# ---------------------------------------------------------------------------
# I4 — no lineage: ordinary upload
# ---------------------------------------------------------------------------


async def test_upload_without_lineage_has_no_source_columns(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-none-{uuid4().hex[:8]}@example.com")

    uploaded = await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id, data=_png(), filename="photo.png", content_type="image/png"
    )

    row = await _frame_row(db_session, uploaded.id)
    assert (row.source_upload_id, row.source_output_id, row.source_timestamp_ms) == (
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# I7 — boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timestamp_ms", [0, _DURATION_MS])
async def test_boundary_timestamps_are_accepted(
    db_session: AsyncSession,
    make_user: UserFactory,
    media_ingestor: MediaIngestService,
    timestamp_ms: int,
) -> None:
    user = await make_user(email=f"frame-edge-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user)

    uploaded = await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(LibraryAssetSource.UPLOAD, source.id, timestamp_ms),
    )

    assert (await _frame_row(db_session, uploaded.id)).source_timestamp_ms == timestamp_ms


# ---------------------------------------------------------------------------
# I6 — every failure is one error, found before any media work
# ---------------------------------------------------------------------------


async def _assert_rejected(
    db_session: AsyncSession,
    media_ingestor: MediaIngestService,
    user: User,
    lineage: FrameLineage,
    expected: FrameLineageReason,
    *,
    content_type: str = "image/png",
) -> None:
    storage = _storage()
    before = await _count_uploads(db_session, user)
    with (
        patch.object(media_ingestor, "prepare_image", wraps=media_ingestor.prepare_image) as prep,
        pytest.raises(UserContentFrameLineageError) as exc,
    ):
        await _service(db_session, media_ingestor, user, storage).upload_image(
            user_id=user.id,
            data=_png(),
            filename="frame.png",
            content_type=content_type,
            lineage=lineage,
        )
    assert exc.value.reason is expected
    # Validated *before* any media work: nothing prepared, stored, or inserted.
    prep.assert_not_called()
    storage.upload.assert_not_called()
    assert await _count_uploads(db_session, user) == before


async def test_nonexistent_source_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-missing-{uuid4().hex[:8]}@example.com")
    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, uuid4(), 0),
        FrameLineageReason.SOURCE_UNAVAILABLE,
    )


async def test_foreign_users_source_is_rejected_like_a_missing_one(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    owner = await make_user(email=f"frame-owner-{uuid4().hex[:8]}@example.com")
    attacker = await make_user(email=f"frame-attacker-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, owner)

    await _assert_rejected(
        db_session,
        media_ingestor,
        attacker,
        _lineage(LibraryAssetSource.UPLOAD, source.id, 0),
        FrameLineageReason.SOURCE_UNAVAILABLE,
    )


async def test_other_product_source_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-prod-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user, product_id="synthara")

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, source.id, 0),
        FrameLineageReason.SOURCE_UNAVAILABLE,
    )


async def test_thumbnail_source_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-thumb-{uuid4().hex[:8]}@example.com")
    video = await _video_upload(db_session, user)
    thumb = await _upload_thumbnail(db_session, video)

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, thumb.id, 0),
        FrameLineageReason.SOURCE_UNAVAILABLE,
    )


async def test_image_source_is_rejected(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_user_image: Callable[..., Coroutine[Any, Any, UserImage]],
    media_ingestor: MediaIngestService,
) -> None:
    user = await make_user(email=f"frame-img-{uuid4().hex[:8]}@example.com")
    image = await make_user_image(user=user)

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, image.id, 0),
        FrameLineageReason.SOURCE_NOT_VIDEO,
    )


async def test_timestamp_past_duration_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-late-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user)

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, source.id, _DURATION_MS + 1),
        FrameLineageReason.TIMESTAMP_OUT_OF_RANGE,
    )


async def test_source_with_unknown_duration_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-nodur-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user, duration_ms=None)

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, source.id, 0),
        FrameLineageReason.SOURCE_DURATION_UNKNOWN,
    )


async def test_video_file_with_lineage_is_rejected(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-vid-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user)

    await _assert_rejected(
        db_session,
        media_ingestor,
        user,
        _lineage(LibraryAssetSource.UPLOAD, source.id, 0),
        FrameLineageReason.VIDEO_FILE,
        content_type="video/mp4",
    )


# ---------------------------------------------------------------------------
# I8 — an upload source's retention slides; an output source is untouched
# ---------------------------------------------------------------------------


async def test_upload_source_and_its_thumbnails_have_their_retention_slid(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-ttl-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user, expires_in=timedelta(hours=2))
    thumb = await _upload_thumbnail(db_session, source)
    other = await _video_upload(db_session, user, expires_in=timedelta(hours=2))
    other_before = other.expires_at
    source_id, thumb_id, other_id = source.id, thumb.id, other.id

    await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(LibraryAssetSource.UPLOAD, source.id, 100),
    )

    floor = datetime.now(UTC) + timedelta(days=_RETENTION_DAYS) - timedelta(minutes=5)
    refreshed_source = await _reload(db_session, UserImage, source_id)
    refreshed_thumb = await _reload(db_session, UserImage, thumb_id)
    refreshed_other = await _reload(db_session, UserImage, other_id)
    assert refreshed_source.expires_at >= floor
    assert refreshed_thumb.expires_at >= floor
    assert refreshed_other.expires_at == other_before  # an unrelated upload is never touched


async def test_output_source_retention_is_not_touched(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: Callable[..., Coroutine[Any, Any, Any]],
    media_ingestor: MediaIngestService,
) -> None:
    user = await make_user(email=f"frame-ttl-out-{uuid4().hex[:8]}@example.com")
    source = await _video_output(db_session, user, make_job, expires_in=timedelta(hours=2))
    before = source.expires_at
    source_id = source.id

    await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(LibraryAssetSource.OUTPUT, source.id, 100),
    )

    refreshed = await _reload(db_session, GenerationOutput, source_id)
    assert refreshed.expires_at == before


async def test_rejected_lineage_does_not_slide_retention(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-ttl-bad-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user, expires_in=timedelta(hours=2))
    before = source.expires_at
    source_id = source.id

    with pytest.raises(UserContentFrameLineageError):
        await _service(db_session, media_ingestor, user).upload_image(
            user_id=user.id,
            data=_png(),
            filename="frame.png",
            content_type="image/png",
            lineage=_lineage(LibraryAssetSource.UPLOAD, source.id, _DURATION_MS + 1),
        )

    refreshed = await _reload(db_session, UserImage, source_id)
    assert refreshed.expires_at == before


# ---------------------------------------------------------------------------
# I9 — the frame shows up in the source's library lineage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_kind", [LibraryAssetSource.UPLOAD, LibraryAssetSource.OUTPUT])
async def test_frame_appears_in_source_frame_descendants(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: Callable[..., Coroutine[Any, Any, Any]],
    media_ingestor: MediaIngestService,
    source_kind: LibraryAssetSource,
) -> None:
    user = await make_user(email=f"frame-lin-{uuid4().hex[:8]}@example.com")
    source: UserImage | GenerationOutput = (
        await _video_upload(db_session, user)
        if source_kind is LibraryAssetSource.UPLOAD
        else await _video_output(db_session, user, make_job)
    )

    uploaded = await _service(db_session, media_ingestor, user).upload_image(
        user_id=user.id,
        data=_png(),
        filename="frame.png",
        content_type="image/png",
        lineage=_lineage(source_kind, source.id, 4000),
    )

    descendants = await LibraryRepository(db_session).list_frame_descendants(
        source_kind, source.id, user_id=user.id, product_id=user.product_id, limit=10
    )
    assert [d.id for d in descendants] == [uploaded.id]
    assert descendants[0].source_timestamp_ms == 4000


# ---------------------------------------------------------------------------
# I10 — client bytes: UPLOAD policy, ledger row, sm/md thumbnails
# ---------------------------------------------------------------------------


async def test_lineage_frame_uses_upload_policy_ledger_and_thumbnails(
    db_session: AsyncSession, make_user: UserFactory, media_ingestor: MediaIngestService
) -> None:
    user = await make_user(email=f"frame-pol-{uuid4().hex[:8]}@example.com")
    source = await _video_upload(db_session, user)

    with patch.object(
        media_ingestor, "prepare_image", wraps=media_ingestor.prepare_image
    ) as prepare:
        uploaded = await _service(db_session, media_ingestor, user).upload_image(
            user_id=user.id,
            data=_png(),
            filename="frame.png",
            content_type="image/png",
            lineage=_lineage(LibraryAssetSource.UPLOAD, source.id, 1000),
        )

    # Untrusted client bytes: never the PROVIDER policy the old server worker used.
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["policy"] is ImageIngestPolicy.UPLOAD

    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == uploaded.id)))
        .scalars()
        .all()
    )
    assert len(ledger) == 1
    assert ledger[0].user_id == user.id
    assert ledger[0].source_kind == "upload"

    derivatives = (
        (
            await db_session.execute(
                select(UserImage).where(UserImage.parent_image_id == uploaded.id)
            )
        )
        .scalars()
        .all()
    )
    assert sorted(d.thumbnail_max_edge or 0 for d in derivatives) == [150, 512]
    # Derivatives are plain thumbnails: lineage lives only on the frame itself.
    assert all(
        (d.source_upload_id, d.source_output_id, d.source_timestamp_ms) == (None, None, None)
        for d in derivatives
    )
