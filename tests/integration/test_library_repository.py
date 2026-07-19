"""Integration tests for LibraryRepository against a real database."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest_asyncio

from src.core.enums import OutputMediaType
from src.core.library_ref import LibraryAssetSource
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User

OutputFactory = Callable[..., Coroutine[Any, Any, GenerationOutput]]


@pytest_asyncio.fixture
async def library_repo(db_session: AsyncSession) -> LibraryRepository:
    """LibraryRepository bound to the test session."""
    return LibraryRepository(db_session)


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> OutputFactory:
    """Factory fixture: create a GenerationOutput row and flush it."""

    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        content_type: str = "image/jpeg",
        is_thumbnail: bool = False,
        output_index: int = 0,
        size_bytes: int = 1000,
        product_id: str = "vex",
        output_id: object = None,
        created_at: datetime | None = None,
    ) -> GenerationOutput:
        if user is None:
            user = await make_user(email=f"libout-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id=product_id)
        file_ext = "jpeg" if content_type == "image/jpeg" else "mp4"
        oid = output_id or uuid4()
        out = GenerationOutput(
            id=oid,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.{file_ext}",
            content_type=content_type,
            size_bytes=size_bytes,
            format=file_ext,
            output_index=output_index,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            is_thumbnail=is_thumbnail,
            product_id=product_id,
        )
        if created_at is not None:
            out.created_at = created_at
        db_session.add(out)
        await db_session.flush()
        return out

    return _factory


class TestListAssetsMixedSources:
    async def test_returns_both_uploads_and_outputs(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"mix-{uuid4().hex[:8]}@example.com")
        upload = await make_user_image(user=user)
        output = await make_output(user=user)

        rows = await library_repo.list_assets(user.id, "vex", limit=20)
        ids = {r.id for r in rows}
        assert upload.id in ids
        assert output.id in ids

    async def test_excludes_thumbnail_rows(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"thumb-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed")
        full = await make_output(user=user, job=job, is_thumbnail=False, output_index=0)
        thumb = await make_output(user=user, job=job, is_thumbnail=True, output_index=1)

        rows = await library_repo.list_assets(user.id, "vex", limit=20)
        ids = {r.id for r in rows}
        assert full.id in ids
        assert thumb.id not in ids

    async def test_excludes_incomplete_jobs(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"pend-{uuid4().hex[:8]}@example.com")
        pending_job = await make_job(user=user, status="pending")
        output = await make_output(user=user, job=pending_job)

        rows = await library_repo.list_assets(user.id, "vex", limit=20)
        ids = {r.id for r in rows}
        assert output.id not in ids


class TestListAssetsIsolation:
    async def test_product_isolation(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(
            email=f"prodiso-{uuid4().hex[:8]}@example.com", product_id="synthara"
        )
        await make_user_image(user=user, product_id="synthara")

        rows = await library_repo.list_assets(user.id, "vex", limit=20)
        assert len(rows) == 0

    async def test_user_isolation(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user_a = await make_user(email=f"usera-{uuid4().hex[:8]}@example.com")
        user_b = await make_user(email=f"userb-{uuid4().hex[:8]}@example.com")
        await make_user_image(user=user_b)

        rows = await library_repo.list_assets(user_a.id, "vex", limit=20)
        assert len(rows) == 0


class TestListAssetsFilters:
    async def test_source_filter_upload_only(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"srcup-{uuid4().hex[:8]}@example.com")
        upload = await make_user_image(user=user)
        await make_output(user=user)

        rows = await library_repo.list_assets(
            user.id, "vex", limit=20, source=LibraryAssetSource.UPLOAD
        )
        assert {r.id for r in rows} == {upload.id}
        assert all(r.source == LibraryAssetSource.UPLOAD for r in rows)

    async def test_source_filter_output_only(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"srcout-{uuid4().hex[:8]}@example.com")
        await make_user_image(user=user)
        output = await make_output(user=user)

        rows = await library_repo.list_assets(
            user.id, "vex", limit=20, source=LibraryAssetSource.OUTPUT
        )
        assert {r.id for r in rows} == {output.id}

    async def test_media_type_video_filter(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"vidfilt-{uuid4().hex[:8]}@example.com")
        image_out = await make_output(user=user, content_type="image/jpeg")
        video_out = await make_output(user=user, content_type="video/mp4")

        rows = await library_repo.list_assets(
            user.id, "vex", limit=20, media_type=OutputMediaType.VIDEO
        )
        ids = {r.id for r in rows}
        assert video_out.id in ids
        assert image_out.id not in ids

    async def test_media_type_image_filter(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"imgfilt-{uuid4().hex[:8]}@example.com")
        image_out = await make_output(user=user, content_type="image/jpeg")
        video_out = await make_output(user=user, content_type="video/mp4")

        rows = await library_repo.list_assets(
            user.id, "vex", limit=20, media_type=OutputMediaType.IMAGE
        )
        ids = {r.id for r in rows}
        assert image_out.id in ids
        assert video_out.id not in ids

    async def test_model_filter_excludes_uploads(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"modelfilt-{uuid4().hex[:8]}@example.com")
        await make_user_image(user=user)
        job = await make_job(user=user, status="completed", model="grok-imagine-image")
        output = await make_output(user=user, job=job)

        rows = await library_repo.list_assets(user.id, "vex", limit=20, model="grok-imagine-image")
        ids = {r.id for r in rows}
        assert ids == {output.id}

    async def test_favorite_filter_with_metadata(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"fav-{uuid4().hex[:8]}@example.com")
        favorited = await make_user_image(user=user)
        not_favorited = await make_user_image(user=user)

        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, favorited.id, is_favorite=True
        )

        rows = await library_repo.list_assets(user.id, "vex", limit=20, favorite=True)
        ids = {r.id for r in rows}
        assert favorited.id in ids
        assert not_favorited.id not in ids

    async def test_favorite_filter_without_any_metadata_rows(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """No library_asset_metadata row at all (LEFT JOIN, COALESCE to false)."""
        user = await make_user(email=f"nofav-{uuid4().hex[:8]}@example.com")
        await make_user_image(user=user)

        rows = await library_repo.list_assets(user.id, "vex", limit=20, favorite=True)
        assert len(rows) == 0


class TestOldestSort:
    async def test_oldest_sort_orders_ascending(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        from src.core.enums import LibrarySort

        user = await make_user(email=f"oldest-{uuid4().hex[:8]}@example.com")
        now = datetime.now(UTC)
        older = await make_user_image(user=user, expires_at=now + timedelta(days=1))
        older.created_at = now - timedelta(hours=2)
        newer = await make_user_image(user=user, expires_at=now + timedelta(days=1))
        newer.created_at = now - timedelta(hours=1)
        await db_session.flush()

        rows = await library_repo.list_assets(user.id, "vex", limit=20, sort=LibrarySort.OLDEST)
        ids_in_order = [r.id for r in rows]
        assert ids_in_order.index(older.id) < ids_in_order.index(newer.id)


class TestMixedPagination:
    async def test_three_pages_no_dupes_no_gaps_with_equal_timestamps(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
        db_session: AsyncSession,
    ) -> None:
        from src.api.schemas.pagination import decode_library_cursor, encode_library_cursor

        user = await make_user(email=f"pg-{uuid4().hex[:8]}@example.com")
        base = datetime.now(UTC) - timedelta(days=1)

        all_ids: set[Any] = set()
        # Interleave uploads and outputs sharing the SAME created_at timestamp
        # to exercise the source_rank tie-break.
        for i in range(3):
            ts = base + timedelta(seconds=i)
            upload = await make_user_image(user=user)
            upload.created_at = ts
            output = await make_output(user=user, created_at=ts)
            all_ids.add(upload.id)
            all_ids.add(output.id)
        await db_session.flush()

        seen: list[Any] = []
        cursor: str | None = None
        limit = 2
        for _ in range(10):  # safety bound
            decoded = decode_library_cursor(cursor) if cursor else None
            rows = await library_repo.list_assets(user.id, "vex", limit=limit, cursor=decoded)
            has_more = len(rows) > limit
            page = rows[:limit]
            seen.extend(r.id for r in page)
            if not has_more or not page:
                break
            last = page[-1]
            cursor = encode_library_cursor(last.created_at, last.source.value, last.id)

        assert len(seen) == len(set(seen)), "pagination produced duplicate rows"
        assert set(seen) == all_ids, "pagination missed or invented rows"
