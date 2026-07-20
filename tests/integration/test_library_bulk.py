"""Integration tests for LibraryService.bulk_apply against a real database.

Covers: all-or-nothing validation (P5 — one bad ref fails the whole
request with zero side effects, verified from a second NullPool session),
and happy paths for bulk favorite/project/delete. Also verifies the
retention sweeper's metadata purge is unaffected by the new project_id
column (P2/D14 interplay).
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.library import BulkDelete, BulkSetFavorite, BulkSetProject
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import LibraryBulkValidationError, LibraryService
from src.api.services.library_project import LibraryProjectService
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.db.models.user import User

OutputFactory = Callable[..., Coroutine[Any, Any, GenerationOutput]]


async def _create_user_committed(session: AsyncSession) -> User:
    """Seed a User via a real committing session bound to the shared engine.

    Needed whenever the assertion crosses connections — the SAVEPOINT-scoped
    ``db_session``/``make_user`` fixtures never commit their outer
    transaction, so writes made through them are invisible outside that one
    connection (mirrors the pattern in test_library_metadata.py).
    """
    from src.db.models.user import User as UserModel

    user = UserModel(
        id=uuid4(),
        email=f"bulkiso-{uuid4().hex[:8]}@example.com",
        password_hash="hashed",
        product_id="vex",
    )
    session.add(user)
    await session.flush()
    return user


async def _create_upload_committed(session: AsyncSession, *, user: User) -> Any:
    from src.db.models.storage import UserImage

    img_id = uuid4()
    image = UserImage(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="photo.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=user.product_id,
    )
    session.add(image)
    await session.flush()
    return image


@pytest_asyncio.fixture
async def library_service(db_session: AsyncSession) -> LibraryService:
    return LibraryService(session=db_session)


@pytest_asyncio.fixture
async def project_service(db_session: AsyncSession) -> LibraryProjectService:
    return LibraryProjectService(session=db_session)


@pytest.fixture
def content_proxy() -> ContentProxyService:
    mock_storage = MagicMock()
    mock_storage.delete = AsyncMock()
    settings = MagicMock()
    settings.content_url_ttl = 3600
    return ContentProxyService(storage=mock_storage, settings=settings)


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> OutputFactory:
    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        product_id: str = "vex",
        output_id: object = None,
    ) -> GenerationOutput:
        if user is None:
            user = await make_user(email=f"bulkout-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id=product_id)
        oid = output_id or uuid4()
        out = GenerationOutput(
            id=oid,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.jpeg",
            content_type="image/jpeg",
            size_bytes=1000,
            format="jpeg",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            is_thumbnail=False,
            product_id=product_id,
        )
        db_session.add(out)
        await db_session.flush()
        return out

    return _factory


class TestBulkAllOrNothing:
    async def test_one_malformed_ref_fails_whole_batch(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkbad-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        good_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkSetFavorite(asset_refs=[good_ref, "not-a-valid-ref"], value=True)
        with pytest.raises(LibraryBulkValidationError) as exc_info:
            await library_service.bulk_apply(
                op, user.id, "vex", session=db_session, content_proxy=content_proxy
            )
        assert "not-a-valid-ref" in exc_info.value.invalid_refs

    async def test_one_unowned_ref_fails_whole_batch_zero_side_effects(
        self,
        content_proxy: ContentProxyService,
        db_engine: AsyncEngine,
    ) -> None:
        """A committed second session must see NO favorite applied to the good ref."""
        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with session_factory() as setup_session:
            user = await _create_user_committed(setup_session)
            other_user = await _create_user_committed(setup_session)
            image = await _create_upload_committed(setup_session, user=user)
            foreign_image = await _create_upload_committed(setup_session, user=other_user)
            await setup_session.commit()

        good_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        foreign_ref = format_asset_ref(LibraryAssetSource.UPLOAD, foreign_image.id)

        async with session_factory() as work_session:
            service = LibraryService(session=work_session)
            op = BulkSetFavorite(asset_refs=[good_ref, foreign_ref], value=True)
            with pytest.raises(LibraryBulkValidationError) as exc_info:
                await service.bulk_apply(
                    op, user.id, "vex", session=work_session, content_proxy=content_proxy
                )
            assert foreign_ref in exc_info.value.invalid_refs
            await work_session.rollback()

        async with session_factory() as verify_session:
            repo = LibraryRepository(verify_session)
            metadata = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
            assert metadata is None, "the good ref must not have been favorited"


class TestBulkHappyPaths:
    async def test_bulk_set_favorite(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkfav-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        output = await make_output(user=user)
        refs = [
            format_asset_ref(LibraryAssetSource.UPLOAD, image.id),
            format_asset_ref(LibraryAssetSource.OUTPUT, output.id),
        ]

        op = BulkSetFavorite(asset_refs=refs, value=True)
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result.op == "set_favorite"
        assert result.succeeded == 2
        assert result.failed == 0

        repo = LibraryRepository(db_session)
        upload_meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
        output_meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.OUTPUT, output.id)
        assert upload_meta is not None and upload_meta.is_favorite is True
        assert output_meta is not None and output_meta.is_favorite is True

    async def test_bulk_set_project_assign_and_unassign(
        self,
        library_service: LibraryService,
        project_service: LibraryProjectService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkproj-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        project = await project_service.create(
            user.id, "vex", "Bulk Project", None, session=db_session
        )
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        assign_op = BulkSetProject(asset_refs=[ref], project_id=project.id)
        await library_service.bulk_apply(
            assign_op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        repo = LibraryRepository(db_session)
        meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
        assert meta is not None and meta.project_id == project.id

        unassign_op = BulkSetProject(asset_refs=[ref], project_id=None)
        await library_service.bulk_apply(
            unassign_op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        # bulk_set_project runs a Core-level statement, bypassing ORM change
        # tracking — the already-loaded `meta` instance needs an explicit
        # refresh (not expire_all(), which would also expire `user` and
        # trip MissingGreenlet on next attribute access outside a session
        # I/O call) before re-reading its project_id.
        await db_session.refresh(meta)
        assert meta.project_id is None

    async def test_bulk_set_project_nonexistent_project_raises_not_found(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        from src.api.services.library import LibraryProjectNotFoundError

        user = await make_user(email=f"bulkprojnf-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkSetProject(asset_refs=[ref], project_id=uuid4())
        with pytest.raises(LibraryProjectNotFoundError):
            await library_service.bulk_apply(
                op, user.id, "vex", session=db_session, content_proxy=content_proxy
            )

    async def test_bulk_delete(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkdel-{uuid4().hex[:8]}@example.com")
        image_a = await make_user_image(user=user)
        image_b = await make_user_image(user=user)
        refs = [
            format_asset_ref(LibraryAssetSource.UPLOAD, image_a.id),
            format_asset_ref(LibraryAssetSource.UPLOAD, image_b.id),
        ]

        op = BulkDelete(asset_refs=refs)
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result.op == "delete"
        assert result.succeeded == 2

        from src.db.repositories.user_image import UserImageRepository

        image_repo = UserImageRepository(db_session)
        assert await image_repo.get(image_a.id) is None
        assert await image_repo.get(image_b.id) is None


class TestBulkDuplicateRefDedup:
    """H1 — duplicate refs must be deduped (first occurrence wins) before
    reaching the multi-row ON CONFLICT statement, never a
    CardinalityViolationError / phantom failed entry."""

    async def test_set_favorite_duplicate_refs_collapse_to_one(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkdupfav-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkSetFavorite(asset_refs=[ref, ref], value=True)
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert len(result.results) == 1
        assert result.succeeded == 1
        assert result.failed == 0
        assert result.results[0].success is True

        repo = LibraryRepository(db_session)
        meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
        assert meta is not None and meta.is_favorite is True

    async def test_set_project_duplicate_refs_mixed_with_unique(
        self,
        library_service: LibraryService,
        project_service: LibraryProjectService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkdupproj-{uuid4().hex[:8]}@example.com")
        image_a = await make_user_image(user=user)
        image_b = await make_user_image(user=user)
        project = await project_service.create(
            user.id, "vex", "Dedup Project", None, session=db_session
        )
        ref_a = format_asset_ref(LibraryAssetSource.UPLOAD, image_a.id)
        ref_b = format_asset_ref(LibraryAssetSource.UPLOAD, image_b.id)

        op = BulkSetProject(asset_refs=[ref_a, ref_b, ref_a], project_id=project.id)
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert len(result.results) == 2
        assert result.succeeded == 2
        assert {r.asset_ref for r in result.results} == {ref_a, ref_b}

    async def test_delete_duplicate_refs_collapse_to_one(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkdupdel-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkDelete(asset_refs=[ref, ref])
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert len(result.results) == 1
        assert result.results[0].success is True
        assert result.succeeded == 1
        assert result.failed == 0

    async def test_mixed_case_uuid_duplicate_collapses(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"bulkdupcase-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        lower_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        upper_ref = f"upload:{str(image.id).upper()}"

        op = BulkSetFavorite(asset_refs=[lower_ref, upper_ref], value=True)
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert len(result.results) == 1
        assert result.succeeded == 1


class TestMetadataPurgeUnaffectedByProjectColumns:
    async def test_retention_sweep_purges_metadata_with_project_assigned(
        self,
        db_engine: AsyncEngine,
    ) -> None:
        from src.api.services.content_retention import ContentRetentionService
        from src.db.models.user import User as UserModel

        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with session_factory() as session:
            user = UserModel(
                id=uuid4(),
                email=f"purgeproj-{uuid4().hex[:8]}@example.com",
                password_hash="hashed",
                product_id="vex",
            )
            session.add(user)
            await session.flush()

            from src.db.models.storage import UserImage

            img_id = uuid4()
            expired_image = UserImage(
                id=img_id,
                user_id=user.id,
                storage_key=f"users/{user.id}/uploads/{img_id}.png",
                original_filename="photo.png",
                content_type="image/png",
                size_bytes=100,
                format="png",
                expires_at=datetime.now(UTC) - timedelta(days=1),
                product_id="vex",
            )
            session.add(expired_image)
            await session.flush()

            project_service = LibraryProjectService(session=session)
            project = await project_service.create(
                user.id, "vex", "Purge Project", None, session=session
            )

            repo = LibraryRepository(session)
            await repo.upsert_metadata(
                user.id,
                "vex",
                LibraryAssetSource.UPLOAD,
                expired_image.id,
                project_id=project.id,
            )
            await session.commit()

        mock_storage = MagicMock()
        mock_storage.delete_many = AsyncMock(return_value=1)
        service = ContentRetentionService(
            session_factory=session_factory,
            storage=mock_storage,
            batch_size=100,
            max_batches_per_run=5,
        )
        await service.sweep()

        async with session_factory() as verify_session:
            from sqlalchemy import select

            result = await verify_session.execute(
                select(LibraryAssetMetadata).where(
                    LibraryAssetMetadata.asset_type == "upload",
                    LibraryAssetMetadata.asset_id == expired_image.id,
                )
            )
            assert result.scalar_one_or_none() is None

            # The project row itself must survive — only the asset+metadata were purged.
            from src.db.repositories.library_project import LibraryProjectRepository

            still_there = await LibraryProjectRepository(verify_session).get(
                project.id, user_id=user.id, product_id="vex"
            )
            assert still_there is not None
