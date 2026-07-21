"""Integration tests for Library Phase 3 tags against a real database.

Covers: tag CRUD owner/product isolation + case-insensitive 409;
replace-set semantics on patch_asset (add/remove/clear/idempotent re-set);
tag_id list filter; page hydration returns name-ordered tags; bulk
add/remove all-or-nothing with a foreign tag_id; duplicate pairs collapse;
delete_asset purges tag rows; retention sweep purges tag rows; tag CASCADE
on tag deletion removes join rows and empties the list filter.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.library import BulkAddTags, BulkRemoveTags, LibraryAssetPatch
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import (
    LibraryService,
    LibraryTagNotFoundError,
    LibraryValidationError,
)
from src.api.services.library_tag import LibraryTagNameConflictError, LibraryTagService
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.models.library import LibraryAssetTag
from src.db.repositories.library_tag import LibraryTagRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.db.models.storage import GenerationJob, GenerationOutput
    from src.db.models.user import User

OutputFactory = Callable[..., Coroutine[Any, Any, "GenerationOutput"]]

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def library_service(db_session: AsyncSession) -> LibraryService:
    return LibraryService(session=db_session)


@pytest_asyncio.fixture
async def tag_service(db_session: AsyncSession) -> LibraryTagService:
    return LibraryTagService(session=db_session)


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
    ) -> GenerationOutput:
        from src.db.models.storage import GenerationOutput as GenerationOutputModel

        if user is None:
            user = await make_user(email=f"tagout-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id=product_id)
        oid = uuid4()
        out = GenerationOutputModel(
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


class TestTagCrud:
    async def test_create_and_get(
        self,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagcrud-{uuid4().hex[:8]}@example.com")
        tag = await tag_service.create(user.id, "vex", "sunset", session=db_session)
        fetched = await tag_service.get(tag.id, user.id, "vex", session=db_session)
        assert fetched is not None
        assert fetched.name == "sunset"

    async def test_case_insensitive_name_conflict_raises_409_error(
        self,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagconflict-{uuid4().hex[:8]}@example.com")
        await tag_service.create(user.id, "vex", "Sunset", session=db_session)
        with pytest.raises(LibraryTagNameConflictError):
            await tag_service.create(user.id, "vex", "sunset", session=db_session)

    async def test_owner_isolation_cannot_fetch_other_users_tag(
        self,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        owner = await make_user(email=f"tagowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"tagother-{uuid4().hex[:8]}@example.com")
        tag = await tag_service.create(owner.id, "vex", "private", session=db_session)

        fetched = await tag_service.get(tag.id, other.id, "vex", session=db_session)
        assert fetched is None

    async def test_product_isolation_same_user_different_product(
        self,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagprod-{uuid4().hex[:8]}@example.com")
        tag = await tag_service.create(user.id, "vex", "cross-product", session=db_session)

        fetched = await tag_service.get(tag.id, user.id, "synthara", session=db_session)
        assert fetched is None

    async def test_delete_then_cascade_removes_join_rows(
        self,
        tag_service: LibraryTagService,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagcascade-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "cascade-me", session=db_session)

        asset_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        await library_service.patch_asset(
            asset_ref,
            LibraryAssetPatch(tag_ids=[tag.id]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )

        deleted = await tag_service.delete(tag.id, user.id, "vex", session=db_session)
        assert deleted is True

        result = await db_session.execute(
            select(LibraryAssetTag).where(LibraryAssetTag.tag_id == tag.id)
        )
        assert result.scalar_one_or_none() is None

        # The list filter for the now-deleted tag must simply return nothing
        # (not error) — the id no longer resolves to any owned tag.
        page = await library_service.list_assets(
            user.id, "vex", VEX_CONFIG, session=db_session, tag_id=tag.id
        )
        assert page.items == []


class TestReplaceSetSemantics:
    async def test_add_then_clear_then_idempotent_reset(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagreplace-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag_a = await tag_service.create(user.id, "vex", "a", session=db_session)
        tag_b = await tag_service.create(user.id, "vex", "b", session=db_session)
        asset_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        # Set [a, b].
        detail = await library_service.patch_asset(
            asset_ref,
            LibraryAssetPatch(tag_ids=[tag_a.id, tag_b.id]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert detail is not None
        assert {t.id for t in detail.tags} == {tag_a.id, tag_b.id}

        # Replace with just [a] — b must be removed.
        detail = await library_service.patch_asset(
            asset_ref,
            LibraryAssetPatch(tag_ids=[tag_a.id]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert detail is not None
        assert {t.id for t in detail.tags} == {tag_a.id}

        # Clear entirely via [].
        detail = await library_service.patch_asset(
            asset_ref,
            LibraryAssetPatch(tag_ids=[]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert detail is not None
        assert detail.tags == ()

        # Re-setting the same (empty) set again is idempotent — no error.
        detail = await library_service.patch_asset(
            asset_ref,
            LibraryAssetPatch(tag_ids=[]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert detail is not None
        assert detail.tags == ()

    async def test_over_20_tags_rejected(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagcap-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag_ids = []
        for i in range(21):
            tag = await tag_service.create(user.id, "vex", f"tag-{i}", session=db_session)
            tag_ids.append(tag.id)
        asset_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        with pytest.raises(LibraryValidationError):
            await library_service.patch_asset(
                asset_ref,
                LibraryAssetPatch(tag_ids=tag_ids),
                user.id,
                "vex",
                VEX_CONFIG,
                session=db_session,
            )

    async def test_foreign_tag_id_raises_not_found(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagforeign-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        asset_ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        with pytest.raises(LibraryTagNotFoundError):
            await library_service.patch_asset(
                asset_ref,
                LibraryAssetPatch(tag_ids=[uuid4()]),
                user.id,
                "vex",
                VEX_CONFIG,
                session=db_session,
            )


class TestTagIdListFilter:
    async def test_filters_to_tagged_assets_only(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagfilter-{uuid4().hex[:8]}@example.com")
        tagged = await make_user_image(user=user)
        untagged = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "filterme", session=db_session)

        await library_service.patch_asset(
            format_asset_ref(LibraryAssetSource.UPLOAD, tagged.id),
            LibraryAssetPatch(tag_ids=[tag.id]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )

        page = await library_service.list_assets(
            user.id, "vex", VEX_CONFIG, session=db_session, tag_id=tag.id
        )
        ids = {item.asset_ref for item in page.items}
        assert format_asset_ref(LibraryAssetSource.UPLOAD, tagged.id) in ids
        assert format_asset_ref(LibraryAssetSource.UPLOAD, untagged.id) not in ids

    async def test_page_hydration_returns_name_ordered_tags(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagorder-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag_z = await tag_service.create(user.id, "vex", "zebra", session=db_session)
        tag_a = await tag_service.create(user.id, "vex", "apple", session=db_session)

        await library_service.patch_asset(
            format_asset_ref(LibraryAssetSource.UPLOAD, image.id),
            LibraryAssetPatch(tag_ids=[tag_z.id, tag_a.id]),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )

        page = await library_service.list_assets(user.id, "vex", VEX_CONFIG, session=db_session)
        item = next(i for i in page.items if i.asset_ref.endswith(str(image.id)))
        assert [t.name for t in item.tags] == ["apple", "zebra"]


class TestBulkTagOps:
    async def test_bulk_add_then_remove(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagbulk-{uuid4().hex[:8]}@example.com")
        image_a = await make_user_image(user=user)
        image_b = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "batch", session=db_session)
        refs = [
            format_asset_ref(LibraryAssetSource.UPLOAD, image_a.id),
            format_asset_ref(LibraryAssetSource.UPLOAD, image_b.id),
        ]

        add_op = BulkAddTags(asset_refs=refs, tag_ids=[tag.id])
        result = await library_service.bulk_apply(
            add_op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result.op == "add_tags"
        assert result.succeeded == 2

        repo = LibraryTagRepository(db_session)
        tags_map = await repo.batch_tags_for_assets(
            [("upload", image_a.id), ("upload", image_b.id)], user_id=user.id, product_id="vex"
        )
        assert tag.id in {t.id for t in tags_map.get(("upload", image_a.id), [])}
        assert tag.id in {t.id for t in tags_map.get(("upload", image_b.id), [])}

        remove_op = BulkRemoveTags(asset_refs=refs, tag_ids=[tag.id])
        result = await library_service.bulk_apply(
            remove_op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result.op == "remove_tags"
        assert result.succeeded == 2

        tags_map = await repo.batch_tags_for_assets(
            [("upload", image_a.id), ("upload", image_b.id)], user_id=user.id, product_id="vex"
        )
        assert tags_map == {}

    async def test_re_add_is_idempotent(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagidempotent-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "idempotent", session=db_session)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkAddTags(asset_refs=[ref], tag_ids=[tag.id])
        for _ in range(2):
            result = await library_service.bulk_apply(
                op, user.id, "vex", session=db_session, content_proxy=content_proxy
            )
            assert result.succeeded == 1

    async def test_foreign_tag_id_fails_whole_batch_zero_side_effects(
        self,
        content_proxy: ContentProxyService,
        db_engine: AsyncEngine,
    ) -> None:
        """P5/H1: a foreign tag_id must fail the whole request with no writes,
        verified from a second NullPool session (mirrors test_library_bulk.py)."""
        from src.db.models.storage import UserImage
        from src.db.models.user import User as UserModel

        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with session_factory() as setup_session:
            user = UserModel(
                id=uuid4(),
                email=f"tagbulkforeign-{uuid4().hex[:8]}@example.com",
                password_hash="hashed",
                product_id="vex",
            )
            other_user = UserModel(
                id=uuid4(),
                email=f"tagbulkforeignother-{uuid4().hex[:8]}@example.com",
                password_hash="hashed",
                product_id="vex",
            )
            setup_session.add_all([user, other_user])
            await setup_session.flush()

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
                product_id="vex",
            )
            setup_session.add(image)
            await setup_session.flush()

            tag_service = LibraryTagService(session=setup_session)
            foreign_tag = await tag_service.create(
                other_user.id, "vex", "not-yours", session=setup_session
            )
            await setup_session.commit()

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        async with session_factory() as work_session:
            service = LibraryService(session=work_session)
            op = BulkAddTags(asset_refs=[ref], tag_ids=[foreign_tag.id])
            with pytest.raises(LibraryTagNotFoundError):
                await service.bulk_apply(
                    op, user.id, "vex", session=work_session, content_proxy=content_proxy
                )
            await work_session.rollback()

        async with session_factory() as verify_session:
            repo = LibraryTagRepository(verify_session)
            tags_map = await repo.batch_tags_for_assets(
                [("upload", image.id)], user_id=user.id, product_id="vex"
            )
            assert tags_map == {}

    async def test_duplicate_asset_refs_and_tag_ids_collapse(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagdup-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "dup", session=db_session)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        op = BulkAddTags(asset_refs=[ref, ref], tag_ids=[tag.id, tag.id])
        result = await library_service.bulk_apply(
            op, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert len(result.results) == 1
        assert result.succeeded == 1


class TestDeleteAssetPurgesTags:
    async def test_delete_asset_purges_tag_rows(
        self,
        library_service: LibraryService,
        tag_service: LibraryTagService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"tagdeleteasset-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        tag = await tag_service.create(user.id, "vex", "gone-soon", session=db_session)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        await library_service.patch_asset(
            ref, LibraryAssetPatch(tag_ids=[tag.id]), user.id, "vex", VEX_CONFIG, session=db_session
        )

        deleted = await library_service.delete_asset(
            ref, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert deleted is True

        result = await db_session.execute(
            select(LibraryAssetTag).where(
                LibraryAssetTag.asset_type == "upload", LibraryAssetTag.asset_id == image.id
            )
        )
        assert result.scalar_one_or_none() is None

        # The tag row itself survives — only the join row was purged.
        still_there = await tag_service.get(tag.id, user.id, "vex", session=db_session)
        assert still_there is not None


class TestRetentionSweepPurgesTags:
    async def test_sweep_purges_tag_rows_for_expired_upload(
        self,
        db_engine: AsyncEngine,
    ) -> None:
        from src.api.services.content_retention import ContentRetentionService
        from src.db.models.storage import UserImage
        from src.db.models.user import User as UserModel

        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with session_factory() as session:
            user = UserModel(
                id=uuid4(),
                email=f"tagpurge-{uuid4().hex[:8]}@example.com",
                password_hash="hashed",
                product_id="vex",
            )
            session.add(user)
            await session.flush()

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

            tag_service = LibraryTagService(session=session)
            tag = await tag_service.create(user.id, "vex", "expiring", session=session)

            repo = LibraryTagRepository(session)
            await repo.set_asset_tags(
                "upload", expired_image.id, [tag.id], user_id=user.id, product_id="vex"
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
            result = await verify_session.execute(
                select(LibraryAssetTag).where(
                    LibraryAssetTag.asset_type == "upload",
                    LibraryAssetTag.asset_id == expired_image.id,
                )
            )
            assert result.scalar_one_or_none() is None

            # The tag row itself survives sweep — only the join row was purged.
            still_there = await LibraryTagRepository(verify_session).get(
                tag.id, user_id=user.id, product_id="vex"
            )
            assert still_there is not None


class TestTagNameCaseInsensitiveConstraintAtDbLevel:
    async def test_direct_insert_of_colliding_name_raises_integrity_error(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        """Belt-and-suspenders: the functional unique index itself, not just
        the service-layer nested-transaction handling."""
        from src.db.models.library import LibraryTag as LibraryTagModel

        user = await make_user(email=f"tagdblevel-{uuid4().hex[:8]}@example.com")
        db_session.add(LibraryTagModel(id=uuid4(), product_id="vex", user_id=user.id, name="Dup"))
        await db_session.flush()

        db_session.add(LibraryTagModel(id=uuid4(), product_id="vex", user_id=user.id, name="dup"))
        with pytest.raises(IntegrityError):
            await db_session.flush()
