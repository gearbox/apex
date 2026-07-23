"""Integration tests for LibraryService against a real database.

Covers asset detail malformed/cross-user refs, favorite idempotency, patch
tri-state semantics, and delete (via ContentProxyService with a mocked R2
storage backend — DB behavior is real, only the object-storage boundary is
stubbed, matching the ContentProxyService unit test conventions).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.library import LibraryAssetPatch
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.library import (
    LibraryProjectNotFoundError,
    LibraryService,
    LibraryValidationError,
)
from src.api.services.library_project import LibraryProjectService
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.storage import GenerationJob, GenerationOutput
    from src.db.models.user import User


async def _create_user_committed(session: AsyncSession) -> User:
    """Seed a User via a real committing session bound to the shared engine.

    Needed whenever the assertion/mutation crosses connections — the
    SAVEPOINT-scoped ``db_session``/``make_user`` fixtures never durably
    commit their outer transaction, so writes made through them are
    invisible outside that one connection (mirrors the pattern in
    test_library_bulk.py).
    """
    from src.db.models.user import User as UserModel

    user = UserModel(
        id=uuid4(),
        email=f"svciso-{uuid4().hex[:8]}@example.com",
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


@pytest.fixture
def content_proxy() -> ContentProxyService:
    mock_storage = MagicMock()
    mock_storage.delete = AsyncMock()
    settings = MagicMock()
    settings.content_url_ttl = 3600
    return ContentProxyService(storage=mock_storage, settings=settings)


@pytest_asyncio.fixture
async def project_service(db_session: AsyncSession) -> LibraryProjectService:
    return LibraryProjectService(session=db_session)


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> Callable[..., Coroutine[Any, Any, GenerationOutput]]:
    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        product_id: str = "vex",
    ) -> GenerationOutput:
        from src.db.models.storage import GenerationOutput as GenerationOutputModel

        if user is None:
            user = await make_user(email=f"svcout-{uuid4().hex[:8]}@example.com")
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


class TestGetAssetDetailEdgeCases:
    async def test_malformed_ref_returns_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"malformed-{uuid4().hex[:8]}@example.com")
        result = await library_service.get_asset_detail(
            "not-a-valid-ref", user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is None

    async def test_unknown_source_prefix_returns_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"unknownsrc-{uuid4().hex[:8]}@example.com")
        result = await library_service.get_asset_detail(
            f"generation:{uuid4()}", user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is None

    async def test_cross_user_ref_returns_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        owner = await make_user(email=f"owner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=owner)

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        result = await library_service.get_asset_detail(
            ref, other.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is None

    async def test_cross_product_ref_returns_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(
            email=f"crossprod-{uuid4().hex[:8]}@example.com", product_id="synthara"
        )
        image = await make_user_image(user=user, product_id="synthara")

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        result = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is None

    async def test_valid_upload_ref_returns_detail(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"valid-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        result = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.asset_ref == ref
        assert result.source == LibraryAssetSource.UPLOAD
        assert result.descendants.job_count == 0
        assert result.descendants.frame_count == 0


class TestFavoriteIdempotency:
    async def test_set_favorite_true_twice_is_idempotent(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"favtrue-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        first = await library_service.set_favorite(ref, True, user.id, "vex", session=db_session)
        second = await library_service.set_favorite(ref, True, user.id, "vex", session=db_session)
        assert first is True
        assert second is True

        detail = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert detail is not None
        assert detail.is_favorite is True

    async def test_set_favorite_false_twice_is_idempotent(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"favfalse-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        first = await library_service.set_favorite(ref, False, user.id, "vex", session=db_session)
        second = await library_service.set_favorite(ref, False, user.id, "vex", session=db_session)
        assert first is True
        assert second is True

    async def test_set_favorite_missing_asset_returns_false(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"favmissing-{uuid4().hex[:8]}@example.com")
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, uuid4())
        result = await library_service.set_favorite(ref, True, user.id, "vex", session=db_session)
        assert result is False


class TestPatchAssetTriState:
    async def test_absent_display_title_is_noop(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchabsent-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        await library_service.set_favorite(ref, True, user.id, "vex", session=db_session)
        patch = LibraryAssetPatch()  # display_title absent -> msgspec.UNSET
        assert patch.display_title is msgspec.UNSET

        result = await library_service.patch_asset(
            ref, patch, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.display_title is None
        assert result.is_favorite is True  # untouched by the no-op patch

    async def test_null_display_title_clears_existing_value(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchnull-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        await library_service.patch_asset(
            ref,
            LibraryAssetPatch(display_title="Original Title"),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        result = await library_service.patch_asset(
            ref,
            LibraryAssetPatch(display_title=None),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert result is not None
        assert result.display_title is None

    async def test_value_display_title_sets_value(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchvalue-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        result = await library_service.patch_asset(
            ref,
            LibraryAssetPatch(display_title="  My Sunset Photo  "),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert result is not None
        assert result.display_title == "My Sunset Photo"  # stripped

    async def test_empty_string_normalizes_to_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchempty-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        result = await library_service.patch_asset(
            ref,
            LibraryAssetPatch(display_title="   "),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert result is not None
        assert result.display_title is None

    async def test_over_255_chars_raises_validation_error(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchtoolong-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        with pytest.raises(LibraryValidationError):
            await library_service.patch_asset(
                ref,
                LibraryAssetPatch(display_title="x" * 256),
                user.id,
                "vex",
                VEX_CONFIG,
                session=db_session,
            )

    async def test_malformed_ref_returns_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"patchmalformed-{uuid4().hex[:8]}@example.com")
        result = await library_service.patch_asset(
            "not-valid",
            LibraryAssetPatch(display_title="x"),
            user.id,
            "vex",
            VEX_CONFIG,
            session=db_session,
        )
        assert result is None

    async def test_both_fields_patch_performs_single_upsert(
        self,
        library_service: LibraryService,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        """M2 — both fields present must resolve to exactly one upsert_metadata
        call (single statement, single fields=[...] log event) rather than two
        sequential calls for display_title and project_id."""
        user = await make_user(email=f"patchboth-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        project = await project_service.create(
            user.id, "vex", "Patch Both Project", None, session=db_session
        )
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        with patch.object(
            LibraryRepository,
            "upsert_metadata",
            autospec=True,
            side_effect=LibraryRepository.upsert_metadata,
        ) as spy:
            result = await library_service.patch_asset(
                ref,
                LibraryAssetPatch(display_title="Both Fields", project_id=project.id),
                user.id,
                "vex",
                VEX_CONFIG,
                session=db_session,
            )
        assert spy.call_count == 1
        assert spy.call_args.kwargs["display_title"] == "Both Fields"
        assert spy.call_args.kwargs["project_id"] == project.id

        assert result is not None
        assert result.display_title == "Both Fields"
        assert result.project_id == project.id

        repo = LibraryRepository(db_session)
        meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
        assert meta is not None
        assert meta.display_title == "Both Fields"
        assert meta.project_id == project.id

    async def test_project_404_leaves_display_title_unchanged(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        """M2 — validate-then-mutate: a project_id 404 must abort before any
        DB write, so a title present in the same patch is never applied."""
        user = await make_user(email=f"patch404-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        with pytest.raises(LibraryProjectNotFoundError):
            await library_service.patch_asset(
                ref,
                LibraryAssetPatch(display_title="Should Not Apply", project_id=uuid4()),
                user.id,
                "vex",
                VEX_CONFIG,
                session=db_session,
            )

        repo = LibraryRepository(db_session)
        meta = await repo.get_metadata(user.id, "vex", LibraryAssetSource.UPLOAD, image.id)
        assert meta is None


class TestDeleteAsset:
    async def test_delete_upload_purges_metadata_and_content(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"del-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        await library_service.set_favorite(ref, True, user.id, "vex", session=db_session)

        deleted = await library_service.delete_asset(
            ref, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert deleted is True

        # Underlying row gone
        from src.db.repositories.user_image import UserImageRepository

        assert await UserImageRepository(db_session).get(image.id, user_id=user.id) is None

        # Metadata purged too (D14 mirrored at the service level)
        remaining = await LibraryRepository(db_session).get_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id
        )
        assert remaining is None

    async def test_delete_already_deleted_returns_false(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"deltwice-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        first = await library_service.delete_asset(
            ref, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        second = await library_service.delete_asset(
            ref, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert first is True
        assert second is False

    async def test_delete_cross_user_returns_false(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        owner = await make_user(email=f"delowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"delother-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=owner)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        result = await library_service.delete_asset(
            ref, other.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result is False

    async def test_delete_type_mismatch_treated_as_not_found(
        self,
        library_service: LibraryService,
        content_proxy: ContentProxyService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        """A ref claiming 'output' for an id that is actually an upload must
        be treated as not-found (typed pre-check), never fall through and
        delete the upload anyway."""
        user = await make_user(email=f"delmismatch-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        wrong_ref = format_asset_ref(LibraryAssetSource.OUTPUT, image.id)

        result = await library_service.delete_asset(
            wrong_ref, user.id, "vex", session=db_session, content_proxy=content_proxy
        )
        assert result is False

        from src.db.repositories.user_image import UserImageRepository

        # The upload must still exist — the mismatched ref must not have
        # fallen through to delete it via the other table.
        assert await UserImageRepository(db_session).get(image.id, user_id=user.id) is not None

    async def test_content_not_found_rolls_back_metadata_purge(
        self,
        content_proxy: ContentProxyService,
        db_engine: AsyncEngine,
    ) -> None:
        """M1 — the metadata purge is flushed before delete_content so it can
        ride in delete_content's commit. If delete_content then raises
        ContentNotFoundError (forced here via patch.object to simulate the
        race), delete_asset must roll the flushed purge back itself — the
        metadata row (and the still-undeleted asset) must survive.

        Uses a real committing session_factory bound to the shared engine
        (rather than the SAVEPOINT-scoped ``db_session`` fixture) so the
        rollback-vs-commit distinction under test is real — a single shared
        ``db_session`` never durably commits mid-test (each ``session.commit()``
        is a logical no-op against the fixture's own outer SAVEPOINT), so a
        later ``session.rollback()`` there would undo unrelated prior work
        too, not just the one purge. This mirrors the production request
        lifecycle: one fresh session per call, real commit/rollback.
        """
        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        async with session_factory() as setup_session:
            user = await _create_user_committed(setup_session)
            image = await _create_upload_committed(setup_session, user=user)
            ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
            await LibraryRepository(setup_session).upsert_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
            )
            await setup_session.commit()

        async with session_factory() as work_session:
            service = LibraryService(session=work_session)
            with patch.object(
                content_proxy, "delete_content", side_effect=ContentNotFoundError("forced")
            ):
                result = await service.delete_asset(
                    ref, user.id, "vex", session=work_session, content_proxy=content_proxy
                )
            assert result is False

        async with session_factory() as verify_session:
            from src.db.repositories.user_image import UserImageRepository

            # The asset was never actually deleted (delete_content was
            # forced to fail before doing anything) — it must still be there.
            assert (
                await UserImageRepository(verify_session).get(image.id, user_id=user.id) is not None
            )

            # And the flushed-but-rolled-back purge must not have wiped the
            # pre-existing favorite metadata.
            remaining = await LibraryRepository(verify_session).get_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, image.id
            )
            assert remaining is not None
            assert remaining.is_favorite is True


class TestGetUploadDetailLineage:
    async def test_source_output_owned_by_other_user_degrades_to_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        """L2 — the lineage source-output lookup must be scoped to user_id.
        A foreign-owned source_output_id is a miss, degrading gracefully to
        source_asset_ref=None instead of leaking cross-user lineage."""
        from src.db.models.storage import UserImage as UserImageModel

        owner = await make_user(email=f"lineageowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"lineageother-{uuid4().hex[:8]}@example.com")
        foreign_output = await make_output(user=other)

        img_id = uuid4()
        image = UserImageModel(
            id=img_id,
            user_id=owner.id,
            product_id=owner.product_id,
            storage_key=f"users/{owner.id}/uploads/{img_id}.png",
            original_filename="frame.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            source_output_id=foreign_output.id,
            source_timestamp_ms=1500,
        )
        db_session.add(image)
        await db_session.flush()

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        result = await library_service.get_asset_detail(
            ref, owner.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref is None
        assert result.lineage.source_job_id is None
        assert result.lineage.source_timestamp_ms == 1500

    async def test_source_upload_owned_by_other_user_degrades_to_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        """L2 — mirrors the source_output_id cross-user test: a foreign-owned
        source_upload_id must degrade to source_asset_ref=None rather than
        leaking a reference to an upload the caller can't access."""
        from src.db.models.storage import UserImage as UserImageModel

        owner = await make_user(email=f"lineageowner2-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"lineageother2-{uuid4().hex[:8]}@example.com")
        foreign_upload = await make_user_image(user=other)

        img_id = uuid4()
        image = UserImageModel(
            id=img_id,
            user_id=owner.id,
            product_id=owner.product_id,
            storage_key=f"users/{owner.id}/uploads/{img_id}.png",
            original_filename="frame.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            source_upload_id=foreign_upload.id,
            source_timestamp_ms=1500,
        )
        db_session.add(image)
        await db_session.flush()

        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
        result = await library_service.get_asset_detail(
            ref, owner.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref is None
        assert result.lineage.source_job_id is None
        assert result.lineage.source_timestamp_ms == 1500


class TestGetOutputDetailLineage:
    """Remediation for the lineage-remediation prompt: _get_output_detail
    previously hardcoded lineage=None for every generated output, hiding
    provenance for i2i/i2v/v2v outputs. Mirrors TestGetUploadDetailLineage."""

    async def test_output_with_input_image_returns_upload_lineage(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"outlineage1-{uuid4().hex[:8]}@example.com")
        input_image = await make_user_image(user=user)
        job = await make_job(user=user, status="completed", generation_type="i2i")
        job.input_image_id = input_image.id
        db_session.add(job)
        await db_session.flush()
        output = await make_output(user=user, job=job)

        ref = format_asset_ref(LibraryAssetSource.OUTPUT, output.id)
        result = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref == format_asset_ref(
            LibraryAssetSource.UPLOAD, input_image.id
        )
        assert result.lineage.source_job_id is None
        assert result.lineage.source_timestamp_ms is None

    async def test_output_with_source_output_returns_output_lineage(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"outlineage2-{uuid4().hex[:8]}@example.com")
        source_output = await make_output(user=user)
        job = await make_job(user=user, status="completed", generation_type="i2i")
        job.source_output_id = source_output.id
        db_session.add(job)
        await db_session.flush()
        output = await make_output(user=user, job=job)

        ref = format_asset_ref(LibraryAssetSource.OUTPUT, output.id)
        result = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref == format_asset_ref(
            LibraryAssetSource.OUTPUT, source_output.id
        )
        assert result.lineage.source_job_id == source_output.job_id
        assert result.lineage.source_timestamp_ms is None

    async def test_source_output_owned_by_other_user_degrades_to_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        """L2 — mirrors TestGetUploadDetailLineage's cross-user test: a
        foreign-owned source_output_id must degrade to source_asset_ref=None
        and source_job_id=None rather than leaking cross-user lineage, and
        must never fall back to job.source_job_id."""
        owner = await make_user(email=f"outlineage3owner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"outlineage3other-{uuid4().hex[:8]}@example.com")
        foreign_output = await make_output(user=other)
        job = await make_job(user=owner, status="completed", generation_type="i2i")
        job.source_output_id = foreign_output.id
        db_session.add(job)
        await db_session.flush()
        output = await make_output(user=owner, job=job)

        ref = format_asset_ref(LibraryAssetSource.OUTPUT, output.id)
        result = await library_service.get_asset_detail(
            ref, owner.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref is None
        assert result.lineage.source_job_id is None

    async def test_input_image_owned_by_other_user_degrades_to_none(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        """L2 — mirrors test_source_output_owned_by_other_user_degrades_to_none
        for the input_image_id branch: a foreign-owned input_image_id must
        degrade to source_asset_ref=None rather than leaking a reference to
        an upload the caller can't access."""
        owner = await make_user(email=f"outlineage5owner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"outlineage5other-{uuid4().hex[:8]}@example.com")
        foreign_upload = await make_user_image(user=other)
        job = await make_job(user=owner, status="completed", generation_type="i2i")
        job.input_image_id = foreign_upload.id
        db_session.add(job)
        await db_session.flush()
        output = await make_output(user=owner, job=job)

        ref = format_asset_ref(LibraryAssetSource.OUTPUT, output.id)
        result = await library_service.get_asset_detail(
            ref, owner.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is not None
        assert result.lineage.source_asset_ref is None
        assert result.lineage.source_job_id is None

    async def test_t2i_output_returns_none_lineage(
        self,
        library_service: LibraryService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: Callable[..., Coroutine[Any, Any, GenerationOutput]],
        db_session: AsyncSession,
    ) -> None:
        """Regression guard: a plain t2i/t2v job has neither input_image_id
        nor source_output_id, so lineage must stay None."""
        user = await make_user(email=f"outlineage4-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed", generation_type="t2i")
        output = await make_output(user=user, job=job)

        ref = format_asset_ref(LibraryAssetSource.OUTPUT, output.id)
        result = await library_service.get_asset_detail(
            ref, user.id, "vex", VEX_CONFIG, session=db_session
        )
        assert result is not None
        assert result.lineage is None
