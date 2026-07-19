"""Integration tests for LibraryService against a real database.

Covers asset detail malformed/cross-user refs, favorite idempotency, patch
tri-state semantics, and delete (via ContentProxyService with a mocked R2
storage backend — DB behavior is real, only the object-storage boundary is
stubbed, matching the ContentProxyService unit test conventions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest
import pytest_asyncio

from src.api.schemas.library import LibraryAssetPatch
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import LibraryService, LibraryValidationError
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User


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
