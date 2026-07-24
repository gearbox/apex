"""Tests for ContentProxyService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.core.config import Settings


def _make_service(ttl: int = 3600) -> ContentProxyService:
    """Build a ContentProxyService with a mock R2 storage."""
    mock_r2 = MagicMock()
    settings = MagicMock(spec=Settings)
    settings.content_url_ttl = ttl
    return ContentProxyService(storage=mock_r2, settings=settings)


class TestContentProxyServiceTtl:
    def test_ttl_default(self) -> None:
        service = _make_service(ttl=10800)
        assert service.ttl == 10800

    def test_ttl_custom(self) -> None:
        service = _make_service(ttl=3600)
        assert service.ttl == 3600


class TestResolveOutput:
    async def test_correct_owner_returns_storage_key(self) -> None:
        output_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_output = MagicMock()
        mock_output.id = output_id
        mock_output.storage_key = "users/x/outputs/job/file.jpeg"
        mock_output.product_id = product_id
        mock_output.size_bytes = 54321

        mock_repo = AsyncMock()
        mock_repo.get.return_value = mock_output

        service = _make_service()
        session = AsyncMock()

        with patch("src.api.services.content_proxy.OutputRepository", return_value=mock_repo):
            storage_key, etag, size_bytes = await service.resolve_output(
                output_id, user_id=user_id, product_id=product_id, session=session
            )

        assert storage_key == "users/x/outputs/job/file.jpeg"
        assert etag == str(output_id)
        assert size_bytes == 54321

    async def test_wrong_user_raises(self) -> None:
        output_id = uuid4()
        user_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get.return_value = None  # not found for this user

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.OutputRepository", return_value=mock_repo),
            pytest.raises(ContentNotFoundError),
        ):
            await service.resolve_output(
                output_id, user_id=user_id, product_id="vex", session=session
            )

    async def test_wrong_product_raises(self) -> None:
        output_id = uuid4()
        user_id = uuid4()

        mock_output = MagicMock()
        mock_output.id = output_id
        mock_output.storage_key = "users/x/outputs/job/file.jpeg"
        mock_output.product_id = "synthara"  # wrong product

        mock_repo = AsyncMock()
        mock_repo.get.return_value = mock_output

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.OutputRepository", return_value=mock_repo),
            pytest.raises(ContentNotFoundError),
        ):
            await service.resolve_output(
                output_id, user_id=user_id, product_id="vex", session=session
            )


class TestResolveUpload:
    async def test_correct_owner_returns_storage_key(self) -> None:
        image_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_image = MagicMock()
        mock_image.id = image_id
        mock_image.storage_key = "users/x/uploads/file.png"
        mock_image.product_id = product_id
        mock_image.size_bytes = 12345

        mock_repo = AsyncMock()
        mock_repo.get.return_value = mock_image

        service = _make_service()
        session = AsyncMock()

        with patch("src.api.services.content_proxy.UserImageRepository", return_value=mock_repo):
            storage_key, etag, size_bytes = await service.resolve_upload(
                image_id, user_id=user_id, product_id=product_id, session=session
            )

        assert storage_key == "users/x/uploads/file.png"
        assert etag == str(image_id)
        assert size_bytes == 12345

    async def test_wrong_user_raises(self) -> None:
        image_id = uuid4()
        user_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get.return_value = None

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.UserImageRepository", return_value=mock_repo),
            pytest.raises(ContentNotFoundError),
        ):
            await service.resolve_upload(
                image_id, user_id=user_id, product_id="vex", session=session
            )

    async def test_wrong_product_raises(self) -> None:
        image_id = uuid4()
        user_id = uuid4()

        mock_image = MagicMock()
        mock_image.id = image_id
        mock_image.storage_key = "users/x/uploads/file.png"
        mock_image.product_id = "synthara"

        mock_repo = AsyncMock()
        mock_repo.get.return_value = mock_image

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.UserImageRepository", return_value=mock_repo),
            pytest.raises(ContentNotFoundError),
        ):
            await service.resolve_upload(
                image_id, user_id=user_id, product_id="vex", session=session
            )


class TestDeleteContent:
    """Tests for ContentProxyService.delete_content."""

    async def test_delete_output_success(self) -> None:
        """Deleting a generation output removes R2 object and DB row."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_output = MagicMock()
        mock_output.id = content_id
        mock_output.storage_key = "users/x/outputs/job/file.png"
        mock_output.product_id = product_id

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output
        mock_output_repo.delete.return_value = True

        mock_image_repo = AsyncMock()

        service = _make_service()
        service._storage.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        service._storage.delete.assert_awaited_once_with("users/x/outputs/job/file.png")
        mock_output_repo.delete.assert_awaited_once_with(content_id, user_id=user_id)

    async def test_delete_upload_success(self) -> None:
        """Deleting a user upload removes R2 object and DB row."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_upload = MagicMock()
        mock_upload.id = content_id
        mock_upload.storage_key = "users/x/uploads/file.png"
        mock_upload.product_id = product_id

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = None  # not an output

        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = mock_upload
        mock_image_repo.delete.return_value = True

        service = _make_service()
        service._storage.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        service._storage.delete.assert_awaited_once_with("users/x/uploads/file.png")
        mock_image_repo.delete.assert_awaited_once_with(content_id, user_id=user_id)

    async def test_delete_not_found_raises(self) -> None:
        """Deleting unknown content raises ContentNotFoundError."""
        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = None
        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = None

        service = _make_service()
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
            pytest.raises(ContentNotFoundError),
        ):
            await service.delete_content(
                uuid4(),
                user_id=uuid4(),
                product_id="vex",
                session=session,
            )

    async def test_delete_wrong_product_raises(self) -> None:
        """Content owned by user but wrong product raises ContentNotFoundError."""
        content_id = uuid4()
        user_id = uuid4()

        mock_output = MagicMock()
        mock_output.id = content_id
        mock_output.storage_key = "users/x/outputs/job/file.png"
        mock_output.product_id = "synthara"  # different product

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output
        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = None

        service = _make_service()
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
            pytest.raises(ContentNotFoundError),
        ):
            await service.delete_content(
                content_id,
                user_id=user_id,
                product_id="vex",  # requesting as vex
                session=session,
            )

    async def test_delete_wrong_user_raises(self) -> None:
        """Content not owned by requesting user raises ContentNotFoundError."""
        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = None  # user_id filter rejects
        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = None

        service = _make_service()
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
            pytest.raises(ContentNotFoundError),
        ):
            await service.delete_content(
                uuid4(),
                user_id=uuid4(),
                product_id="vex",
                session=session,
            )

    async def test_delete_output_purges_derivative_r2_objects(self) -> None:
        """Deleting an output with thumbnails removes all derivative R2 objects."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_output = MagicMock()
        mock_output.id = content_id
        mock_output.storage_key = "users/x/outputs/job/file.png"
        mock_output.product_id = product_id

        derivative_sm = MagicMock()
        derivative_sm.storage_key = "users/x/outputs/job/thumb_sm.webp"

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output
        mock_output_repo.list_derivatives.return_value = [derivative_sm]
        mock_output_repo.delete.return_value = True
        mock_image_repo = AsyncMock()

        service = _make_service()
        service._storage.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        assert service._storage.delete.await_count == 2
        service._storage.delete.assert_any_await("users/x/outputs/job/thumb_sm.webp")
        service._storage.delete.assert_any_await("users/x/outputs/job/file.png")

    async def test_delete_upload_purges_derivative_r2_objects(self) -> None:
        """Deleting an upload with thumbnails removes all derivative R2 objects."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_upload = MagicMock()
        mock_upload.id = content_id
        mock_upload.storage_key = "users/x/uploads/file.png"
        mock_upload.product_id = product_id

        derivative_sm = MagicMock()
        derivative_sm.storage_key = "users/x/uploads/thumb_sm_file.webp"
        derivative_md = MagicMock()
        derivative_md.storage_key = "users/x/uploads/thumb_md_file.webp"

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = None
        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = mock_upload
        mock_image_repo.list_derivatives.return_value = [derivative_sm, derivative_md]
        mock_image_repo.delete.return_value = True

        service = _make_service()
        service._storage.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        assert service._storage.delete.await_count == 3
        service._storage.delete.assert_any_await("users/x/uploads/thumb_sm_file.webp")
        service._storage.delete.assert_any_await("users/x/uploads/thumb_md_file.webp")
        service._storage.delete.assert_any_await("users/x/uploads/file.png")

    async def test_delete_output_fallthrough_to_upload(self) -> None:
        """When ID matches an upload but not an output, upload is deleted."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        # Output lookup returns record but wrong product
        mock_output = MagicMock()
        mock_output.product_id = "synthara"
        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output

        # Upload lookup succeeds
        mock_upload = MagicMock()
        mock_upload.id = content_id
        mock_upload.storage_key = "users/x/uploads/file.png"
        mock_upload.product_id = product_id
        mock_image_repo = AsyncMock()
        mock_image_repo.get.return_value = mock_upload
        mock_image_repo.delete.return_value = True

        service = _make_service()
        service._storage.delete = AsyncMock(return_value=True)  # type: ignore[method-assign]
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        mock_image_repo.delete.assert_awaited_once()

    async def test_delete_commits_before_touching_r2(self) -> None:
        """DB delete + commit must happen before any R2 delete call — the DB
        is the source of truth; R2 cleanup is best-effort afterward."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_output = MagicMock()
        mock_output.id = content_id
        mock_output.storage_key = "users/x/outputs/job/file.png"
        mock_output.product_id = product_id

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output
        mock_output_repo.delete.return_value = True
        mock_image_repo = AsyncMock()

        call_order: list[str] = []
        mock_output_repo.delete.side_effect = lambda *a, **k: call_order.append("db_delete")  # noqa: ARG005

        service = _make_service()
        service._storage.delete = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: call_order.append("r2_delete")  # noqa: ARG005
        )
        session = AsyncMock()
        session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert call_order.index("db_delete") < call_order.index("commit")
        assert call_order.index("commit") < call_order.index("r2_delete")

    async def test_delete_r2_failure_does_not_raise_and_db_row_stays_deleted(self) -> None:
        """If R2 delete raises after the DB row is already committed gone,
        the error is logged and swallowed — not raised to the caller — since
        the DB is already the source of truth and the request must not fail."""
        content_id = uuid4()
        user_id = uuid4()
        product_id = "vex"

        mock_output = MagicMock()
        mock_output.id = content_id
        mock_output.storage_key = "users/x/outputs/job/file.png"
        mock_output.product_id = product_id

        mock_output_repo = AsyncMock()
        mock_output_repo.get.return_value = mock_output
        mock_output_repo.delete.return_value = True
        mock_image_repo = AsyncMock()

        service = _make_service()
        service._storage.delete = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("R2 unreachable")
        )
        session = AsyncMock()

        with (
            patch(
                "src.api.services.content_proxy.OutputRepository",
                return_value=mock_output_repo,
            ),
            patch(
                "src.api.services.content_proxy.UserImageRepository",
                return_value=mock_image_repo,
            ),
        ):
            # Must not raise despite the R2 failure.
            result = await service.delete_content(
                content_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )

        assert result is True
        mock_output_repo.delete.assert_awaited_once_with(content_id, user_id=user_id)
        session.commit.assert_awaited_once()


class TestSettingsContentUrlTtl:
    def test_default_ttl(self) -> None:
        """content_url_ttl defaults to 10800 in Settings."""
        from src.core.config import Settings

        settings = Settings(
            jwt_secret_key="a_valid_test_secret_key_that_is_long_enough_256bits",
        )
        assert settings.content_url_ttl == 10800
