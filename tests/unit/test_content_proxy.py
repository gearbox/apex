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

        mock_repo = AsyncMock()
        mock_repo.get_output.return_value = mock_output

        service = _make_service()
        session = AsyncMock()

        with patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo):
            storage_key, etag = await service.resolve_output(
                output_id, user_id=user_id, product_id=product_id, session=session
            )

        assert storage_key == "users/x/outputs/job/file.jpeg"
        assert etag == str(output_id)

    async def test_wrong_user_raises(self) -> None:
        output_id = uuid4()
        user_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_output.return_value = None  # not found for this user

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo),
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
        mock_repo.get_output.return_value = mock_output

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo),
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

        mock_repo = AsyncMock()
        mock_repo.get_user_image.return_value = mock_image

        service = _make_service()
        session = AsyncMock()

        with patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo):
            storage_key, etag = await service.resolve_upload(
                image_id, user_id=user_id, product_id=product_id, session=session
            )

        assert storage_key == "users/x/uploads/file.png"
        assert etag == str(image_id)

    async def test_wrong_user_raises(self) -> None:
        image_id = uuid4()
        user_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_user_image.return_value = None

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo),
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
        mock_repo.get_user_image.return_value = mock_image

        service = _make_service()
        session = AsyncMock()

        with (
            patch("src.api.services.content_proxy.StorageRepository", return_value=mock_repo),
            pytest.raises(ContentNotFoundError),
        ):
            await service.resolve_upload(
                image_id, user_id=user_id, product_id="vex", session=session
            )


class TestSettingsContentUrlTtl:
    def test_default_ttl(self) -> None:
        """content_url_ttl defaults to 10800 in Settings."""
        from src.core.config import Settings

        settings = Settings(
            jwt_secret_key="a_valid_test_secret_key_that_is_long_enough_256bits",
        )
        assert settings.content_url_ttl == 10800
