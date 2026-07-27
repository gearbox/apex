"""Tests covering StorageService Protocol stub bodies."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.api.services.storage.base import StorageService

pytestmark = pytest.mark.unit


class _ConcreteStorage(StorageService):
    """Minimal concrete implementation that inherits all Protocol stub bodies."""


class TestStorageServiceProtocolStubs:
    """Calling Protocol stub bodies (the `...` expressions) for coverage."""

    async def test_upload_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        mock = MagicMock()
        result = await svc.upload(
            user_id=uuid4(),
            data=b"",
            content_type="image/jpeg",
            storage_type=mock,
        )
        assert result is None

    async def test_download_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.download("some/key")
        assert result is None

    async def test_get_presigned_url_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.get_presigned_url("some/key")
        assert result is None

    async def test_delete_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.delete("some/key")
        assert result is None

    async def test_delete_many_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.delete_many(["key1", "key2"])
        assert result is None

    async def test_exists_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.exists("some/key")
        assert result is None

    async def test_list_user_files_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.list_user_files(uuid4())
        assert result is None

    def test_build_storage_key_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        mock = MagicMock()
        result = svc.build_storage_key(
            user_id=uuid4(),
            file_id=uuid4(),
            storage_type=mock,
            format=mock,
        )
        assert result is None

    async def test_health_check_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.health_check()
        assert result is None

    async def test_close_stub(self) -> None:
        svc = _ConcreteStorage()  # pyright: ignore[reportAbstractUsage]
        result = await svc.close()
        assert result is None
