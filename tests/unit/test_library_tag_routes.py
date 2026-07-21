"""Unit tests for LibraryTagController route handlers (mocked LibraryTagService).

Calls handler functions directly via ``Controller.method.fn(...)`` — same
convention as test_library_project_routes.py — to exercise routing logic
(status codes, error envelopes, NotFoundException) without a DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import NotFoundException
from litestar.status_codes import HTTP_201_CREATED, HTTP_400_BAD_REQUEST

from src.api.routes.library_tag import LibraryTagController
from src.api.schemas.library import LibraryTag, LibraryTagCreate, LibraryTagPatch
from src.api.schemas.pagination import CursorPage
from src.api.services.library_tag import LibraryTagValidationError

pytestmark = pytest.mark.unit


def _tag(**overrides: object) -> LibraryTag:
    now = datetime.now(UTC)
    defaults: dict[str, object] = {
        "id": uuid4(),
        "name": "sunset",
        "created_at": now,
        "updated_at": now,
    } | overrides
    return LibraryTag(**defaults)  # type: ignore[arg-type]


class TestListTags:
    async def test_invalid_cursor_returns_400(self) -> None:
        service = AsyncMock()
        service.list_tags.side_effect = ValueError("bad cursor")

        response = await LibraryTagController.list_tags.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "invalid_cursor"

    async def test_success_returns_page(self) -> None:
        page: CursorPage[LibraryTag] = CursorPage(
            items=[], limit=30, has_more=False, next_cursor=None
        )
        service = AsyncMock()
        service.list_tags.return_value = page

        response = await LibraryTagController.list_tags.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.content is page


class TestCreateTag:
    async def test_success_returns_created(self) -> None:
        tag = _tag()
        service = AsyncMock()
        service.create.return_value = tag

        response = await LibraryTagController.create_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=LibraryTagCreate(name="sunset"),
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.status_code == HTTP_201_CREATED
        assert response.content is tag

    async def test_validation_error_returns_400(self) -> None:
        service = AsyncMock()
        service.create.side_effect = LibraryTagValidationError("name must be 1-50 characters")

        response = await LibraryTagController.create_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=LibraryTagCreate(name="   x"),
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "validation_error"


class TestGetTag:
    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.get.return_value = None

        with pytest.raises(NotFoundException):
            await LibraryTagController.get_tag.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                tag_id=uuid4(),
                session=MagicMock(),
                library_tag_service=service,
            )

    async def test_found_returns_tag(self) -> None:
        tag = _tag()
        service = AsyncMock()
        service.get.return_value = tag

        response = await LibraryTagController.get_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            tag_id=tag.id,
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response is tag


class TestPatchTag:
    async def test_validation_error_returns_400(self) -> None:
        service = AsyncMock()
        service.patch.side_effect = LibraryTagValidationError("name must be 1-50 characters")

        response = await LibraryTagController.patch_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            tag_id=uuid4(),
            data=LibraryTagPatch(name="   "),
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "validation_error"

    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.patch.return_value = None

        with pytest.raises(NotFoundException):
            await LibraryTagController.patch_tag.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                tag_id=uuid4(),
                data=LibraryTagPatch(name="new name"),
                session=MagicMock(),
                library_tag_service=service,
            )

    async def test_success_returns_tag(self) -> None:
        tag = _tag(name="new name")
        service = AsyncMock()
        service.patch.return_value = tag

        response = await LibraryTagController.patch_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            tag_id=tag.id,
            data=LibraryTagPatch(name="new name"),
            session=MagicMock(),
            library_tag_service=service,
        )
        assert response.content is tag


class TestDeleteTag:
    async def test_found_deletes(self) -> None:
        service = AsyncMock()
        service.delete.return_value = True

        await LibraryTagController.delete_tag.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            tag_id=uuid4(),
            session=MagicMock(),
            library_tag_service=service,
        )
        service.delete.assert_awaited_once()

    async def test_not_found_raises_404(self) -> None:
        service = AsyncMock()
        service.delete.return_value = False

        with pytest.raises(NotFoundException):
            await LibraryTagController.delete_tag.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                tag_id=uuid4(),
                session=MagicMock(),
                library_tag_service=service,
            )
