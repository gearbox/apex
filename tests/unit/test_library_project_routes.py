"""Unit tests for LibraryProjectController route handlers (mocked LibraryProjectService).

Calls handler functions directly via ``Controller.method.fn(...)`` — same
convention as test_library_routes.py — to exercise routing logic (status
codes, error envelopes, NotFoundException) without a DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import NotFoundException
from litestar.status_codes import HTTP_201_CREATED, HTTP_400_BAD_REQUEST

from src.api.routes.library_project import LibraryProjectController
from src.api.schemas.library import LibraryProject, LibraryProjectCreate, LibraryProjectPatch
from src.api.schemas.pagination import CursorPage
from src.api.services.library_project import LibraryProjectValidationError

pytestmark = pytest.mark.unit


def _project(**overrides: object) -> LibraryProject:
    now = datetime.now(UTC)
    defaults: dict[str, object] = {
        "id": uuid4(),
        "name": "Vacation Photos",
        "description": None,
        "created_at": now,
        "updated_at": now,
    } | overrides
    return LibraryProject(**defaults)  # type: ignore[arg-type]


class TestListProjects:
    async def test_invalid_cursor_returns_400(self) -> None:
        service = AsyncMock()
        service.list_projects.side_effect = ValueError("bad cursor")

        response = await LibraryProjectController.list_projects.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "invalid_cursor"

    async def test_success_returns_page(self) -> None:
        page: CursorPage[LibraryProject] = CursorPage(
            items=[], limit=30, has_more=False, next_cursor=None
        )
        service = AsyncMock()
        service.list_projects.return_value = page

        response = await LibraryProjectController.list_projects.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.content is page


class TestCreateProject:
    async def test_success_returns_created(self) -> None:
        project = _project()
        service = AsyncMock()
        service.create.return_value = project

        response = await LibraryProjectController.create_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=LibraryProjectCreate(name="Vacation Photos"),
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.status_code == HTTP_201_CREATED
        assert response.content is project

    async def test_validation_error_returns_400(self) -> None:
        service = AsyncMock()
        service.create.side_effect = LibraryProjectValidationError("name must be 1-100 characters")

        response = await LibraryProjectController.create_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            data=LibraryProjectCreate(name="   "),
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "validation_error"


class TestGetProject:
    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.get.return_value = None

        with pytest.raises(NotFoundException):
            await LibraryProjectController.get_project.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                project_id=uuid4(),
                session=MagicMock(),
                library_project_service=service,
            )

    async def test_found_returns_project(self) -> None:
        project = _project()
        service = AsyncMock()
        service.get.return_value = project

        response = await LibraryProjectController.get_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            project_id=project.id,
            session=MagicMock(),
            library_project_service=service,
        )
        assert response is project


class TestPatchProject:
    async def test_validation_error_returns_400(self) -> None:
        service = AsyncMock()
        service.patch.side_effect = LibraryProjectValidationError("name must be 1-100 characters")

        response = await LibraryProjectController.patch_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            project_id=uuid4(),
            data=LibraryProjectPatch(name="   "),
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content.error == "validation_error"

    async def test_none_returns_404(self) -> None:
        service = AsyncMock()
        service.patch.return_value = None

        with pytest.raises(NotFoundException):
            await LibraryProjectController.patch_project.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                project_id=uuid4(),
                data=LibraryProjectPatch(name="New Name"),
                session=MagicMock(),
                library_project_service=service,
            )

    async def test_success_returns_project(self) -> None:
        project = _project(name="New Name")
        service = AsyncMock()
        service.patch.return_value = project

        response = await LibraryProjectController.patch_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            project_id=project.id,
            data=LibraryProjectPatch(name="New Name"),
            session=MagicMock(),
            library_project_service=service,
        )
        assert response.content is project


class TestDeleteProject:
    async def test_found_deletes(self) -> None:
        service = AsyncMock()
        service.delete.return_value = True

        await LibraryProjectController.delete_project.fn(
            MagicMock(),
            current_user_id=uuid4(),
            product_id="vex",
            project_id=uuid4(),
            session=MagicMock(),
            library_project_service=service,
        )
        service.delete.assert_awaited_once()

    async def test_not_found_raises_404(self) -> None:
        service = AsyncMock()
        service.delete.return_value = False

        with pytest.raises(NotFoundException):
            await LibraryProjectController.delete_project.fn(
                MagicMock(),
                current_user_id=uuid4(),
                product_id="vex",
                project_id=uuid4(),
                session=MagicMock(),
                library_project_service=service,
            )
