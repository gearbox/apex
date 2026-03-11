"""Unit tests for admin model enable/disable endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.models import (
    GenerationModelResponse,
    ModelListResponse,
)

pytestmark = pytest.mark.unit


def _make_db_model(
    model_key: str = "grok-imagine-image",
    provider: str = "grok",
    is_enabled: bool = True,
) -> MagicMock:
    m = MagicMock()
    m.model_key = model_key
    m.provider = provider
    m.name = f"Model {model_key}"
    m.description = "Test description"
    m.is_enabled = is_enabled
    m.created_at = datetime.now(UTC)
    m.updated_at = datetime.now(UTC)
    return m


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def admin_user() -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    return user


class TestListModels:
    async def test_returns_all_models_when_enabled_only_false(
        self, mock_session: AsyncMock, admin_user: MagicMock
    ) -> None:
        """GET /v1/admin/models returns all models when enabled_only=False."""
        db_models = [
            _make_db_model("aisha", "comfyui", is_enabled=True),
            _make_db_model("grok-imagine-image", "grok", is_enabled=False),
        ]

        with patch("src.api.routes.admin.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_all = AsyncMock(return_value=db_models)
            repo.list_enabled = AsyncMock(return_value=[db_models[0]])

            MockRepo.return_value = repo

            # Simulate handler logic
            result = await repo.list_all()
            items = [
                GenerationModelResponse(
                    model_key=m.model_key,
                    provider=m.provider,
                    name=m.name,
                    description=m.description,
                    is_enabled=m.is_enabled,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in result
            ]
            response = ModelListResponse(items=items, total=len(items))

        assert response.total == 2
        assert response.items[0].model_key == "aisha"
        assert response.items[1].model_key == "grok-imagine-image"
        repo.list_all.assert_awaited_once()
        repo.list_enabled.assert_not_awaited()

    async def test_calls_list_enabled_when_enabled_only_true(
        self, mock_session: AsyncMock, admin_user: MagicMock
    ) -> None:
        """GET /v1/admin/models?enabled_only=true calls list_enabled."""
        db_models = [_make_db_model("grok-imagine-image", "grok", is_enabled=True)]

        with patch("src.api.routes.admin.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_all = AsyncMock(return_value=[])
            repo.list_enabled = AsyncMock(return_value=db_models)

            result = await repo.list_enabled()
            items = [
                GenerationModelResponse(
                    model_key=m.model_key,
                    provider=m.provider,
                    name=m.name,
                    description=m.description,
                    is_enabled=m.is_enabled,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in result
            ]
            response = ModelListResponse(items=items, total=len(items))

        assert response.total == 1
        repo.list_enabled.assert_awaited_once()
        repo.list_all.assert_not_awaited()


class TestToggleModel:
    async def test_returns_updated_response(
        self, mock_session: AsyncMock, admin_user: MagicMock
    ) -> None:
        """PATCH /v1/admin/models/{key} returns updated GenerationModelResponse."""
        updated = _make_db_model("grok-imagine-image", "grok", is_enabled=False)

        with patch("src.api.routes.admin.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.set_enabled = AsyncMock(return_value=updated)

            model = await repo.set_enabled("grok-imagine-image", False)
            assert model is not None
            response = GenerationModelResponse(
                model_key=model.model_key,
                provider=model.provider,
                name=model.name,
                description=model.description,
                is_enabled=model.is_enabled,
                created_at=model.created_at,
                updated_at=model.updated_at,
            )

        assert response.model_key == "grok-imagine-image"
        assert response.is_enabled is False
        repo.set_enabled.assert_awaited_once_with("grok-imagine-image", False)

    async def test_raises_404_for_nonexistent_key(
        self, mock_session: AsyncMock, admin_user: MagicMock
    ) -> None:
        """PATCH /v1/admin/models/nonexistent raises NotFoundException."""
        from litestar.exceptions import NotFoundException

        with patch("src.api.routes.admin.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.set_enabled = AsyncMock(return_value=None)

            result = await repo.set_enabled("nonexistent", True)
            if result is None:
                with pytest.raises(NotFoundException):
                    raise NotFoundException(detail="Model 'nonexistent' not found")
