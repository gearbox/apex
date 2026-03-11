"""Unit tests for GenerationModelRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.db.repositories.generation_model import GenerationModelRepository

pytestmark = pytest.mark.unit


def _make_model(
    model_key: str = "grok-imagine-image",
    provider: str = "grok",
    is_enabled: bool = True,
) -> MagicMock:
    m = MagicMock()
    m.id = uuid4()
    m.model_key = model_key
    m.provider = provider
    m.name = f"Model {model_key}"
    m.description = "Test model"
    m.is_enabled = is_enabled
    m.created_at = datetime.now(UTC)
    m.updated_at = datetime.now(UTC)
    return m


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    return session


class TestListAll:
    async def test_returns_all_rows(self, mock_session: AsyncMock) -> None:
        """list_all returns all models regardless of is_enabled."""
        models = [_make_model("aisha", "comfyui"), _make_model("grok-imagine-image", "grok")]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = models
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        rows = await repo.list_all()

        assert list(rows) == models
        mock_session.execute.assert_awaited_once()


class TestListEnabled:
    async def test_applies_where_is_enabled_true(self, mock_session: AsyncMock) -> None:
        """list_enabled should filter to only enabled rows."""
        enabled = [_make_model("grok-imagine-image", "grok", is_enabled=True)]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = enabled
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        rows = await repo.list_enabled()

        assert list(rows) == enabled
        # Confirm WHERE clause is in the compiled statement
        stmt = mock_session.execute.call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "is_enabled" in compiled


class TestGetByKey:
    async def test_returns_none_when_not_found(self, mock_session: AsyncMock) -> None:
        """get_by_key returns None for an unknown key."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        result = await repo.get_by_key("nonexistent")

        assert result is None

    async def test_returns_model_when_found(self, mock_session: AsyncMock) -> None:
        """get_by_key returns the model when found."""
        model = _make_model("grok-imagine-image")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = model
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        result = await repo.get_by_key("grok-imagine-image")

        assert result is model


class TestSetEnabled:
    async def test_returns_none_when_key_not_found(self, mock_session: AsyncMock) -> None:
        """set_enabled returns None when the model_key does not exist."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        result = await repo.set_enabled("nonexistent", True)

        assert result is None
        mock_session.flush.assert_not_awaited()

    async def test_updates_is_enabled_and_updated_at(self, mock_session: AsyncMock) -> None:
        """set_enabled mutates the row and flushes the session."""
        model = _make_model("grok-imagine-image", is_enabled=True)
        original_updated_at = model.updated_at

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = model
        mock_session.execute = AsyncMock(return_value=result_mock)

        repo = GenerationModelRepository(mock_session)
        returned = await repo.set_enabled("grok-imagine-image", False)

        assert returned is model
        assert model.is_enabled is False
        # updated_at should have been set to a new datetime
        assert model.updated_at != original_updated_at
        mock_session.flush.assert_awaited_once()
