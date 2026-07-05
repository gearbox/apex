"""Unit tests for DatabaseManager and global DB helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.db.session as session_module
from src.db.session import DatabaseManager, close_db, get_db_manager, init_db

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_db_manager() -> object:
    """Restore global _db_manager after each test."""
    original = session_module._db_manager
    yield
    session_module._db_manager = original


class TestDatabaseManagerProperties:
    def test_engine_property_returns_engine(self) -> None:
        with (
            patch("src.db.session.create_async_engine") as mock_engine_fn,
            patch("src.db.session.async_sessionmaker"),
        ):
            mock_engine = MagicMock()
            mock_engine_fn.return_value = mock_engine
            mgr = DatabaseManager("postgresql+asyncpg://test:test@localhost/test")

        assert mgr.engine is mock_engine

    def test_session_factory_property(self) -> None:
        with (
            patch("src.db.session.create_async_engine"),
            patch("src.db.session.async_sessionmaker") as mock_maker,
        ):
            mock_factory = MagicMock()
            mock_maker.return_value = mock_factory
            mgr = DatabaseManager("postgresql+asyncpg://test:test@localhost/test")

        assert mgr.session_factory is mock_factory


class TestHealthCheck:
    async def test_returns_true_on_success(self) -> None:
        from collections.abc import AsyncIterator
        from contextlib import asynccontextmanager

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        @asynccontextmanager
        async def _fake_connect() -> AsyncIterator[AsyncMock]:
            yield mock_conn

        mock_engine = MagicMock()
        mock_engine.connect = _fake_connect

        with (
            patch("src.db.session.create_async_engine", return_value=mock_engine),
            patch("src.db.session.async_sessionmaker"),
        ):
            mgr = DatabaseManager("postgresql+asyncpg://test:test@localhost/test")
            result = await mgr.health_check()

        assert result is True

    async def test_returns_false_on_exception(self) -> None:
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionError("no DB")

        with (
            patch("src.db.session.create_async_engine", return_value=mock_engine),
            patch("src.db.session.async_sessionmaker"),
        ):
            mgr = DatabaseManager("postgresql+asyncpg://test:test@localhost/test")
            result = await mgr.health_check()

        assert result is False


class TestGetSession:
    async def test_get_session_yields_session(self) -> None:
        mock_engine = MagicMock()
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        with (
            patch("src.db.session.create_async_engine", return_value=mock_engine),
            patch("src.db.session.async_sessionmaker") as mock_maker,
        ):
            mock_factory = MagicMock()
            mock_factory.return_value = mock_session
            mock_maker.return_value = mock_factory

            mgr = DatabaseManager("postgresql+asyncpg://test:test@localhost/test")
            collected: list[object] = []
            async for s in mgr.get_session():
                collected.append(s)

        assert len(collected) == 1
        assert collected[0] is mock_session


class TestGetDbManager:
    def test_raises_when_not_initialized(self) -> None:
        session_module._db_manager = None
        with pytest.raises(RuntimeError, match="not initialized"):
            get_db_manager()

    def test_returns_manager_when_initialized(self) -> None:
        mock_mgr = MagicMock()
        session_module._db_manager = mock_mgr
        assert get_db_manager() is mock_mgr


class TestInitDb:
    def test_creates_and_stores_manager(self) -> None:
        with (
            patch("src.db.session.create_async_engine"),
            patch("src.db.session.async_sessionmaker"),
        ):
            mgr = init_db("postgresql+asyncpg://test:test@localhost/test")

        assert session_module._db_manager is mgr
        assert isinstance(mgr, DatabaseManager)


class TestCloseDb:
    async def test_closes_and_clears_manager(self) -> None:
        mock_mgr = AsyncMock(spec=DatabaseManager)
        session_module._db_manager = mock_mgr

        await close_db()

        mock_mgr.close.assert_awaited_once()
        assert session_module._db_manager is None

    async def test_noop_when_manager_is_none(self) -> None:
        session_module._db_manager = None
        # Should not raise
        await close_db()
