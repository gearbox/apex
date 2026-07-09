"""Migration drift checks for Alembic autogenerate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic.config import Config

from alembic import command

if TYPE_CHECKING:
    import pytest
    from sqlalchemy.ext.asyncio import AsyncEngine


def test_alembic_autogenerate_has_no_model_drift(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when current SQLAlchemy metadata no longer matches migrated schema."""
    assert db_engine is not None  # ensures session-scoped Alembic upgrade has run
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")

    command.check(Config("alembic.ini"))
