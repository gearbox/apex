"""Migration drift checks for Alembic autogenerate."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest
    from sqlalchemy.ext.asyncio import AsyncEngine


def test_alembic_autogenerate_has_no_model_drift(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when Alembic's production import graph sees model/migration drift."""
    assert db_engine is not None  # ensures session-scoped Alembic upgrade has run
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "check"],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, f"alembic check failed:\n{result.stdout}\n{result.stderr}"
