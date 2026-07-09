"""Migration drift checks for Alembic autogenerate."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def test_alembic_autogenerate_has_no_model_drift(
    test_database_url: str,
    db_engine: AsyncEngine,
) -> None:
    """Fail when current SQLAlchemy metadata no longer matches migrated schema."""
    assert db_engine is not None  # ensures session-scoped Alembic upgrade has run
    env = {**os.environ, "DATABASE_URL": test_database_url, "DEBUG": "false"}

    result = subprocess.run(
        ["alembic", "check"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Alembic autogenerate detected model/schema drift.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
