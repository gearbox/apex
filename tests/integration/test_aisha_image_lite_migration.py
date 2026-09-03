"""Integration coverage for the aisha-image-lite seed migration.

The schema fixture is already at head, so this exercises revision 037's
upgrade() and downgrade() directly inside a rolled-back transaction. Calling
upgrade() twice verifies its data inserts are safe to replay after a partial
failure; Alembic's revision tracking would otherwise skip the second call.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "037_seed_aisha_image_lite.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_037", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_aisha_image_lite_seed_is_idempotent_and_reverses_cleanly(
    db_engine: AsyncEngine,
) -> None:
    """Revision 037 can be replayed and removes exactly its seed rows."""
    migration = _load_migration()

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        try:

            def _reset_and_run_upgrade_twice(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    # The session fixture is at head. Start from the state immediately
                    # before this seed to verify the first insert as well as its replay.
                    migration.downgrade()
                    migration.upgrade()
                    migration.upgrade()

            await connection.run_sync(_reset_and_run_upgrade_twice)

            models = await connection.execute(
                text(
                    "SELECT id::text, provider, name, is_enabled FROM generation_models "
                    "WHERE model_key = 'aisha-image-lite'"
                )
            )
            assert models.all() == [
                (
                    migration._MODEL_ID_AISHA_IMAGE_LITE,
                    "aisha",
                    "Aisha Lite",
                    False,
                )
            ]

            pricing = await connection.execute(
                text(
                    "SELECT id::text, provider, generation_type, token_cost FROM pricing_catalog "
                    "WHERE model = 'aisha-image-lite'"
                )
            )
            assert pricing.all() == [
                (
                    migration._PRICE_ID_AISHA_IMG_LITE_T2I,
                    "aisha",
                    "t2i",
                    1,
                )
            ]

            def _run_downgrade(sync_connection: Connection) -> None:
                context = MigrationContext.configure(sync_connection)
                with Operations.context(context):
                    migration.downgrade()

            await connection.run_sync(_run_downgrade)

            model_count = await connection.execute(
                text("SELECT count(*) FROM generation_models WHERE model_key = 'aisha-image-lite'")
            )
            assert model_count.scalar_one() == 0

            pricing_count = await connection.execute(
                text("SELECT count(*) FROM pricing_catalog WHERE model = 'aisha-image-lite'")
            )
            assert pricing_count.scalar_one() == 0
        finally:
            await transaction.rollback()
