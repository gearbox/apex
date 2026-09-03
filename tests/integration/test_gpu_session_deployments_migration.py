"""Integration test for migration 039's gpu_session_deployments backfill and round-trip.

Seeds gpu_sessions rows against the pre-migration (038) schema — one active,
one provisioning, one stopped — then runs the real ``alembic upgrade``/
``downgrade`` commands and asserts:

- invariant 11 (backfill): exactly two deployment rows are created (for the
  active and provisioning sessions), with the right statuses and identity
  copied from the source session; none for the stopped session.
- invariant 12 (round-trip): downgrading to 038 restores
  ``readiness_marker_node_class`` (from each session's primary deployment)
  and ``ix_gpu_sessions_active_user_model``; re-upgrading to 039 re-backfills
  the same shape.

Synchronous test function, same pattern as test_payment_description_migration.py:
alembic's ``command.upgrade``/``command.downgrade`` call ``asyncio.run()``
internally, so they can't run inside an ``async def`` test under
pytest-asyncio. Async DB access is wrapped in its own ``asyncio.run()`` per
phase; the shared ``db_engine`` uses NullPool, so handing it across separate
event loops here is safe.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from alembic.config import Config
from sqlalchemy import text

from alembic import command

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def test_migration_039_backfills_and_round_trips(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    assert db_engine is not None  # The session fixture has upgraded the schema to head.
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    user_id = uuid4()
    active_session_id = uuid4()
    provisioning_session_id = uuid4()
    stopped_session_id = uuid4()
    active_operation_id = uuid4()
    started_at = datetime(2026, 1, 1, tzinfo=UTC)

    async def _seed() -> None:
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO users "
                    "(id, email, password_hash, role, is_active, subscription_tier, "
                    "locale, product_id) "
                    "VALUES (:id, :email, 'x', 'user', true, 'free', 'en', 'vex')"
                ),
                {"id": user_id, "email": f"p2-migration-{user_id}@example.com"},
            )
            await conn.execute(
                text(
                    "INSERT INTO gpu_sessions "
                    "(id, user_id, product_id, bundle_name, bundle_version, model_type, "
                    " readiness_marker_node_class, status, bootstrap_operation_id, started_at) "
                    "VALUES (:id, :user_id, 'vex', 'wan_2.2_i2v', '260105-01', 'aisha-image', "
                    " 'WanVideoSampler', 'active', :operation_id, :started_at)"
                ),
                {
                    "id": active_session_id,
                    "user_id": user_id,
                    "operation_id": active_operation_id,
                    "started_at": started_at,
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO gpu_sessions "
                    "(id, user_id, product_id, bundle_name, bundle_version, model_type, "
                    " readiness_marker_node_class, status) "
                    "VALUES (:id, :user_id, 'vex', 'zit_cyberrealistic', NULL, "
                    " 'aisha-image-lite', NULL, 'provisioning')"
                ),
                {"id": provisioning_session_id, "user_id": user_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO gpu_sessions "
                    "(id, user_id, product_id, bundle_name, bundle_version, model_type, "
                    " readiness_marker_node_class, status, started_at) "
                    "VALUES (:id, :user_id, 'vex', 'wan_2.2_i2v', NULL, 'aisha-image', "
                    " NULL, 'stopped', :started_at)"
                ),
                {"id": stopped_session_id, "user_id": user_id, "started_at": started_at},
            )

    async def _deployments_for(session_id: object) -> list[dict[str, object]]:
        async with db_engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT status, is_primary, bundle_name, bundle_version, model_type, "
                    "readiness_marker_node_class, provision_operation_id, activated_at "
                    "FROM gpu_session_deployments WHERE session_id = :session_id"
                ),
                {"session_id": session_id},
            )
            return [dict(row._mapping) for row in result.all()]

    async def _session_readiness_marker(session_id: object) -> str | None:
        async with db_engine.connect() as conn:
            result = await conn.execute(
                text("SELECT readiness_marker_node_class FROM gpu_sessions WHERE id = :id"),
                {"id": session_id},
            )
            return result.scalar_one()

    async def _schema_shape() -> tuple[set[str], set[str], set[str]]:
        async with db_engine.connect() as conn:
            session_columns = set(
                (
                    await conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'gpu_sessions'"
                        )
                    )
                ).scalars()
            )
            session_indexes = set(
                (
                    await conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE tablename = 'gpu_sessions'")
                    )
                ).scalars()
            )
            deployment_indexes = set(
                (
                    await conn.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE tablename = 'gpu_session_deployments'"
                        )
                    )
                ).scalars()
            )
            return session_columns, session_indexes, deployment_indexes

    async def _cleanup() -> None:
        async with db_engine.begin() as conn:
            # ON DELETE CASCADE on gpu_sessions.user_id and
            # gpu_session_deployments.{session_id,user_id} takes the rest with it.
            await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})

    try:
        # --- invariant 11: backfill -------------------------------------
        command.downgrade(config, "038")
        asyncio.run(_seed())

        command.upgrade(config, "039")

        active_deployments = asyncio.run(_deployments_for(active_session_id))
        assert len(active_deployments) == 1
        d = active_deployments[0]
        assert d["status"] == "active"
        assert d["is_primary"] is True
        assert d["bundle_name"] == "wan_2.2_i2v"
        assert d["bundle_version"] == "260105-01"
        assert d["model_type"] == "aisha-image"
        assert d["readiness_marker_node_class"] == "WanVideoSampler"
        assert d["provision_operation_id"] == active_operation_id
        assert d["activated_at"] is not None

        provisioning_deployments = asyncio.run(_deployments_for(provisioning_session_id))
        assert len(provisioning_deployments) == 1
        d = provisioning_deployments[0]
        assert d["status"] == "deploying"
        assert d["is_primary"] is True
        assert d["model_type"] == "aisha-image-lite"
        assert d["activated_at"] is None

        stopped_deployments = asyncio.run(_deployments_for(stopped_session_id))
        assert stopped_deployments == []

        session_columns, session_indexes, deployment_indexes = asyncio.run(_schema_shape())
        assert "readiness_marker_node_class" not in session_columns
        assert "ix_gpu_sessions_active_user_model" not in session_indexes
        assert "ix_gpu_session_deployments_live_user_model" in deployment_indexes
        assert "ix_gpu_session_deployments_session" in deployment_indexes
        assert "ix_gpu_session_deployments_routing" in deployment_indexes

        # --- invariant 12: round-trip ------------------------------------
        command.downgrade(config, "038")

        session_columns, session_indexes, _ = asyncio.run(_schema_shape())
        assert "readiness_marker_node_class" in session_columns
        assert "ix_gpu_sessions_active_user_model" in session_indexes

        # Restored from each session's primary deployment.
        assert asyncio.run(_session_readiness_marker(active_session_id)) == "WanVideoSampler"
        assert asyncio.run(_session_readiness_marker(provisioning_session_id)) is None
        # Never had a deployment — nothing to restore from, stays NULL.
        assert asyncio.run(_session_readiness_marker(stopped_session_id)) is None

        # Re-upgrading re-backfills the same shape from the same source data.
        command.upgrade(config, "039")

        active_deployments = asyncio.run(_deployments_for(active_session_id))
        assert len(active_deployments) == 1
        assert active_deployments[0]["status"] == "active"
        assert active_deployments[0]["readiness_marker_node_class"] == "WanVideoSampler"

        provisioning_deployments = asyncio.run(_deployments_for(provisioning_session_id))
        assert len(provisioning_deployments) == 1
        assert provisioning_deployments[0]["status"] == "deploying"

        assert asyncio.run(_deployments_for(stopped_session_id)) == []
    finally:
        # Integration tests share the migrated schema, so always restore head.
        command.upgrade(config, "head")
        asyncio.run(_cleanup())
