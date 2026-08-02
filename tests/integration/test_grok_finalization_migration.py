"""Data-preserving upgrade coverage for migration 033.

These tests intentionally start at revision 032: the duplicate-output repair
must be proven against the pre-unique-index shape rather than an already
constrained head schema.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from alembic import command


def _storage_key(ids: dict[str, UUID], name: str) -> str:
    return f"migration/{ids['job'].hex}/{name}.png"


async def _seed_duplicate_group(
    database_url: str,
    *,
    ambiguous_metadata: bool = False,
) -> dict[str, UUID]:
    """Seed one canonical output and two same-bucket duplicates at revision 032."""
    engine = create_async_engine(database_url)
    ids = {
        "user": uuid4(),
        "job": uuid4(),
        "canonical": uuid4(),
        "duplicate_1": uuid4(),
        "duplicate_2": uuid4(),
        "derivative_job": uuid4(),
        "derivative_image": uuid4(),
        "metadata_1": uuid4(),
        "metadata_2": uuid4(),
        "tag_shared": uuid4(),
        "tag_distinct": uuid4(),
    }
    now = datetime.now(UTC)
    product_id = "vex"

    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO users (id, email, password_hash, product_id)
                VALUES (:id, :email, 'migration-test-hash', :product_id)
                """
            ),
            {
                "id": ids["user"],
                "email": f"migration-{ids['user'].hex[:12]}@example.com",
                "product_id": product_id,
            },
        )
        await connection.execute(
            text(
                """
                INSERT INTO generation_jobs (
                    id, user_id, name, status, generation_type, provider,
                    prompt, product_id
                ) VALUES (
                    :id, :user_id, 'duplicate source', 'completed', 't2i', 'grok',
                    'migration fixture', :product_id
                )
                """
            ),
            {"id": ids["job"], "user_id": ids["user"], "product_id": product_id},
        )
        for output_id, storage_key, created_at in (
            (ids["canonical"], _storage_key(ids, "canonical"), now - timedelta(minutes=3)),
            (ids["duplicate_1"], _storage_key(ids, "duplicate-1"), now - timedelta(minutes=2)),
            (ids["duplicate_2"], _storage_key(ids, "duplicate-2"), now - timedelta(minutes=1)),
        ):
            await connection.execute(
                text(
                    """
                    INSERT INTO generation_outputs (
                        id, user_id, job_id, storage_key, content_type, size_bytes,
                        format, output_index, product_id, created_at, expires_at
                    ) VALUES (
                        :id, :user_id, :job_id, :storage_key, 'image/png', 1,
                        'png', 0, :product_id, :created_at, :expires_at
                    )
                    """
                ),
                {
                    "id": output_id,
                    "user_id": ids["user"],
                    "job_id": ids["job"],
                    "storage_key": storage_key,
                    "product_id": product_id,
                    "created_at": created_at,
                    "expires_at": now + timedelta(days=7),
                },
            )
        await connection.execute(
            text(
                """
                INSERT INTO generation_jobs (
                    id, user_id, name, status, generation_type, provider,
                    prompt, product_id, source_output_id
                ) VALUES (
                    :id, :user_id, 'derivative', 'completed', 't2i', 'grok',
                    'migration fixture', :product_id, :source_output_id
                )
                """
            ),
            {
                "id": ids["derivative_job"],
                "user_id": ids["user"],
                "product_id": product_id,
                "source_output_id": ids["duplicate_2"],
            },
        )
        await connection.execute(
            text(
                """
                INSERT INTO user_images (
                    id, user_id, storage_key, original_filename, content_type,
                    size_bytes, format, product_id, expires_at, source_output_id
                ) VALUES (
                    :id, :user_id, :storage_key, 'derivative.png',
                    'image/png', 1, 'png', :product_id, :expires_at, :source_output_id
                )
                """
            ),
            {
                "id": ids["derivative_image"],
                "user_id": ids["user"],
                "storage_key": _storage_key(ids, "derivative"),
                "product_id": product_id,
                "expires_at": now + timedelta(days=7),
                "source_output_id": ids["duplicate_2"],
            },
        )
        metadata_rows = [(ids["metadata_1"], ids["duplicate_1"])]
        if ambiguous_metadata:
            metadata_rows.append((ids["metadata_2"], ids["duplicate_2"]))
        for metadata_id, output_id in metadata_rows:
            await connection.execute(
                text(
                    """
                    INSERT INTO library_asset_metadata (
                        id, product_id, user_id, asset_type, asset_id, display_title
                    ) VALUES (:id, :product_id, :user_id, 'output', :asset_id, 'fixture')
                    """
                ),
                {
                    "id": metadata_id,
                    "product_id": product_id,
                    "user_id": ids["user"],
                    "asset_id": output_id,
                },
            )
        for tag_id, name in ((ids["tag_shared"], "shared"), (ids["tag_distinct"], "distinct")):
            await connection.execute(
                text(
                    """
                    INSERT INTO library_tags (id, product_id, user_id, name)
                    VALUES (:id, :product_id, :user_id, :name)
                    """
                ),
                {"id": tag_id, "product_id": product_id, "user_id": ids["user"], "name": name},
            )
        for tag_id, output_id in (
            (ids["tag_shared"], ids["duplicate_1"]),
            (ids["tag_shared"], ids["duplicate_2"]),
            (ids["tag_distinct"], ids["duplicate_2"]),
        ):
            await connection.execute(
                text(
                    """
                    INSERT INTO library_asset_tags (
                        tag_id, asset_type, asset_id, product_id, user_id
                    ) VALUES (:tag_id, 'output', :asset_id, :product_id, :user_id)
                    """
                ),
                {
                    "tag_id": tag_id,
                    "asset_id": output_id,
                    "product_id": product_id,
                    "user_id": ids["user"],
                },
            )
    await engine.dispose()
    return ids


async def _query_upgrade_result(database_url: str, ids: dict[str, UUID]) -> dict[str, object]:
    engine = create_async_engine(database_url)
    async with engine.connect() as connection:
        output_ids = set(
            (
                await connection.scalars(
                    text(
                        "SELECT id FROM generation_outputs WHERE id = ANY(:ids) ORDER BY id"
                    ).bindparams(ids=[ids["canonical"], ids["duplicate_1"], ids["duplicate_2"]])
                )
            ).all()
        )
        metadata_asset_id = await connection.scalar(
            text("SELECT asset_id FROM library_asset_metadata WHERE id = :id"),
            {"id": ids["metadata_1"]},
        )
        tag_ids = set(
            (
                await connection.scalars(
                    text(
                        """
                        SELECT tag_id FROM library_asset_tags
                        WHERE asset_type = 'output' AND asset_id = :asset_id
                        """
                    ),
                    {"asset_id": ids["canonical"]},
                )
            ).all()
        )
        derivative_job_source = await connection.scalar(
            text("SELECT source_output_id FROM generation_jobs WHERE id = :id"),
            {"id": ids["derivative_job"]},
        )
        derivative_image_source = await connection.scalar(
            text("SELECT source_output_id FROM user_images WHERE id = :id"),
            {"id": ids["derivative_image"]},
        )
        cleanup_rows = (
            await connection.execute(
                text(
                    """
                    SELECT product_id, storage_key FROM storage_cleanup_records
                    WHERE storage_key IN (:duplicate_1, :duplicate_2)
                    ORDER BY storage_key
                    """
                ),
                {
                    "duplicate_1": _storage_key(ids, "duplicate-1"),
                    "duplicate_2": _storage_key(ids, "duplicate-2"),
                },
            )
        ).all()
    await engine.dispose()
    return {
        "output_ids": output_ids,
        "metadata_asset_id": metadata_asset_id,
        "tag_ids": tag_ids,
        "derivative_job_source": derivative_job_source,
        "derivative_image_source": derivative_image_source,
        "cleanup_rows": cleanup_rows,
    }


def test_migration_033_remaps_three_output_duplicates_losslessly(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set-valued tags merge, lineage remaps, and cleanup rows retain product."""
    assert db_engine is not None
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")
    command.downgrade(config, "032")
    try:
        ids = asyncio.run(_seed_duplicate_group(test_database_url))
        command.upgrade(config, "033")
        result = asyncio.run(_query_upgrade_result(test_database_url, ids))

        assert result["output_ids"] == {ids["canonical"]}
        assert result["metadata_asset_id"] == ids["canonical"]
        assert result["tag_ids"] == {ids["tag_shared"], ids["tag_distinct"]}
        assert result["derivative_job_source"] == ids["canonical"]
        assert result["derivative_image_source"] == ids["canonical"]
        assert result["cleanup_rows"] == [
            ("vex", _storage_key(ids, "duplicate-1")),
            ("vex", _storage_key(ids, "duplicate-2")),
        ]
    finally:
        command.upgrade(config, "head")


def test_migration_033_rejects_ambiguous_duplicate_metadata_before_mutation(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two duplicate metadata rows targetting one canonical output abort safely."""
    assert db_engine is not None
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")
    command.downgrade(config, "032")
    ids: dict[str, UUID] | None = None
    try:
        ids = asyncio.run(_seed_duplicate_group(test_database_url, ambiguous_metadata=True))
        with pytest.raises(Exception, match="ambiguous output duplicate metadata"):
            command.upgrade(config, "033")
        # Upgrade rolled back before persistent mutation: both duplicate rows
        # and their metadata are still present in the revision-032 schema.
        engine = create_async_engine(test_database_url)

        async def verify_preflight_rollback() -> tuple[int, int]:
            async with engine.connect() as connection:
                outputs = await connection.scalar(
                    text(
                        """
                        SELECT count(*) FROM generation_outputs
                        WHERE id = ANY(:ids)
                        """
                    ).bindparams(ids=[ids["canonical"], ids["duplicate_1"], ids["duplicate_2"]])
                )
                metadata = await connection.scalar(
                    text("SELECT count(*) FROM library_asset_metadata WHERE id IN (:one, :two)"),
                    {"one": ids["metadata_1"], "two": ids["metadata_2"]},
                )
            await engine.dispose()
            return int(outputs), int(metadata)

        assert asyncio.run(verify_preflight_rollback()) == (3, 2)
    finally:
        if ids is not None:
            # Remove one conflicting metadata row so the cleanup upgrade can
            # restore the shared test database to head.
            async def repair() -> None:
                engine = create_async_engine(test_database_url)
                async with engine.begin() as connection:
                    await connection.execute(
                        text("DELETE FROM library_asset_metadata WHERE id = :id"),
                        {"id": ids["metadata_2"]},
                    )
                await engine.dispose()

            asyncio.run(repair())
        command.upgrade(config, "head")


@pytest.mark.parametrize(
    ("corruption", "error_message"),
    [
        ("metadata", "cross-scope output metadata"),
        ("tag", "cross-scope output tag membership"),
    ],
)
def test_migration_033_rejects_cross_scope_library_references_before_mutation(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    error_message: str,
) -> None:
    """Migration 033 must not remap a library row across product boundaries."""
    assert db_engine is not None
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")
    command.downgrade(config, "032")
    ids: dict[str, UUID] | None = None
    try:
        ids = asyncio.run(_seed_duplicate_group(test_database_url))

        async def introduce_corruption() -> None:
            engine = create_async_engine(test_database_url)
            async with engine.begin() as connection:
                if corruption == "metadata":
                    await connection.execute(
                        text(
                            "UPDATE library_asset_metadata "
                            "SET product_id = 'synthara' WHERE id = :id"
                        ),
                        {"id": ids["metadata_1"]},
                    )
                else:
                    await connection.execute(
                        text(
                            "UPDATE library_asset_tags SET product_id = 'synthara' "
                            "WHERE tag_id = :tag_id AND asset_id = :asset_id"
                        ),
                        {"tag_id": ids["tag_shared"], "asset_id": ids["duplicate_1"]},
                    )
            await engine.dispose()

        asyncio.run(introduce_corruption())
        with pytest.raises(Exception, match=error_message):
            command.upgrade(config, "033")

        async def verify_no_mutation() -> int:
            engine = create_async_engine(test_database_url)
            async with engine.connect() as connection:
                outputs = await connection.scalar(
                    text("SELECT count(*) FROM generation_outputs WHERE job_id = :job_id"),
                    {"job_id": ids["job"]},
                )
            await engine.dispose()
            return int(outputs)

        assert asyncio.run(verify_no_mutation()) == 3
    finally:
        if ids is not None:

            async def repair() -> None:
                engine = create_async_engine(test_database_url)
                async with engine.begin() as connection:
                    if corruption == "metadata":
                        await connection.execute(
                            text(
                                "UPDATE library_asset_metadata SET product_id = 'vex' WHERE id = :id"
                            ),
                            {"id": ids["metadata_1"]},
                        )
                    else:
                        await connection.execute(
                            text(
                                "UPDATE library_asset_tags SET product_id = 'vex' "
                                "WHERE tag_id = :tag_id AND asset_id = :asset_id"
                            ),
                            {"tag_id": ids["tag_shared"], "asset_id": ids["duplicate_1"]},
                        )
                await engine.dispose()

            asyncio.run(repair())
        command.upgrade(config, "head")
