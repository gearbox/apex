"""Migration 060: the artist chat tables and the LLM usage ledger (invariant 18, R2).

The upgrade creates ``studio_artist_chats``, ``studio_artist_chat_messages``,
``studio_memory_drafts`` and ``studio_llm_usage`` with their CHECKs, composite owner FKs and
indexes; the downgrade drops the four tables in reverse order and touches nothing else. The usage
ledger's only foreign key is to ``users`` (nothing a studio delete reaches), and the upgrade
backfills the runs of the last hour into it. After ``downgrade 059 && upgrade head`` the models and
the migrations do not drift (the ``alembic check`` of the PR checklist, run here through the same
comparison). The migration never imports application code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.core.enums import StudioLlmUsageKind
from src.db.models.base import Base

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

    from tests.studio_runs_support import RunWorld

_PATH = Path(__file__).resolve().parents[2] / "alembic" / "versions" / "060_studio_artist_chat.py"
_TABLES = (
    "studio_artist_chats",
    "studio_artist_chat_messages",
    "studio_memory_drafts",
    "studio_llm_usage",
)
_CHECKS = {
    "studio_artist_chats": {
        "ck_studio_artist_chats_message_count",
        "ck_studio_artist_chats_next_position",
    },
    "studio_artist_chat_messages": {
        "ck_studio_chat_messages_role",
        "ck_studio_chat_messages_status",
        "ck_studio_chat_messages_user_complete",
        "ck_studio_chat_messages_client_id_user",
        "ck_studio_chat_messages_price_artist",
        "ck_studio_chat_messages_price",
        "ck_studio_chat_messages_error_code",
        "ck_studio_chat_messages_finished",
        "ck_studio_chat_messages_position",
        "ck_studio_chat_messages_revision",
    },
    "studio_memory_drafts": {
        "ck_studio_drafts_status",
        "ck_studio_drafts_draft",
        "ck_studio_drafts_error_code",
        "ck_studio_drafts_finished",
        "ck_studio_drafts_price",
        "ck_studio_drafts_revision",
    },
    "studio_llm_usage": {"ck_studio_llm_usage_kind"},
}
_OWNER_FKS = {
    "studio_artist_chats": {
        "fk_studio_artist_chats_artist_owner": (
            "FOREIGN KEY (artist_id, user_id, product_id)",
            "REFERENCES studio_artists(id, user_id, product_id)",
        )
    },
    "studio_artist_chat_messages": {
        "fk_studio_artist_chat_messages_chat_owner": (
            "FOREIGN KEY (chat_id, user_id, product_id)",
            "REFERENCES studio_artist_chats(id, user_id, product_id)",
        )
    },
    "studio_memory_drafts": {
        "fk_studio_memory_drafts_message_owner": (
            "FOREIGN KEY (source_message_id, user_id, product_id)",
            "REFERENCES studio_artist_chat_messages(id, user_id, product_id)",
        ),
        "fk_studio_memory_drafts_chat_owner": (
            "FOREIGN KEY (chat_id, user_id, product_id)",
            "REFERENCES studio_artist_chats(id, user_id, product_id)",
        ),
    },
    "studio_llm_usage": {},  # no owner FK: its one foreign key is to ``users``
}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_060", _PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Connection, action: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_load(), action)()


async def _exists(connection: AsyncConnection, table: str) -> bool:
    return (
        await connection.execute(text("SELECT to_regclass(:name) IS NOT NULL"), {"name": table})
    ).scalar_one()


async def _checks(connection: AsyncConnection, table: str) -> set[str]:
    return {
        row[0]
        for row in await connection.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = to_regclass(:table) AND contype = 'c'"
            ),
            {"table": table},
        )
    }


async def _index_definition(connection: AsyncConnection, name: str) -> str:
    return (
        await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": name}
        )
    ).scalar_one()


def test_revision_chain() -> None:
    migration = _load()
    assert migration.revision == "060"
    assert migration.down_revision == "059"


async def test_the_tables_have_their_checks_and_cascading_owner_foreign_keys(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        for table in _TABLES:
            assert await _exists(connection, table), table
            assert await _checks(connection, table) == _CHECKS[table], table
            foreign_keys = {
                row[0]: (row[1], row[2])
                for row in await connection.execute(
                    text(
                        "SELECT conname, confdeltype::text, pg_get_constraintdef(oid) "
                        "FROM pg_constraint WHERE conrelid = to_regclass(:table) AND contype = 'f'"
                    ),
                    {"table": table},
                )
            }
            # every FK of a chat table is ON DELETE CASCADE: the user, and the parent with the
            # owner pair (the usage ledger has the user only)
            assert {on_delete for on_delete, _ in foreign_keys.values()} == {"c"}, table
            for name, parts in _OWNER_FKS[table].items():
                assert name in foreign_keys, name
                for part in parts:
                    assert part in foreign_keys[name][1], (name, part)


async def test_the_composite_fk_targets_are_unique_keys(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        names = {
            row[0]
            for row in await connection.execute(
                text("SELECT conname FROM pg_constraint WHERE contype = 'u'")
            )
        }
    assert {
        "uq_studio_artists_id_owner",  # from 059
        "uq_studio_artist_chats_id_owner",
        "uq_studio_artist_chat_messages_id_owner",
        "uq_studio_chat_messages_position",
        "uq_studio_chat_messages_client_id",
        "uq_studio_memory_drafts_user_idempotency",
    } <= names


async def test_the_one_pending_reply_index_is_partial_and_unique(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        definition = await _index_definition(connection, "uq_studio_chat_one_pending_reply")
        draft_definition = await _index_definition(
            connection, "uq_studio_memory_drafts_one_pending"
        )
    assert definition.startswith("CREATE UNIQUE INDEX")
    assert "(chat_id)" in definition
    assert "role)::text = 'artist'" in definition
    assert "status)::text = 'pending'" in definition
    assert draft_definition.startswith("CREATE UNIQUE INDEX")
    assert "(source_message_id)" in draft_definition
    assert "status)::text = 'pending'" in draft_definition


async def test_the_indexes_serve_the_documented_queries(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        chats = await _index_definition(connection, "ix_studio_artist_chats_list")
        position = await _index_definition(connection, "ix_studio_chat_messages_chat_position")
        sweep = await _index_definition(connection, "ix_studio_chat_messages_pending")
        draft_sweep = await _index_definition(connection, "ix_studio_memory_drafts_pending")
        message_user = await _index_definition(connection, "ix_studio_artist_chat_messages_user_id")
        draft_user = await _index_definition(connection, "ix_studio_memory_drafts_user_id")
        existing = {
            row[0]
            for row in await connection.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE indexname IN ('ix_studio_chat_messages_user_created', "
                    "'ix_studio_memory_drafts_user_created')"
                )
            )
        }
    assert chats.endswith("(artist_id, updated_at DESC, id DESC)")
    assert position.endswith(('(chat_id, "position" DESC)', "(chat_id, position DESC)"))
    assert "(created_at)" in sweep
    assert "status)::text = 'pending'" in sweep
    assert "status)::text = 'pending'" in draft_sweep
    # The hourly limit no longer reads these tables (it counts the usage ledger), so the indexes
    # that existed for it are gone; the users ON DELETE CASCADE keeps a plain user_id index.
    assert existing == set()
    assert message_user.endswith("USING btree (user_id)")
    assert draft_user.endswith("USING btree (user_id)")


async def test_downgrade_drops_the_four_tables_and_nothing_else_and_upgrade_restores_them(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        await connection.run_sync(_run, "downgrade")
        for table in _TABLES:
            assert not await _exists(connection, table), table
        # what 059 made (and what the ledger's rows were backfilled from) is still there
        assert await _exists(connection, "studio_artist_memories")
        assert await _exists(connection, "studio_artists")
        assert await _exists(connection, "studio_runs")

        await connection.run_sync(_run, "upgrade")
        for table in _TABLES:
            assert await _exists(connection, table), table
            assert await _checks(connection, table) == _CHECKS[table], table
        await transaction.rollback()


def _drift(connection: Connection) -> list[str]:
    """What ``alembic check`` would report for the tables this migration touches."""
    diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    flat = [str(item) for diff in diffs for item in (diff if isinstance(diff, list) else [diff])]
    return [
        item for item in flat if "chat" in item or "memory_draft" in item or "llm_usage" in item
    ]


async def test_after_downgrade_and_upgrade_the_models_and_the_migration_do_not_drift(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert await connection.run_sync(_drift) == []
        await transaction.rollback()


async def _seed(connection: AsyncConnection) -> tuple[object, object, object]:
    user_id, artist_id, chat_id = uuid4(), uuid4(), uuid4()
    await connection.execute(
        text(
            "INSERT INTO users (id, email, password_hash, product_id) "
            "VALUES (:id, :email, 'x', 'vex')"
        ),
        {"id": user_id, "email": f"mig-{uuid4().hex}@example.com"},
    )
    await connection.execute(
        text(
            "INSERT INTO studio_artists (id, user_id, product_id, name) "
            "VALUES (:id, :user, 'vex', 'A')"
        ),
        {"id": artist_id, "user": user_id},
    )
    await connection.execute(
        text(
            "INSERT INTO studio_artist_chats (id, user_id, product_id, artist_id, title) "
            "VALUES (:id, :user, 'vex', :artist, 't')"
        ),
        {"id": chat_id, "user": user_id, "artist": artist_id},
    )
    return user_id, artist_id, chat_id


_USER = (
    "INSERT INTO studio_artist_chat_messages (id, user_id, product_id, chat_id, position, role, "
    "status, client_message_id, finished_at) VALUES "
    "(:id, :user, 'vex', :chat, :pos, :role, :status, :cid, {finished})"
)


@pytest.mark.parametrize(
    ("overrides", "finished"),
    [
        ({"role": "robot"}, "now()"),
        ({"status": "deleted"}, "now()"),
        ({"status": "pending"}, "NULL"),  # a user message is born complete
        ({"pos": 0}, "now()"),
        ({"cid": None}, "now()"),  # a user message has a client id
        ({}, "NULL"),  # a terminal row has finished_at
    ],
    ids=["role", "status", "user-pending", "position", "client-id", "finished"],
)
async def test_the_message_checks_hold_in_the_database(
    db_engine: AsyncEngine, overrides: dict[str, object], finished: str
) -> None:
    """The CHECKs are the backstop behind the service: a bad row cannot be written at all."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        user_id, _, chat_id = await _seed(connection)
        values: dict[str, object] = {
            "id": uuid4(),
            "user": user_id,
            "chat": chat_id,
            "pos": 1,
            "role": "user",
            "status": "complete",
            "cid": uuid4(),
        }
        values.update(overrides)
        with pytest.raises(IntegrityError):
            await connection.execute(text(_USER.format(finished=finished)), values)
        await transaction.rollback()


async def test_a_valid_pair_and_a_valid_draft_are_accepted_and_cascade_with_the_chat(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        user_id, _, chat_id = await _seed(connection)
        user_message, reply = uuid4(), uuid4()
        await connection.execute(
            text(_USER.format(finished="now()")),
            {
                "id": user_message,
                "user": user_id,
                "chat": chat_id,
                "pos": 1,
                "role": "user",
                "status": "complete",
                "cid": uuid4(),
            },
        )
        await connection.execute(
            text(
                "INSERT INTO studio_artist_chat_messages (id, user_id, product_id, chat_id, "
                "position, role, status, price_tokens) VALUES "
                "(:id, :user, 'vex', :chat, 2, 'artist', 'pending', 1)"
            ),
            {"id": reply, "user": user_id, "chat": chat_id},
        )
        # a second pending reply in the same chat is refused by the partial unique index
        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(
                    text(
                        "INSERT INTO studio_artist_chat_messages (id, user_id, product_id, "
                        "chat_id, position, role, status, price_tokens) VALUES "
                        "(:id, :user, 'vex', :chat, 3, 'artist', 'pending', 1)"
                    ),
                    {"id": uuid4(), "user": user_id, "chat": chat_id},
                )
        draft_id = uuid4()
        await connection.execute(
            text(
                "INSERT INTO studio_memory_drafts (id, user_id, product_id, chat_id, "
                "source_message_id, status, idempotency_key, price_tokens) VALUES "
                "(:id, :user, 'vex', :chat, :msg, 'pending', 'k', 1)"
            ),
            {"id": draft_id, "user": user_id, "chat": chat_id, "msg": user_message},
        )
        # a draft is "complete" exactly when it has its content
        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(
                    text(
                        "UPDATE studio_memory_drafts SET status = 'complete', finished_at = now() "
                        "WHERE id = :id"
                    ),
                    {"id": draft_id},
                )

        await connection.execute(
            text("DELETE FROM studio_artist_chats WHERE id = :id"), {"id": chat_id}
        )
        for table in ("studio_artist_chat_messages", "studio_memory_drafts"):
            remaining = (
                await connection.execute(
                    text(f"SELECT count(*) FROM {table} WHERE chat_id = :id"),  # noqa: S608
                    {"id": chat_id},
                )
            ).scalar_one()
            assert remaining == 0, table
        await transaction.rollback()


# ---------------------------------------------------------------------------------------------
# the usage ledger (R2)
# ---------------------------------------------------------------------------------------------


async def test_the_usage_ledger_has_its_check_its_window_index_and_only_the_users_foreign_key(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        check = (
            await connection.execute(
                text(
                    "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE conname = 'ck_studio_llm_usage_kind'"
                )
            )
        ).scalar_one()
        window = await _index_definition(connection, "ix_studio_llm_usage_window")
        references = {
            row[0]
            for row in await connection.execute(
                text(
                    "SELECT confrelid::regclass::text FROM pg_constraint "
                    "WHERE conrelid = 'studio_llm_usage'::regclass AND contype = 'f'"
                )
            )
        }
        columns = {
            row[0]: (row[1], row[2])
            for row in await connection.execute(
                text(
                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'studio_llm_usage'"
                )
            )
        }
    for kind in StudioLlmUsageKind:  # generated from the enum, like every other CHECK
        assert f"'{kind.value}'" in check
    assert window.endswith("USING btree (user_id, product_id, kind, created_at)")
    # the point of the table: no chat, message, draft, track or run can reach it
    assert references == {"users"}
    assert columns == {
        "id": ("uuid", "NO"),
        "user_id": ("uuid", "NO"),
        "product_id": ("character varying", "NO"),
        "kind": ("character varying", "NO"),
        "created_at": ("timestamp with time zone", "NO"),
    }


async def test_an_unknown_usage_kind_is_refused_and_deleting_the_user_removes_the_rows(
    db_engine: AsyncEngine,
) -> None:
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        user_id, _, _ = await _seed(connection)
        insert = text(
            "INSERT INTO studio_llm_usage (id, user_id, product_id, kind) "
            "VALUES (:id, :user, 'vex', :kind)"
        )
        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(
                    insert, {"id": uuid4(), "user": user_id, "kind": "image_generation"}
                )
        await connection.execute(insert, {"id": uuid4(), "user": user_id, "kind": "chat_message"})
        await connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        remaining = (
            await connection.execute(
                text("SELECT count(*) FROM studio_llm_usage WHERE user_id = :id"), {"id": user_id}
            )
        ).scalar_one()
        assert remaining == 0
        await transaction.rollback()


async def test_the_upgrade_backfills_the_runs_of_the_last_hour_and_only_those(
    run_world: RunWorld, db_engine: AsyncEngine
) -> None:
    """A deploy inside the hour must not hand out a fresh run allowance (R2).

    One run is recent, one is two hours old (the clock is moved with SQL), one belongs to a
    bystander: after ``downgrade && upgrade`` the ledger holds exactly the two last-hour runs, as
    ``run`` rows carrying the run's own ``created_at``.
    """
    actor = await run_world.add_actor()
    other = await run_world.add_actor()
    tracks = [await run_world.new_track(actor, f"T{i}") for i in range(2)]
    recent = await run_world.submit(actor, tracks[0])
    old = await run_world.submit(actor, tracks[1])
    bystander = await run_world.submit(other, await run_world.new_track(other))
    async with run_world.factory() as session, session.begin():
        await session.execute(
            text("UPDATE studio_runs SET created_at = now() - interval '2 hours' WHERE id = :id"),
            {"id": old.run.id},
        )

    async def backfilled(connection: AsyncConnection, user_id: UUID) -> list[tuple[str, object]]:
        rows = await connection.execute(
            text(
                "SELECT kind, created_at FROM studio_llm_usage WHERE user_id = :id "
                "ORDER BY created_at"
            ),
            {"id": user_id},
        )
        return [(row[0], row[1]) for row in rows]

    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        await connection.run_sync(_run, "downgrade")
        await connection.run_sync(_run, "upgrade")
        assert await backfilled(connection, actor.user_id) == [
            ("run", (await run_world.run(recent.run.id)).created_at)
        ]
        assert await backfilled(connection, other.user_id) == [
            ("run", (await run_world.run(bystander.run.id)).created_at)
        ]
        await transaction.rollback()
