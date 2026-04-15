"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
from logging.config import fileConfig

from alembic.operations.ops import AlterColumnOp, MigrateOperation, MigrationScript, ModifyTableOps
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.db.models import Base

# Alembic Config object for access to .ini values
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = Base.metadata

if _db_url := os.environ.get("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _is_comment_only_alter(op: AlterColumnOp) -> bool:
    """Return True if an AlterColumnOp changes nothing except the comment."""
    return (
        op.modify_comment is not False
        and op.modify_nullable is None
        and op.modify_server_default is False
        and op.modify_name is None
        and op.modify_type is None
    )


def _filter_ops(ops_list: list[MigrateOperation]) -> list[MigrateOperation]:
    """Strip comment-only AlterColumnOps; drop empty ModifyTableOps."""
    filtered: list[MigrateOperation] = []
    for op in ops_list:
        if isinstance(op, ModifyTableOps):
            op.ops = [
                t
                for t in op.ops
                if not (isinstance(t, AlterColumnOp) and _is_comment_only_alter(t))
            ]
            if op.ops:
                filtered.append(op)
        else:
            filtered.append(op)
    return filtered


def _drop_comment_only_ops(
    context: object,  # noqa: ARG001
    revision: object,  # noqa: ARG001
    directives: list[MigrationScript],
) -> None:
    """Remove AlterColumnOp entries that only change column comments.

    Column comments in migrations are documentation aids and are not
    reflected in SQLAlchemy models, so they would always show up as
    false-positive drift. This hook suppresses them from autogenerate.
    """
    for script in directives:
        for upgrade_ops in script.upgrade_ops_list:
            upgrade_ops.ops = _filter_ops(upgrade_ops.ops)
        for downgrade_ops in script.downgrade_ops_list:
            downgrade_ops.ops = _filter_ops(downgrade_ops.ops)


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        process_revision_directives=_drop_comment_only_ops,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode with actual database connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
