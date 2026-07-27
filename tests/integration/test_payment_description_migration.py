"""Integration test for migration 030's payment-description backfill.

Seeds legacy gateway-named ledger rows against the pre-migration (029)
schema, runs the real ``alembic upgrade``/``downgrade`` commands (round-
tripping the shared session-scoped db_engine, same pattern as
test_payment_provider_state_migration.py), and asserts the backfill
neutralizes descriptions, is idempotent, and is fully reversible.

The test function is intentionally synchronous, like
test_payment_provider_state_migration.py: alembic's ``command.upgrade``/
``command.downgrade`` execute env.py, which calls ``asyncio.run()``
internally — invoking them from a running event loop (i.e. from an
``async def`` test under pytest-asyncio) would raise "asyncio.run() cannot
be called from a running event loop". Async DB access (seed/fetch/cleanup)
is wrapped in its own ``asyncio.run()`` per phase instead; the shared
db_engine uses NullPool, so handing it across separate event loops here is
safe (no connection is held open between phases).

``command.upgrade`` tracks the applied revision in ``alembic_version``, so
calling it a second time with the same target ("030") is a no-op rather than
re-executing ``upgrade()`` — it cannot be used to prove the backfill is
idempotent. Instead we re-run just the UPDATE statement (the same
module-level SQL constant ``upgrade()`` itself calls) directly against the
already-upgraded schema.

``user_images.display_filename`` is added by the following revision (031),
not this one (see 030's docstring) — pinning ``command.upgrade``/
``command.downgrade`` calls in this test to "030" therefore never touches
that column. ``_display_filename_column_exists()`` still asserts its
absence after the final downgrade to "029" purely as a defense-in-depth
check that the two migrations' DDL didn't get re-merged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from alembic.config import Config
from sqlalchemy import text

from alembic import command

if TYPE_CHECKING:
    from types import ModuleType

    from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "030_neutral_payment_labels.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_030", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_030_neutralizes_backfills_idempotently_and_reverses(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    assert db_engine is not None
    migration = _load_migration()
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    nowpayments_user_id = uuid4()
    stripe_user_id = uuid4()
    nowpayments_account_id = uuid4()
    stripe_account_id = uuid4()
    nowpayments_payment_id = uuid4()
    nowpayments_partial_payment_id = uuid4()
    stripe_payment_id = uuid4()
    full_credit_id = uuid4()
    partial_credit_id = uuid4()
    stripe_credit_id = uuid4()
    unrelated_debit_id = uuid4()
    no_payment_credit_id = uuid4()

    async def _seed() -> None:
        async with db_engine.begin() as conn:
            for uid in (nowpayments_user_id, stripe_user_id):
                await conn.execute(
                    text(
                        "INSERT INTO users "
                        "(id, email, password_hash, role, is_active, subscription_tier, "
                        "locale, product_id) "
                        "VALUES (:id, :email, 'x', 'user', true, 'free', 'en', 'vex')"
                    ),
                    {"id": uid, "email": f"payer-{uid}@example.com"},
                )
            for account_id, uid in (
                (nowpayments_account_id, nowpayments_user_id),
                (stripe_account_id, stripe_user_id),
            ):
                await conn.execute(
                    text(
                        "INSERT INTO token_accounts "
                        "(id, account_type, user_id, product_id) "
                        "VALUES (:id, 'personal', :user_id, 'vex')"
                    ),
                    {"id": account_id, "user_id": uid},
                )
            for payment_id, account_id, provider, uid in (
                (
                    nowpayments_payment_id,
                    nowpayments_account_id,
                    "nowpayments",
                    nowpayments_user_id,
                ),
                (
                    nowpayments_partial_payment_id,
                    nowpayments_account_id,
                    "nowpayments",
                    nowpayments_user_id,
                ),
                (stripe_payment_id, stripe_account_id, "stripe", stripe_user_id),
            ):
                await conn.execute(
                    text(
                        "INSERT INTO payments "
                        "(id, account_id, payment_provider, external_id, status, "
                        "amount_usd, tokens_granted, product_id, created_by) "
                        "VALUES (:id, :account_id, :provider, :external_id, 'completed', "
                        "10.00, 1000, 'vex', :created_by)"
                    ),
                    {
                        "id": payment_id,
                        "account_id": account_id,
                        "provider": provider,
                        "external_id": f"ext-{payment_id}",
                        "created_by": uid,
                    },
                )

            # Full nowpayments credit — legacy gateway-named description.
            await conn.execute(
                text(
                    "INSERT INTO token_transactions "
                    "(id, account_id, transaction_type, amount, balance_after, "
                    "payment_id, description, product_id) "
                    "VALUES (:id, :account_id, 'credit', 1000, 1000, :payment_id, "
                    "'Token purchase via nowpayments', 'vex')"
                ),
                {
                    "id": full_credit_id,
                    "account_id": nowpayments_account_id,
                    "payment_id": nowpayments_payment_id,
                },
            )
            # Partial nowpayments credit — legacy description with the "(partial)" suffix.
            await conn.execute(
                text(
                    "INSERT INTO token_transactions "
                    "(id, account_id, transaction_type, amount, balance_after, "
                    "payment_id, description, product_id) "
                    "VALUES (:id, :account_id, 'credit', 400, 400, :payment_id, "
                    "'Token purchase via nowpayments (partial)', 'vex')"
                ),
                {
                    "id": partial_credit_id,
                    "account_id": nowpayments_account_id,
                    "payment_id": nowpayments_partial_payment_id,
                },
            )
            # Full stripe credit — legacy gateway-named description.
            await conn.execute(
                text(
                    "INSERT INTO token_transactions "
                    "(id, account_id, transaction_type, amount, balance_after, "
                    "payment_id, description, product_id) "
                    "VALUES (:id, :account_id, 'credit', 1000, 2000, :payment_id, "
                    "'Token purchase via stripe', 'vex')"
                ),
                {
                    "id": stripe_credit_id,
                    "account_id": stripe_account_id,
                    "payment_id": stripe_payment_id,
                },
            )
            # A non-credit transaction with a matching description shape — must be untouched.
            await conn.execute(
                text(
                    "INSERT INTO token_transactions "
                    "(id, account_id, transaction_type, amount, balance_after, "
                    "payment_id, description, product_id) "
                    "VALUES (:id, :account_id, 'debit', -50, 1950, :payment_id, "
                    "'Token purchase via stripe', 'vex')"
                ),
                {
                    "id": unrelated_debit_id,
                    "account_id": stripe_account_id,
                    "payment_id": stripe_payment_id,
                },
            )
            # A credit with payment_id IS NULL — must be untouched (join can't match).
            await conn.execute(
                text(
                    "INSERT INTO token_transactions "
                    "(id, account_id, transaction_type, amount, balance_after, "
                    "description, product_id) "
                    "VALUES (:id, :account_id, 'credit', 100, 2050, "
                    "'Token purchase via stripe', 'vex')"
                ),
                {"id": no_payment_credit_id, "account_id": stripe_account_id},
            )

    async def _fetch_all() -> dict[str, tuple[str, dict[str, object]]]:
        # asyncpg returns jsonb as raw text via a bare text() query — cast to
        # ::text explicitly and json.loads() rather than rely on implicit
        # driver-level decoding.
        ids = {
            "full": full_credit_id,
            "partial": partial_credit_id,
            "stripe": stripe_credit_id,
            "debit": unrelated_debit_id,
            "no_payment": no_payment_credit_id,
        }
        out: dict[str, tuple[str, dict[str, object]]] = {}
        async with db_engine.connect() as conn:
            for key, txn_id in ids.items():
                result = await conn.execute(
                    text(
                        "SELECT description, metadata::text FROM token_transactions WHERE id = :id"
                    ),
                    {"id": txn_id},
                )
                row = result.one()
                out[key] = (row[0], json.loads(row[1]))
        return out

    async def _rerun_neutralize() -> None:
        async with db_engine.begin() as conn:
            await conn.execute(text(migration.DISABLE_IMMUTABILITY_TRIGGER_SQL))
            await conn.execute(text(migration.NEUTRALIZE_DESCRIPTIONS_SQL))
            await conn.execute(text(migration.ENABLE_IMMUTABILITY_TRIGGER_SQL))

    async def _display_filename_column_exists() -> bool:
        async with db_engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'user_images' AND column_name = 'display_filename'"
                )
            )
            return result.first() is not None

    async def _cleanup() -> None:
        async with db_engine.begin() as conn:
            # DELETE is blocked by the same immutability trigger as UPDATE.
            await conn.execute(text(migration.DISABLE_IMMUTABILITY_TRIGGER_SQL))
            await conn.execute(
                text("DELETE FROM token_transactions WHERE id = ANY(:ids)"),
                {
                    "ids": [
                        full_credit_id,
                        partial_credit_id,
                        stripe_credit_id,
                        unrelated_debit_id,
                        no_payment_credit_id,
                    ]
                },
            )
            await conn.execute(text(migration.ENABLE_IMMUTABILITY_TRIGGER_SQL))
            await conn.execute(
                text("DELETE FROM payments WHERE id = ANY(:ids)"),
                {
                    "ids": [
                        nowpayments_payment_id,
                        nowpayments_partial_payment_id,
                        stripe_payment_id,
                    ]
                },
            )
            await conn.execute(
                text("DELETE FROM token_accounts WHERE id = ANY(:ids)"),
                {"ids": [nowpayments_account_id, stripe_account_id]},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = ANY(:ids)"),
                {"ids": [nowpayments_user_id, stripe_user_id]},
            )

    try:
        # Start from the pre-migration schema, then seed legacy-shaped rows.
        command.downgrade(config, "029")
        asyncio.run(_seed())

        # Run the migration under test.
        command.upgrade(config, "030")

        rows = asyncio.run(_fetch_all())
        description, metadata = rows["full"]
        assert description == "Token purchase via crypto payment"
        assert metadata["payment_method"] == "crypto"

        description, metadata = rows["partial"]
        assert description == "Token purchase via crypto payment (partial)"
        assert metadata["payment_method"] == "crypto"

        description, metadata = rows["stripe"]
        assert description == "Token purchase via card payment"
        assert metadata["payment_method"] == "card"

        description, _metadata = rows["debit"]
        assert description == "Token purchase via stripe"

        description, _metadata = rows["no_payment"]
        assert description == "Token purchase via stripe"

        # Idempotence: re-running just the UPDATE (no DDL) changes nothing further.
        asyncio.run(_rerun_neutralize())
        rows = asyncio.run(_fetch_all())
        description, metadata = rows["full"]
        assert description == "Token purchase via crypto payment"
        assert metadata["payment_method"] == "crypto"
        description, metadata = rows["partial"]
        assert description == "Token purchase via crypto payment (partial)"
        assert metadata["payment_method"] == "crypto"

        # Downgrade restores the original gateway-named descriptions and drops the column.
        command.downgrade(config, "029")
        rows = asyncio.run(_fetch_all())

        description, metadata = rows["full"]
        assert description == "Token purchase via nowpayments"
        assert "payment_method" not in metadata

        description, metadata = rows["partial"]
        assert description == "Token purchase via nowpayments (partial)"
        assert "payment_method" not in metadata

        description, metadata = rows["stripe"]
        assert description == "Token purchase via stripe"
        assert "payment_method" not in metadata

        assert not asyncio.run(_display_filename_column_exists())
    finally:
        command.upgrade(config, "head")
        asyncio.run(_cleanup())
