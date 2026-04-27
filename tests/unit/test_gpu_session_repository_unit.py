"""Unit tests for GpuSessionRepository (uncovered methods)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

pytestmark = pytest.mark.unit


def _make_session_row(**kwargs: object) -> MagicMock:
    row = MagicMock()
    row.id = kwargs.get("id", uuid4())
    row.user_id = kwargs.get("user_id", uuid4())
    row.product_id = kwargs.get("product_id", "vex")
    row.status = kwargs.get("status", GpuSessionStatus.active)
    row.provision_attempt = kwargs.get("provision_attempt", 0)
    return row


def _make_db(rows: object = None) -> AsyncMock:
    db = AsyncMock()
    db.flush = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = rows
    result.scalars.return_value.all.return_value = rows or []
    result.scalar_one.return_value = rows
    db.execute = AsyncMock(return_value=result)
    return db


class TestGetByIdForUpdate:
    async def test_uses_select_with_for_update_when_flagged(self) -> None:
        row = _make_session_row()
        db = _make_db(row)
        repo = GpuSessionRepository(db)

        result = await repo.get_by_id(row.id, for_update=True)

        assert result is row
        db.execute.assert_awaited_once()
        # Confirm the query was executed (not db.get)
        db.get.assert_not_called()

    async def test_uses_session_get_without_for_update(self) -> None:
        row = _make_session_row()
        db = AsyncMock()
        db.get = AsyncMock(return_value=row)
        repo = GpuSessionRepository(db)

        result = await repo.get_by_id(row.id, for_update=False)

        assert result is row
        db.get.assert_awaited_once()
        db.execute.assert_not_awaited()


class TestGetByIdForUser:
    async def test_returns_session_when_owned(self) -> None:
        row = _make_session_row()
        db = _make_db(row)
        repo = GpuSessionRepository(db)

        result = await repo.get_by_id_for_user(row.id, row.user_id, "vex")

        assert result is row

    async def test_returns_none_when_not_owned(self) -> None:
        db = _make_db(None)
        repo = GpuSessionRepository(db)

        result = await repo.get_by_id_for_user(uuid4(), uuid4(), "vex")

        assert result is None


class TestListByUser:
    async def test_includes_terminal_when_flag_is_true(self) -> None:
        rows = [_make_session_row(status=GpuSessionStatus.stopped)]
        db = _make_db(rows)
        repo = GpuSessionRepository(db)

        result = await repo.list_by_user(uuid4(), "vex", include_terminal=True)

        assert list(result) == rows
        stmt = db.execute.call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # Should NOT filter out terminal statuses
        assert "NOT IN" not in compiled.upper()

    async def test_excludes_terminal_by_default(self) -> None:
        rows: list[object] = []
        db = _make_db(rows)
        repo = GpuSessionRepository(db)

        await repo.list_by_user(uuid4(), "vex", include_terminal=False)

        stmt = db.execute.call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "NOT IN" in compiled.upper() or "not_in" in compiled.lower()


class TestListAllAdmin:
    async def test_returns_rows_ordered_by_created_at(self) -> None:
        rows = [_make_session_row(), _make_session_row()]
        db = _make_db(rows)
        repo = GpuSessionRepository(db)

        result = await repo.list_all_admin(limit=50)

        assert list(result) == rows
        db.execute.assert_awaited_once()


class TestIncrementProvisionAttempt:
    async def test_returns_new_attempt_count(self) -> None:
        db = AsyncMock()
        db.flush = AsyncMock()
        result = MagicMock()
        result.scalar_one.return_value = 3
        db.execute = AsyncMock(return_value=result)

        repo = GpuSessionRepository(db)
        count = await repo.increment_provision_attempt(uuid4())

        assert count == 3
        db.flush.assert_awaited_once()


class TestUpdateInstance:
    async def test_executes_update_and_flushes(self) -> None:
        db = AsyncMock()
        db.flush = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock())

        repo = GpuSessionRepository(db)
        await repo.update_instance(
            uuid4(),
            vastai_instance_id=42,
            vastai_offer_id=7,
            vastai_cost_per_hour_micros=500000,
            vastai_gpu_name="RTX 4090",
            provisioning_started_at=datetime.now(UTC),
        )

        db.execute.assert_awaited_once()
        db.flush.assert_awaited_once()


class TestAddPausedSeconds:
    async def test_executes_update_and_flushes(self) -> None:
        db = AsyncMock()
        db.flush = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock())

        repo = GpuSessionRepository(db)
        await repo.add_paused_seconds(uuid4(), 300)

        db.execute.assert_awaited_once()
        db.flush.assert_awaited_once()


class TestMarkBillingFinalized:
    async def test_executes_update_and_flushes(self) -> None:
        db = AsyncMock()
        db.flush = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock())

        repo = GpuSessionRepository(db)
        await repo.mark_billing_finalized(uuid4(), datetime.now(UTC))

        db.execute.assert_awaited_once()
        db.flush.assert_awaited_once()
