"""Unit tests for AdminRepository."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.db.repositories.admin import AdminRepository

pytestmark = pytest.mark.unit


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    result.scalar_one_or_none.return_value = None
    result.all.return_value = []
    result.rowcount = 1
    session.execute.return_value = result
    return session


class TestAdminRepository:
    async def test_grant_permission(self) -> None:
        session = _make_session()
        repo = AdminRepository(session)
        grant = MagicMock()
        result = await repo.grant_permission(grant)
        session.add.assert_called_once_with(grant)
        session.flush.assert_awaited_once()
        assert result is grant

    async def test_revoke_permission_returns_true(self) -> None:
        session = _make_session()
        session.execute.return_value.rowcount = 1
        repo = AdminRepository(session)
        result = await repo.revoke_permission(uuid4(), "read_users", "vex")
        assert result is True

    async def test_revoke_permission_returns_false(self) -> None:
        session = _make_session()
        session.execute.return_value.rowcount = 0
        repo = AdminRepository(session)
        result = await repo.revoke_permission(uuid4(), "read_users", "vex")
        assert result is False

    async def test_get_permissions(self) -> None:
        session = _make_session()
        grant = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = [grant]
        repo = AdminRepository(session)
        result = await repo.get_permissions(uuid4(), "vex")
        assert list(result) == [grant]

    async def test_get_permissions_batch_empty(self) -> None:
        session = _make_session()
        repo = AdminRepository(session)
        result = await repo.get_permissions_batch([], "vex")
        assert result == {}
        session.execute.assert_not_called()

    async def test_get_permissions_batch_with_users(self) -> None:
        user1 = uuid4()
        user2 = uuid4()
        session = _make_session()
        session.execute.return_value.all.return_value = [
            (user1, "read_users"),
            (user1, "manage_billing"),
            (user2, "read_users"),
        ]
        repo = AdminRepository(session)
        result = await repo.get_permissions_batch([user1, user2], "vex")
        assert sorted(result[user1]) == sorted(["read_users", "manage_billing"])
        assert result[user2] == ["read_users"]

    async def test_has_permission_true(self) -> None:
        session = _make_session()
        session.execute.return_value.scalar_one_or_none.return_value = uuid4()
        repo = AdminRepository(session)
        result = await repo.has_permission(uuid4(), "read_users", "vex")
        assert result is True

    async def test_has_permission_false(self) -> None:
        session = _make_session()
        session.execute.return_value.scalar_one_or_none.return_value = None
        repo = AdminRepository(session)
        result = await repo.has_permission(uuid4(), "read_users", "vex")
        assert result is False

    async def test_delete_all_permissions(self) -> None:
        session = _make_session()
        session.execute.return_value.rowcount = 3
        repo = AdminRepository(session)
        result = await repo.delete_all_permissions(uuid4(), "vex")
        assert result == 3

    async def test_write_audit(self) -> None:
        session = _make_session()
        repo = AdminRepository(session)
        entry = MagicMock()
        result = await repo.write_audit(entry)
        session.add.assert_called_once_with(entry)
        session.flush.assert_awaited_once()
        assert result is entry

    async def test_get_audit_log_no_filter(self) -> None:
        session = _make_session()
        entry = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = [entry]
        repo = AdminRepository(session)
        result = await repo.get_audit_log("vex")
        assert list(result) == [entry]

    async def test_get_audit_log_with_target_user(self) -> None:
        session = _make_session()
        session.execute.return_value.scalars.return_value.all.return_value = []
        repo = AdminRepository(session)
        result = await repo.get_audit_log("vex", target_user_id=uuid4())
        assert not list(result)
