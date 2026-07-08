"""Unit tests for PushSubscriptionRepository query methods without PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.push_subscription import PushSubscriptionRepository


class FakeResult:
    def __init__(
        self,
        *,
        scalar: object | None = None,
        rows: Sequence[object] | None = None,
        rowcount: int = 0,
    ) -> None:
        self._scalar = scalar
        self._rows = list(rows or [])
        self.rowcount = rowcount

    def scalar_one_or_none(self) -> object | None:
        return self._scalar

    def scalar_one(self) -> object:
        assert self._scalar is not None
        return self._scalar

    def scalars(self) -> FakeResult:
        return self

    def all(self) -> list[object]:
        return self._rows


def _repo_with_result(result: FakeResult) -> tuple[PushSubscriptionRepository, Any]:
    session = MagicMock(spec=AsyncSession)
    cast("Any", session).execute = AsyncMock(return_value=result)
    cast("Any", session).flush = AsyncMock()
    return PushSubscriptionRepository(cast("AsyncSession", session)), session


async def test_upsert_executes_atomic_statement_and_flushes() -> None:
    subscription = MagicMock()
    repo, session = _repo_with_result(FakeResult(scalar=subscription))
    user_id = uuid4()

    result = await repo.upsert(
        user_id=user_id,
        product_id="vex",
        endpoint="https://push.example/endpoint",
        p256dh="p",
        auth="a",
        user_agent="agent",
    )

    assert result is subscription
    cast("Any", session).execute.assert_awaited_once()
    cast("Any", session).flush.assert_awaited_once()


async def test_get_by_endpoint_returns_matching_row() -> None:
    subscription = MagicMock()
    repo, session = _repo_with_result(FakeResult(scalar=subscription))

    result = await repo.get_by_endpoint("https://push.example/endpoint")

    assert result is subscription
    cast("Any", session).execute.assert_awaited_once()


async def test_delete_by_endpoint_returns_rowcount_bool_and_flushes() -> None:
    repo, session = _repo_with_result(FakeResult(rowcount=1))

    deleted = await repo.delete_by_endpoint("https://push.example/endpoint", user_id=uuid4())

    assert deleted is True
    cast("Any", session).execute.assert_awaited_once()
    cast("Any", session).flush.assert_awaited_once()


async def test_delete_by_id_executes_delete_and_flushes() -> None:
    repo, session = _repo_with_result(FakeResult())

    await repo.delete_by_id(uuid4())

    cast("Any", session).execute.assert_awaited_once()
    cast("Any", session).flush.assert_awaited_once()


async def test_list_by_user_returns_scalars() -> None:
    rows = [MagicMock(), MagicMock()]
    repo, session = _repo_with_result(FakeResult(rows=rows))

    result = await repo.list_by_user(uuid4())

    assert result == rows
    cast("Any", session).execute.assert_awaited_once()


async def test_list_batch_uses_cursor_when_provided() -> None:
    rows = [MagicMock()]
    repo, session = _repo_with_result(FakeResult(rows=rows))

    result = await repo.list_batch(limit=10, cursor_id=uuid4())

    assert result == rows
    cast("Any", session).execute.assert_awaited_once()
