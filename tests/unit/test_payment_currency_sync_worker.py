"""Unit tests for PaymentCurrencySyncWorker."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.product_registry import PRODUCT_REGISTRY
from src.workers.payment_currency_sync import PaymentCurrencySyncWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit


def _db_manager() -> MagicMock:
    db_manager = MagicMock()

    @asynccontextmanager
    async def _session() -> AsyncIterator[MagicMock]:
        yield MagicMock()

    db_manager.session = _session
    return db_manager


def _make_worker(sync_service: AsyncMock, **kwargs: object) -> PaymentCurrencySyncWorker:
    defaults: dict[str, object] = {
        "db_manager": _db_manager(),
        "sync_service": sync_service,
        "interval": 10800,
        "redis_client_factory": MagicMock(),
    } | kwargs
    return PaymentCurrencySyncWorker(**defaults)  # type: ignore[arg-type]


async def test_run_once_syncs_every_registered_product() -> None:
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(return_value=[])
    worker = _make_worker(sync_service)

    await worker.run_once()

    assert sync_service.refresh.await_count == len(PRODUCT_REGISTRY)


async def test_run_once_survives_one_product_failing() -> None:
    """Never propagates: a persistently failing product must not stop the
    sweep from reaching the other products this tick or every future tick."""
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=[RuntimeError("boom"), []])
    worker = _make_worker(sync_service)

    await worker.run_once()  # must not raise

    assert sync_service.refresh.await_count == len(PRODUCT_REGISTRY)


async def test_run_once_never_raises_even_if_every_product_fails() -> None:
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=RuntimeError("boom"))
    worker = _make_worker(sync_service)

    await worker.run_once()  # must not raise
