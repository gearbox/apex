"""Periodic background task that syncs the payment currency catalog.

Interval configured via ``Settings.payment_currency_sync_interval_seconds``
(default 3h — dashboard-checked currency lists change rarely; also
triggerable on demand via ``POST /v1/admin/payments/currencies/refresh``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.product_registry import PRODUCT_REGISTRY
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from src.api.services.payment_currency_sync import PaymentCurrencySyncService
    from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)


class PaymentCurrencySyncWorker(PeriodicWorker):
    """Worker that periodically refreshes every product's currency catalog."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager,
        sync_service: PaymentCurrencySyncService,
        interval: int,
        redis_enabled: bool = False,
    ) -> None:
        super().__init__(
            name="payment_currency_sync",
            interval_seconds=interval,
            jitter_seconds=5.0,
            redis_enabled=redis_enabled,
        )
        self._db_manager = db_manager
        self._sync_service = sync_service

    async def run_once(self) -> None:
        """Refresh every registered product's catalog; one product's failure logs and continues.

        Intentionally never lets a single product's exception escape this
        method — a persistently-failing product must not stop the sweep from
        reaching healthy products every tick. Failure counts land in the
        summary log line, nothing is swallowed silently.
        """
        succeeded = 0
        failed = 0
        for product_config in PRODUCT_REGISTRY.values():
            try:
                async with self._db_manager.session() as session:
                    results = await self._sync_service.refresh(product_config, session=session)
            except Exception:
                failed += 1
                logger.warning(
                    "payment_currency_sync.product_failed",
                    product_id=product_config.slug,
                    exc_info=True,
                )
            else:
                succeeded += 1
                logger.debug(
                    "payment_currency_sync.product_ok",
                    product_id=product_config.slug,
                    providers=len(results),
                )

        logger.info(
            "payment_currency_sync.tick_summary",
            products_succeeded=succeeded,
            products_failed=failed,
        )
