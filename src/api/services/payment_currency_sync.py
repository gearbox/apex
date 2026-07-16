"""Sync the DB-cached payment currency catalog from provider discovery endpoints.

Single source of truth stays the provider (D2): this service never invents
availability or metadata, it only mirrors merchant/coins (availability) and
full-currencies (display metadata) into ``payment_currencies`` via
``PaymentCurrencyRepository.sync_catalog``. Callers (the periodic worker and
the admin refresh route) own the session/transaction — this service never
commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec
import structlog

from src.api.services.billing_errors import LogoCacheError, LogoStorageError
from src.api.services.payments.catalog import SupportsCurrencyCatalog
from src.core.product import PaymentProvider
from src.db.repositories.payment_currency import CatalogEntry, PaymentCurrencyRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.payment_currency_logos import LogoCacheService
    from src.api.services.payments.catalog import CurrencyDetails
    from src.api.services.payments.registry import GatewayRegistry
    from src.core.product import ProductConfig
    from src.db.models.billing import PaymentCurrency

logger = structlog.get_logger(__name__)


class SyncResult(msgspec.Struct, frozen=True, kw_only=True):
    """Outcome of syncing one provider's catalog for one product."""

    provider: PaymentProvider
    upserted: int
    deactivated: int


@dataclass
class _LogoTally:
    """Per-run logo-caching counters, threaded through one ticker at a time.

    `storage_unavailable` is the P1-2 circuit breaker: once a `LogoStorageError`
    is seen, every subsequent ticker in the same run skips `ensure_cached`
    entirely instead of retrying a dead storage backend.
    """

    hits: int = 0
    misses: int = 0
    skipped_storage: int = 0
    storage_unavailable: bool = False


class PaymentCurrencySyncService:
    """Discover catalog-capable gateways and reconcile their catalogs into the DB."""

    def __init__(self, registry: GatewayRegistry, logo_cache: LogoCacheService | None) -> None:
        self._registry = registry
        self._logo_cache = logo_cache

    async def refresh(
        self,
        product_config: ProductConfig,
        *,
        session: AsyncSession,
    ) -> list[SyncResult]:
        """Sync every catalog-capable, product-enabled gateway's currencies.

        Raises whatever the first failing provider raises (D6: an endpoint
        failure keeps stale data — the caller decides whether that means
        "abort" (admin route → 502) or "log and move to the next product"
        (worker tick).
        """
        repo = PaymentCurrencyRepository(session)
        now = datetime.now(UTC)
        results: list[SyncResult] = []

        capable_providers = sorted(
            product_config.payment_providers & self._registry.providers,
            key=lambda provider: provider.value,
        )
        for provider in capable_providers:
            gateway = self._registry.get(provider)
            if not isinstance(gateway, SupportsCurrencyCatalog):
                continue
            results.append(await self._sync_one(product_config, gateway, repo=repo, now=now))
        return results

    async def _sync_one(
        self,
        product_config: ProductConfig,
        gateway: SupportsCurrencyCatalog,
        *,
        repo: PaymentCurrencyRepository,
        now: datetime,
    ) -> SyncResult:
        slug = product_config.slug
        provider_value = gateway.provider.value

        try:
            selected = await gateway.list_merchant_currencies(slug)
            details = await gateway.list_full_currencies(slug)
        except Exception:
            logger.warning(
                "payment_currency.sync_failed",
                product_id=slug,
                provider=provider_value,
                exc_info=True,
            )
            raise

        existing_rows = await repo.list_currencies(slug, provider_value, only_available=False)
        existing_by_ticker: dict[str, PaymentCurrency] = {row.ticker: row for row in existing_rows}

        entries: list[CatalogEntry] = []
        metadata_missing = 0
        tally = _LogoTally()
        for ticker in selected:
            info = details.get(ticker)
            if info is None:
                logger.warning(
                    "payment_currency.metadata_missing",
                    product_id=slug,
                    provider=provider_value,
                    ticker=ticker,
                )
                metadata_missing += 1
                entries.append(CatalogEntry(ticker=ticker))
                continue

            logo_key, logo_source_url, logo_synced_at = await self._resolve_logo(
                info,
                ticker=ticker,
                product_id=slug,
                provider=provider_value,
                existing_row=existing_by_ticker.get(ticker),
                tally=tally,
                now=now,
            )
            entries.append(
                CatalogEntry(
                    ticker=ticker,
                    name=info.name,
                    network=info.network,
                    has_metadata=True,
                    logo_key=logo_key,
                    logo_source_url=logo_source_url,
                    logo_synced_at=logo_synced_at,
                )
            )

        upserted, deactivated = await repo.sync_catalog(slug, provider_value, entries, now=now)
        logger.info(
            "payment_currency.sync_ok",
            product_id=slug,
            provider=provider_value,
            upserted=upserted,
            deactivated=deactivated,
            logo_hits=tally.hits,
            logo_misses=tally.misses,
            metadata_missing=metadata_missing,
            logos_disabled=self._logo_cache is None,
            logos_skipped_storage=tally.skipped_storage,
        )
        return SyncResult(provider=gateway.provider, upserted=upserted, deactivated=deactivated)

    async def _resolve_logo(
        self,
        info: CurrencyDetails,
        *,
        ticker: str,
        product_id: str,
        provider: str,
        existing_row: PaymentCurrency | None,
        tally: _LogoTally,
        now: datetime,
    ) -> tuple[str | None, str | None, datetime | None]:
        """Resolve one ticker's logo fields, updating the run-wide `tally` (P1-2).

        Returns (logo_key, logo_source_url, logo_synced_at) — all `None` when
        there's nothing to do (no logo cache/URL, unchanged, or storage is
        down for the rest of this run).
        """
        if self._logo_cache is None or not info.logo_url:
            return None, None, None

        unchanged = (
            existing_row is not None
            and existing_row.logo_key is not None
            and existing_row.logo_source_url == info.logo_url
        )
        if unchanged:
            return None, None, None

        if tally.storage_unavailable:
            tally.skipped_storage += 1
            return None, None, None

        try:
            logo_key, hit = await self._cache_logo(
                self._logo_cache,
                info.logo_url,
                ticker=ticker,
                product_id=product_id,
                provider=provider,
            )
        except LogoStorageError as exc:
            tally.storage_unavailable = True
            tally.skipped_storage += 1
            logger.exception(
                "payment_currency.logo_storage_unavailable",
                product_id=product_id,
                provider=provider,
                error=str(exc),
            )
            return None, None, None

        if hit:
            tally.hits += 1
            return logo_key, info.logo_url, now
        tally.misses += 1
        return None, None, None

    @staticmethod
    async def _cache_logo(
        logo_cache: LogoCacheService,
        logo_url: str,
        *,
        ticker: str,
        product_id: str,
        provider: str,
    ) -> tuple[str | None, bool]:
        """Attempt to cache one currency's logo. Per-logo download/validation failures
        are non-fatal (D10); `LogoStorageError` propagates so the caller can
        short-circuit the rest of the run (P1-2).
        """
        try:
            key = await logo_cache.ensure_cached(logo_url)
        except LogoStorageError:
            raise
        except LogoCacheError as exc:
            logger.warning(
                "payment_currency.logo_failed",
                product_id=product_id,
                provider=provider,
                ticker=ticker,
                reason=str(exc),
            )
            return None, False
        else:
            return key, True
