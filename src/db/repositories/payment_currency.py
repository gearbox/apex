"""Persistence for the DB-cached payment currency catalog.

The table is a cache synced from a payment provider's own discovery
endpoints — never hand-edited. ``sync_catalog`` is a set-reconciliation
upsert: every ticker present in the latest successful sync is marked
available and touched; everything else for that (product, provider) pair
flips to unavailable. Rows are never deleted (reconciliation history).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import structlog
from sqlalchemy import CursorResult, select, update
from sqlalchemy.dialects.postgresql import insert

from src.core.uid import new_id
from src.db.models.billing import PAYMENT_CURRENCY_MAX_LEN, PaymentCurrency

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """One synced ticker's availability plus optional display/logo metadata.

    ``has_metadata`` distinguishes "the provider's full-currencies response
    didn't include this ticker this sync" (``False`` — a transient gap that
    must not erase a previously learned name/network) from "the provider
    included this ticker with these (possibly null) fields" (``True`` — an
    intentional overwrite, including a provider retracting a name to null).
    """

    ticker: str
    name: str | None = None
    network: str | None = None
    has_metadata: bool = False
    logo_key: str | None = None
    logo_source_url: str | None = None
    logo_synced_at: datetime | None = None


class PaymentCurrencyRepository:
    """Reconciliation-style upserts for the ``payment_currencies`` cache table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_currencies(
        self,
        product_id: str,
        provider: str | None = None,
        *,
        only_available: bool,
        include_suppressed: bool = False,
    ) -> Sequence[PaymentCurrency]:
        """List cached currencies for a product, ordered by ticker.

        ``provider=None`` (the default) returns rows across every provider
        for the product — used by the public/admin catalog endpoints, which
        are not provider-scoped. ``include_suppressed=False`` (the default)
        excludes admin-suppressed tickers — the public catalog path
        (``only_available=True, include_suppressed=False``); the admin path
        passes ``include_suppressed=True`` to see everything.
        """
        stmt = select(PaymentCurrency).where(PaymentCurrency.product_id == product_id)
        if provider is not None:
            stmt = stmt.where(PaymentCurrency.provider == provider)
        if only_available:
            stmt = stmt.where(PaymentCurrency.is_available.is_(True))
        if not include_suppressed:
            stmt = stmt.where(PaymentCurrency.is_suppressed.is_(False))
        stmt = stmt.order_by(PaymentCurrency.ticker)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def set_suppressed(
        self,
        product_id: str,
        provider: str,
        ticker: str,
        *,
        is_suppressed: bool,
    ) -> PaymentCurrency | None:
        """Set the suppression flag for one (product, provider, ticker) row.

        Returns ``None`` when no row exists for that ticker (D5: suppression
        requires an already-seen ticker) — the caller maps that to a 404.
        """
        ticker = ticker.strip().upper()
        stmt = select(PaymentCurrency).where(
            PaymentCurrency.product_id == product_id,
            PaymentCurrency.provider == provider,
            PaymentCurrency.ticker == ticker,
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.is_suppressed = is_suppressed
        await self._session.flush()
        return row

    async def is_ticker_suppressed(self, product_id: str, provider: str, ticker: str) -> bool:
        """Cheap single-row lookup for the charge-creation guard.

        An unseen ticker is never suppressed — nothing to check.
        """
        ticker = ticker.strip().upper()
        stmt = select(PaymentCurrency.is_suppressed).where(
            PaymentCurrency.product_id == product_id,
            PaymentCurrency.provider == provider,
            PaymentCurrency.ticker == ticker,
        )
        result = await self._session.execute(stmt)
        value = result.scalar_one_or_none()
        return bool(value)

    async def sync_catalog(
        self,
        product_id: str,
        provider: str,
        entries: Collection[CatalogEntry],
        *,
        now: datetime,
    ) -> tuple[int, int]:
        """Upsert seen tickers as available, deactivate everything else (D3).

        Logo columns (``logo_key``/``logo_source_url``/``logo_synced_at``) are
        written only when an entry carries a freshly-cached logo — an entry
        without logo data never nulls out a previously cached one, so the
        unchanged-logo skip path (D8) doesn't erase history. ``name``/
        ``network`` follow the same pattern via ``entry.has_metadata``: a
        ticker transiently missing from the provider's full-currencies
        response keeps its previously learned display metadata rather than
        being nulled out.

        Returns:
            ``(upserted_count, deactivated_count)``.

        Raises:
            ValueError: ``entries`` is empty, or every entry was skipped for
                being overlong — an empty/fully-rejected merchant list is
                treated as a sync failure (API anomaly), never as grounds to
                mass-deactivate every cached row for this product/provider.
        """
        if not entries:
            raise ValueError(
                "sync_catalog received an empty entry set — refusing to "
                "mass-deactivate; treat an empty merchant list as a sync failure"
            )

        seen_tickers: list[str] = []
        upserted = 0
        for entry in entries:
            ticker = entry.ticker.strip().upper()
            if len(ticker) > PAYMENT_CURRENCY_MAX_LEN:
                logger.warning(
                    "payment_currency.ticker_overlong",
                    product_id=product_id,
                    provider=provider,
                    ticker=ticker,
                )
                continue
            seen_tickers.append(ticker)

            values: dict[str, object] = {
                "id": new_id(),
                "product_id": product_id,
                "provider": provider,
                "ticker": ticker,
                "is_available": True,
                "name": entry.name,
                "network": entry.network,
                "first_seen_at": now,
                "last_seen_at": now,
                "updated_at": now,
            }
            update_values: dict[str, object] = {
                "is_available": True,
                "last_seen_at": now,
                "updated_at": now,
            }
            if entry.has_metadata:
                update_values["name"] = entry.name
                update_values["network"] = entry.network
            if entry.logo_key is not None:
                values["logo_key"] = entry.logo_key
                values["logo_source_url"] = entry.logo_source_url
                values["logo_synced_at"] = entry.logo_synced_at
                update_values["logo_key"] = entry.logo_key
                update_values["logo_source_url"] = entry.logo_source_url
                update_values["logo_synced_at"] = entry.logo_synced_at

            statement = (
                insert(PaymentCurrency)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_payment_currencies",
                    set_=update_values,
                )
            )
            await self._session.execute(statement)
            upserted += 1

        if not seen_tickers:
            raise ValueError(
                "sync_catalog: every entry was skipped as overlong — refusing to mass-deactivate"
            )

        deactivate_stmt = (
            update(PaymentCurrency)
            .where(
                PaymentCurrency.product_id == product_id,
                PaymentCurrency.provider == provider,
                PaymentCurrency.ticker.notin_(seen_tickers),
                PaymentCurrency.is_available.is_(True),
            )
            .values(is_available=False, updated_at=now)
        )
        result = cast("CursorResult[tuple[()]]", await self._session.execute(deactivate_stmt))
        deactivated = result.rowcount or 0

        return upserted, deactivated
