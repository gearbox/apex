"""PostgreSQL reconciliation behavior for PaymentCurrencyRepository.sync_catalog."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.billing import PaymentCurrency
from src.db.repositories.payment_currency import CatalogEntry, PaymentCurrencyRepository

pytestmark = pytest.mark.asyncio

_NOW = datetime.now(UTC)


async def _rows(session: AsyncSession, product_id: str = "vex") -> dict[str, PaymentCurrency]:
    result = await session.execute(
        select(PaymentCurrency).where(PaymentCurrency.product_id == product_id)
    )
    return {row.ticker: row for row in result.scalars().all()}


async def test_first_sync_inserts_available_rows(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    upserted, deactivated = await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="btc", name="Bitcoin", network="BTC")],
        now=_NOW,
    )
    assert (upserted, deactivated) == (1, 0)

    rows = await _rows(db_session)
    assert rows["BTC"].is_available is True
    assert rows["BTC"].name == "Bitcoin"


async def test_second_sync_deactivates_missing_and_keeps_seen(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC"), CatalogEntry(ticker="ETH")],
        now=_NOW,
    )

    upserted, deactivated = await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW
    )
    assert (upserted, deactivated) == (1, 1)

    rows = await _rows(db_session)
    assert rows["BTC"].is_available is True
    assert rows["ETH"].is_available is False


async def test_reactivation_flips_available_back_to_true(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="ETH")], now=_NOW)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    rows = await _rows(db_session)
    assert rows["BTC"].is_available is True


async def test_unique_constraint_prevents_duplicate_rows(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    result = await db_session.execute(
        select(PaymentCurrency).where(
            PaymentCurrency.product_id == "vex",
            PaymentCurrency.provider == "nowpayments",
            PaymentCurrency.ticker == "BTC",
        )
    )
    assert len(result.scalars().all()) == 1


async def test_metadata_update_on_second_sync(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC", name="Old Name")], now=_NOW
    )
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC", name="Bitcoin")], now=_NOW
    )

    rows = await _rows(db_session)
    assert rows["BTC"].name == "Bitcoin"


async def test_empty_entries_raises(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    with pytest.raises(ValueError, match="empty"):
        await repo.sync_catalog("vex", "nowpayments", [], now=_NOW)


async def test_overlong_ticker_is_skipped_not_written(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    overlong = "X" * 25
    upserted, _ = await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC"), CatalogEntry(ticker=overlong)],
        now=_NOW,
    )
    assert upserted == 1

    rows = await _rows(db_session)
    assert overlong not in rows
    assert "BTC" in rows


async def test_entry_without_logo_data_does_not_null_existing_logo_columns(
    db_session: AsyncSession,
) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex",
        "nowpayments",
        [
            CatalogEntry(
                ticker="BTC",
                logo_key="abc123.svg",
                logo_source_url="https://nowpayments.io/btc.svg",
                logo_synced_at=_NOW,
            )
        ],
        now=_NOW,
    )

    # Second sync carries no logo data for this entry (unchanged-logo skip path).
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC", name="Bitcoin")], now=_NOW
    )

    rows = await _rows(db_session)
    assert rows["BTC"].logo_key == "abc123.svg"
    assert rows["BTC"].logo_source_url == "https://nowpayments.io/btc.svg"
    assert rows["BTC"].name == "Bitcoin"


async def test_list_currencies_filters_by_availability(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC"), CatalogEntry(ticker="ETH")], now=_NOW
    )
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    available = await repo.list_currencies("vex", "nowpayments", only_available=True)
    all_rows = await repo.list_currencies("vex", "nowpayments", only_available=False)

    assert [row.ticker for row in available] == ["BTC"]
    assert {row.ticker for row in all_rows} == {"BTC", "ETH"}
