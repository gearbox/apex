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
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC", name="Old Name", has_metadata=True)],
        now=_NOW,
    )
    await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC", name="Bitcoin", has_metadata=True)],
        now=_NOW,
    )

    rows = await _rows(db_session)
    assert rows["BTC"].name == "Bitcoin"


async def test_transient_metadata_gap_preserves_prior_name_and_network(
    db_session: AsyncSession,
) -> None:
    """A ticker briefly missing from full-currencies (has_metadata=False) must not
    strip a display name learned on a previous sync — provider glitches shouldn't
    erase metadata."""
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC", name="Bitcoin", network="BTC", has_metadata=True)],
        now=_NOW,
    )
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    rows = await _rows(db_session)
    assert rows["BTC"].name == "Bitcoin"
    assert rows["BTC"].network == "BTC"


async def test_present_but_null_metadata_overwrites_prior_name(
    db_session: AsyncSession,
) -> None:
    """A provider retracting a name to null (has_metadata=True, name=None) is an
    intentional overwrite, unlike a transient gap."""
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC", name="Bitcoin", has_metadata=True)],
        now=_NOW,
    )
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC", name=None, has_metadata=True)], now=_NOW
    )

    rows = await _rows(db_session)
    assert rows["BTC"].name is None


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
        "vex",
        "nowpayments",
        [CatalogEntry(ticker="BTC", name="Bitcoin", has_metadata=True)],
        now=_NOW,
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


# ---------------------------------------------------------------------------
# Currency suppression (D1-D5): set_suppressed, is_ticker_suppressed,
# list_currencies(include_suppressed), and the D2 sync-never-touches-the-flag
# regression.
# ---------------------------------------------------------------------------


async def test_set_suppressed_round_trip(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    row = await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)
    assert row is not None
    assert row.is_suppressed is True

    row = await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=False)
    assert row is not None
    assert row.is_suppressed is False


async def test_set_suppressed_returns_none_for_unseen_ticker(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    row = await repo.set_suppressed("vex", "nowpayments", "GHOST", is_suppressed=True)
    assert row is None


async def test_set_suppressed_normalizes_ticker_case(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

    row = await repo.set_suppressed("vex", "nowpayments", "  btc  ", is_suppressed=True)
    assert row is not None
    assert row.ticker == "BTC"
    assert row.is_suppressed is True


async def test_is_ticker_suppressed_true_false_and_unseen(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC"), CatalogEntry(ticker="ETH")], now=_NOW
    )
    await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)

    assert await repo.is_ticker_suppressed("vex", "nowpayments", "BTC") is True
    assert await repo.is_ticker_suppressed("vex", "nowpayments", "ETH") is False
    assert await repo.is_ticker_suppressed("vex", "nowpayments", "GHOST") is False
    # Case-insensitive lookup, mirroring the gateway's strip+uppercase normalization.
    assert await repo.is_ticker_suppressed("vex", "nowpayments", "btc") is True


async def test_list_currencies_include_suppressed_matrix(db_session: AsyncSession) -> None:
    repo = PaymentCurrencyRepository(db_session)
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC"), CatalogEntry(ticker="ETH")], now=_NOW
    )
    await repo.sync_catalog(
        "vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW
    )  # ETH -> unavailable
    await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)

    # Public path: only_available=True, include_suppressed=False (default).
    public = await repo.list_currencies("vex", "nowpayments", only_available=True)
    assert not [row.ticker for row in public]

    # Admin path: only_available=False, include_suppressed=True — sees everything.
    admin = await repo.list_currencies(
        "vex", "nowpayments", only_available=False, include_suppressed=True
    )
    assert {row.ticker for row in admin} == {"BTC", "ETH"}

    # only_available=True with include_suppressed=True still excludes the unavailable ETH
    # but includes the available-but-suppressed BTC.
    available_incl_suppressed = await repo.list_currencies(
        "vex", "nowpayments", only_available=True, include_suppressed=True
    )
    assert [row.ticker for row in available_incl_suppressed] == ["BTC"]

    # only_available=False with include_suppressed=False (default) excludes suppressed BTC
    # but includes unavailable ETH.
    all_not_suppressed = await repo.list_currencies("vex", "nowpayments", only_available=False)
    assert [row.ticker for row in all_not_suppressed] == ["ETH"]


class TestSuppressionSurvivesSync:
    """D2 regression: sync_catalog must never read or write is_suppressed."""

    async def test_suppression_survives_upsert_touch(self, db_session: AsyncSession) -> None:
        repo = PaymentCurrencyRepository(db_session)
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)
        await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)

        # A normal sync that re-touches BTC (still seen) must not clear suppression.
        await repo.sync_catalog(
            "vex", "nowpayments", [CatalogEntry(ticker="BTC", name="Bitcoin")], now=_NOW
        )

        rows = await _rows(db_session)
        assert rows["BTC"].is_suppressed is True
        assert rows["BTC"].is_available is True

    async def test_suppression_survives_deactivation_pass(self, db_session: AsyncSession) -> None:
        repo = PaymentCurrencyRepository(db_session)
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)
        await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)

        # BTC missing from this sync -> deactivated (is_available flips false).
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="ETH")], now=_NOW)

        rows = await _rows(db_session)
        assert rows["BTC"].is_available is False
        assert rows["BTC"].is_suppressed is True

    async def test_suppression_survives_reappearance(self, db_session: AsyncSession) -> None:
        repo = PaymentCurrencyRepository(db_session)
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)
        await repo.set_suppressed("vex", "nowpayments", "BTC", is_suppressed=True)

        # Deactivate, then bring BTC back — is_available flips true, suppression stays.
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="ETH")], now=_NOW)
        await repo.sync_catalog("vex", "nowpayments", [CatalogEntry(ticker="BTC")], now=_NOW)

        rows = await _rows(db_session)
        assert rows["BTC"].is_available is True
        assert rows["BTC"].is_suppressed is True
