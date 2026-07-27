"""Tests for topup_description() — the single ledger-wording site."""

from __future__ import annotations

import pytest

from src.api.services.payments.descriptions import topup_description
from src.core.product import PaymentMethodKind, PaymentProvider

pytestmark = pytest.mark.unit


def test_full_credit_crypto() -> None:
    description = topup_description(PaymentMethodKind.CRYPTO, partial=False)
    assert description == "Token purchase via crypto payment"


def test_partial_credit_card() -> None:
    assert (
        topup_description(PaymentMethodKind.CARD, partial=True)
        == "Token purchase via card payment (partial)"
    )


@pytest.mark.parametrize("provider", list(PaymentProvider))
@pytest.mark.parametrize("partial", [True, False])
def test_never_leaks_gateway_name(provider: PaymentProvider, partial: bool) -> None:
    description = topup_description(provider.method_kind, partial=partial)
    assert provider.value not in description
