"""Gateway registry contract tests."""

from unittest.mock import AsyncMock

import pytest

from src.api.dependencies.common import build_gateway_registry
from src.api.services.billing_errors import UnsupportedProviderError
from src.api.services.payments.registry import GatewayRegistry
from src.core.config import Settings
from src.core.product import PaymentProvider

pytestmark = pytest.mark.unit


def _gateway(provider: PaymentProvider) -> AsyncMock:
    gateway = AsyncMock()
    gateway.provider = provider
    return gateway


def test_registry_rejects_duplicate_provider() -> None:
    with pytest.raises(ValueError, match="Duplicate payment gateway"):
        GatewayRegistry([_gateway(PaymentProvider.STRIPE), _gateway(PaymentProvider.STRIPE)])


def test_registry_rejects_unknown_provider() -> None:
    registry = GatewayRegistry([])
    with pytest.raises(UnsupportedProviderError):
        registry.get(PaymentProvider.STRIPE)


def test_default_registry_is_complete() -> None:
    settings = Settings(jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes")
    assert build_gateway_registry(settings).providers == frozenset(PaymentProvider)
