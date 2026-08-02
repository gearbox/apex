"""Unit tests for the provider billing policy registry.

Complements the end-to-end coverage in test_grok_client.py (which exercises
the actual raise sites) with focused tests against the policy layer itself.
"""

from __future__ import annotations

from typing import Literal

import pytest

from src.api.services.generation.provider_billing_policy import (
    ProviderBillingPolicy,
    ProviderBillingPolicyRegistry,
)
from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider


def _failure(
    kind: ProviderFailureKind,
    *,
    provider: Provider = Provider.GROK,
    provider_request_accepted: bool | None = True,
) -> ProviderFailure:
    return ProviderFailure(
        kind=kind,
        provider=provider,
        sanitized_message=ProviderFailure.safe_message_for_kind(kind),
        provider_request_accepted=provider_request_accepted,
    )


_CHARGING_REGISTRY = ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")


def test_empty_result_set_is_not_billable() -> None:
    """A zero-image response is unproven provider malfunction (D1), not
    evidence content was generated but undeliverable."""
    failure = _failure(ProviderFailureKind.MALFORMED_RESPONSE)

    assert _CHARGING_REGISTRY.apply(failure).billable is False


def test_missing_url_is_not_billable() -> None:
    failure = _failure(ProviderFailureKind.MALFORMED_RESPONSE)

    assert _CHARGING_REGISTRY.apply(failure).billable is False


def test_undecodable_base64_is_not_billable() -> None:
    """A corrupt payload proves nothing about whether xAI produced a valid
    image — it cannot be shown to the user and stays non-billable (D1)."""
    failure = _failure(ProviderFailureKind.MALFORMED_RESPONSE)

    assert _CHARGING_REGISTRY.apply(failure).billable is False


@pytest.mark.parametrize(
    ("setting", "expected_billable"),
    [("charge", True), ("refund", False)],
)
def test_undelivered_output_is_billable_and_configurable(
    setting: Literal["charge", "refund"],
    expected_billable: bool,
) -> None:
    registry = ProviderBillingPolicyRegistry.with_grok_moderation_policy("refund", setting)
    failure = _failure(ProviderFailureKind.OUTPUT_NOT_DELIVERED)

    assert registry.apply(failure).billable is expected_billable


@pytest.mark.parametrize("kind", list(ProviderFailureKind))
@pytest.mark.parametrize("provider_request_accepted", [True, False, None])
def test_unregistered_provider_is_never_billable(
    kind: ProviderFailureKind,
    provider_request_accepted: bool | None,
) -> None:
    """The registry's fallback for a provider absent from ``policies`` must
    charge nothing — providers stay financially independent (D3, invariant
    13). ``Provider.AISHA`` is never registered by
    ``with_grok_moderation_policy``."""
    failure = _failure(
        kind,
        provider=Provider.AISHA,
        provider_request_accepted=provider_request_accepted,
    )

    assert _CHARGING_REGISTRY.apply(failure).billable is False


def test_default_policy_instance_charges_nothing() -> None:
    """D3: ``ProviderBillingPolicy()``'s default instance must not arm a
    charge for the next provider that adopts the normalized ProviderFailure
    path."""
    policy = ProviderBillingPolicy()

    for kind in ProviderFailureKind:
        failure = _failure(kind, provider=Provider.AISHA)
        assert policy.is_failure_billable(failure) is False


def test_ambiguous_acceptance_is_never_billable() -> None:
    """Invariant 3: provider_request_accepted is None never promotes to a charge."""
    for kind in (ProviderFailureKind.MODERATION_REJECTED, ProviderFailureKind.OUTPUT_NOT_DELIVERED):
        failure = _failure(kind, provider_request_accepted=None)
        assert _CHARGING_REGISTRY.apply(failure).billable is False
