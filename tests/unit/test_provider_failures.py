"""Unit tests for the normalized ProviderFailure taxonomy and disclosure logic."""

from __future__ import annotations

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider

_BILLED_NOTICE = "This generation was charged because the provider processed the request."

# Both charge flags on so every billable-capable kind actually comes back billable.
_CHARGING_REGISTRY = ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")


def test_every_billable_kind_discloses_the_charge() -> None:
    """Invariant 12: failure.billable is True implies the public message
    contains the billing disclosure. Looping over every kind — driven
    through the real policy, not hand-set — means a future kind that the
    policy makes billable but that public_message_for_failure forgets to
    disclose fails this test rather than shipping a silent charge."""
    for kind in ProviderFailureKind:
        failure = ProviderFailure(
            kind=kind,
            provider=Provider.GROK,
            sanitized_message=ProviderFailure.safe_message_for_kind(kind),
            provider_request_accepted=True,
        )
        resolved = _CHARGING_REGISTRY.apply(failure)
        message = ProviderFailure.public_message_for_failure(resolved)

        if resolved.billable:
            assert _BILLED_NOTICE in message, f"{kind} is billable but discloses no charge"
        else:
            assert _BILLED_NOTICE not in message, f"{kind} is not billable but claims a charge"


def test_non_billable_message_never_claims_a_charge() -> None:
    for kind in ProviderFailureKind:
        message = ProviderFailure.public_message_for_failure(
            ProviderFailure(
                kind=kind,
                provider=Provider.GROK,
                sanitized_message=ProviderFailure.safe_message_for_kind(kind),
                billable=False,
            )
        )
        assert _BILLED_NOTICE not in message


def test_output_not_delivered_message_is_billed_notice_when_billable() -> None:
    failure = ProviderFailure(
        kind=ProviderFailureKind.OUTPUT_NOT_DELIVERED,
        provider=Provider.GROK,
        sanitized_message=ProviderFailure.safe_message_for_kind(
            ProviderFailureKind.OUTPUT_NOT_DELIVERED
        ),
        billable=True,
    )

    assert _BILLED_NOTICE in ProviderFailure.public_message_for_failure(failure)
