"""Provider-specific billing decisions for normalized generation failures."""

from __future__ import annotations

import dataclasses
from typing import Literal

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderBillingPolicy:
    """Billing policy for failures reported by one generation provider.

    Every field defaults to non-billable. The registry's fallback instance
    (returned for a provider absent from ``policies``) must charge nothing —
    see ``ProviderBillingPolicyRegistry.apply`` and invariant 13.
    """

    charge_on_moderation_rejection: bool = False
    charge_on_undelivered_output: bool = False

    def is_failure_billable(self, failure: ProviderFailure) -> bool:
        """Return whether the existing reservation must remain spent.

        Unknown acceptance is never promoted to a charge. This preserves the
        existing compensation behavior for ambiguous infrastructure failures.

        Billability follows evidence, not the coarse ``MALFORMED_RESPONSE``
        kind: an empty result set, a missing URL, or an undecodable payload
        are all malformed responses but prove nothing about delivery, so
        they are never billed here. Only ``OUTPUT_NOT_DELIVERED`` — raised
        exclusively at the two call sites with documented evidence that
        content was generated but not fetchable — is billable, and only
        when the provider's flag is on.
        """
        if failure.provider_request_accepted is not True:
            return False
        if failure.kind is ProviderFailureKind.MODERATION_REJECTED:
            return self.charge_on_moderation_rejection
        if failure.kind is ProviderFailureKind.OUTPUT_NOT_DELIVERED:
            return self.charge_on_undelivered_output
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderBillingPolicyRegistry:
    """Request/worker-injected provider billing rules.

    Financial behaviour must not come from a module-global default: API
    processes and the standalone poller must use the setting they were started
    with.  The registry also keeps providers independent as more policies are
    added.
    """

    policies: dict[Provider, ProviderBillingPolicy]

    @classmethod
    def with_grok_moderation_policy(
        cls,
        moderation_billing_policy: Literal["charge", "refund"],
        undelivered_output_billing_policy: Literal["charge", "refund"] = "charge",
    ) -> ProviderBillingPolicyRegistry:
        return cls(
            policies={
                Provider.GROK: ProviderBillingPolicy(
                    charge_on_moderation_rejection=moderation_billing_policy == "charge",
                    charge_on_undelivered_output=undelivered_output_billing_policy == "charge",
                )
            }
        )

    def apply(self, failure: ProviderFailure) -> ProviderFailure:
        # The fallback ``ProviderBillingPolicy()`` for a provider absent from
        # ``policies`` charges nothing (see its docstring) — the registry's
        # whole purpose is keeping providers financially independent, and an
        # unconfigured provider must never be silently armed for a charge.
        policy = self.policies.get(failure.provider, ProviderBillingPolicy())
        return dataclasses.replace(failure, billable=policy.is_failure_billable(failure))


DEFAULT_PROVIDER_BILLING_POLICIES = ProviderBillingPolicyRegistry.with_grok_moderation_policy(
    "charge", "charge"
)


def apply_provider_billing_policy(
    failure: ProviderFailure,
    *,
    registry: ProviderBillingPolicyRegistry | None = None,
) -> ProviderFailure:
    """Return ``failure`` annotated with the applicable billing decision."""
    return (registry or DEFAULT_PROVIDER_BILLING_POLICIES).apply(failure)
