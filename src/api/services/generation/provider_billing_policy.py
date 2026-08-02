"""Provider-specific billing decisions for normalized generation failures."""

from __future__ import annotations

import dataclasses
from typing import Literal

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderBillingPolicy:
    """Billing policy for failures reported by one generation provider."""

    charge_on_moderation_rejection: bool = False

    def is_failure_billable(self, failure: ProviderFailure) -> bool:
        """Return whether the existing reservation must remain spent.

        Unknown acceptance is never promoted to a charge. This preserves the
        existing compensation behavior for ambiguous infrastructure failures.
        """
        return failure.provider_request_accepted is True and (
            failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
            or (
                self.charge_on_moderation_rejection
                and failure.kind is ProviderFailureKind.MODERATION_REJECTED
            )
        )


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
        cls, moderation_billing_policy: Literal["charge", "refund"]
    ) -> ProviderBillingPolicyRegistry:
        return cls(
            policies={
                Provider.GROK: ProviderBillingPolicy(
                    charge_on_moderation_rejection=moderation_billing_policy == "charge"
                )
            }
        )

    def apply(self, failure: ProviderFailure) -> ProviderFailure:
        policy = self.policies.get(failure.provider, ProviderBillingPolicy())
        return dataclasses.replace(failure, billable=policy.is_failure_billable(failure))


DEFAULT_PROVIDER_BILLING_POLICIES = ProviderBillingPolicyRegistry.with_grok_moderation_policy(
    "charge"
)


def apply_provider_billing_policy(
    failure: ProviderFailure,
    *,
    registry: ProviderBillingPolicyRegistry | None = None,
) -> ProviderFailure:
    """Return ``failure`` annotated with the applicable billing decision."""
    return (registry or DEFAULT_PROVIDER_BILLING_POLICIES).apply(failure)
