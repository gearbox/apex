"""Provider-specific billing decisions for normalized generation failures."""

from __future__ import annotations

import dataclasses

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
        return (
            self.charge_on_moderation_rejection
            and failure.kind == ProviderFailureKind.MODERATION_REJECTED
            and failure.provider_request_accepted is True
        )


_DEFAULT_POLICY = ProviderBillingPolicy()
_POLICIES: dict[Provider, ProviderBillingPolicy] = {
    # Production observation: Grok bills moderation rejections after it has
    # accepted the request. Keep the debit instead of issuing compensation.
    Provider.GROK: ProviderBillingPolicy(charge_on_moderation_rejection=True),
}


def apply_provider_billing_policy(failure: ProviderFailure) -> ProviderFailure:
    """Return ``failure`` annotated with the applicable billing decision."""
    policy = _POLICIES.get(failure.provider, _DEFAULT_POLICY)
    return dataclasses.replace(failure, billable=policy.is_failure_billable(failure))
