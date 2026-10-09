"""Billing-policy registry builders for tests that only care about the Grok flags."""

from __future__ import annotations

from typing import Literal

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.core.config import Settings

Policy = Literal["charge", "refund"]


def grok_policy_registry(
    moderation: Policy = "charge", undelivered: Policy = "charge"
) -> ProviderBillingPolicyRegistry:
    """``ProviderBillingPolicyRegistry.from_settings`` with explicit Grok policies."""
    return ProviderBillingPolicyRegistry.from_settings(
        Settings.model_construct(
            grok_moderation_billing_policy=moderation,
            grok_undelivered_output_billing_policy=undelivered,
        )
    )
