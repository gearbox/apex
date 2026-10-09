"""Builders for ``GrokJobService`` that keep tests independent of the materializer wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.generation.provider_billing_policy import DEFAULT_PROVIDER_BILLING_POLICIES
from src.api.services.grok.factory import build_grok_job_service

if TYPE_CHECKING:
    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
    from src.api.services.grok import GrokClient
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.media_ingest import MediaIngestor
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.storage import R2StorageService


def make_grok_job_service(
    grok_client: GrokClient,
    storage: R2StorageService,
    retention_days: int = 7,
    *,
    billing_service: BillingService | None = None,
    event_bus: EventBus | None = None,
    ops_event_bus: OpsEventBus | None = None,
    max_poll_time: int = 600,
    finalization_lease_seconds: int = 120,
    media_ingestor: MediaIngestor,
    billing_policy: ProviderBillingPolicyRegistry | None = None,
) -> GrokJobService:
    """Build a ``GrokJobService`` (and its materializer) from plain values."""
    return build_grok_job_service(
        grok_client=grok_client,
        storage=storage,
        media_ingestor=media_ingestor,
        billing_service=billing_service,
        event_bus=event_bus,
        ops_event_bus=ops_event_bus,
        billing_policy=billing_policy or DEFAULT_PROVIDER_BILLING_POLICIES,
        retention_days=retention_days,
        max_poll_time=max_poll_time,
        finalization_lease_seconds=finalization_lease_seconds,
    )
