"""Builders for ``GrokJobService`` that keep tests independent of the materializer wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from src.api.services.generation.provider_billing_policy import DEFAULT_PROVIDER_BILLING_POLICIES
from src.api.services.grok.factory import build_grok_job_service
from tests.video_receipt_support import stub_receipts

if TYPE_CHECKING:
    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
    from src.api.services.grok import GrokClient
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.media_ingest import MediaIngestor
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.storage import R2StorageService
    from src.api.services.video_jobs.receipts import AcceptedTaskReceipts


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
    provider_video_ingestor: MediaIngestor | None = None,
    max_output_bytes: int = 200 * 1024 * 1024,
    billing_policy: ProviderBillingPolicyRegistry | None = None,
    receipts: AcceptedTaskReceipts | None = None,
) -> GrokJobService:
    """Build a ``GrokJobService`` (and its materializer) from plain values."""
    return build_grok_job_service(
        grok_client=grok_client,
        storage=storage,
        media_ingestor=media_ingestor,
        provider_video_ingestor=provider_video_ingestor or media_ingestor,
        max_output_bytes=max_output_bytes,
        billing_service=billing_service,
        event_bus=event_bus,
        ops_event_bus=ops_event_bus,
        billing_policy=billing_policy or DEFAULT_PROVIDER_BILLING_POLICIES,
        retention_days=retention_days,
        max_poll_time=max_poll_time,
        finalization_lease_seconds=finalization_lease_seconds,
        receipts=receipts or stub_receipts(),
    )


def streaming_http_client(content: bytes = b"video-bytes", status: int = 200) -> httpx.AsyncClient:
    """A real ``httpx`` client over a mock transport (the materializer streams downloads)."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=content))
    )
