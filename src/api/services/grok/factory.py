"""Construction of a ``GrokJobService`` together with its video materializer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.grok.job_service import GrokJobService
from src.api.services.media_ingest.profile_log import VideoProfileOrigin
from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
from src.api.services.video_jobs.transition import make_transition_service_factory
from src.core.enums import Provider

if TYPE_CHECKING:
    import httpx

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
    from src.api.services.grok import GrokClient
    from src.api.services.media_ingest import MediaIngestor
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.storage import R2StorageService


def build_grok_job_service(
    *,
    grok_client: GrokClient,
    storage: R2StorageService,
    media_ingestor: MediaIngestor,
    provider_video_ingestor: MediaIngestor,
    max_output_bytes: int,
    billing_service: BillingService | None,
    event_bus: EventBus | None,
    ops_event_bus: OpsEventBus | None,
    billing_policy: ProviderBillingPolicyRegistry,
    retention_days: int,
    max_poll_time: int,
    finalization_lease_seconds: int,
) -> GrokJobService:
    """Build the Grok job service and the materializer that finishes its videos.

    ``media_ingestor`` prepares user-facing image inputs/outputs; the materializer
    uses ``provider_video_ingestor`` (provider-output byte budget) for generated
    videos, with ``max_output_bytes`` capping the download to the same budget.
    """
    holder: list[GrokJobService] = []

    def http_client() -> httpx.AsyncClient:
        return holder[0].http_client

    materializer = RemoteVideoMaterializer(
        provider=Provider.GROK,
        profile_origin=VideoProfileOrigin.GROK_OUTPUT,
        storage=storage,
        media_ingestor=provider_video_ingestor,
        http_client_provider=http_client,
        retention_days=retention_days,
        finalization_lease_seconds=finalization_lease_seconds,
        max_output_bytes=max_output_bytes,
        billing_policy=billing_policy,
        transition_service_factory=make_transition_service_factory(
            billing_service=billing_service,
            event_bus=event_bus,
            ops_event_bus=ops_event_bus,
        ),
    )
    service = GrokJobService(
        grok_client=grok_client,
        storage=storage,
        retention_days=retention_days,
        materializer=materializer,
        billing_service=billing_service,
        event_bus=event_bus,
        ops_event_bus=ops_event_bus,
        max_poll_time=max_poll_time,
        media_ingestor=media_ingestor,
        billing_policy=billing_policy,
    )
    holder.append(service)
    return service
