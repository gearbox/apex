"""Shared Vast.ai instance provisioning logic.

Extracted from GpuSessionService._provision_instance so that both the service
(initial start_session) and GpuProvisioningWorker (retry path) use the same
offer-walk + create_instance loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import structlog

from src.api.services.vastai.exceptions import OfferTakenError, VastAIError

if TYPE_CHECKING:
    from src.api.services.vastai.client import VastAIClient
    from src.api.services.vastai.schemas import VastAIOffer

logger = structlog.get_logger(__name__)

_VASTAI_IMAGE = "vastai/comfy:latest"
# Branch in URL must match ACS_AISHA_BRANCH default in Settings.aisha_branch — keep them in sync.
_ONSTART_CMD = (
    "curl -sL https://raw.githubusercontent.com/gearbox/aisha/master/scripts/onstart.sh | bash"
)


async def provision_vastai_instance(
    *,
    vastai_client: VastAIClient,
    offers: Sequence[VastAIOffer],
    disk_gb: int,
    env: dict[str, str],
    image: str = _VASTAI_IMAGE,
    onstart_cmd: str = _ONSTART_CMD,
    max_retries: int,
) -> tuple[int, VastAIOffer]:
    """Try to create a Vast.ai instance, walking down cheapest offers on OfferTakenError.

    Bounded by ``max_retries``. Returns ``(instance_id, selected_offer)`` on success.
    On persistent failure, raises either the last non-OfferTaken exception or a
    generic ``VastAIError`` if every candidate offer was taken.

    Caller is responsible for tunnel/other-resource cleanup on failure.

    Args:
        vastai_client: VastAIClient instance.
        offers: Sorted list of offers (cheapest first).
        disk_gb: Disk allocation for the instance.
        env: Environment variables dict (SECURITY: never log this — contains tokens).
        image: Docker image to use.
        onstart_cmd: Shell command to run on instance start.
        max_retries: Max number of offers to try.

    Returns:
        Tuple of (instance_id, selected_offer).

    Raises:
        VastAIError: If max_retries <= 0 or all offers were taken/errored.
    """
    if max_retries <= 0:
        raise VastAIError(f"max_node_provisioning_retries must be >= 1, got {max_retries}")

    candidates = offers[:max_retries]
    last_exc: Exception | None = None

    for attempt, offer in enumerate(candidates):
        try:
            instance_id = await vastai_client.create_instance(
                offer.id,
                image=image,
                disk_gb=disk_gb,
                env=env,
                onstart_cmd=onstart_cmd,
            )
        except OfferTakenError:
            remaining = len(candidates) - attempt - 1
            logger.warning(
                "gpu_session.provision.offer_taken_retry",
                offer_id=offer.id,
                attempt=attempt + 1,
                remaining_offers=remaining,
            )
            continue
        except Exception as exc:
            last_exc = exc
            break

        logger.info(
            "gpu_session.provision.instance_created",
            instance_id=instance_id,
            offer_id=offer.id,
            dph_micros=offer.dph_total_micros,
        )
        return instance_id, offer

    if last_exc is not None:
        raise last_exc
    raise VastAIError("All offers were taken — no instance could be created after retries")
