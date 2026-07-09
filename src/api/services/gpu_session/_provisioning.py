"""Shared Vast.ai instance provisioning logic.

Extracted from GpuSessionService._provision_instance so that both the service
(initial start_session) and GpuProvisioningWorker (retry path) use the same
offer-walk + create_instance loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.vastai.exceptions import OfferTakenError, VastAIError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.api.services.vastai.client import VastAIClient
    from src.api.services.vastai.schemas import VastAIOffer

logger = structlog.get_logger(__name__)

# No :latest tag exists. Pin to a specific version; bumping requires deliberate review
# because it changes CUDA/PyTorch/Python versions. Check hub.docker.com/r/vastai/comfy/tags.
# TODO: move to bundle.yaml so each bundle pins its own image (bundle-driven docker image).
# This TODO is deferred because right now we use vastai templates with comfyui/cuda preset.
_VASTAI_IMAGE = "vastai/comfy:v0.15.1-cuda-12.9-py312"


def make_onstart_cmd(branch: str) -> str:
    """Build the onstart shell command for a given Aisha branch.

    The URL template lives here — single source of truth — so callers only
    need to pass the branch (from Settings.aisha_branch). This prevents the
    curl URL from drifting out of sync with ACS_AISHA_BRANCH.
    """
    return f"curl -sL https://raw.githubusercontent.com/gearbox/aisha/{branch}/scripts/onstart.sh | bash"


async def provision_vastai_instance(
    *,
    vastai_client: VastAIClient,
    offers: Sequence[VastAIOffer],
    disk_gb: int,
    env: dict[str, str],
    template_hash_id: str | None,
    image: str = _VASTAI_IMAGE,
    onstart_cmd: str | None = None,
    offer_walk_depth: int,
) -> tuple[int, VastAIOffer]:
    """Try to create a Vast.ai instance, walking down cheapest offers on OfferTakenError.

    Bounded by ``offer_walk_depth``. Returns ``(instance_id, selected_offer)`` on success.
    On persistent failure, raises either the last non-OfferTaken exception or a
    generic ``VastAIError`` if every candidate offer was taken.

    Caller is responsible for tunnel/other-resource cleanup on failure.

    Args:
        vastai_client: VastAIClient instance.
        offers: Sorted list of offers (cheapest first).
        disk_gb: Disk allocation for the instance.
        env: Environment variables dict (SECURITY: never log this — contains tokens).
        template_hash_id: Vast.ai template hash. When set, image and onstart_cmd are
            omitted (template provides them). None uses the legacy image+onstart path.
        image: Docker image to use (legacy path when template_hash_id is None).
        onstart_cmd: Shell command for the legacy path. Ignored when template_hash_id set.
        offer_walk_depth: Max number of offers to try within a single provisioning attempt.

    Returns:
        Tuple of (instance_id, selected_offer).

    Raises:
        VastAIError: If offer_walk_depth <= 0 or all offers were taken/errored.
    """
    if offer_walk_depth <= 0:
        raise VastAIError(f"offer_walk_depth must be >= 1, got {offer_walk_depth}")

    candidates = offers[:offer_walk_depth]
    last_exc: Exception | None = None

    for attempt, offer in enumerate(candidates):
        try:
            if template_hash_id is not None:
                instance_id = await vastai_client.create_instance(
                    offer.id,
                    disk_gb=disk_gb,
                    env=env,
                    template_hash_id=template_hash_id,
                )
            else:
                instance_id = await vastai_client.create_instance(
                    offer.id,
                    disk_gb=disk_gb,
                    env=env,
                    image=image,
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
