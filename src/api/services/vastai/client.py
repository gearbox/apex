"""Vast.ai async REST API client."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar, cast

import httpx
import msgspec
import structlog

from src.core.bundle_config import HardwareRequirements

from .exceptions import (
    InstanceNotFoundError,
    NoCapacityError,
    OfferTakenError,
    VastAIError,
    VastAIPaymentError,
    VastAIRateLimitError,
)
from .schemas import (
    CreateInstanceResponse,
    GetInstanceResponse,
    SearchOffersResponse,
    VastAIInstance,
    VastAIOffer,
)

if TYPE_CHECKING:
    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_T = TypeVar("_T", bound=msgspec.Struct)


class VastAIClient:
    """Async client for the Vast.ai REST API.

    Handles GPU instance lifecycle: search → create → monitor → stop/start → destroy.
    """

    _VASTAI_API_BASE = "https://console.vast.ai/api/v0"

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, settings: Settings) -> None:
        self._client = http_client
        self._api_key = api_key
        self._settings = settings

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _decode_response(self, resp: httpx.Response, schema: type[_T], context: str) -> _T:
        """Decode response body via msgspec, wrapping ValidationError as VastAIError.

        Callers should only encounter domain exceptions from this client.
        msgspec.ValidationError leaking out exposes an internal parsing detail
        and breaks the invariant that ``except VastAIError`` catches everything
        this client can raise.
        """
        try:
            return msgspec.json.decode(resp.content, type=schema)
        except msgspec.ValidationError as exc:
            body_preview = resp.text[:200].replace("\n", " ")
            raise VastAIError(
                f"{context}: response did not match {schema.__name__} schema — {exc}; "
                f"body preview: {body_preview!r}",
                status_code=resp.status_code,
            ) from exc

    def _raise_for_status(self, resp: httpx.Response, context: str) -> None:
        if resp.status_code < 400:
            return
        raise VastAIError(
            f"{context}: HTTP {resp.status_code} — {resp.text[:256]}",
            status_code=resp.status_code,
        )

    def _parse_retry_after(self, resp: httpx.Response) -> float:
        """Parse Retry-After header; cap to settings.vastai_max_retry_after_seconds.

        Vast.ai sends Retry-After as an integer second count. Fall back to
        1.0s if the header is missing or malformed.
        """
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return 1.0
        try:
            seconds = float(raw)
        except ValueError:
            logger.warning(
                "vastai.retry_after_unparseable",
                raw_value=raw[:64],
            )
            return 1.0
        return min(max(seconds, 0.0), self._settings.vastai_max_retry_after_seconds)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        context: str,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue an HTTP request, retrying on 429 with Retry-After.

        Handles 429 consistently across all Vast.ai endpoints. Non-429
        responses are returned as-is; callers do their own status checks.
        """
        attempts_remaining = self._settings.vastai_max_429_retries
        while True:
            kwargs: dict[str, Any] = {"headers": self._auth_headers()}
            if json_body is not None:
                kwargs["json"] = json_body
            resp = cast(httpx.Response, await getattr(self._client, method)(url, **kwargs))

            if resp.status_code != 429:
                return resp

            retry_after = self._parse_retry_after(resp)
            if attempts_remaining <= 0:
                logger.warning(
                    "vastai.rate_limit_exhausted",
                    context=context,
                    retry_after_seconds=retry_after,
                )
                raise VastAIRateLimitError(
                    f"{context}: rate limit retries exhausted (last Retry-After={retry_after}s)",
                    retry_after_seconds=retry_after,
                )

            logger.info(
                "vastai.rate_limit_retry",
                context=context,
                retry_after_seconds=retry_after,
                attempts_remaining=attempts_remaining,
            )
            await asyncio.sleep(retry_after)
            attempts_remaining -= 1

    async def search_offers(
        self, hardware: HardwareRequirements, *, limit: int = 10
    ) -> list[VastAIOffer]:
        """Search for GPU offers matching hardware requirements.

        Translates HardwareRequirements into Vast.ai search filters.
        Results are sorted by cost (cheapest first).

        Raises:
            NoCapacityError: If no offers match.
            VastAIError: On API errors.
        """
        try:
            cuda_min = float(hardware.cuda_min_version)
        except ValueError as exc:
            # HardwareRequirements.cuda_min_version should be validated upstream
            # by BundleIndexService, but surface a clear error just in case.
            raise VastAIError(
                f"Invalid cuda_min_version {hardware.cuda_min_version!r}: "
                f"must be a numeric string (e.g. '12.1')"
            ) from exc

        payload: dict[str, Any] = {
            "gpu_name": {"in": list(hardware.gpu_whitelist)},
            "num_gpus": {"gte": hardware.num_gpus},
            "disk_space": {"gte": hardware.min_disk_gb},
            "inet_up": {"gte": hardware.min_network_upload_mbps},
            "inet_down": {"gte": hardware.min_network_download_mbps},
            "cuda_max_good": {"gte": cuda_min},
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "ondemand",
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        resp = await self._request_with_retry(
            "post",
            f"{self._VASTAI_API_BASE}/bundles/",
            context="search_offers",
            json_body=payload,
        )
        self._raise_for_status(resp, "search_offers")
        result = self._decode_response(resp, SearchOffersResponse, "search_offers")
        if not result.offers:
            logger.warning("vastai.search_offers.no_capacity", hardware=str(hardware))
            raise NoCapacityError("No GPU offers match the requested hardware requirements")
        logger.info(
            "vastai.search_offers.found",
            count=len(result.offers),
            cheapest_dph_micros=result.offers[0].dph_total_micros,
        )
        return result.offers

    async def create_instance(
        self,
        offer_id: int,
        *,
        disk_gb: int,
        env: dict[str, str],
        template_hash_id: str | None = None,
        image: str | None = None,
        onstart_cmd: str | None = None,
    ) -> int:
        """Create a Vast.ai instance from an offer.

        Args:
            offer_id: Offer ID from search results.
            disk_gb: Disk allocation in GB.
            env: Environment variables dict (includes port mappings as "-p X:X": "1").
            template_hash_id: Vast.ai template hash. When set, image and onstart_cmd
                are omitted (template provides them).
            image: Docker image (e.g. "vastai/comfy:latest"). Required when
                template_hash_id is None.
            onstart_cmd: Onstart script command. Optional when using template.

        Returns:
            Instance ID.

        Raises:
            ValueError: If neither template_hash_id nor image is provided.
            OfferTakenError: If the offer was already rented.
            VastAIPaymentError: If account balance is insufficient.
            VastAIError: On other API errors.
        """
        payload: dict[str, Any] = {
            "disk": disk_gb,
            "runtype": "ssh_direct",
            "env": env,
        }
        if template_hash_id is not None:
            payload["template_hash_id"] = template_hash_id
        else:
            if image is None:
                raise ValueError("create_instance requires either template_hash_id or image")
            payload["image"] = image
            if onstart_cmd is not None:
                payload["onstart"] = onstart_cmd
        resp = await self._request_with_retry(
            "put",
            f"{self._VASTAI_API_BASE}/asks/{offer_id}/",
            context=f"create_instance(offer_id={offer_id})",
            json_body=payload,
        )
        if resp.status_code == 409:
            logger.warning("vastai.create_instance.offer_taken", offer_id=offer_id)
            raise OfferTakenError(
                f"Offer {offer_id} was already rented", status_code=resp.status_code
            )
        if resp.status_code == 402:
            logger.warning("vastai.create_instance.payment_error", offer_id=offer_id)
            raise VastAIPaymentError(
                "Vast.ai account has insufficient credits", status_code=resp.status_code
            )
        self._raise_for_status(resp, f"create_instance(offer_id={offer_id})")
        result = self._decode_response(
            resp, CreateInstanceResponse, f"create_instance(offer_id={offer_id})"
        )
        instance_id = result.new_contract
        logger.info(
            "vastai.create_instance.success",
            offer_id=offer_id,
            instance_id=instance_id,
        )
        return instance_id

    async def get_instance(self, instance_id: int) -> VastAIInstance | None:
        """Get current state of an instance, or ``None`` if Vast.ai has no detail yet.

        Returns ``None`` when Vast.ai responds with ``{"instances": null}`` — observed
        transiently when an instance is mid-creation, the container image is still
        pulling, or as a tombstone for recently-destroyed instances. Callers should
        treat ``None`` as "not ready yet" and fall through to their timeout logic.

        Raises:
            InstanceNotFoundError: If instance doesn't exist (HTTP 404).
            VastAIError: On other API errors.
        """
        resp = await self._request_with_retry(
            "get",
            f"{self._VASTAI_API_BASE}/instances/{instance_id}/",
            context=f"get_instance(instance_id={instance_id})",
        )
        if resp.status_code == 404:
            raise InstanceNotFoundError(f"Instance {instance_id} not found", status_code=404)
        self._raise_for_status(resp, f"get_instance(instance_id={instance_id})")
        # The /instances/{id}/ endpoint wraps the instance in an "instances" key (despite
        # being a single-instance lookup). See https://docs.vast.ai/api/show-instance and
        # vastai/async_/client.py:190-204 in the official SDK.
        envelope = self._decode_response(
            resp, GetInstanceResponse, f"get_instance(instance_id={instance_id})"
        )
        if envelope.instances is None:
            logger.info(
                "vastai.get_instance.no_detail_yet",
                instance_id=instance_id,
            )
            return None
        logger.info(
            "vastai.get_instance.success",
            instance_id=instance_id,
            actual_status=envelope.instances.actual_status,
            cur_state=envelope.instances.cur_state,
        )
        return envelope.instances

    async def stop_instance(self, instance_id: int) -> None:
        """Stop (pause) an instance — retains disk, stops GPU billing.

        Raises:
            VastAIError: On API errors.
        """
        resp = await self._request_with_retry(
            "put",
            f"{self._VASTAI_API_BASE}/instances/{instance_id}/",
            context=f"stop_instance(instance_id={instance_id})",
            json_body={"state": "stopped"},
        )
        self._raise_for_status(resp, f"stop_instance(instance_id={instance_id})")
        logger.info("vastai.stop_instance.success", instance_id=instance_id)

    async def start_instance(self, instance_id: int) -> None:
        """Start (resume) a stopped instance.

        Raises:
            VastAIError: On API errors.
        """
        resp = await self._request_with_retry(
            "put",
            f"{self._VASTAI_API_BASE}/instances/{instance_id}/",
            context=f"start_instance(instance_id={instance_id})",
            json_body={"state": "running"},
        )
        self._raise_for_status(resp, f"start_instance(instance_id={instance_id})")
        logger.info("vastai.start_instance.success", instance_id=instance_id)

    async def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance permanently. Idempotent — ignores 404.

        Raises:
            VastAIError: On non-404 API errors.
        """
        resp = await self._request_with_retry(
            "delete",
            f"{self._VASTAI_API_BASE}/instances/{instance_id}/",
            context=f"destroy_instance(instance_id={instance_id})",
        )
        if resp.status_code == 404:
            logger.info("vastai.destroy_instance.already_gone", instance_id=instance_id)
            return
        self._raise_for_status(resp, f"destroy_instance(instance_id={instance_id})")
        logger.info("vastai.destroy_instance.success", instance_id=instance_id)
