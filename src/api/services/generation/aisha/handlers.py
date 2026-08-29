"""Media-specific handlers used by the Aisha generation-provider facade."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, Protocol

import structlog

from src.api.schemas.generation import DEFAULT_NEGATIVE_PROMPT, GenerationRequest
from src.api.services.bundle_index import BundleIndexService, BundleNotFoundError
from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.service import FeatureNotSupportedError, ProviderResponseError
from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    validate_tunnel_hostname,
)
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.api.services.image_normalization import (
    ImageNormalizationError,
    ImageTooLargeError,
    ensure_comfyui_input,
    read_image_dimensions,
    sniff_format,
)
from src.api.services.workflow.contract import MediaSlot
from src.core.enums import AspectRatio, JobStatus, MediaKind, Provider, Sampler, Scheduler
from src.core.resolution import TIER_MEGAPIXELS, resolve_dimensions
from src.core.uid import new_id
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.api.services.gpu_session.service import GpuSessionService
    from src.api.services.storage import R2StorageService
    from src.api.services.workflow import WorkflowService
    from src.api.services.workflow.contract import BundleCapabilities
    from src.core.generation_config import BundleGenerationConfig

logger = structlog.get_logger(__name__)

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_SAFE_FILENAME_LEN = 64


def _aspect_log_value(effective_aspect: AspectRatio | tuple[int, int]) -> str:
    if isinstance(effective_aspect, AspectRatio):
        return effective_aspect.value
    w, h = effective_aspect
    return f"{w}:{h}"


def _sanitize_filename(name: str | None) -> str:
    """Reduce ``name`` to a safe-for-filesystem leaf name."""
    if not name:
        return "image"
    leaf = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    sanitized = _SAFE_FILENAME_CHARS.sub("_", leaf)[:_MAX_SAFE_FILENAME_LEN]
    return sanitized or "image"


class UnsupportedMediaError(FeatureNotSupportedError):
    """A bundle output kind has no registered Aisha submit handler."""


class AishaMediaHandler(Protocol):
    """Per-output-media submit strategy owned by the Aisha facade."""

    media: ClassVar[MediaKind]

    def validate(
        self,
        request: UnifiedGenerationRequest,
        capabilities: BundleCapabilities | None,
    ) -> None:
        """Validate media-specific request semantics."""
        ...

    async def prepare_inputs(
        self,
        source_media: Sequence[ResolvedSourceMedia],
        *,
        client: ComfyUIClient,
        job_id: UUID,
        gpu_session_id: UUID,
        resolved_input: tuple[bytes, str] | None = None,
    ) -> dict[MediaSlot, list[str]]:
        """Upload resolved source media and return filenames keyed by slot."""
        ...

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        source_media: Sequence[ResolvedSourceMedia] = (),
    ) -> ProviderSubmitResult:
        """Submit a request for this output medium."""
        ...


class AishaImageGenerationHandler:
    """Own the image-specific Aisha/ComfyUI generation flow."""

    media: ClassVar[MediaKind] = MediaKind.IMAGE

    def __init__(
        self,
        workflow_service: WorkflowService,
        gpu_session_service: GpuSessionService | None,
        bundle_index: BundleIndexService,
        r2_storage: R2StorageService | None,
        tunnel_domain: str,
        tunnel_hostname_allowed_prefix: str | None,
        max_input_megapixels: float,
    ) -> None:
        self._workflow = workflow_service
        self._gpu_session_service = gpu_session_service
        self._bundle_index = bundle_index
        self._r2 = r2_storage
        self._tunnel_domain = tunnel_domain
        self._tunnel_hostname_allowed_prefix = tunnel_hostname_allowed_prefix
        self._max_input_megapixels = max_input_megapixels

    def validate(
        self,
        request: UnifiedGenerationRequest,
        capabilities: BundleCapabilities | None,
    ) -> None:
        """Image flow has no additional validation beyond its bound bundle."""

    @staticmethod
    def _resolve_int(value: int | None, default: int, min_val: int, max_val: int, name: str) -> int:
        if value is None:
            return default
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} {value} out of range [{min_val}, {max_val}] for this model")
        return value

    @staticmethod
    def _resolve_float(
        value: float | None, default: float, min_val: float, max_val: float, name: str
    ) -> float:
        if value is None:
            return default
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} {value} out of range [{min_val}, {max_val}] for this model")
        return value

    @staticmethod
    def _resolve_enum(
        value: Sampler | Scheduler | None,
        default: Sampler | Scheduler,
        allowed: frozenset[Sampler] | frozenset[Scheduler],
        name: str,
    ) -> Sampler | Scheduler:
        if value is None:
            return default
        if allowed and value not in allowed:
            raise ValueError(
                f"{name} {value!r} is not allowed for this model; allowed: "
                f"{sorted(v.value for v in allowed)}"
            )
        return value

    async def _resolve_input_image(
        self,
        source_media: Sequence[ResolvedSourceMedia],
    ) -> tuple[bytes, str] | None:
        """Download and normalize one resolved source image for ComfyUI."""
        if not source_media:
            return None
        if len(source_media) != 1:
            raise FeatureNotSupportedError("Aisha accepts exactly one source media asset")
        if self._r2 is None:
            raise ProviderResponseError(
                "Image-to-image is unavailable on this deployment. "
                "Please contact support if this persists."
            )

        source = source_media[0]
        image_bytes = await self._r2.download(source.storage_key)
        filename = source.storage_key.rsplit("/", 1)[-1]
        try:
            normalized = await ensure_comfyui_input(
                image_bytes, max_megapixels=self._max_input_megapixels
            )
        except ImageTooLargeError as e:
            logger.warning(
                "aisha.input_too_large",
                sniffed=sniff_format(image_bytes).value,
                size_bytes=len(image_bytes),
                megapixels=e.megapixels,
                limit=e.limit,
                source_position=source.position,
            )
            raise ValueError("Input image exceeds maximum pixel count") from e
        except ImageNormalizationError as e:
            logger.warning(
                "aisha.input_normalization_failed",
                sniffed=sniff_format(image_bytes).value,
                size_bytes=len(image_bytes),
                source_position=source.position,
                error=str(e),
            )
            raise ValueError("Input image is not decodable") from e

        stem = filename.rsplit(".", 1)[0]
        return normalized.data, f"{stem}.{normalized.format.value}"

    @staticmethod
    async def _resolve_effective_aspect_ratio(
        request: UnifiedGenerationRequest,
        i2i_input: tuple[bytes, str] | None,
    ) -> AspectRatio | tuple[int, int]:
        """Resolve an explicit aspect, source aspect, or the 1:1 image default."""
        if request.aspect_ratio is not None:
            return request.aspect_ratio
        if i2i_input is not None:
            try:
                src_w, src_h = await read_image_dimensions(i2i_input[0])
            except ImageNormalizationError as e:
                raise ValueError("Input image is not decodable") from e
            gcd = math.gcd(src_w, src_h)
            return (src_w // gcd, src_h // gcd)
        return AspectRatio.RATIO_1_1

    async def prepare_inputs(
        self,
        source_media: Sequence[ResolvedSourceMedia],
        *,
        client: ComfyUIClient,
        job_id: UUID,
        gpu_session_id: UUID,
        resolved_input: tuple[bytes, str] | None = None,
    ) -> dict[MediaSlot, list[str]]:
        """Upload source media and return the declared reference-slot mapping."""
        if not source_media:
            return {}
        i2i_input = resolved_input
        if i2i_input is None:
            i2i_input = await self._resolve_input_image(source_media)
        if i2i_input is None:
            return {}

        image_bytes, source_filename = i2i_input
        safe_source = _sanitize_filename(source_filename)
        stem, _, ext = safe_source.rpartition(".")
        comfyui_filename = f"input_{stem or safe_source}_{str(job_id)[:8]}.{ext or 'png'}"
        upload_result = await client.upload_image(
            image_data=image_bytes,
            filename=comfyui_filename,
        )
        stored_filename = upload_result.get("name", comfyui_filename)
        logger.info(
            "aisha.i2i.image_uploaded",
            job_id=str(job_id),
            gpu_session_id=str(gpu_session_id),
            requested_name=comfyui_filename,
            stored_name=stored_filename,
            bytes=len(image_bytes),
        )
        return {MediaSlot.REFERENCE: [stored_filename]}

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        source_media: Sequence[ResolvedSourceMedia] = (),
    ) -> ProviderSubmitResult:
        """Resolve the session, configure the image workflow, and queue it."""
        job_id = new_id()
        if self._gpu_session_service is None:
            raise NoActiveSessionError(
                "GPU sessions are not configured on this server. "
                "Vast.ai and Cloudflare Tunnel settings are required."
            )
        gpu_session = await self._gpu_session_service.get_active_session_for_model(
            user_id=user_id,
            product_id=product_id,
            model_type=request.model,
        )
        if gpu_session is None:
            raise NoActiveSessionError(
                f"No active GPU session for model {request.model.value}. "
                "Start a session first via POST /v1/sessions."
            )

        # Resolve the source before billing so invalid input cannot leave a debit.
        i2i_input = await self._resolve_input_image(source_media)
        effective_aspect = await self._resolve_effective_aspect_ratio(request, i2i_input)
        gen_cfg: BundleGenerationConfig = self._bundle_index.get_generation_config(
            gpu_session.bundle_name, gpu_session.bundle_version
        )

        tier = request.image_resolution or gen_cfg.defaults.resolution
        has_explicit_dims = request.width is not None and request.height is not None
        dims = resolve_dimensions(
            aspect_ratio=effective_aspect,
            max_megapixels=gen_cfg.constraints.max_megapixels,
            latent_multiple=gen_cfg.constraints.latent_multiple,
            max_edge=gen_cfg.constraints.max_edge,
            tier=None if has_explicit_dims else tier,
            explicit_width=request.width,
            explicit_height=request.height,
        )
        if not has_explicit_dims:
            requested_mp = TIER_MEGAPIXELS[tier]
            if dims.megapixels < requested_mp * 0.9:
                logger.info(
                    "generation.resolution.clamped",
                    tier=tier.value,
                    requested_mp=requested_mp,
                    resolved_mp=round(dims.megapixels, 3),
                    max_megapixels=gen_cfg.constraints.max_megapixels,
                    aspect_ratio=_aspect_log_value(effective_aspect),
                )

        steps = self._resolve_int(
            request.steps,
            gen_cfg.defaults.steps,
            gen_cfg.constraints.min_steps,
            gen_cfg.constraints.max_steps,
            "steps",
        )
        cfg = self._resolve_float(
            request.cfg,
            gen_cfg.defaults.cfg,
            gen_cfg.constraints.min_cfg,
            gen_cfg.constraints.max_cfg,
            "cfg",
        )
        sampler = self._resolve_enum(
            request.sampler,
            gen_cfg.defaults.sampler,
            gen_cfg.constraints.allowed_samplers,
            "sampler",
        )
        scheduler = self._resolve_enum(
            request.scheduler,
            gen_cfg.defaults.scheduler,
            gen_cfg.constraints.allowed_schedulers,
            "scheduler",
        )
        denoise = request.denoise if request.denoise is not None else gen_cfg.defaults.denoise

        legacy_request = GenerationRequest(
            prompt=request.prompt,
            name=request.name,
            negative_prompt=request.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            height=dims.height,
            width=dims.width,
            aspect_ratio=request.aspect_ratio,
            model_type=request.model,
            generation_type=request.generation_type,
            max_images=request.n,
            seed=request.seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler.value,
            scheduler=scheduler.value,
            denoise=denoise,
        )

        db_job = await JobRepository(session).create(
            id=job_id,
            user_id=user_id,
            name=legacy_request.name or "Untitled",
            prompt=request.prompt,
            generation_type=request.generation_type,
            status=JobStatus.PENDING,
            provider=Provider.AISHA,
            model=request.model.value,
            aspect_ratio=request.aspect_ratio.value if request.aspect_ratio else None,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
            gpu_session_id=gpu_session.id,
        )

        reserve_result = await billing_service.check_and_reserve(
            account_id,
            token_cost,
            job_id,
            metadata={
                "type": "generation",
                "provider": Provider.AISHA.value,
                "generation_type": request.generation_type.value,
                "model": request.model.value,
            },
            description="Generation charge",
            session=session,
            product_id=product_id,
        )
        if db_job is not None:
            db_job.token_cost = token_cost
            db_job.debit_transaction_id = reserve_result.txn.id
        await session.flush()

        hostname = (gpu_session.tunnel_hostname or "").lower()
        try:
            validate_tunnel_hostname(
                hostname,
                allowed_suffix=f".{self._tunnel_domain}",
                allowed_prefix=self._tunnel_hostname_allowed_prefix,
            )
        except InvalidTunnelHostnameError as e:
            logger.exception(
                "aisha.tunnel_hostname.allowlist_violation",
                job_id=str(job_id),
                gpu_session_id=str(gpu_session.id),
                tunnel_hostname=hostname,
            )
            if db_job is not None:
                db_job.status = JobStatus.FAILED
                db_job.error_message = "Generation backend session is misconfigured."
                await session.flush()
            raise ProviderResponseError(
                "Your generation session is misconfigured. "
                "Please start a new session and try again."
            ) from e

        client = ComfyUIClient(f"https://{hostname}")
        try:
            await client.connect()
            media_filenames = await self.prepare_inputs(
                source_media,
                client=client,
                job_id=job_id,
                gpu_session_id=gpu_session.id,
                resolved_input=i2i_input,
            )
            try:
                bundle_dir = self._bundle_index.get_bundle_path(gpu_session.bundle_name)
            except BundleNotFoundError as exc:
                raise NoActiveSessionError(
                    f"Bundle '{gpu_session.bundle_name}' not found in index. "
                    "Wait for the bundle index to sync and try again."
                ) from exc
            bound = self._workflow.load(bundle_dir, gpu_session.bundle_version)
            configured = self._workflow.apply(
                bound,
                request=legacy_request,
                media_filenames=media_filenames,
                filename_prefix=f"gen_{str(job_id)[:8]}",
                bundle_name=gpu_session.bundle_name,
                bundle_version=gpu_session.bundle_version,
            )
            result = await client.queue_prompt(configured)
        finally:
            await client.close()

        if prompt_id := result.get("prompt_id"):
            if db_job is not None:
                db_job.status = JobStatus.QUEUED
                db_job.started_at = datetime.now(UTC)
                db_job.external_request_id = prompt_id
        elif db_job is not None:
            db_job.status = JobStatus.FAILED
            db_job.error_message = "Generation backend returned an unexpected response."
            await session.flush()
            logger.error(
                "aisha.queue_prompt.no_prompt_id",
                job_id=str(job_id),
                gpu_session_id=str(gpu_session.id),
                tunnel_hostname=gpu_session.tunnel_hostname,
                response_keys=list(result.keys()) if isinstance(result, dict) else None,
            )
            raise ProviderResponseError(
                "The generation backend returned an unexpected response. "
                "Your tokens have been refunded. "
                "Please try again or report the error if it persists."
            )

        if db_job is None:
            raise ValueError("Failed to create Aisha job record")
        await session.flush()
        return ProviderSubmitResult(
            job=db_job,
            balance_after=reserve_result.txn.balance_after,
            balance_event=reserve_result.event,
        )
