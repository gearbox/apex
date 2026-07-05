"""Aisha generation provider — adapter for ComfyUI WorkflowService."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.schemas.generation import DEFAULT_NEGATIVE_PROMPT, GenerationRequest
from src.api.services.bundle_index import BundleIndexService, BundleNotFoundError
from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.generation.service import ProviderResponseError
from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    validate_tunnel_hostname,
)
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.api.services.image_normalization import (
    ImageNormalizationError,
    ensure_comfyui_input,
    sniff_format,
)
from src.api.services.storage import R2StorageService
from src.api.services.workflow_service import WorkflowService
from src.core.enums import JobStatus, ModelType, Provider, Sampler, Scheduler
from src.core.generation_config import BundleGenerationConfig
from src.core.resolution import TIER_MEGAPIXELS, resolve_dimensions
from src.core.uid import new_id
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

# Sanitization for ComfyUI input filenames. ComfyUI writes uploaded files to
# its `input/` folder by name; any path separators or shell-meta characters in
# a user-supplied filename could escape that directory. We strip directory
# components, restrict to a conservative ASCII charset, and cap length.
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_SAFE_FILENAME_LEN = 64


def _sanitize_filename(name: str | None) -> str:
    """Reduce ``name`` to a safe-for-filesystem leaf name.

    - Strips directory components (``../``, ``foo/bar``) by taking the
      basename only.
    - Replaces any character outside ``[A-Za-z0-9._-]`` with ``_``.
    - Caps length at 64 chars to bound disk-name reasonableness.
    - Falls back to ``"image"`` if the input is None, empty, or fully
      sanitized to nothing (e.g. ``"///"``).

    The schema currently declares ``UserImage.original_filename`` and
    ``Output.storage_key`` NOT NULL, so ``None`` shouldn't be reachable
    in production today — but accepting ``str | None`` costs one
    falsy-check and defends against future schema relaxations or
    code paths that might introduce nullable filename sources.

    The job_id suffix added by the caller still guarantees uniqueness
    across concurrent jobs; this helper just guards against malicious
    or malformed input names propagating to the GPU node's filesystem.
    """
    if not name:
        return "image"
    # Take the basename — defeats absolute paths, ``../`` traversal, and
    # OS-mismatched separators.
    leaf = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    sanitized = _SAFE_FILENAME_CHARS.sub("_", leaf)[:_MAX_SAFE_FILENAME_LEN]
    return sanitized or "image"


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.gpu_session.service import GpuSessionService
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)


class AishaGenerationProvider:
    """Adapts the ComfyUI stack (WorkflowService) to the GenerationProvider protocol.

    Each submit() call resolves the user's active GPU session, builds a
    per-request ComfyUIClient targeting that session's tunnel hostname,
    and closes the client when done — no shared client state.
    """

    def __init__(
        self,
        workflow_service: WorkflowService,
        gpu_session_service: GpuSessionService | None,
        bundle_index: BundleIndexService,
        r2_storage: R2StorageService | None = None,
        tunnel_domain: str = "",
        tunnel_hostname_allowed_prefix: str | None = None,
    ) -> None:
        self._workflow = workflow_service
        self._gpu_session_service = gpu_session_service
        # R2 is used to fetch source bytes for image-to-image: we download
        # the user image / prior output, then upload it to ComfyUI's input
        # folder via the per-session tunnel. Optional only because some
        # test setups don't wire it; in production it must be configured.
        self._r2 = r2_storage
        # Allowlist suffix for the tunnel hostname. Defense in depth: the write
        # path (CloudflareTunnelClient.create_session_tunnel) only ever produces
        # hostnames of the form ``{id_short}.{tunnel_domain}``, but we re-check
        # at outbound-request time so a corrupted DB row, bad migration, or a
        # future code path that writes the column can't cause SSRF.
        self._tunnel_domain = tunnel_domain
        self._tunnel_hostname_allowed_prefix = tunnel_hostname_allowed_prefix
        self._bundle_index = bundle_index

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """Aisha-specific validation beyond what the enum provides."""
        if request.model == ModelType.AISHA_VIDEO:
            raise ValueError("Aisha video generation is not yet available via the unified endpoint")

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
                f"{name} {value!r} is not allowed for this model; allowed: {sorted(v.value for v in allowed)}"
            )
        return value

    async def _resolve_input_image(
        self,
        request: UnifiedGenerationRequest,
        session: AsyncSession,
    ) -> tuple[bytes, str] | None:
        """Resolve the I2I input image to (bytes, original_filename).

        Mirrors ``GrokGenerationProvider._resolve_input_url`` but downloads
        the actual bytes (ComfyUI is behind our tunnel and cannot fetch a
        presigned URL from R2 — we have to push the bytes through). Returns
        None when the request has no input image (T2I path).

        Source priority matches Grok:
        1. ``source_output_id`` — a prior generation's output
        2. ``input_image_id`` — a user-uploaded image

        Mutual exclusion is also enforced at the orchestrator layer; the
        defensive check here turns provider misuse (a future direct caller
        bypassing the orchestrator) into a clear local error rather than a
        silently-ignored field.
        """
        # Defensive: orchestrator already rejects this combination, but a
        # future direct caller of the provider deserves a clear local error
        # rather than a silently-ignored input_image_id.
        if request.source_output_id is not None and request.input_image_id is not None:
            raise ValueError("source_output_id and input_image_id are mutually exclusive")

        # Fast path for T2I: no input image, no R2 dependency exercised.
        if request.source_output_id is None and request.input_image_id is None:
            return None

        if self._r2 is None:
            # Should never happen in production (DI always wires R2), but
            # if we got here for I2I without R2 it's a config error, not a
            # user error. Surface as ProviderResponseError with an
            # infrastructure-agnostic message.
            raise ProviderResponseError(
                "Image-to-image is unavailable on this deployment. "
                "Please contact support if this persists."
            )

        if request.source_output_id is not None:
            output = await OutputRepository(session).get(request.source_output_id)
            if output is None:
                raise ValueError(f"Source output {request.source_output_id} not found")
            image_bytes = await self._r2.download(output.storage_key)
            filename = output.storage_key.rsplit("/", 1)[-1]
        else:
            # input_image_id is not None at this point (mutual exclusion enforced
            # above and the early-return handled the both-None case).
            assert request.input_image_id is not None
            user_image = await UserImageRepository(session).get(request.input_image_id)
            if user_image is None:
                raise ValueError(f"Input image {request.input_image_id} not found")
            image_bytes = await self._r2.download(user_image.storage_key)
            filename = user_image.original_filename

        # ComfyUI bridge: anything that isn't PNG/JPEG (including static WebP,
        # which Grok outputs and ComfyUI handles unreliably) is converted to
        # PNG here. This covers the source_output_id -> Grok-WEBP-output remix
        # path that upload-time normalization cannot fix.
        try:
            normalized = await ensure_comfyui_input(image_bytes)
        except ImageNormalizationError as e:
            logger.warning(
                "aisha.input_normalization_failed",
                sniffed=sniff_format(image_bytes).value,
                size_bytes=len(image_bytes),
                source_output_id=str(request.source_output_id)
                if request.source_output_id
                else None,
                input_image_id=str(request.input_image_id) if request.input_image_id else None,
                error=str(e),
            )
            raise ValueError("Input image is not decodable") from e

        stem = filename.rsplit(".", 1)[0]
        filename = f"{stem}.{normalized.format.value}"
        return normalized.data, filename

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
        source_output_id: UUID | None = None,
    ) -> GenerationJob:
        """Resolve GPU session, build per-request ComfyUI client, queue workflow."""
        job_id = new_id()

        # 1. Resolve active GPU session for this model
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

        # I2I bridge: fetch input image bytes from R2 BEFORE billing.
        # Resolution failures (missing row, mutually-exclusive inputs,
        # missing R2 dependency) are user/config errors that should not
        # leave a debit on the ledger requiring later refund. Validation
        # via ValueError surfaces as a 4xx; ProviderResponseError surfaces
        # as a 5xx. Either way: no reservation made, nothing to refund.
        i2i_input = await self._resolve_input_image(request, session)

        # Resolve per-model generation config from bundle.yaml (before billing)
        gen_cfg: BundleGenerationConfig = self._bundle_index.get_generation_config(
            gpu_session.bundle_name, gpu_session.bundle_version
        )

        # Resolve image dimensions
        tier = request.image_resolution or gen_cfg.defaults.resolution
        has_explicit_dims = request.width is not None and request.height is not None
        dims = resolve_dimensions(
            aspect_ratio=request.aspect_ratio,
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
                    aspect_ratio=request.aspect_ratio.value,
                )

        # Resolve + validate sampler params (raises ValueError → 4xx, no reservation made)
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

        # Map unified request -> legacy GenerationRequest for workflow_service
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

        # Create DB job record
        db_job = await JobRepository(session).create(
            id=job_id,
            user_id=user_id,
            name=legacy_request.name or "Untitled",
            prompt=request.prompt,
            generation_type=request.generation_type,
            status=JobStatus.PENDING,
            provider=Provider.AISHA,
            model=request.model.value,
            aspect_ratio=request.aspect_ratio.value,
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
            input_image_id=request.input_image_id,
            gpu_session_id=gpu_session.id,
        )

        # Billing reservation
        txn = await billing_service.check_and_reserve(
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
            db_job.debit_transaction_id = txn.id
        await session.flush()

        # 2. Build a session-scoped ComfyUI client.
        # SSRF guard: validate tunnel hostname against the configured domain.
        hostname = (gpu_session.tunnel_hostname or "").lower()
        try:
            validate_tunnel_hostname(
                hostname,
                allowed_suffix=f".{self._tunnel_domain}",
                allowed_prefix=self._tunnel_hostname_allowed_prefix,
            )
        except InvalidTunnelHostnameError as e:
            logger.error(
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

        base_url = f"https://{hostname}"
        client = ComfyUIClient(base_url)
        try:
            await client.connect()

            # If this is an I2I request, push the source bytes to ComfyUI's
            # input folder before queuing the workflow. ComfyUI returns the
            # filename it stored, which we wire into the LoadImage node via
            # apply_parameters(input_image_1=...). Job-scoped prefix prevents
            # concurrent jobs on the same node from clobbering each other.
            uploaded_filename: str | None = None
            if i2i_input is not None:
                image_bytes, source_filename = i2i_input
                # Sanitize the source filename: it derives from user-supplied
                # original_filename or an R2 storage_key tail, neither of
                # which is guaranteed safe for ComfyUI's input/ folder. We
                # strip any directory components and restrict to a
                # conservative ASCII charset before composing the upload
                # name. The job_id suffix preserves uniqueness across
                # concurrent jobs on the same node.
                safe_source = _sanitize_filename(source_filename)
                stem, _, ext = safe_source.rpartition(".")
                comfyui_filename = f"input_{stem or safe_source}_{str(job_id)[:8]}.{ext or 'png'}"
                upload_result = await client.upload_image(
                    image_data=image_bytes,
                    filename=comfyui_filename,
                )
                # ComfyUI returns the actual stored name; trust it over our request.
                uploaded_filename = upload_result.get("name", comfyui_filename)
                logger.info(
                    "aisha.i2i.image_uploaded",
                    job_id=str(job_id),
                    gpu_session_id=str(gpu_session.id),
                    requested_name=comfyui_filename,
                    stored_name=uploaded_filename,
                    bytes=len(image_bytes),
                )

            # Build and queue workflow from the provisioned bundle's cache
            try:
                bundle_dir = self._bundle_index.get_bundle_path(gpu_session.bundle_name)
            except BundleNotFoundError as exc:
                raise NoActiveSessionError(
                    f"Bundle '{gpu_session.bundle_name}' not found in index. "
                    "Wait for the bundle index to sync and try again."
                ) from exc
            workflow = self._workflow.load_workflow_from_bundle(
                bundle_dir, gpu_session.bundle_version
            )
            self._workflow.validate_workflow(workflow)
            self._workflow.inject_checkpoint(
                workflow,
                gpu_session.bundle_name,
                gpu_session.bundle_version,
                session_id=str(gpu_session.id),
            )
            configured = self._workflow.apply_parameters(
                workflow=workflow,
                request=legacy_request,
                input_image_1=uploaded_filename,
                filename_prefix=f"gen_{str(job_id)[:8]}",
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
            # The backend accepted the request but returned no job handle we can
            # poll. Mark the job FAILED so the user can see the failure via
            # GET /v1/jobs/{id}, flush so the error_message persists alongside
            # the debit, then raise. The orchestrator's except-path refunds the
            # debit and wraps this into a user-visible GenerationError. We
            # deliberately keep the user-facing message infrastructure-agnostic;
            # the structlog entry below captures the backend detail for ops.
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
        return db_job
