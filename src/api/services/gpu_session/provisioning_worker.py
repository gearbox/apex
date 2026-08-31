"""Background worker: advances non-terminal GPU sessions toward active or failed.

Started in app lifespan (Phase 1F). One sweep per
``settings.gpu_provision_poll_interval_seconds`` (default 15s).

Per sweep, processes sessions in these statuses:
  - pending      → poll Vast.ai until 'running', transition to provisioning
  - provisioning → probe ComfyUI via CF tunnel, transition to active
                   (on timeout: retry with new node OR fail)
  - resuming     → probe ComfyUI via CF tunnel, transition to active
                   (on timeout: fail — no retry for resume)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import secrets
import time
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import structlog

from src.api.services.jobs.sweep import JobSweepFailure
from src.api.services.vastai.exceptions import InstanceNotFoundError
from src.core.enums import STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES, GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository
from src.workers.base import PeriodicWorker

from ._env_builder import build_acs_env
from ._events import publish_status_event
from ._provisioning import make_onstart_cmd, provision_vastai_instance

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BillingService
    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.api.services.event_bus import EventBus
    from src.api.services.gpu_session.node_cooldown import NodeCooldownStore
    from src.api.services.jobs.sweep import JobSweepService
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.vastai.client import VastAIClient
    from src.api.services.vastai.schemas import VastAIInstance
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

_PROBE_TIMEOUT_SECONDS = 10.0

# Reason strings passed to _retry_or_fail — defined as constants so the terminal guard
# and every call site share a single source of truth (no typo risk, grep-friendly).
_REASON_PROVISIONING_TIMEOUT = "provisioning_timeout"
_REASON_PENDING_TIMEOUT = "pending_timeout"
_REASON_PENDING_TIMEOUT_AFTER_ERRORS = "pending_timeout_after_errors"
# Retryable: stall = no callbacks for stall_timeout; node_reported_failure = node said it failed.
# Both are eligible for recreation (unlike provisioning_timeout which stays terminal).
_REASON_PROVISIONING_STALLED = "provisioning_stalled"
_REASON_NODE_REPORTED_FAILURE = "node_reported_failure"
_READINESS_MODEL_TYPE = "checkpoints"

# Vast.ai actual_status values that mean the container is dead and will never reach 'running'.
# Sourced from Vast.ai docs: https://docs.vast.ai/sdk/python/quickstart — "if actual_status
# becomes 'exited' (container crashed), 'unknown' (no heartbeat from host), or 'offline'
# (host disconnected), it will never reach 'running'."
_TERMINAL_ACTUAL_STATUSES = frozenset({"exited", "unknown", "offline"})

# Vast.ai cur_state values that, combined with a non-'running' actual_status, mean the
# instance is dead. Observed empirically when a container fails to start (port allocation
# error etc.): actual_status='created' with cur_state='stopped'.
_TERMINAL_CUR_STATES = frozenset({"stopped", "exited"})


def _match_checkpoint(expected: str, available: list[str]) -> tuple[bool, bool]:
    """Return (matched, exact).

    ComfyUI lists checkpoints relative to the checkpoints dir, so a nested file
    appears as 'subdir/name.safetensors' while bundle.yaml declares the basename.
    Match exactly first; fall back to basename comparison (which also absorbs any
    stray './' or category prefix). exact=False on a basename-only match signals
    a subfolder placement that the workflow model-input application cannot
    address until the bundle declares that relative filename.
    """
    if expected in available:
        return True, True
    exp_base = PurePosixPath(expected).name
    if any(PurePosixPath(a).name == exp_base for a in available):
        return True, False
    return False, False


def _classify_terminal_state(instance: VastAIInstance) -> str | None:
    """Return a short reason string if the instance can never reach 'running', else None.

    Reasons are stable strings suitable for log filtering:
    'exited', 'unknown', 'offline', 'stopped_before_running'.

    actual_status takes precedence over cur_state — it's the more authoritative signal.
    """
    if instance.actual_status in _TERMINAL_ACTUAL_STATUSES:
        return instance.actual_status  # 'exited' | 'unknown' | 'offline'
    # Guard: if actual_status == 'running', trust it even if cur_state lags behind.
    if instance.actual_status != "running" and instance.cur_state in _TERMINAL_CUR_STATES:
        return "stopped_before_running"
    return None


def _require_cf_tunnel_id(session: GpuSession) -> str:
    if session.cf_tunnel_id is None:
        raise ValueError("session has no cf_tunnel_id")
    return session.cf_tunnel_id


class GpuProvisioningWorker(PeriodicWorker):
    """Advances non-terminal GPU sessions toward their terminal states.

    Started in app lifespan (Phase 1F). One sweep per
    ``settings.gpu_provision_poll_interval_seconds`` (default 15s).

    Per sweep, processes sessions in these statuses:
      - pending      → poll Vast.ai until 'running', transition to provisioning
      - provisioning → probe ComfyUI via CF tunnel, transition to active
                       (on timeout: retry with new node OR fail)
      - resuming     → probe ComfyUI via CF tunnel, transition to active
                       (on timeout: fail — no retry for resume)
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        vastai_client: VastAIClient,
        cf_client: CloudflareTunnelClient,
        bundle_index: BundleIndexService,
        http_client: httpx.AsyncClient,
        settings: Settings,
        cooldown_store: NodeCooldownStore,
        billing_service: BillingService | None = None,
        event_bus: EventBus | None = None,
        job_sweep_service: JobSweepService | None = None,
        ops_event_bus: OpsEventBus | None = None,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="gpu_provisioning",
            interval_seconds=settings.gpu_provision_poll_interval_seconds,
            jitter_seconds=2.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._session_factory = session_factory
        self._vastai = vastai_client
        self._cf = cf_client
        self._bundles = bundle_index
        self._http = http_client
        self._settings = settings
        self._cooldown = cooldown_store
        self._billing_service = billing_service
        self._event_bus = event_bus
        self._job_sweep = job_sweep_service
        self._ops_event_bus = ops_event_bus

    async def run_once(self) -> None:
        """One full sweep: query non-terminal sessions and advance each."""
        started = time.monotonic()

        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            sessions = list(
                await repo.list_by_status(
                    GpuSessionStatus.pending,
                    GpuSessionStatus.provisioning,
                    GpuSessionStatus.resuming,
                )
            )

        if not sessions:
            return

        counts: dict[str, int] = {}
        for s in sessions:
            counts[s.status] = counts.get(s.status, 0) + 1

        logger.info(
            "gpu_provisioning_worker.sweep_start",
            total=len(sessions),
            **counts,
        )

        if len(sessions) > 50:
            logger.warning(
                "gpu_provisioning_worker.large_session_count",
                count=len(sessions),
            )

        # Bound per-sweep concurrency. Settings enforces ge=1, so this is always
        # a valid positive int in production; tests that pass a misconstructed
        # Settings mock will fail fast here, which is the right behavior.
        semaphore = asyncio.Semaphore(self._settings.gpu_provision_worker_concurrency)

        async def _bounded(session: GpuSession) -> None:
            async with semaphore:
                await self._advance(session)

        results = await asyncio.gather(
            *[_bounded(s) for s in sessions],
            return_exceptions=True,
        )

        for s, result in zip(sessions, results, strict=True):
            if result is not None and isinstance(result, BaseException):
                logger.error(
                    "gpu_provisioning_worker.advance_error",
                    session_id=str(s.id),
                    exc_info=result,
                )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info("gpu_provisioning_worker.sweep_done", duration_ms=elapsed_ms)

    async def _advance(self, session: GpuSession) -> None:
        """Dispatch to per-status handler."""
        if session.status == GpuSessionStatus.pending:
            await self._advance_pending(session)
        elif session.status == GpuSessionStatus.provisioning:
            await self._advance_provisioning(session)
        elif session.status == GpuSessionStatus.resuming:
            await self._advance_resuming(session)

    # ------------------------------------------------------------------
    # Status-specific handlers
    # ------------------------------------------------------------------

    async def _advance_pending(self, session: GpuSession) -> None:
        """Poll Vast.ai until actual_status == 'running', then → provisioning."""
        if session.vastai_instance_id is None:
            logger.error(
                "gpu_session.provision.pending_without_instance_id",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "pending without vastai_instance_id")
            return

        instance: VastAIInstance | None
        try:
            instance = await self._vastai.get_instance(session.vastai_instance_id)
        except InstanceNotFoundError:
            logger.exception(
                "gpu_session.provision.instance_vanished",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )
            await self._mark_failed(session, "instance not found on Vast.ai")
            return
        except Exception:
            logger.exception(
                "gpu_session.provision.get_instance_error",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )
            # Transient error — fall through to timeout check rather than
            # silently swallowing so a stuck session eventually terminates.
            if self._provision_timeout_exceeded(session):
                logger.warning(
                    "gpu_session.provision.pending_timeout_after_errors",
                    session_id=str(session.id),
                )
                await self._retry_or_fail(session, reason=_REASON_PENDING_TIMEOUT_AFTER_ERRORS)
            return

        # Fast-fail on terminal Vast.ai states — no 20-min timeout wait.
        if await self._handle_terminal_state_if_any(session, instance, stage="pending"):
            return

        if instance is None or instance.actual_status != "running":
            if self._provision_timeout_exceeded(session):
                logger.warning(
                    "gpu_session.provision.pending_timeout",
                    session_id=str(session.id),
                )
                await self._retry_or_fail(session, reason=_REASON_PENDING_TIMEOUT)
            return

        await self._transition(
            session,
            new_status=GpuSessionStatus.provisioning,
            log_event="gpu_session.provision.instance_running",
        )

    async def _advance_provisioning(self, session: GpuSession) -> None:
        """Probe ComfyUI; on success → active; on terminal state → fail; on timeout → retry or fail."""
        if session.vastai_instance_id is None:
            await self._mark_failed(session, "provisioning without vastai_instance_id")
            return

        # Defensive fast-fail: did the container die after reaching 'running'?
        # get_instance errors are logged but don't break the probe path — the
        # ComfyUI probe is the authoritative health signal at this stage.
        instance: VastAIInstance | None
        try:
            instance = await self._vastai.get_instance(session.vastai_instance_id)
        except Exception as exc:
            logger.warning(
                "gpu_session.provision.vastai_get_instance_failed",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
                stage="provisioning",
                error=str(exc),
            )
            instance = None

        if await self._handle_terminal_state_if_any(session, instance, stage="provisioning"):
            return

        # Stall check (retryable): only applies when callbacks have been flowing.
        # If last_progress_at is null (old aisha, unset env, network gap), we fall
        # back to the fixed ceiling below — identical to pre-Phase-2 behavior.
        if session.last_progress_at is not None:
            stall_elapsed = datetime.now(UTC) - session.last_progress_at
            if stall_elapsed >= timedelta(
                seconds=self._settings.gpu_provision_stall_timeout_seconds
            ):
                logger.warning(
                    "gpu_session.provision.stalled",
                    session_id=str(session.id),
                    stall_seconds=int(stall_elapsed.total_seconds()),
                    stall_timeout=self._settings.gpu_provision_stall_timeout_seconds,
                )
                await self._retry_or_fail(session, reason=_REASON_PROVISIONING_STALLED)
                return

        # Fixed ceiling (terminal backstop): applies whether or not callbacks flow.
        if self._provision_timeout_exceeded(session):
            logger.warning(
                "gpu_session.provision.provisioning_timeout",
                session_id=str(session.id),
            )
            await self._retry_or_fail(session, reason=_REASON_PROVISIONING_TIMEOUT)
            return

        # Node-reported failure: receiver wrote provisioning_phase="failed"; worker
        # picks it up on next sweep and routes into the existing retry machinery.
        if session.provisioning_phase == "failed":
            logger.warning(
                "gpu_session.provision.node_reported_failure",
                session_id=str(session.id),
                error=str((session.provisioning_progress or {}).get("error", "")),
            )
            await self._retry_or_fail(session, reason=_REASON_NODE_REPORTED_FAILURE)
            return

        reachable = await self._probe_comfyui(session)
        if reachable:
            await self._mark_active(session)
        elif session.provisioning_phase == "ready":
            # Node reported terminal-ready but our authoritative probe disagrees:
            # strong signal of a provisioning defect (e.g. wrong/absent checkpoint).
            logger.warning(
                "gpu_session.provision.ready_probe_mismatch",
                session_id=str(session.id),
                bundle_name=session.bundle_name,
                bundle_version=session.bundle_version,
            )

    async def _advance_resuming(self, session: GpuSession) -> None:
        """Probe ComfyUI; on success → active; on timeout → fail (no retry for resume)."""
        if self._resuming_timeout_exceeded(session):
            logger.warning(
                "gpu_session.resume.timeout",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "resume_timeout")
            return

        reachable = await self._probe_comfyui(session)
        if reachable:
            await self._mark_active(session)

    # ------------------------------------------------------------------
    # Terminal-state helper
    # ------------------------------------------------------------------

    async def _handle_terminal_state_if_any(
        self,
        session: GpuSession,
        instance: VastAIInstance | None,
        *,
        stage: str,
    ) -> bool:
        """If the instance is in a terminal Vast.ai state, log and fast-fail via _retry_or_fail.

        Returns True if the session was failed, False if the caller should continue.

        Args:
            session: The session being advanced.
            instance: Result of get_instance, or None on transient errors.
            stage: Lifecycle stage ('pending' or 'provisioning') — included as a structured
                log field to distinguish callers without splitting the event name.
        """
        if instance is None:
            return False
        terminal_reason = _classify_terminal_state(instance)
        if terminal_reason is None:
            return False
        logger.warning(
            "gpu_session.provision.vastai_terminal_state",
            session_id=str(session.id),
            instance_id=session.vastai_instance_id,
            stage=stage,
            actual_status=instance.actual_status,
            cur_state=instance.cur_state,
            reason=terminal_reason,
        )
        await self._retry_or_fail(session, reason=f"vastai_terminal_state:{terminal_reason}")
        return True

    # ------------------------------------------------------------------
    # ComfyUI probe
    # ------------------------------------------------------------------

    async def _probe_comfyui(self, session: GpuSession) -> bool:
        """Probe ComfyUI for readiness via the CF tunnel.

        Returns True only when all applicable conditions hold:
        1. HTTP 200 from /object_info.
        2. Checkpoint presence (when bundle declares exactly one checkpoint):
           the declared filename must appear in ComfyUI's available list.
        3. Node-class marker (when configured): the marker class must be registered.
        4. Degradation backstop: if neither check is applicable, log a WARNING
           and return True on the 200 (unverifiable bundle, gap made loud).
        """
        if session.tunnel_hostname is None:
            logger.warning(
                "gpu_session.provision.probe_no_hostname",
                session_id=str(session.id),
            )
            return False

        url = f"https://{session.tunnel_hostname}/object_info"
        start = time.monotonic()
        try:
            resp = await self._http.get(url, timeout=_PROBE_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.info(
                "gpu_session.provision.probe_failed",
                session_id=str(session.id),
                hostname=session.tunnel_hostname,
                error_class=exc.__class__.__name__,
            )
            return False
        latency_ms = int((time.monotonic() - start) * 1000)

        if resp.status_code != 200:
            logger.info(
                "gpu_session.provision.probe",
                session_id=str(session.id),
                status_code=resp.status_code,
                reachable=False,
            )
            return False

        # Determine which checks are applicable before parsing JSON.
        expected = self._bundles.get_model_filenames(
            session.bundle_name, session.bundle_version, _READINESS_MODEL_TYPE
        )
        if expected is None:
            logger.warning(
                "gpu_session.provision.probe_checkpoint_lookup_failed",
                session_id=str(session.id),
                bundle_name=session.bundle_name,
                bundle_version=session.bundle_version,
            )
            return False
        checkpoint_check_applicable = len(expected) == 1
        marker = session.readiness_marker_node_class
        marker_check_applicable = marker is not None

        if not checkpoint_check_applicable:
            logger.info(
                "gpu_session.provision.probe_checkpoint_check_skipped",
                session_id=str(session.id),
                bundle_name=session.bundle_name,
                bundle_version=session.bundle_version,
                declared_count=len(expected),
            )
            # Degradation backstop: nothing concrete to verify → return on 200 alone.
            if not marker_check_applicable:
                logger.warning(
                    "gpu_session.provision.probe_unverifiable",
                    session_id=str(session.id),
                    bundle_name=session.bundle_name,
                    bundle_version=session.bundle_version,
                    hostname=session.tunnel_hostname,
                )
                return True

        # At least one check is applicable — parse the JSON body.
        try:
            parsed = msgspec.json.decode(resp.content)
        except msgspec.DecodeError as exc:
            logger.warning(
                "gpu_session.provision.probe_invalid_json",
                session_id=str(session.id),
                error=str(exc),
            )
            return False

        if not isinstance(parsed, dict):
            logger.warning(
                "gpu_session.provision.probe_unexpected_response",
                session_id=str(session.id),
                response_type=type(parsed).__name__,
            )
            return False

        # Step 2: checkpoint presence (single-checkpoint bundles only).
        if checkpoint_check_applicable:
            # Defensively extract available list; /object_info shape:
            # parsed["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            # is the list of filenames available to ComfyUI.
            shape_ok = True
            available: list[Any] = []
            ckpt_info = parsed.get("CheckpointLoaderSimple")
            if not isinstance(ckpt_info, dict):
                shape_ok = False
            else:
                inputs = ckpt_info.get("input")
                if not isinstance(inputs, dict):
                    shape_ok = False
                else:
                    required = inputs.get("required")
                    if not isinstance(required, dict):
                        shape_ok = False
                    else:
                        ckpt_name_entry = required.get("ckpt_name")
                        if not isinstance(ckpt_name_entry, list) or len(ckpt_name_entry) < 1:
                            shape_ok = False
                        else:
                            available = ckpt_name_entry[0]
                            if not isinstance(available, list):
                                shape_ok = False

            if not shape_ok:
                logger.warning(
                    "gpu_session.provision.probe_object_info_shape",
                    session_id=str(session.id),
                    bundle_name=session.bundle_name,
                    expected_checkpoint=expected[0],
                )
                return False

            matched, exact = _match_checkpoint(expected[0], available)
            if not matched:
                logger.info(
                    "gpu_session.provision.probe_checkpoint_missing",
                    session_id=str(session.id),
                    bundle_name=session.bundle_name,
                    expected=expected[0],
                    available_count=len(available),
                    available_sample=available[:5],
                )
                return False
            if not exact:
                # Readiness passes on basename, but the bundle must declare the
                # exposed subpath for its model-input application to use it.
                logger.warning(
                    "gpu_session.provision.probe_checkpoint_path_mismatch",
                    session_id=str(session.id),
                    bundle_name=session.bundle_name,
                    expected=expected[0],
                    available_sample=available[:5],
                )

        # Step 3: node-class marker (when configured).
        if marker_check_applicable:
            if marker not in parsed:
                class_names = list(parsed.keys())
                logger.info(
                    "gpu_session.provision.probe_marker_missing",
                    session_id=str(session.id),
                    marker=marker,
                    total_classes_registered=len(class_names),
                    sample_classes=class_names[:5],
                )
                return False
            logger.info(
                "gpu_session.provision.probe_marker_found",
                session_id=str(session.id),
                marker=marker,
                latency_ms=latency_ms,
            )

        logger.info(
            "gpu_session.provision.probe_ready",
            session_id=str(session.id),
            latency_ms=latency_ms,
            checkpoint_verified=checkpoint_check_applicable,
            marker_verified=marker_check_applicable,
        )
        return True

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _retry_or_fail(self, session: GpuSession, *, reason: str) -> None:
        """Attempt to provision a fresh Vast.ai node, OR mark failed if budget exhausted.

        With provisioning_recreation_attempts=1 (default), the first failure is terminal —
        no autonomous recreation. The user starts a new session if they want to try again.
        This prevents 'sessions waking up at midnight and burning credits' after the user
        has given up.
        """
        # Step 1: destroy the failed instance best-effort
        if session.vastai_instance_id is not None:
            try:
                await self._vastai.destroy_instance(session.vastai_instance_id)
            except InstanceNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "gpu_session.provision.retry_destroy_failed",
                    session_id=str(session.id),
                    instance_id=session.vastai_instance_id,
                )

        # Cooldown the machine regardless of whether this failure is terminal —
        # the user's next manual session must not land on the same broken node.
        if session.vastai_machine_id is not None:
            await self._cooldown.record_failure(session.vastai_machine_id, reason=reason)
        else:
            logger.debug("gpu_session.node_cooldown.no_machine_id", session_id=str(session.id))

        # Step 2: atomically increment attempt counter
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            new_attempt = await repo.increment_provision_attempt(session.id)

        # Attempts are 1-indexed: attempt=1 is the original; attempt=2 would be the
        # first recreation. With provisioning_recreation_attempts=1 (default), the
        # original attempt's failure is terminal — no recreation ever fires.

        # provisioning_timeout is always terminal: re-downloading the full model on a
        # fresh node hits the exact same wall (same slow network, same bundle size).
        # provisioning_stalled and node_reported_failure are retryable (bad node, not bad bundle).
        if (
            reason == _REASON_PROVISIONING_TIMEOUT
            or new_attempt >= self._settings.provisioning_recreation_attempts + 1
        ):
            logger.warning(
                "gpu_session.provision.attempts_exhausted",
                session_id=str(session.id),
                attempt=new_attempt,
                limit=self._settings.provisioning_recreation_attempts,
                reason=reason,
            )
            await self._mark_failed(
                session, f"attempts_exhausted after {new_attempt} attempts ({reason})"
            )
            return

        # Step 3: search for a new offer with the same hardware requirements
        logger.info(
            "gpu_session.provision.retry",
            session_id=str(session.id),
            attempt=new_attempt,
            reason=reason,
        )

        try:
            bundle = self._bundles.resolve_bundle_override(
                f"{session.bundle_name}:{session.bundle_version}"
                if session.bundle_version
                else session.bundle_name
            )
        except Exception:
            logger.exception(
                "gpu_session.provision.retry_bundle_resolve_failed",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "retry_bundle_resolve_failed")
            return

        try:
            offers = await self._vastai.search_offers(
                bundle.hardware, limit=self._settings.vastai_offer_search_limit
            )
        except Exception:
            logger.exception(
                "gpu_session.provision.retry_search_failed",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "retry_search_failed: no capacity")
            return

        # Re-fetch the tunnel token from CF (not stored on the session row)
        try:
            tunnel_token = await self._cf.get_tunnel_token(_require_cf_tunnel_id(session))
        except Exception:
            logger.exception(
                "gpu_session.provision.retry_get_token_failed",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "retry_get_token_failed")
            return

        if not self._settings.ai_bundles_github_token:
            logger.error(
                "gpu_session.provision.retry_config_error",
                session_id=str(session.id),
                reason="ai_bundles_github_token is empty; CLI clone will fail",
            )
            await self._mark_failed(session, "misconfigured: ai_bundles_github_token is empty")
            return

        # Generate a fresh callback token so the destroyed node's leaked token is dead.
        # The hash is written to the DB atomically with the new instance info below.
        fresh_callback_token = secrets.token_urlsafe(48)
        fresh_callback_token_hash = hashlib.sha256(fresh_callback_token.encode()).hexdigest()

        # SECURITY: never log env — contains tunnel_token, callback_token, hf_token,
        # civitai_api_token, and ACS_GITHUB_TOKEN.
        env = build_acs_env(
            settings=self._settings,
            session_id=session.id,
            bundle_name=session.bundle_name,
            bundle_version=session.bundle_version,
            comfyui_port=bundle.hardware.comfyui_port,
            tunnel_token=tunnel_token,
            callback_token=fresh_callback_token,
        )

        try:
            instance_id, selected_offer = await provision_vastai_instance(
                vastai_client=self._vastai,
                offers=offers,
                disk_gb=bundle.hardware.min_disk_gb,
                env=env,
                template_hash_id=bundle.hardware.template_hash_id,
                cooldown_store=self._cooldown,
                onstart_cmd=make_onstart_cmd(self._settings.aisha_branch),
                offer_walk_depth=self._settings.provisioning_offer_walk_depth,
            )
        except Exception:
            logger.exception(
                "gpu_session.provision.retry_create_failed",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "retry_create_failed: instance creation error")
            return

        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            current = await repo.get_by_id(session.id, for_update=True)
            if current is None:
                return
            if current.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
                # User stopped the session mid-retry — clean up the new instance
                with contextlib.suppress(Exception):
                    await self._vastai.destroy_instance(instance_id)
                return
            # Rotate the callback token so the old (destroyed) node's token is invalidated.
            await repo.update_callback_token_hash(session.id, fresh_callback_token_hash)
            await repo.update_instance(
                session.id,
                vastai_instance_id=instance_id,
                vastai_offer_id=selected_offer.id,
                vastai_cost_per_hour_micros=selected_offer.dph_total_micros,
                vastai_gpu_name=selected_offer.gpu_name,
                vastai_machine_id=selected_offer.machine_id,
                provisioning_started_at=now,
            )
            await repo.update_status(session.id, GpuSessionStatus.pending)

        logger.info(
            "gpu_session.provision.retry_instance_created",
            session_id=str(session.id),
            instance_id=instance_id,
            attempt=new_attempt,
        )

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    async def _transition(
        self,
        session: GpuSession,
        *,
        new_status: GpuSessionStatus,
        log_event: str,
        **extra_fields: object,
    ) -> None:
        """Re-load under SELECT FOR UPDATE, validate status, then write new_status."""
        previous_status = str(session.status)
        # Pop error_message so it reaches the SSE event as well as the DB row.
        error_message = extra_fields.get("error_message")
        if error_message is not None and not isinstance(error_message, str):
            raise TypeError("error_message must be a str")
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            current = await repo.get_by_id(session.id, for_update=True)
            if current is None:
                return
            if current.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
                logger.info(
                    "gpu_session.provision.skip_transition_stale",
                    session_id=str(session.id),
                    observed_status=current.status,
                    intended_status=new_status.value,
                )
                return
            previous_status = str(current.status)
            await repo.update_status(session.id, new_status, **extra_fields)
            current.status = new_status

        logger.info(log_event, session_id=str(session.id), new_status=new_status.value)
        await self._publish_status_event(
            current,
            previous_status=previous_status,
            error_message=error_message,
        )

    async def _mark_active(self, session: GpuSession) -> None:
        """Transition to active; set started_at if not already set (preserve for resuming)."""
        extra: dict[str, Any] = {}
        if session.started_at is None:
            extra["started_at"] = datetime.now(UTC)
        # Stamp last_reachable_at at activation so the reconciler grace window
        # starts from when the session became usable, not from created_at
        # (provisioning takes ~15 min, so created_at would be stale immediately).
        extra["last_reachable_at"] = datetime.now(UTC)
        await self._transition(
            session,
            new_status=GpuSessionStatus.active,
            log_event="gpu_session.provision.active",
            **extra,
        )

    async def _mark_failed(self, session: GpuSession, reason: str) -> None:
        """Destroy instance + delete tunnel (best-effort), refund billing, transition to failed."""
        # Sweep in-flight jobs before the status transition so the user-visible
        # state stays consistent: jobs go FAILED, then session goes FAILED.
        if self._job_sweep is not None:
            await self._job_sweep.sweep_session_best_effort(
                session_id=session.id,
                product_id=session.product_id,
                failure=JobSweepFailure(internal_reason=f"GPU session failed: {reason}"),
                log_event="gpu_session.provision.job_sweep",
            )

        if session.vastai_instance_id is not None:
            try:
                await self._vastai.destroy_instance(session.vastai_instance_id)
            except InstanceNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "gpu_session.provision.final_destroy_failed",
                    session_id=str(session.id),
                )

        if session.cf_tunnel_id and session.cf_dns_record_id:
            try:
                await self._cf.delete_session_tunnel(session.cf_tunnel_id, session.cf_dns_record_id)
            except Exception:
                logger.exception(
                    "gpu_session.provision.final_tunnel_delete_failed",
                    session_id=str(session.id),
                )

        await self._transition(
            session,
            new_status=GpuSessionStatus.failed,
            log_event="gpu_session.provision.failed",
            error_message=reason[:500],
        )

        # Issue full refund of base reservation — session never became usable
        if self._billing_service is not None:
            try:
                async with self._session_factory() as db, db.begin():
                    refund_result = await self._billing_service.refund(
                        job_id=session.id,
                        description=f"GPU session failed: {reason[:200]}",
                        session=db,
                        product_id=session.product_id,
                        user_id=session.user_id,
                    )
                if self._event_bus is not None:
                    await self._event_bus.publish_balance(refund_result.event)
                logger.info(
                    "gpu_session.provision.refund_issued",
                    session_id=str(session.id),
                )
            except Exception:
                logger.exception(
                    "gpu_session.provision.refund_failed",
                    session_id=str(session.id),
                )

    # ------------------------------------------------------------------
    # SSE event publishing
    # ------------------------------------------------------------------

    async def _publish_status_event(
        self,
        session: GpuSession,
        previous_status: str,
        error_message: str | None = None,
    ) -> None:
        """Fire-and-forget SSE publish. Delegates to the shared helper."""
        await publish_status_event(
            self._event_bus,
            session,
            previous_status=previous_status,
            error_message=error_message,
            ops_event_bus=self._ops_event_bus,
        )

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def _provision_timeout_exceeded(self, session: GpuSession) -> bool:
        base = session.provisioning_started_at or session.created_at
        elapsed = datetime.now(UTC) - base
        return elapsed > timedelta(seconds=self._settings.gpu_provision_timeout_seconds)

    def _resuming_timeout_exceeded(self, session: GpuSession) -> bool:
        base = session.resumed_at
        if base is None:
            # resume_session always sets resumed_at — treat missing as timed out
            return True
        elapsed = datetime.now(UTC) - base
        return elapsed > timedelta(minutes=self._settings.gpu_resume_timeout_minutes)
