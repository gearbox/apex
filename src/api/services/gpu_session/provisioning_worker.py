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
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from src.api.services.vastai.exceptions import InstanceNotFoundError
from src.api.services.vastai.schemas import VastAIInstance
from src.core.enums import GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

from ._env_builder import build_acs_env
from ._events import publish_status_event
from ._provisioning import make_onstart_cmd, provision_vastai_instance

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BillingService
    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.api.services.event_bus import EventBus
    from src.api.services.jobs.sweep import JobSweepService
    from src.api.services.vastai.client import VastAIClient
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

_PROBE_TIMEOUT_SECONDS = 10.0
_TERMINAL_STATUSES = frozenset({GpuSessionStatus.stopped, GpuSessionStatus.failed})

# Vast.ai actual_status values that mean the container is dead and will never reach 'running'.
# Sourced from Vast.ai docs: https://docs.vast.ai/sdk/python/quickstart — "if actual_status
# becomes 'exited' (container crashed), 'unknown' (no heartbeat from host), or 'offline'
# (host disconnected), it will never reach 'running'."
_TERMINAL_ACTUAL_STATUSES = frozenset({"exited", "unknown", "offline"})

# Vast.ai cur_state values that, combined with a non-'running' actual_status, mean the
# instance is dead. Observed empirically when a container fails to start (port allocation
# error etc.): actual_status='created' with cur_state='stopped'.
_TERMINAL_CUR_STATES = frozenset({"stopped", "exited"})


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


class GpuProvisioningWorker:
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
        billing_service: BillingService | None = None,
        event_bus: EventBus | None = None,
        job_sweep_service: JobSweepService | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._vastai = vastai_client
        self._cf = cf_client
        self._bundles = bundle_index
        self._http = http_client
        self._settings = settings
        self._billing_service = billing_service
        self._event_bus = event_bus
        self._job_sweep = job_sweep_service
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the provisioning worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "gpu_provisioning_worker.started",
            interval_s=self._settings.gpu_provision_poll_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the provisioning worker."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("gpu_provisioning_worker.stopped")

    async def _run_loop(self) -> None:
        """Main periodic loop."""
        while self._running:
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("gpu_provisioning_worker.sweep_error")
            # Always sleep even on errors to avoid tight-loop on transient failures
            await asyncio.sleep(self._settings.gpu_provision_poll_interval_seconds)

    async def _sweep_once(self) -> None:
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
                logger.exception(
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
            logger.error(
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
                await self._retry_or_fail(session, reason="pending_timeout_after_errors")
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
                await self._retry_or_fail(session, reason="pending_timeout")
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

        if self._provision_timeout_exceeded(session):
            logger.warning(
                "gpu_session.provision.provisioning_timeout",
                session_id=str(session.id),
            )
            await self._retry_or_fail(session, reason="provisioning_timeout")
            return

        reachable = await self._probe_comfyui(session)
        if reachable:
            await self._mark_active(session)

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
        """Return True if ComfyUI is reachable via the tunnel, False otherwise."""
        if session.tunnel_hostname is None:
            logger.warning(
                "gpu_session.provision.probe_no_hostname",
                session_id=str(session.id),
            )
            return False

        url = f"https://{session.tunnel_hostname}/object_info"
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

        reachable = resp.status_code == 200
        logger.info(
            "gpu_session.provision.probe",
            session_id=str(session.id),
            status_code=resp.status_code,
            reachable=reachable,
        )
        return reachable

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

        # Step 2: atomically increment attempt counter
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            new_attempt = await repo.increment_provision_attempt(session.id)

        # Attempts are 1-indexed: attempt=1 is the original; attempt=2 would be the
        # first recreation. With provisioning_recreation_attempts=1 (default), the
        # original attempt's failure is terminal — no recreation ever fires.
        if new_attempt >= self._settings.provisioning_recreation_attempts + 1:
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
            offers = await self._vastai.search_offers(bundle.hardware)
        except Exception:
            logger.exception(
                "gpu_session.provision.retry_search_failed",
                session_id=str(session.id),
            )
            await self._mark_failed(session, "retry_search_failed: no capacity")
            return

        # Re-fetch the tunnel token from CF (not stored on the session row)
        try:
            if session.cf_tunnel_id is None:
                raise ValueError("session has no cf_tunnel_id")
            tunnel_token = await self._cf.get_tunnel_token(session.cf_tunnel_id)
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

        # SECURITY: never log env — contains tunnel_token, callback_token, hf_token,
        # civitai_api_token, and ACS_GITHUB_TOKEN.
        env = build_acs_env(
            settings=self._settings,
            session_id=session.id,
            bundle_name=session.bundle_name,
            bundle_version=session.bundle_version,
            comfyui_port=bundle.hardware.comfyui_port,
            tunnel_token=tunnel_token,
            callback_token=session.callback_token or "",
        )

        try:
            instance_id, selected_offer = await provision_vastai_instance(
                vastai_client=self._vastai,
                offers=offers,
                disk_gb=bundle.hardware.min_disk_gb,
                env=env,
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
            if current.status in _TERMINAL_STATUSES | {GpuSessionStatus.stopping}:
                # User stopped the session mid-retry — clean up the new instance
                with contextlib.suppress(Exception):
                    await self._vastai.destroy_instance(instance_id)
                return
            await repo.update_instance(
                session.id,
                vastai_instance_id=instance_id,
                vastai_offer_id=selected_offer.id,
                vastai_cost_per_hour_micros=selected_offer.dph_total_micros,
                vastai_gpu_name=selected_offer.gpu_name,
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
        **extra_fields: Any,
    ) -> None:
        """Re-load under SELECT FOR UPDATE, validate status, then write new_status."""
        previous_status = str(session.status)
        # Pop error_message so it reaches the SSE event as well as the DB row.
        error_message = extra_fields.get("error_message")
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            current = await repo.get_by_id(session.id, for_update=True)
            if current is None:
                return
            if current.status in _TERMINAL_STATUSES | {GpuSessionStatus.stopping}:
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
                reason=f"GPU session failed: {reason}",
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
                    await self._billing_service.refund(
                        job_id=session.id,
                        description=f"GPU session failed: {reason[:200]}",
                        session=db,
                        product_id=session.product_id,
                        user_id=session.user_id,
                    )
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
        )

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def _provision_timeout_exceeded(self, session: GpuSession) -> bool:
        base = session.provisioning_started_at or session.created_at
        elapsed = datetime.now(UTC) - base
        return elapsed > timedelta(minutes=self._settings.gpu_provision_timeout_minutes)

    def _resuming_timeout_exceeded(self, session: GpuSession) -> bool:
        base = session.resumed_at
        if base is None:
            # resume_session always sets resumed_at — treat missing as timed out
            return True
        elapsed = datetime.now(UTC) - base
        return elapsed > timedelta(minutes=self._settings.gpu_resume_timeout_minutes)
