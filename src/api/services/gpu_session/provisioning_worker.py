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
from src.core.enums import GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

from ._provisioning import provision_vastai_instance

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.api.services.vastai.client import VastAIClient
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

_PROBE_TIMEOUT_SECONDS = 10.0
_TERMINAL_STATUSES = frozenset({GpuSessionStatus.stopped, GpuSessionStatus.failed})


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
    ) -> None:
        self._session_factory = session_factory
        self._vastai = vastai_client
        self._cf = cf_client
        self._bundles = bundle_index
        self._http = http_client
        self._settings = settings
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
            return  # transient — try again next sweep

        if instance.actual_status != "running":
            if self._pending_timeout_exceeded(session):
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
        """Probe ComfyUI; on success → active; on timeout → retry or fail."""
        if self._provisioning_timeout_exceeded(session):
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
        """Try provisioning a new Vast.ai node; mark failed if retries exhausted."""
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

        if new_attempt > self._settings.max_node_provisioning_retries:
            logger.warning(
                "gpu_session.provision.retry_exhausted",
                session_id=str(session.id),
                provision_attempt=new_attempt,
                reason=reason,
            )
            await self._mark_failed(session, f"retry_exhausted after {new_attempt} attempts")
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

        comfyui_port = bundle.hardware.comfyui_port
        env: dict[str, str] = {
            "ACS_BUNDLE": session.bundle_name,
            "ACS_BUNDLE_VERSION": session.bundle_version or "current",
            "ACS_CF_TUNNEL_TOKEN": tunnel_token,
            "ACS_APEX_SESSION_ID": str(session.id),
            "ACS_APEX_CALLBACK_URL": self._settings.apex_callback_url,
            "ACS_APEX_CALLBACK_TOKEN": session.callback_token or "",
            "ACS_HF_TOKEN": self._settings.hf_token,
            "ACS_CIVITAI_API_TOKEN": self._settings.civitai_api_token,
            f"-p {comfyui_port}:{comfyui_port}": "1",
        }

        try:
            instance_id, selected_offer = await provision_vastai_instance(
                vastai_client=self._vastai,
                offers=offers,
                disk_gb=bundle.hardware.min_disk_gb,
                env=env,
                max_retries=self._settings.max_node_provisioning_retries,
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
            await repo.update_status(session.id, new_status, **extra_fields)

        logger.info(log_event, session_id=str(session.id), new_status=new_status.value)
        # TODO: publish GPU_SESSION_STATUS_CHANGED event (Phase 1F)

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
        """Destroy instance + delete tunnel (best-effort), then transition to failed."""
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

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def _pending_timeout_exceeded(self, session: GpuSession) -> bool:
        base = session.provisioning_started_at or session.created_at
        elapsed = datetime.now(UTC) - base
        return elapsed > timedelta(minutes=self._settings.gpu_provision_timeout_minutes)

    def _provisioning_timeout_exceeded(self, session: GpuSession) -> bool:
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
