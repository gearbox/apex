"""GPU session lifecycle service."""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy.exc import IntegrityError

from src.api.services.bundle_index import BundleIndexService
from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import NoCapacityError, OfferTakenError, VastAIError
from src.core.enums import GpuSessionStatus, ModelType
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.repositories.gpu_session import GpuSessionRepository

from .exceptions import (
    GpuSessionError,
    InvalidSessionStateError,
    SessionAlreadyExistsError,
)
from .schemas import StopConfirmation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_STOPPABLE_STATUSES = frozenset(
    {
        GpuSessionStatus.active,
        GpuSessionStatus.stale,
        GpuSessionStatus.paused,
    }
)

_VASTAI_IMAGE = "vastai/comfy:latest"
_ONSTART_CMD = (
    "curl -sL https://raw.githubusercontent.com/gearbox/aisha/main/scripts/onstart.sh | bash"
)


class GpuSessionService:
    """Synchronous orchestration of GPU session lifecycle.

    This service is the single entry point for session lifecycle operations
    called from HTTP routes. It does NOT handle background polling — that
    belongs to GpuProvisioningWorker (Phase 1E-2).

    Responsibilities:
    - Start: resolve bundle → create CF tunnel → search Vast.ai →
             create instance → persist session (status=pending)
    - Pause: stop Vast.ai instance, retain tunnel + disk (status=paused)
    - Resume: start Vast.ai instance (status=resuming)
    - Stop: two-call confirmation → destroy instance → delete tunnel (status=stopped)
    - Get / list / routing queries
    """

    def __init__(
        self,
        *,
        vastai_client: VastAIClient,
        cf_client: CloudflareTunnelClient,
        bundle_index: BundleIndexService,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._vastai = vastai_client
        self._cf = cf_client
        self._bundles = bundle_index
        self._session_factory = session_factory
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Public lifecycle methods                                             #
    # ------------------------------------------------------------------ #

    async def start_session(
        self,
        *,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
        bundle_override: str | None = None,
    ) -> GpuSession:
        """Create a new GPU session and trigger provisioning.

        Steps (in order, with cleanup on failure at each step):
        1. Resolve bundle (default via model_type OR override if provided)
        2. Pre-check uniqueness via get_non_terminal_for_model (fast fail)
        3. Generate session_id (UUIDv7) and callback_token (token_urlsafe)
        4. Create CF tunnel + DNS (CloudflareTunnelClient.create_session_tunnel)
           → on failure: nothing to clean up yet, re-raise
        5. Search Vast.ai for offers matching hardware requirements
           → on failure (NoCapacityError): delete tunnel, re-raise
        6. Pick cheapest offer, create Vast.ai instance with env dict
           → on OfferTakenError: try next offer up to max_node_provisioning_retries
           → on persistent failure: destroy any partially-created instance,
             delete tunnel, re-raise
        7. Persist GpuSession row with status='pending' and all populated fields
           → on IntegrityError (race with concurrent start_session):
             destroy instance, delete tunnel, raise SessionAlreadyExistsError
        8. Return the persisted GpuSession

        Args:
            user_id: Owner user.
            product_id: Product scope ("vex" or "synthara").
            model_type: ModelType for which the session is being started.
            bundle_override: Optional admin-only "name" or "name:version" spec.
                If provided, takes precedence over model_type's default bundle.

        Returns:
            The created GpuSession with status='pending'.

        Raises:
            BundleNotFoundError: Unknown model_type or bundle override.
            SessionAlreadyExistsError: User already has a non-terminal session for
                this (product, model_type).
            NoCapacityError: No Vast.ai offers match requirements.
            TunnelCreationError / DNSRecordError: CF tunnel setup failed.
            VastAIError: Instance creation failed after all retries.
        """
        logger.info(
            "gpu_session.start.begun",
            user_id=str(user_id),
            product_id=product_id,
            model_type=model_type.value,
            bundle_override=bundle_override,
        )

        # Step 1: resolve bundle (no DB, no external API calls)
        if bundle_override is not None:
            bundle = self._bundles.resolve_bundle_override(bundle_override)
        else:
            bundle = self._bundles.resolve_bundle(model_type.value)
        hardware = bundle.hardware
        logger.info(
            "gpu_session.start.bundle_resolved",
            bundle_name=bundle.bundle_name,
            bundle_version=bundle.bundle_version,
        )

        # Step 2: pre-check uniqueness (short read-only session, fast fail)
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            existing = await repo.get_non_terminal_for_model(user_id, product_id, model_type.value)

        if existing is not None:
            raise SessionAlreadyExistsError(
                f"User {user_id} already has a non-terminal session for "
                f"model_type={model_type.value} in product={product_id}"
            )

        # Step 3: generate IDs (no external calls)
        session_id = new_id()
        session_id_short = str(session_id)[:8]
        callback_token = secrets.token_urlsafe(48)

        # Step 4: create CF tunnel + DNS
        try:
            tunnel_id, tunnel_token, dns_record_id, hostname = await self._cf.create_session_tunnel(
                session_id_short, hardware.comfyui_port
            )
        except Exception:
            logger.exception(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="TunnelError",
                phase="tunnel_creation",
            )
            raise
        logger.info(
            "gpu_session.start.tunnel_created",
            tunnel_id=tunnel_id,
            hostname=hostname,
        )

        # Step 5: search Vast.ai offers
        try:
            offers = await self._vastai.search_offers(hardware)
        except NoCapacityError:
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            logger.error(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="NoCapacityError",
                phase="search_offers",
            )
            raise
        logger.info(
            "gpu_session.start.offers_found",
            count=len(offers),
            cheapest_dph_micros=offers[0].dph_total_micros,
        )

        # Build env dict (SECURITY: never log this dict — contains tunnel_token, callback_token,
        # hf_token, civitai_api_token)
        env: dict[str, str] = {
            "ACS_BUNDLE": bundle.bundle_name,
            "ACS_BUNDLE_VERSION": bundle.bundle_version or "current",
            "ACS_CF_TUNNEL_TOKEN": tunnel_token,
            "ACS_APEX_SESSION_ID": str(session_id),
            "ACS_APEX_CALLBACK_URL": self._settings.apex_callback_url,
            "ACS_APEX_CALLBACK_TOKEN": callback_token,
            "ACS_HF_TOKEN": self._settings.hf_token,
            "ACS_CIVITAI_API_TOKEN": self._settings.civitai_api_token,
            "-p 18188:18188": "1",
        }

        # Step 6: create Vast.ai instance with retry loop
        max_retries = self._settings.max_node_provisioning_retries
        instance_id: int | None = None
        selected_offer = None
        last_exc: Exception | None = None

        for attempt, offer in enumerate(offers[:max_retries]):
            try:
                instance_id = await self._vastai.create_instance(
                    offer.id,
                    image=_VASTAI_IMAGE,
                    disk_gb=hardware.min_disk_gb,
                    env=env,
                    onstart_cmd=_ONSTART_CMD,
                )
                selected_offer = offer
                logger.info(
                    "gpu_session.start.instance_created",
                    instance_id=instance_id,
                    offer_id=offer.id,
                    dph_micros=offer.dph_total_micros,
                )
                break
            except OfferTakenError:
                remaining = len(offers[:max_retries]) - attempt - 1
                logger.warning(
                    "gpu_session.start.offer_taken_retry",
                    offer_id=offer.id,
                    attempt=attempt + 1,
                    remaining_offers=remaining,
                )
            except Exception as exc:
                last_exc = exc
                break

        if instance_id is None or selected_offer is None:
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            logger.error(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="VastAIError",
                phase="create_instance",
            )
            if last_exc is not None:
                raise last_exc
            raise VastAIError("All offers were taken — no instance could be created after retries")

        # Step 7: persist session (separate write transaction)
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            try:
                session_row = await repo.create(
                    id=session_id,
                    user_id=user_id,
                    product_id=product_id,
                    status=GpuSessionStatus.pending,
                    bundle_name=bundle.bundle_name,
                    bundle_version=bundle.bundle_version,
                    model_type=model_type.value,
                    cf_tunnel_id=tunnel_id,
                    cf_dns_record_id=dns_record_id,
                    tunnel_hostname=hostname,
                    vastai_instance_id=instance_id,
                    vastai_offer_id=selected_offer.id,
                    vastai_cost_per_hour_micros=selected_offer.dph_total_micros,
                    vastai_gpu_name=selected_offer.gpu_name,
                    callback_token=callback_token,
                )
            except IntegrityError:
                # Race with a concurrent start_session call that slipped past our pre-check.
                # Clean up external resources so they don't become orphans.
                await self._destroy_instance_best_effort(instance_id)
                await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
                raise SessionAlreadyExistsError(
                    f"Concurrent session creation conflict for user={user_id}, "
                    f"model_type={model_type.value}"
                ) from None

        logger.info(
            "gpu_session.start.persisted",
            session_id=str(session_id),
            status=GpuSessionStatus.pending,
        )
        # TODO: publish GPU_SESSION_STATUS_CHANGED event (wire up in Phase 1F)
        return session_row

    async def pause_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Pause an active session — stops the Vast.ai instance but retains disk.

        Only allowed from status='active'. Tunnel and DNS record are retained so
        resume is fast. GPU billing stops immediately; only storage charges continue.

        Steps:
        1. Load session with ownership check (for_update=True for SELECT FOR UPDATE)
        2. Verify status == 'active' — else raise InvalidSessionStateError
        3. Call vastai_client.stop_instance(vastai_instance_id)
        4. Update status → 'paused', set paused_at = now()

        Args:
            session_id: Session to pause.
            user_id: Requesting user (ownership check).
            product_id: Product scope.

        Returns:
            Updated GpuSession with status='paused'.

        Raises:
            GpuSessionError: Session not found or not owned by user.
            InvalidSessionStateError: Session not in 'active' state.
            VastAIError: Propagated if stop_instance fails (session stays 'active').
        """
        logger.info("gpu_session.pause.begun", session_id=str(session_id))

        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            session_row = await repo.get_by_id(session_id, for_update=True)
            if (
                session_row is None
                or session_row.user_id != user_id
                or session_row.product_id != product_id
            ):
                raise GpuSessionError(f"Session {session_id} not found or access denied")

            if session_row.status != GpuSessionStatus.active:
                raise InvalidSessionStateError(
                    f"Cannot pause session in state '{session_row.status}'",
                    current_status=session_row.status,
                    operation="pause",
                )

            # VastAI call inside the transaction: if it fails, exception propagates,
            # transaction rolls back, and session stays 'active' (no partial state).
            if session_row.vastai_instance_id is None:
                raise GpuSessionError(f"Session {session_id} has no Vast.ai instance to pause")
            await self._vastai.stop_instance(session_row.vastai_instance_id)

            now = datetime.now(UTC)
            await repo.update_status(session_id, GpuSessionStatus.paused, paused_at=now)
            session_row.status = GpuSessionStatus.paused
            session_row.paused_at = now

        logger.info("gpu_session.pause.done", session_id=str(session_id))
        # TODO: publish GPU_SESSION_STATUS_CHANGED event (wire up in Phase 1F)
        return session_row

    async def resume_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Resume a paused session — restarts the Vast.ai instance.

        Only allowed from status='paused'. Transitions to 'resuming'; the
        provisioning worker (Phase 1E-2) will poll for ComfyUI readiness and
        transition to 'active' when the instance is back online.

        Steps:
        1. Load session with ownership check
        2. Verify status == 'paused' — else raise InvalidSessionStateError
        3. Call vastai_client.start_instance(vastai_instance_id)
        4. Update status → 'resuming', set resumed_at = now()

        Args:
            session_id: Session to resume.
            user_id: Requesting user (ownership check).
            product_id: Product scope.

        Returns:
            Updated GpuSession with status='resuming'.

        Raises:
            GpuSessionError: Session not found or not owned by user.
            InvalidSessionStateError: Session not in 'paused' state.
            VastAIError: Propagated if start_instance fails (session stays 'paused').
        """
        logger.info("gpu_session.resume.begun", session_id=str(session_id))

        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            session_row = await repo.get_by_id(session_id, for_update=True)
            if (
                session_row is None
                or session_row.user_id != user_id
                or session_row.product_id != product_id
            ):
                raise GpuSessionError(f"Session {session_id} not found or access denied")

            if session_row.status != GpuSessionStatus.paused:
                raise InvalidSessionStateError(
                    f"Cannot resume session in state '{session_row.status}'",
                    current_status=session_row.status,
                    operation="resume",
                )

            # VastAI call inside the transaction: if it fails, session stays 'paused'.
            if session_row.vastai_instance_id is None:
                raise GpuSessionError(f"Session {session_id} has no Vast.ai instance to resume")
            await self._vastai.start_instance(session_row.vastai_instance_id)

            now = datetime.now(UTC)
            await repo.update_status(session_id, GpuSessionStatus.resuming, resumed_at=now)
            session_row.status = GpuSessionStatus.resuming
            session_row.resumed_at = now

        logger.info("gpu_session.resume.done", session_id=str(session_id))
        # TODO: publish GPU_SESSION_STATUS_CHANGED event (wire up in Phase 1F)
        return session_row

    async def stop_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        confirmed: bool = False,
    ) -> GpuSession | StopConfirmation:
        """Stop a session permanently (two-call confirmation flow).

        Allowed from status in {'active', 'stale', 'paused'}.

        First call (confirmed=False):
            Returns a StopConfirmation with cost summary. No state changes.
            Caller should display this to the user and re-invoke with
            confirmed=True to proceed.

        Second call (confirmed=True):
            1. Load session (for_update=True) and verify stoppable state
            2. Update status → 'stopping'
            3. Destroy Vast.ai instance (best-effort: logs + continues on errors)
            4. Delete CF tunnel + DNS (best-effort: logs + continues on errors)
            5. Update status → 'stopped', set stopped_at = now()

        Args:
            session_id: Session to stop.
            user_id: Requesting user (ownership check).
            product_id: Product scope.
            confirmed: When False, return cost confirmation; when True, execute stop.

        Returns:
            StopConfirmation when confirmed=False.
            GpuSession (stopped) when confirmed=True.

        Raises:
            GpuSessionError: Session not found or not owned.
            InvalidSessionStateError: Session is in a non-stoppable state
                ('pending', 'provisioning', 'resuming', 'stopping', 'stopped', 'failed').
        """
        if not confirmed:
            return await self._stop_confirmation(session_id, user_id, product_id)
        return await self._stop_confirmed(session_id, user_id, product_id)

    async def get_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession | None:
        """Get session with ownership filter. Returns None if not found or not owned.

        Args:
            session_id: Session UUID.
            user_id: Owner filter.
            product_id: Product filter.

        Returns:
            GpuSession or None.
        """
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            return await repo.get_by_id_for_user(session_id, user_id, product_id)

    async def get_active_session_for_model(
        self,
        *,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
    ) -> GpuSession | None:
        """Used by AishaGenerationProvider (Phase 1F) for routing.

        Returns only sessions with status='active'. Paused/stale/provisioning
        sessions are NOT routable for generation.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: Model type filter.

        Returns:
            Active GpuSession or None.
        """
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            return await repo.get_active_for_model(user_id, product_id, model_type.value)

    async def list_user_sessions(
        self,
        *,
        user_id: UUID,
        product_id: str,
        include_terminal: bool = False,
    ) -> Sequence[GpuSession]:
        """List user's sessions ordered by created_at DESC.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            include_terminal: When True, includes 'stopped' and 'failed' sessions.

        Returns:
            Sequence of GpuSession ordered by created_at DESC.
        """
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            return await repo.list_by_user(user_id, product_id, include_terminal=include_terminal)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _stop_confirmation(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> StopConfirmation:
        """Build stop confirmation DTO without changing any state."""
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            session_row = await repo.get_by_id_for_user(session_id, user_id, product_id)

        if session_row is None:
            raise GpuSessionError(f"Session {session_id} not found or access denied")

        if session_row.status not in _STOPPABLE_STATUSES:
            raise InvalidSessionStateError(
                f"Cannot stop session in state '{session_row.status}'",
                current_status=session_row.status,
                operation="stop",
            )

        now = datetime.now(UTC)
        start = (
            session_row.started_at if session_row.started_at is not None else session_row.created_at
        )
        if session_row.status == GpuSessionStatus.paused and session_row.paused_at is not None:
            end = session_row.paused_at
        else:
            end = now
        active_duration_seconds = int((end - start).total_seconds())

        logger.info("gpu_session.stop.confirmation_requested", session_id=str(session_id))
        return StopConfirmation(
            session_id=session_id,
            model_type=session_row.model_type,
            bundle_name=session_row.bundle_name,
            vastai_gpu_name=session_row.vastai_gpu_name,
            vastai_cost_per_hour_micros=session_row.vastai_cost_per_hour_micros,
            active_duration_seconds=active_duration_seconds,
            message=(
                f"Stopping this session will permanently destroy the GPU node. "
                f"Active time: {active_duration_seconds // 3600}h "
                f"{(active_duration_seconds % 3600) // 60}m. "
                "Confirm to proceed."
            ),
        )

    async def _stop_confirmed(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Execute confirmed stop: teardown external resources and mark stopped."""
        logger.info("gpu_session.stop.begun", session_id=str(session_id))

        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            session_row = await repo.get_by_id(session_id, for_update=True)
            if (
                session_row is None
                or session_row.user_id != user_id
                or session_row.product_id != product_id
            ):
                raise GpuSessionError(f"Session {session_id} not found or access denied")

            if session_row.status not in _STOPPABLE_STATUSES:
                raise InvalidSessionStateError(
                    f"Cannot stop session in state '{session_row.status}'",
                    current_status=session_row.status,
                    operation="stop",
                )

            await repo.update_status(session_id, GpuSessionStatus.stopping)
            session_row.status = GpuSessionStatus.stopping

            teardown_had_errors = False

            # Destroy Vast.ai instance (best-effort — must not block 'stopped' transition)
            if session_row.vastai_instance_id is not None:
                try:
                    await self._vastai.destroy_instance(session_row.vastai_instance_id)
                except Exception:
                    teardown_had_errors = True
                    logger.warning(
                        "gpu_session.stop.teardown_instance_failed",
                        session_id=str(session_id),
                        instance_id=session_row.vastai_instance_id,
                    )

            # Delete CF tunnel + DNS (best-effort)
            if session_row.cf_tunnel_id is not None and session_row.cf_dns_record_id is not None:
                try:
                    await self._cf.delete_session_tunnel(
                        session_row.cf_tunnel_id,
                        session_row.cf_dns_record_id,
                    )
                except Exception:
                    teardown_had_errors = True
                    logger.warning(
                        "gpu_session.stop.teardown_tunnel_failed",
                        session_id=str(session_id),
                        tunnel_id=session_row.cf_tunnel_id,
                    )

            # Always transition to 'stopped' regardless of teardown failures —
            # orphaned resources are cleaned by OrphanedTunnelCleanupWorker (Phase 1E-2)
            now = datetime.now(UTC)
            await repo.update_status(session_id, GpuSessionStatus.stopped, stopped_at=now)
            session_row.status = GpuSessionStatus.stopped
            session_row.stopped_at = now

        logger.info(
            "gpu_session.stop.done",
            session_id=str(session_id),
            teardown_had_errors=teardown_had_errors,
        )
        # TODO: publish GPU_SESSION_STATUS_CHANGED event (wire up in Phase 1F)
        return session_row

    async def _delete_tunnel_best_effort(self, tunnel_id: str, dns_record_id: str) -> None:
        """Delete CF tunnel and DNS record, logging failures without re-raising."""
        try:
            await self._cf.delete_session_tunnel(tunnel_id, dns_record_id)
        except Exception:
            logger.exception(
                "gpu_session.start.tunnel_cleanup_failed",
                tunnel_id=tunnel_id,
            )

    async def _destroy_instance_best_effort(self, instance_id: int) -> None:
        """Destroy Vast.ai instance, logging failures without re-raising."""
        try:
            await self._vastai.destroy_instance(instance_id)
        except Exception:
            logger.exception(
                "gpu_session.start.instance_cleanup_failed",
                instance_id=instance_id,
            )
