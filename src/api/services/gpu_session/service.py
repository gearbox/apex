"""GPU session lifecycle service."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.services.jobs.sweep import JobSweepFailure
from src.api.services.vastai.exceptions import NoCapacityError
from src.core.enums import (
    LIVE_DEPLOYMENT_STATUSES,
    STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES,
    DeploymentStatus,
    GpuSessionStatus,
    ModelType,
    OperationKind,
)
from src.core.uid import new_id
from src.db.repositories.billing import BillingRepository
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository
from src.db.repositories.job import JobRepository

from ._env_builder import build_acs_env
from ._events import publish_status_event
from ._provisioning import make_onstart_cmd, provision_vastai_instance
from .exceptions import (
    GpuSessionError,
    InvalidSessionStateError,
    SessionAlreadyExistsError,
    SessionHasInFlightJobsError,
)
from .schemas import StopConfirmation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.api.services.event_bus import EventBus
    from src.api.services.gpu_session.node_cooldown import NodeCooldownStore
    from src.api.services.jobs.sweep import JobSweepService
    from src.api.services.ops_event_bus import OpsEventBus
    from src.api.services.vastai.client import VastAIClient
    from src.api.services.vastai.schemas import VastAIOffer
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession
    from src.db.models.gpu_session_deployment import GpuSessionDeployment

logger = structlog.get_logger(__name__)

# Stop is a destructive operator-safety action — must always succeed for any
# session that hasn't already terminated. Terminal states are handled idempotently
# by stop_session itself (return the existing row, no work).
#
# Mirrors Vast.ai's own destroy semantics: destroying a non-existent instance is
# a no-op (404 treated as success).
_STOPPABLE_STATUSES = frozenset(
    {
        GpuSessionStatus.pending,
        GpuSessionStatus.provisioning,
        GpuSessionStatus.resuming,
        GpuSessionStatus.active,
        GpuSessionStatus.stale,
        GpuSessionStatus.paused,
    }
)

# Minimum billable duration for a GPU session in minutes. The base reservation
# at start time covers exactly this much usage.
_MIN_BILLABLE_MINUTES = 5


def billable_minutes_for_active_seconds(active_seconds: int) -> int:
    """Convert active seconds to billable minutes (per-started-minute, with floor).

    Per-started-minute means a session running 5m01s is billed as 6 minutes;
    any partial second of a minute counts as a full minute. The floor matches
    the base reservation so a freshly-started session that's stopped within
    seconds still costs exactly the reservation.

    Sentinel cases:
        0   → 5 (floor)
        1   → 5 (floor)
        300 → 5 (exact)
        301 → 6 (rounded up; the next started minute counts)
        359 → 6
        360 → 6 (exact)
    """
    rounded_up = (max(0, active_seconds) + 59) // 60
    return max(_MIN_BILLABLE_MINUTES, rounded_up)


class SessionDurations(msgspec.Struct, kw_only=True, frozen=True):
    """Two-bucket duration breakdown for a GPU session.

    These are the two billable quantities. Today only ``active_seconds`` has
    a per-minute rate (GPU compute); in a later phase ``paused_seconds`` will
    be multiplied by the storage rate (Vast.ai charges us for disk while a
    node is stopped but not destroyed).

    Invariants:
    - Both values are non-negative.
    - ``active_seconds + paused_seconds`` equals the session's total wall-clock
      lifetime from start to the relevant endpoint (stopped_at, paused_at if
      currently paused, or ``now`` if neither), minus any time before
      ``started_at`` was set.
    """

    active_seconds: int
    paused_seconds: int


def compute_session_durations(session: GpuSession, *, now: datetime) -> SessionDurations:
    """Split a session's wall-clock lifetime into active vs paused buckets.

    Single source of truth for both the stop-confirmation preview and the
    post-stop finalize-billing calculation. Previously these two paths had
    independent formulas and drifted.

    ``total_paused_seconds`` on the session row is maintained on resume AND
    on stop-from-paused (see ``_stop_confirmed``). This helper reads that
    column as-is and does NOT try to derive an in-flight pause: if the column
    is wrong, billing is wrong. Column maintenance is the write-path's job.

    End-of-lifetime timestamp:
        - If the session is stopped, use ``stopped_at``.
        - Else if the session is currently paused, use ``paused_at``
          (the preview path; rendering active duration frozen at pause start).
        - Else use ``now``.
    """
    start = session.started_at or session.created_at

    if session.stopped_at is not None:
        end = session.stopped_at
    elif session.status == GpuSessionStatus.paused and session.paused_at is not None:
        end = session.paused_at
    else:
        end = now

    total_seconds = max(0, int((end - start).total_seconds()))
    # Defensive cast: total_paused_seconds is NOT NULL DEFAULT 0 in the schema,
    # so this should always be an int, but `int(... or 0)` keeps the helper
    # robust against handcrafted in-memory rows in tests and any future column
    # migration that briefly allows NULL.
    paused_seconds = max(0, int(session.total_paused_seconds or 0))
    # Clamp against total in case of clock-skew or column drift, so active
    # never goes negative and paused never exceeds wall-clock.
    paused_seconds = min(paused_seconds, total_seconds)
    active_seconds = total_seconds - paused_seconds
    return SessionDurations(active_seconds=active_seconds, paused_seconds=paused_seconds)


# Re-exported for backward compat with any code that imported from this module directly.
__all__ = [
    "GpuSessionService",
    "SessionDurations",
    "billable_minutes_for_active_seconds",
    "compute_session_durations",
]


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
        billing_service: BillingService,
        cooldown_store: NodeCooldownStore,
        event_bus: EventBus | None = None,
        job_sweep_service: JobSweepService | None = None,
        ops_event_bus: OpsEventBus | None = None,
    ) -> None:
        self._vastai = vastai_client
        self._cf = cf_client
        self._bundles = bundle_index
        self._session_factory = session_factory
        self._settings = settings
        self._billing_service = billing_service
        self._cooldown = cooldown_store
        self._event_bus = event_bus
        self._job_sweep = job_sweep_service
        self._ops_event_bus = ops_event_bus

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
        account_id: UUID,
    ) -> GpuSession:
        """Create a new GPU session and trigger provisioning.

        Steps (in order, with cleanup on failure at each step):
        1. Resolve bundle (default via model_type OR override if provided)
        2. Pre-check uniqueness via GpuSessionDeploymentRepository.get_live_for_model (fast fail)
        3. Generate session_id (UUIDv7) and callback_token (token_urlsafe)
        1.5. Assert sufficient balance BEFORE creating any external resources
        4. Create CF tunnel + DNS (CloudflareTunnelClient.create_session_tunnel)
           → on failure: nothing to clean up yet, re-raise
        5. Search Vast.ai for offers matching hardware requirements
           → on failure (NoCapacityError): delete tunnel, re-raise
        6. Pick cheapest offer, create Vast.ai instance with env dict
           → on OfferTakenError: try next offer up to offer_walk_depth candidates
           → on persistent failure: destroy any partially-created instance,
             delete tunnel, re-raise
        7. Persist GpuSession row with status='pending' and all populated fields
           7.5. In same transaction: reserve tokens via self._billing_service.check_and_reserve
           → on IntegrityError (race with concurrent start_session):
             destroy instance, delete tunnel, raise SessionAlreadyExistsError
           → on check_and_reserve failure: destroy instance, delete tunnel, re-raise
        8. Publish GPU_SESSION_STATUS_CHANGED event (fire-and-forget)
        9. Return the persisted GpuSession
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
            deployment_repo = GpuSessionDeploymentRepository(db)
            existing = await deployment_repo.get_live_for_model(
                user_id, product_id, model_type.value
            )

        if existing is not None:
            raise SessionAlreadyExistsError(
                f"User {user_id} already has a non-terminal session for "
                f"model_type={model_type.value} in product={product_id}"
            )

        # Step 3: generate IDs (no external calls)
        session_id = new_id()
        bootstrap_operation_id = new_id()
        deployment_id = new_id()
        session_id_short = str(session_id)[:8]
        callback_token = secrets.token_urlsafe(48)
        # SECURITY: only the hash is persisted; plaintext goes into the env and is never logged.
        callback_token_hash = hashlib.sha256(callback_token.encode()).hexdigest()

        # Step 1.5: assert sufficient balance BEFORE creating external resources.
        # The base reservation is derived from the per-minute rate so start-time
        # and finalize-time calculations can never drift apart if ops changes
        # billing settings while sessions are running.
        token_cost = _MIN_BILLABLE_MINUTES * self._settings.gpu_session_tokens_per_minute
        async with self._session_factory() as db:
            await self._billing_service.assert_sufficient_balance(
                account_id, token_cost, session=db
            )

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

        # Validate config before any Vast.ai API calls (fail fast on deployment misconfiguration)
        if not self._settings.ai_bundles_github_token:
            logger.error(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="ConfigurationError",
                phase="env_validation",
                reason="ai_bundles_github_token is empty; ai-bundles is private and CLI clone will fail",
            )
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            raise NoCapacityError(
                "Apex is misconfigured: ai_bundles_github_token is empty. "
                "Set AI_BUNDLES_GITHUB_TOKEN (pydantic-settings maps this to the field automatically) "
                "to a GitHub PAT with read access to gearbox/ai-bundles."
            )

        # Step 5: search Vast.ai offers
        try:
            offers = await self._vastai.search_offers(
                hardware, limit=self._settings.vastai_offer_search_limit
            )
        except NoCapacityError:
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            logger.exception(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="NoCapacityError",
                phase="search_offers",
            )
            raise

        # Defensive: search_offers should raise NoCapacityError on an empty list
        if not offers:
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            logger.error(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class="NoCapacityError",
                phase="search_offers",
            )
            raise NoCapacityError(
                "Vast.ai search_offers returned an empty list (no offers match hardware)"
            )

        logger.info(
            "gpu_session.start.offers_found",
            count=len(offers),
            cheapest_dph_micros=offers[0].dph_total_micros,
        )

        # SECURITY: never log env — contains tunnel_token, callback_token, hf_token,
        # civitai_api_token, and ACS_GITHUB_TOKEN.
        env = build_acs_env(
            settings=self._settings,
            session_id=session_id,
            operation_id=bootstrap_operation_id,
            bundle_name=bundle.bundle_name,
            bundle_version=bundle.bundle_version,
            comfyui_port=hardware.comfyui_port,
            tunnel_token=tunnel_token,
            callback_token=callback_token,
        )

        # Step 6: create Vast.ai instance with retry loop
        try:
            instance_id, selected_offer = await self._provision_instance(
                offers, hardware.min_disk_gb, env, hardware.template_hash_id
            )
        except Exception as exc:
            await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
            logger.exception(
                "gpu_session.start.failed",
                user_id=str(user_id),
                model_type=model_type.value,
                error_class=exc.__class__.__name__,
                phase="create_instance",
            )
            raise

        # Step 7: persist session + deployment + billing reservation in a single transaction
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            operation_repo = GpuSessionOperationRepository(db)
            deployment_repo = GpuSessionDeploymentRepository(db)
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
                    vastai_machine_id=selected_offer.machine_id,
                    callback_token_hash=callback_token_hash,
                    account_id=account_id,
                    bootstrap_operation_id=bootstrap_operation_id,
                )
                await operation_repo.create(
                    id=bootstrap_operation_id,
                    session_id=session_id,
                    product_id=product_id,
                    kind=OperationKind.session_bootstrap,
                    target_bundle=bundle.bundle_name,
                    target_bundle_version=bundle.bundle_version,
                    target_mode="full",
                )
                # The primary deployment is created with the session — see D6/D21.
                # This insert is what the new partial unique index actually guards;
                # a race that slips past the step-2 pre-check lands here as an
                # IntegrityError, not on the session insert above.
                await deployment_repo.create(
                    id=deployment_id,
                    session_id=session_id,
                    user_id=user_id,
                    product_id=product_id,
                    model_type=model_type.value,
                    bundle_name=bundle.bundle_name,
                    bundle_version=bundle.bundle_version,
                    readiness_marker_node_class=(
                        bundle.readiness_marker.node_class
                        if bundle.readiness_marker is not None
                        else None
                    ),
                    status=DeploymentStatus.deploying,
                    provision_operation_id=bootstrap_operation_id,
                    is_primary=True,
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

            # Step 7.5: billing reservation in the same transaction
            try:
                reserve_result = await self._billing_service.check_and_reserve(
                    account_id,
                    token_cost,
                    session_id,
                    metadata={
                        "type": "gpu_session_reservation",
                        "model_type": model_type.value,
                        "bundle_name": bundle.bundle_name,
                    },
                    description="GPU session base reservation",
                    session=db,
                    product_id=product_id,
                    user_id=user_id,
                )
            except Exception:
                # Balance changed between assert and reserve — clean up external resources
                await self._destroy_instance_best_effort(instance_id)
                await self._delete_tunnel_best_effort(tunnel_id, dns_record_id)
                raise

        if self._event_bus is not None:
            await self._event_bus.publish_balance(reserve_result.event)

        logger.info(
            "gpu_session.start.persisted",
            session_id=str(session_id),
            status=GpuSessionStatus.pending,
        )
        await self._publish_status_event(session_row, previous_status="none")
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
            session_row = await self._load_session_for_update(
                repo,
                session_id,
                user_id,
                product_id,
                frozenset({GpuSessionStatus.active}),
                "pause",
            )

            # Reject pause if any Aisha jobs are in-flight on this session.
            # Count is read with the row lock held to avoid TOCTOU — a job that
            # is QUEUED at this moment can't transition to RUNNING via the poller
            # because the poller would need its own DB session and would see
            # whatever we set here (pausing → excluded from active-only filter).
            job_repo = JobRepository(db)
            in_flight = await job_repo.count_in_flight_for_session(session_row.id)
            if in_flight > 0:
                raise SessionHasInFlightJobsError(
                    session_id=session_row.id,
                    in_flight_count=in_flight,
                )

            # VastAI call inside the transaction: if it fails, exception propagates,
            # transaction rolls back, and session stays 'active' (no partial state).
            # The row lock prevents concurrent pause/stop races on the same instance —
            # correctness matters more than marginal throughput on sub-2s calls.
            if session_row.vastai_instance_id is None:
                raise GpuSessionError(f"Session {session_id} has no Vast.ai instance to pause")
            await self._vastai.stop_instance(session_row.vastai_instance_id)

            await self._set_status(
                db, repo, session_row, GpuSessionStatus.paused, paused_at=datetime.now(UTC)
            )

        logger.info("gpu_session.pause.done", session_id=str(session_id))
        await self._publish_status_event(session_row, previous_status="active")
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
            session_row = await self._load_session_for_update(
                repo,
                session_id,
                user_id,
                product_id,
                frozenset({GpuSessionStatus.paused}),
                "resume",
            )

            # VastAI call inside the transaction: if it fails, session stays 'paused'.
            # Row lock prevents resume+stop races (same reasoning as pause_session).
            if session_row.vastai_instance_id is None:
                raise GpuSessionError(f"Session {session_id} has no Vast.ai instance to resume")
            await self._vastai.start_instance(session_row.vastai_instance_id)

            # Accumulate paused duration before transitioning
            if session_row.paused_at is not None:
                paused_seconds = int((datetime.now(UTC) - session_row.paused_at).total_seconds())
                await repo.add_paused_seconds(session_row.id, paused_seconds)
                session_row.total_paused_seconds = session_row.total_paused_seconds + paused_seconds

            await self._set_status(
                db, repo, session_row, GpuSessionStatus.resuming, resumed_at=datetime.now(UTC)
            )

        logger.info("gpu_session.resume.done", session_id=str(session_id))
        await self._publish_status_event(session_row, previous_status="paused")
        return session_row

    async def stop_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        confirmed: bool = False,
        reason: str | None = None,
    ) -> GpuSession | StopConfirmation:
        """Stop a session permanently. Always succeeds for any non-terminal state;
        idempotent for terminal states.

        Pre-active sessions (pending/provisioning — started_at IS NULL) are stopped
        immediately without the confirmation dialog and receive a full refund.

        Active/stale/paused/resuming sessions with started_at IS NOT NULL go through
        the two-call confirmation flow (confirmed=False → StopConfirmation preview;
        confirmed=True → execute stop with finalize-billing).

        Terminal states (stopping/stopped/failed) return the existing row immediately
        as a no-op; no work is performed.

        Args:
            session_id: Session to stop.
            user_id: Requesting user (ownership check).
            product_id: Product scope.
            confirmed: When False, return cost confirmation; when True, execute stop.
                       Ignored for pre-active and terminal sessions.

        Returns:
            StopConfirmation when confirmed=False for an active/stale/paused session.
            GpuSession (stopped or already-terminal) in all other cases.

        Raises:
            GpuSessionError: Session not found or not accessible.

        Concurrency model:
            This method does an initial unlocked read for state-based routing
            (terminal / pre-active / active), then dispatches to a helper that
            takes its own row lock. The race window between the unlocked read
            and the lock is intentional and handled:

            - Terminal-race: the appropriate helper detects the row is terminal
              under lock and returns idempotently (see
              ``gpu_session.stop.raced_to_terminal`` and
              ``gpu_session.stop_pre_active.raced_to_terminal`` log events).
            - Active-race (pre-active → active): ``_stop_pre_active`` falls
              through to ``_stop_confirmed`` under its lock (see
              ``gpu_session.stop_pre_active.raced_to_active``).

            Holding a lock across the routing decision would force
            ``SELECT FOR UPDATE`` on idempotent terminal reads and on the
            dry-run ``confirmed=False`` preview, both of which are read-only
            paths that don't need a lock. The two-round-trip pattern is the
            right trade-off for this workload.
        """
        # Initial load (no lock) for state-based routing.
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            session_row = await repo.get_by_id_for_user(session_id, user_id, product_id)
        if session_row is None:
            raise GpuSessionError(f"Session {session_id} not found or not accessible")

        # Terminal states: idempotent return — no work needed.
        if session_row.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
            logger.info(
                "gpu_session.stop.idempotent_terminal",
                session_id=str(session_id),
                status=str(session_row.status),
            )
            return session_row

        # Pre-active sessions never reached 'active', so started_at is NULL.
        # Skip the confirmation dialog — no billable usage occurred — and refund fully.
        if session_row.started_at is None:
            return await self._stop_pre_active(session_id, user_id, product_id, reason=reason)

        # Active/stale/paused/resuming with started_at IS NOT NULL: normal
        # two-call confirmation flow.
        if not confirmed:
            return await self._stop_confirmation(session_id, user_id, product_id)
        return await self._stop_confirmed(session_id, user_id, product_id, reason=reason)

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
    ) -> tuple[GpuSession, GpuSessionDeployment] | None:
        """Used by AishaGenerationProvider for routing.

        Returns only a session+deployment pair where the session is 'active'
        AND the deployment is 'active' — a single joined read (see
        GpuSessionDeploymentRepository.get_routable), so a session that flips
        away from active between two sequential reads can never be routed to.
        Paused/stale/provisioning sessions are NOT routable for generation.

        Model identity for generation (bundle_name/bundle_version) must be read
        from the returned deployment, not the session — see D18.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: Model type filter.

        Returns:
            (GpuSession, GpuSessionDeployment) tuple or None.
        """
        async with self._session_factory() as db:
            deployment_repo = GpuSessionDeploymentRepository(db)
            return await deployment_repo.get_routable(user_id, product_id, model_type.value)

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

    def _verify_ownership(
        self,
        session_row: GpuSession | None,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Raise GpuSessionError if the session isn't owned by (user_id, product_id).

        Returns the validated session row. Kept as a synchronous helper because
        it performs no I/O — the caller is responsible for opening the DB session
        and loading the row (typically with ``for_update=True``).
        """
        if (
            session_row is None
            or session_row.user_id != user_id
            or session_row.product_id != product_id
        ):
            raise GpuSessionError(f"Session {session_id} not found or access denied")
        return session_row

    def _ensure_status(
        self,
        session_row: GpuSession,
        allowed: frozenset[GpuSessionStatus],
        operation: str,
    ) -> None:
        """Raise InvalidSessionStateError if session is not in an allowed state."""
        if session_row.status not in allowed:
            raise InvalidSessionStateError(
                f"Cannot {operation} session in state '{session_row.status}'",
                current_status=session_row.status,
                operation=operation,
            )

    async def _load_session_for_update(
        self,
        repo: GpuSessionRepository,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        allowed_statuses: frozenset[GpuSessionStatus],
        operation: str,
    ) -> GpuSession:
        """Combined row-lock load + ownership check + status guard.

        Caller must already be inside ``async with db.begin()`` so the
        ``for_update=True`` lock stays active until the caller commits/rolls back.
        """
        session_row = self._verify_ownership(
            await repo.get_by_id(session_id, for_update=True),
            session_id,
            user_id,
            product_id,
        )
        self._ensure_status(session_row, allowed_statuses, operation)
        return session_row

    async def _set_status(
        self,
        db: AsyncSession,
        repo: GpuSessionRepository,
        session_row: GpuSession,
        new_status: GpuSessionStatus,
        *,
        paused_at: datetime | None = None,
        resumed_at: datetime | None = None,
        stopped_at: datetime | None = None,
    ) -> None:
        """Persist new_status + optional timestamp, and mirror onto the in-memory row.

        Extra timestamp fields are only forwarded to the repo when non-None so
        we don't accidentally clobber existing values with NULL.

        One of the two D15 lifecycle-cascade chokepoints (the other is
        GpuProvisioningWorker._transition): a transition to 'stopped' or
        'failed' flips every live deployment to 'removed'/'failed' in the same
        transaction. Pause/resume (paused/resuming) deliberately do not cascade
        — D17, a paused Vast.ai instance keeps its disk, so the deployment stays
        'active'.
        """
        extra: dict[str, datetime] = {}
        if paused_at is not None:
            extra["paused_at"] = paused_at
        if resumed_at is not None:
            extra["resumed_at"] = resumed_at
        if stopped_at is not None:
            extra["stopped_at"] = stopped_at

        await repo.update_status(session_row.id, new_status, **extra)
        session_row.status = new_status
        if paused_at is not None:
            session_row.paused_at = paused_at
        if resumed_at is not None:
            session_row.resumed_at = resumed_at
        if stopped_at is not None:
            session_row.stopped_at = stopped_at

        if new_status in (GpuSessionStatus.stopped, GpuSessionStatus.failed):
            to_status = (
                DeploymentStatus.removed
                if new_status == GpuSessionStatus.stopped
                else DeploymentStatus.failed
            )
            deployment_repo = GpuSessionDeploymentRepository(db)
            await deployment_repo.mark_status(
                session_row.id,
                from_statuses=LIVE_DEPLOYMENT_STATUSES,
                to_status=to_status,
                at=stopped_at or datetime.now(UTC),
            )

    async def _teardown_external_resources(
        self, session_row: GpuSession, *, log_prefix: str
    ) -> bool:
        """Best-effort teardown of Vast.ai instance + CF tunnel for a session.

        Always continues on failure so the DB status transition can proceed,
        but RETRIES the destroy call before giving up. The orphan sweeper
        is the long-tail recovery for whatever still slips through.

        Returns True if any teardown step failed (caller may want to log it).
        """
        had_errors = False
        if session_row.vastai_instance_id is not None:
            if not await self._destroy_with_retry(
                session_row.vastai_instance_id, log_prefix=log_prefix, session_id=session_row.id
            ):
                had_errors = True
            else:
                await self._mark_instance_destroyed(session_row.id)

        if session_row.cf_tunnel_id is not None and session_row.cf_dns_record_id is not None:
            try:
                await self._cf.delete_session_tunnel(
                    session_row.cf_tunnel_id,
                    session_row.cf_dns_record_id,
                )
            except Exception:
                had_errors = True
                logger.warning(
                    "%s.teardown_tunnel_failed",
                    log_prefix,
                    session_id=str(session_row.id),
                    tunnel_id=session_row.cf_tunnel_id,
                )
        return had_errors

    async def _destroy_with_retry(
        self, instance_id: int, *, log_prefix: str, session_id: UUID
    ) -> bool:
        """Try destroy_instance up to N times with exponential backoff.

        Returns True on success, False after exhausting retries.
        """
        attempts = self._settings.vastai_destroy_retry_attempts
        backoff_seconds = 1.0
        for attempt in range(1, attempts + 1):
            try:
                await self._vastai.destroy_instance(instance_id)
                if attempt > 1:
                    logger.info(
                        "%s.teardown_instance_recovered",
                        log_prefix,
                        session_id=str(session_id),
                        instance_id=instance_id,
                        attempt=attempt,
                    )
            except Exception as exc:
                if attempt == attempts:
                    logger.exception(
                        "%s.teardown_instance_failed",
                        log_prefix,
                        session_id=str(session_id),
                        instance_id=instance_id,
                        attempts=attempts,
                    )
                    return False
                logger.info(
                    "%s.teardown_instance_retry",
                    log_prefix,
                    session_id=str(session_id),
                    instance_id=instance_id,
                    attempt=attempt,
                    backoff_seconds=backoff_seconds,
                    error=str(exc),
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 5.0)
            else:
                return True
        return False  # unreachable, satisfies mypy

    async def _mark_instance_destroyed(self, session_id: UUID) -> None:
        """Record that destroy_instance succeeded for this session.

        The orphan sweeper uses this to skip sessions whose instance is confirmed gone.
        """
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            await repo.mark_instance_destroyed(session_id, datetime.now(UTC))

    async def _stop_confirmation(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> StopConfirmation:
        """Build stop confirmation DTO without changing any state."""
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            session_row = self._verify_ownership(
                await repo.get_by_id_for_user(session_id, user_id, product_id),
                session_id,
                user_id,
                product_id,
            )

        self._ensure_status(session_row, _STOPPABLE_STATUSES, "stop")

        durations = compute_session_durations(session_row, now=datetime.now(UTC))
        active_duration_seconds = durations.active_seconds
        paused_duration_seconds = durations.paused_seconds

        logger.info("gpu_session.stop.confirmation_requested", session_id=str(session_id))
        return StopConfirmation(
            session_id=session_id,
            model_type=session_row.model_type,
            vastai_gpu_name=session_row.vastai_gpu_name,
            vastai_cost_per_hour_micros=session_row.vastai_cost_per_hour_micros,
            active_duration_seconds=active_duration_seconds,
            paused_duration_seconds=paused_duration_seconds,
            message=(
                f"Stopping this session will permanently destroy the GPU node. "
                f"Active time: {active_duration_seconds // 3600}h "
                f"{(active_duration_seconds % 3600) // 60}m. "
                "Confirm to proceed."
            ),
        )

    async def _stop_pre_active(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        reason: str | None = None,
    ) -> GpuSession:
        """Stop a session that never became active (started_at IS NULL).

        No billable usage occurred. The base reservation is fully refunded.
        Any infrastructure provisioned so far (Vast.ai instance, CF tunnel)
        is torn down best-effort. Idempotent w.r.t. infrastructure: missing
        resources (Vast.ai 404, missing tunnel ID) are silently OK.

        Concurrency note: if the provisioning worker races to mark the session
        active between our initial load and the lock here, we fall through to
        the normal _stop_confirmed path (started_at will be non-NULL under lock).
        """
        logger.info("gpu_session.stop_pre_active.start", session_id=str(session_id))

        previous_status: str = ""  # set in TX1 unless we race-return early

        # TX1: verify + transition to 'stopping' (mutex against concurrent stops).
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            session_row = self._verify_ownership(
                await repo.get_by_id(session_id, for_update=True),
                session_id,
                user_id,
                product_id,
            )
            # Re-check status under lock — another stop may have beaten us.
            if session_row.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
                logger.info(
                    "gpu_session.stop_pre_active.raced_to_terminal",
                    session_id=str(session_id),
                    status=str(session_row.status),
                )
                return session_row
            # Race: provisioning worker promoted session to active between initial
            # load and this lock. Fall through to normal stop path.
            if session_row.started_at is not None:
                logger.info(
                    "gpu_session.stop_pre_active.raced_to_active",
                    session_id=str(session_id),
                )
            else:
                previous_status = str(session_row.status)
                await self._set_status(db, repo, session_row, GpuSessionStatus.stopping)

        # If the session raced to active under the lock, use the normal stop path.
        if session_row.started_at is not None:
            return await self._stop_confirmed(session_id, user_id, product_id, reason=reason)

        await self._publish_status_event(session_row, previous_status=previous_status)

        # Teardown — best-effort, errors logged not raised.
        await self._teardown_external_resources(
            session_row, log_prefix="gpu_session.stop_pre_active"
        )

        # Refund the base reservation in full — the user never got a working session.
        if session_row.account_id is not None:
            try:
                async with self._session_factory() as db, db.begin():
                    refund_result = await self._billing_service.refund(
                        session_row.id,
                        description="GPU session pre-active stop: session never became active",
                        session=db,
                        product_id=session_row.product_id,
                        user_id=session_row.user_id,
                    )
                if self._event_bus is not None:
                    await self._event_bus.publish_balance(refund_result.event)
                logger.info(
                    "gpu_session.stop_pre_active.refunded",
                    session_id=str(session_id),
                )
            except Exception:
                # Don't block the session transition on a refund failure — the
                # billing reconciler will catch up.
                logger.exception(
                    "gpu_session.stop_pre_active.refund_failed",
                    session_id=str(session_id),
                )

        # TX2: finalize to 'stopped'.
        stop_now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            reloaded = await repo.get_by_id(session_id, for_update=True)
            if reloaded is None:  # pragma: no cover
                raise GpuSessionError(f"Session {session_id} disappeared during pre-active stop")
            session_row = reloaded
            await self._set_status(
                db, repo, session_row, GpuSessionStatus.stopped, stopped_at=stop_now
            )
            await repo.mark_billing_finalized(session_id, stop_now)

        logger.info("gpu_session.stop_pre_active.success", session_id=str(session_id))
        await self._publish_status_event(session_row, previous_status="stopping", reason=reason)
        return session_row

    async def _stop_confirmed(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        reason: str | None = None,
    ) -> GpuSession:
        """Execute confirmed stop: teardown external resources and mark stopped.

        Concurrency note: the ``stopping`` intermediate status acts as a mutex.
        Once the first transaction commits with status='stopping', any concurrent
        ``_stop_confirmed`` call on the same session will fail ``_ensure_status``
        (stopping is not in _STOPPABLE_STATUSES). This lets us release the row
        lock during the slow external teardown (destroy instance + delete tunnel)
        without risking duplicate teardown calls.
        """
        logger.info("gpu_session.stop.begun", session_id=str(session_id))

        # Capture a single "now" for the whole stop sequence so the paused-time
        # rollup in TX1 and the stopped_at timestamp in TX2 are consistent. This
        # avoids classifying the external-teardown window as active time.
        stop_now = datetime.now(UTC)

        # Transaction 1 (short): verify + transition to 'stopping' (the mutex).
        # If stopping from 'paused', roll the in-flight pause duration into
        # total_paused_seconds so the column stays authoritative for the
        # post-stop billing calculation. Without this rollup, stop-from-paused
        # would bill the idle pause interval as active GPU time.
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            # Lock without status filter — terminal-race handled below as an
            # idempotent return; the pre-active race ("started_at became NULL")
            # cannot happen for sessions routed here via stop_session.
            session_row = self._verify_ownership(
                await repo.get_by_id(session_id, for_update=True),
                session_id,
                user_id,
                product_id,
            )
            # If the row raced to terminal between routing-read and this lock,
            # return idempotently. Matches stop_session's contract: stop always
            # succeeds; terminal states are no-ops.
            if session_row.status in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES:
                logger.info(
                    "gpu_session.stop.raced_to_terminal",
                    session_id=str(session_id),
                    status=str(session_row.status),
                )
                return session_row
            # Programming-error guard: _stop_confirmed is only reachable via
            # stop_session for sessions with started_at IS NOT NULL. If this fires
            # in production, something bypassed the public API.
            if session_row.started_at is None:
                raise RuntimeError(
                    "_stop_confirmed reached with started_at=NULL — caller bypassed stop_session"
                )
            previous_status = str(session_row.status)

            if session_row.status == GpuSessionStatus.paused and session_row.paused_at is not None:
                in_flight_pause_seconds = max(
                    0, int((stop_now - session_row.paused_at).total_seconds())
                )
                if in_flight_pause_seconds > 0:
                    await repo.add_paused_seconds(session_row.id, in_flight_pause_seconds)
                    # Keep the in-memory row aligned with the DB so downstream
                    # reads (and _finalize_billing) see the updated total.
                    session_row.total_paused_seconds += in_flight_pause_seconds
                    logger.info(
                        "gpu_session.stop.pause_rolled_up",
                        session_id=str(session_id),
                        in_flight_pause_seconds=in_flight_pause_seconds,
                        total_paused_seconds=session_row.total_paused_seconds,
                    )

            await self._set_status(db, repo, session_row, GpuSessionStatus.stopping)

        await self._publish_status_event(
            session_row, previous_status=previous_status, reason=reason
        )

        # Sweep in-flight jobs immediately after TX1 commits.
        #
        # Why here: TX1 committed status='stopping'; the poller's active-only
        # filter now excludes this session, so no new poll completions succeed.
        # Failing jobs early gives users clear feedback. If the poller races to
        # complete a job before our sweep, the conditional UPDATE inside
        # transition_to_failed is a no-op and the user keeps the result.
        #
        # Why best-effort: a sweep failure MUST NOT block external teardown.
        # Worst case: jobs hit the 30-min age timeout in the poller.
        if self._job_sweep is not None:
            await self._job_sweep.sweep_session_best_effort(
                session_id=session_row.id,
                product_id=session_row.product_id,
                failure=JobSweepFailure(
                    internal_reason="GPU session stopped before job completed."
                ),
                log_event="gpu_session.stop.job_sweep",
            )

        # External teardown — outside any DB transaction. Best-effort; errors are
        # logged but don't block the final 'stopped' transition. Orphaned resources
        # are cleaned by OrphanedTunnelCleanupWorker (Phase 1E-2).
        teardown_had_errors = await self._teardown_external_resources(
            session_row, log_prefix="gpu_session.stop"
        )

        # Transaction 2 (short): finalize status='stopped'. We re-load the row
        # in this transaction to attach it to the new session (the earlier
        # session_row is detached after its transaction closed) and to pick up
        # any concurrent column updates.
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            reloaded = await repo.get_by_id(session_id, for_update=True)
            if reloaded is None:  # pragma: no cover — race with DB deletion
                raise GpuSessionError(f"Session {session_id} disappeared during stop teardown")
            session_row = reloaded
            await self._set_status(
                db, repo, session_row, GpuSessionStatus.stopped, stopped_at=stop_now
            )

        logger.info(
            "gpu_session.stop.done",
            session_id=str(session_id),
            teardown_had_errors=teardown_had_errors,
        )
        await self._publish_status_event(session_row, previous_status="stopping", reason=reason)

        # Billing finalization — runs AFTER TX2 commits, outside any transaction.
        # If this fails the session is already stopped; billing drift is reconciled separately.
        if session_row.account_id is not None:
            await self._finalize_billing(session_row)

        return session_row

    async def _finalize_billing(self, session_row: GpuSession) -> None:
        """Compute actual usage and create overage debit or partial refund.

        Billing is per-started-minute (ceiling), with a 5-minute floor that
        matches the base reservation. A session running 5m01s is billed as
        6 minutes; the partial second of the sixth minute is fully charged.

        **Ledger-as-authority.** The refund/overage math uses the actual debit
        amount from the ledger (``get_debit_for_job(session.id).amount``), not
        a freshly-computed reservation amount. This makes over-refund impossible
        by construction even if ops changes ``gpu_session_tokens_per_minute``
        while sessions are running — we can only refund down to zero net charge
        on what was actually debited.

        The base reservation uses ``session_row.id`` as ``job_id`` so that
        full refund-on-failure (``BillingService.refund(job_id=...)``) and the
        single-debit lookup (``get_debit_for_job``) keep working. The overage
        debit, when present, uses ``job_id=None`` and records the parent
        session in metadata. This preserves the one-debit-per-job invariant
        on the base reservation while still giving us a complete audit trail
        via the metadata link.

        **Failure semantics.** On success, ``billing_finalized_at`` is stamped
        on the session row. On failure, it stays NULL and a phase-2 reconciler
        worker will retry (see ``billing_finalized_at`` column docstring). A
        single in-line retry with fixed backoff handles transient DB/connection
        hiccups; persistent failures are handed off to the reconciler.
        """
        account_id = session_row.account_id
        if account_id is None:
            return

        # total_paused_seconds is authoritative by this point (resume and
        # stop-from-paused both roll their pauses into the column). The helper
        # returns a two-bucket split; today we bill only active_seconds, but
        # paused_seconds is already separated out for upcoming storage billing.
        durations = compute_session_durations(session_row, now=datetime.now(UTC))
        billable_minutes = billable_minutes_for_active_seconds(durations.active_seconds)
        billable_tokens = billable_minutes * self._settings.gpu_session_tokens_per_minute

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                await self._apply_finalize_billing(
                    session_row=session_row,
                    billable_tokens=billable_tokens,
                    billable_minutes=billable_minutes,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "gpu_session.billing.finalization_attempt_failed",
                        session_id=str(session_row.id),
                        attempt=attempt,
                        error=str(last_exc),
                    )
                    # Fixed 1s backoff — keeps this synchronous stop path short.
                    # Anything that doesn't recover in 1s is a bug or persistent
                    # issue; the reconciler is the right recovery channel.
                    await asyncio.sleep(1.0)
            else:
                return  # success — billing_finalized_at stamped inside

        # Both attempts failed. Leave billing_finalized_at NULL for the
        # reconciler to pick up. Log at ERROR level for alerting.
        logger.error(
            "gpu_session.billing.finalization_deferred",
            session_id=str(session_row.id),
            billable_tokens=billable_tokens,
            billable_minutes=billable_minutes,
            error=str(last_exc) if last_exc else "unknown",
        )

    async def finalize_billing_for_session(self, session_row: GpuSession) -> bool:
        """Public wrapper around ``_finalize_billing`` for the reconciler worker.

        Runs the same in-line retry logic as the stop path, then re-checks
        the row's ``billing_finalized_at`` column to determine whether the
        attempt succeeded. Returns True on success, False otherwise.

        Encapsulating the success check here (rather than in the worker)
        keeps the worker's loop linear and avoids exposing the private
        ``_finalize_billing`` method as part of any cross-component contract.

        The re-fetch costs one extra query per candidate, paid for by
        removing the worker's identical re-query — net zero query count.
        """
        await self._finalize_billing(session_row)
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            refreshed = await repo.get_by_id(session_row.id)
        return refreshed is not None and refreshed.billing_finalized_at is not None

    async def _apply_finalize_billing(
        self,
        *,
        session_row: GpuSession,
        billable_tokens: int,
        billable_minutes: int,
    ) -> None:
        """One finalization attempt — isolates the transactional work.

        Compares ``billable_tokens`` against total settled tokens for the session
        (base reservation + any metered debits from SessionCreditGuard) and creates
        a clamped overage debit or partial refund as appropriate.

        Uses ``get_settled_tokens_for_session`` rather than just the base reservation
        amount so that metered debits during the session lifetime are not double-charged.

        Overage debits are recorded via ``settle_session_usage`` at full cost; balance
        may go negative (debt recorded, not prevented). Prevention of new work lives
        in ``check_and_reserve``.

        Stamps ``billing_finalized_at`` on success so the reconciler knows to skip
        this session.
        """
        pending_event: BalanceEvent | None = None
        async with self._session_factory() as db, db.begin():
            repo = BillingRepository(db)

            # Look up the base reservation to find the account and original debit amount.
            debit = await repo.get_debit_for_job(session_row.id)
            if debit is None:
                # No base reservation found. This can happen if the reservation
                # step failed between external resource creation and commit, OR
                # if an admin manually reversed the debit. Nothing to reconcile.
                logger.warning(
                    "gpu_session.billing.no_debit_found",
                    session_id=str(session_row.id),
                )
                await GpuSessionRepository(db).mark_billing_finalized(
                    session_row.id, datetime.now(UTC)
                )
                return

            account_id = debit.account_id
            original_debit = abs(debit.amount)

            # Total settled = base reservation + all metered debits from credit guard.
            # This is the authoritative "already charged" figure.
            total_settled = await repo.get_settled_tokens_for_session(session_row.id)

            if billable_tokens > total_settled:
                # Remaining overage after metered debits — record full cost; may drive balance negative.
                overage = billable_tokens - total_settled
                settle_result = await self._billing_service.settle_session_usage(
                    account_id,
                    overage,
                    session_id=session_row.id,
                    model_type=session_row.model_type,
                    session=db,
                    product_id=session_row.product_id,
                    user_id=session_row.user_id,
                    extra_metadata={
                        "type": "gpu_session_overage",
                        "bundle_name": session_row.bundle_name,
                        "billable_minutes": billable_minutes,
                    },
                )
                pending_event = settle_result.event
                logger.info(
                    "gpu_session.billing.overage_charged",
                    session_id=str(session_row.id),
                    overage_tokens=overage,
                    settled_tokens=settle_result.settled_tokens,
                    billable_minutes=billable_minutes,
                    total_settled=total_settled,
                    original_debit=original_debit,
                )
            elif billable_tokens < total_settled:
                # Over-charged relative to actual usage (e.g. short session with metered debits).
                # Refund against the base reservation; cap at original_debit to prevent over-refund.
                overpaid = total_settled - billable_tokens
                refund_amount = min(overpaid, original_debit)
                if refund_amount > 0:
                    refund_result = await self._billing_service.partial_refund(
                        session_row.id,
                        refund_amount,
                        description=(
                            f"GPU session partial refund: used {billable_minutes}min at current rate"
                        ),
                        session=db,
                        product_id=session_row.product_id,
                        user_id=session_row.user_id,
                    )
                    pending_event = refund_result.event
                    logger.info(
                        "gpu_session.billing.partial_refund",
                        session_id=str(session_row.id),
                        refund_tokens=refund_amount,
                        billable_minutes=billable_minutes,
                        total_settled=total_settled,
                        original_debit=original_debit,
                    )

            await GpuSessionRepository(db).mark_billing_finalized(session_row.id, datetime.now(UTC))

        if self._event_bus is not None:
            await self._event_bus.publish_balance(pending_event)

    async def _publish_status_event(
        self,
        session: GpuSession,
        previous_status: str,
        error_message: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Fire-and-forget SSE publish. Delegates to the shared helper."""
        await publish_status_event(
            self._event_bus,
            session,
            previous_status=previous_status,
            error_message=error_message,
            reason=reason,
            ops_event_bus=self._ops_event_bus,
        )

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

    async def _provision_instance(
        self,
        offers: Sequence[VastAIOffer],
        disk_gb: int,
        env: dict[str, str],
        template_hash_id: str | None,
    ) -> tuple[int, VastAIOffer]:
        """Thin wrapper around the shared provision_vastai_instance helper."""
        return await provision_vastai_instance(
            vastai_client=self._vastai,
            offers=offers,
            disk_gb=disk_gb,
            env=env,
            template_hash_id=template_hash_id,
            cooldown_store=self._cooldown,
            onstart_cmd=make_onstart_cmd(self._settings.aisha_branch),
            offer_walk_depth=self._settings.provisioning_offer_walk_depth,
        )
