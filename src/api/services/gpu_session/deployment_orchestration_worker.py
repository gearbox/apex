"""Background worker: drives a deployment through provision -> restart -> active (P4).

OperationEventService is a pure writer (P1); the state machine for what happens
*after* a command reaches a terminal state belongs to a worker (D32). This worker is
the P4 counterpart to GpuSessionCommandSweepWorker: each tick it looks for deployments
whose current command finished and advances them, exactly mirroring the shape (and the
LeaderLease) of every other PeriodicWorker in this codebase.

Eight independent steps per tick, in this order so a batch that finishes and drains in
the same tick doesn't wait a full interval for its restart:
  0. Recover a deployment's provision pointer lost between commit and enqueue.
  1. Reap an attach that committed but never enqueued a provision command.
  1b. Reap a removal stranded without its removal command.
  2. Provision finished  -> pending_restart=true (success) or 'failed' (failure).
  3. Session drain       -> enqueue one comfyui_restart per readiness marker.
  4. Restart finished    -> 'active' (success) or 'failed', no retry (D34).
  5. Removal finished    -> 'removed' (success) or back to 'active' (failure).
  6. Reconcile orphaned routing suspensions after every restart owner is terminal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.services.gpu_session._events import publish_deployment_event
from src.api.services.gpu_session.command_payload import RestartCommand
from src.api.services.gpu_session.command_service import CommandEnqueueSessionError
from src.core.enums import TERMINAL_COMMAND_STATUSES, CommandStatus, DeploymentStatus, OperationKind
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_deployment import (
    RESTART_COHORT_SANITY_LIMIT,
    GpuSessionDeploymentRepository,
)
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository
from src.db.repositories.job import JobRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.event_bus import EventBus
    from src.api.services.gpu_session.command_service import GpuSessionCommandService
    from src.core.config import Settings
    from src.db.models.gpu_session_command import GpuSessionCommand
    from src.db.models.gpu_session_deployment import GpuSessionDeployment
    from src.db.models.gpu_session_operation import GpuSessionOperation

logger = structlog.get_logger(__name__)

# CO5: aisha's BundleRemover.remove() (gearbox/aisha @ a7a65d864231c31709124dfe3c6c74a99ea5a0a9,
# src/ai_content_service/remover.py) raises RemovalError(f"bundle {bundle_name!r} is not
# recorded in residency; resident bundles: {names}") when the bundle was already removed from
# this node's residency manifest. A D23 re-claim after an agent crash can legitimately run the
# removal command twice; the second run's failure must not fail an operation whose weights are
# already gone.
_ALREADY_REMOVED_ERROR_MARKER = "is not recorded in residency"
_ORPHANED_DEPLOYMENT_GRACE = timedelta(minutes=1)


class _RestartCohortChangedError(RuntimeError):
    """The candidate cohort changed before its restart pointer was stamped."""


def _require_restart_pointer_update(updated: int, expected: int) -> None:
    """Abort the transaction unless every cohort member received the pointer."""
    if updated != expected:
        raise _RestartCohortChangedError


def _is_already_removed_error(error: str | None) -> bool:
    return error is not None and _ALREADY_REMOVED_ERROR_MARKER in error


class DeploymentOrchestrationWorker(PeriodicWorker):
    """Advances P4 deployments through provision -> restart -> active/removed."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        command_service: GpuSessionCommandService,
        settings: Settings,
        event_bus: EventBus | None = None,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="deployment_orchestration",
            interval_seconds=settings.gpu_deployment_orchestration_interval_seconds,
            jitter_seconds=2.0,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._session_factory = session_factory
        self._command_service = command_service
        self._settings = settings
        self._event_bus = event_bus

    async def run_once(self) -> None:
        await self._recover_deployments_missing_provision_pointer()
        await self._reap_orphaned_deployments_without_provision_command()
        await self._reap_orphaned_removals_without_command()
        await self._process_provision_results()
        await self._process_restart_enqueue()
        await self._process_restart_results()
        await self._process_removal_results()
        await self._reconcile_orphaned_routing_suspensions()

    # ------------------------------------------------------------------
    # Step 0 — repair a lost provision pointer before the reaper can see it
    # ------------------------------------------------------------------

    async def _recover_deployments_missing_provision_pointer(self) -> None:
        """N3: attach's create() and enqueue_batch() commit in separate transactions,
        so a process death between them can leave a deployment with a real, queued
        provision command but a null provision_operation_id. Repairing the pointer
        here — before the orphan reaper runs below in the same tick — means the
        reaper only ever fires on a deployment that genuinely has no command.
        """
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_deployments_missing_provision_pointer()
        for deployment in candidates:
            async with self._session_factory() as db:
                command = await GpuSessionCommandRepository(db).get_latest_by_deployment_and_kind(
                    deployment.id, OperationKind.bundle_provision
                )
            if command is None:
                continue
            async with self._session_factory() as db, db.begin():
                recovered = await GpuSessionDeploymentRepository(db).recover_provision_pointer(
                    deployment.id, operation_id=command.operation_id, batch_id=command.batch_id
                )
            if not recovered:
                continue
            logger.warning(
                "gpu_session.deployment.provision_pointer_recovered",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
                operation_id=str(command.operation_id),
            )

    # ------------------------------------------------------------------
    # Step 1 — committed attach with no provision command
    # ------------------------------------------------------------------

    async def _reap_orphaned_deployments_without_provision_command(self) -> None:
        now = datetime.now(UTC)
        created_before = now - _ORPHANED_DEPLOYMENT_GRACE
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_orphaned_deployments_without_provision_command(created_before=created_before)
        for deployment in candidates:
            async with self._session_factory() as db, db.begin():
                applied = await GpuSessionDeploymentRepository(
                    db
                ).fail_orphaned_deployment_without_provision_command(
                    deployment.id,
                    created_before=created_before,
                    at=now,
                )
            if not applied:
                continue
            deployment.status = DeploymentStatus.failed
            deployment.removed_at = now
            logger.error(
                "gpu_session.deployment.orphaned_no_command",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(
                self._event_bus,
                deployment,
                error_message="provision command was not enqueued",
            )

    async def _reap_orphaned_removals_without_command(self) -> None:
        """Return old removing rows to active if no removal command was persisted."""
        now = datetime.now(UTC)
        created_before = now - _ORPHANED_DEPLOYMENT_GRACE
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_orphaned_removals_without_command(created_before=created_before)
        for deployment in candidates:
            async with self._session_factory() as db, db.begin():
                applied = await GpuSessionDeploymentRepository(
                    db
                ).restore_orphaned_removal_without_command(
                    deployment.id, created_before=created_before
                )
            if not applied:
                continue
            deployment.status = DeploymentStatus.active
            logger.error(
                "gpu_session.deployment.removing_no_command",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            # S1: explicit None, not the fallback — a row returned to 'active'
            # because its removal command was never persisted genuinely has no
            # governing operation; reporting a stale restart_operation_id here
            # would be worse than reporting nothing.
            await publish_deployment_event(
                self._event_bus,
                deployment,
                operation_id=None,
                error_message="removal command was not enqueued",
            )

    # ------------------------------------------------------------------
    # Step 2 — provision finished
    # ------------------------------------------------------------------

    async def _process_provision_results(self) -> None:
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_deploying_awaiting_provision_result()
        for deployment in candidates:
            try:
                await self._process_one_provision_result(deployment)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.provision_result_error",
                    deployment_id=str(deployment.id),
                )

    async def _process_one_provision_result(self, deployment: GpuSessionDeployment) -> None:
        # S4: fetch the command and the operation it governs in the same
        # session — one fewer connection checkout than fetching them
        # separately, since the operation lookup only ever needs command.id.
        async with self._session_factory() as db:
            command = await GpuSessionCommandRepository(db).get_latest_by_deployment_and_kind(
                deployment.id, OperationKind.bundle_provision
            )
            if command is None or command.status not in TERMINAL_COMMAND_STATUSES:
                return
            operation = await GpuSessionOperationRepository(db).get(command.operation_id)

        succeeded = command.status == CommandStatus.succeeded
        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            applied = await GpuSessionDeploymentRepository(db).resolve_provision_outcome(
                deployment.id, succeeded=succeeded, at=now
            )
        if not applied:
            return
        if succeeded:
            deployment.pending_restart = True
            logger.info(
                "gpu_session.deployment.pending_restart",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(self._event_bus, deployment, operation=operation)
        else:
            deployment.status = DeploymentStatus.failed
            deployment.removed_at = now
            logger.warning(
                "gpu_session.deployment.provision_failed",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
                command_status=str(command.status),
                error=command.error,
            )
            await publish_deployment_event(
                self._event_bus, deployment, operation=operation, error_message=command.error
            )

    # ------------------------------------------------------------------
    # Step 3 — session drain -> enqueue restart
    # ------------------------------------------------------------------

    async def _process_restart_enqueue(self) -> None:
        await self._reap_pending_restart_missing_batch_id()
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_pending_restart_awaiting_operation()
        session_ids = {deployment.session_id for deployment in candidates}
        for session_id in session_ids:
            try:
                async with self._session_factory() as db:
                    members = await GpuSessionDeploymentRepository(
                        db
                    ).list_pending_restart_awaiting_operation_for_session(session_id)
                if len(members) > RESTART_COHORT_SANITY_LIMIT:
                    logger.error(
                        "gpu_session.deployment.restart_cohort_implausible",
                        session_id=str(session_id),
                        count=len(members),
                    )
                    continue
                await self._process_one_restart_session(session_id, members)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.restart_enqueue_error", session_id=str(session_id)
                )

    async def _reap_pending_restart_missing_batch_id(self) -> None:
        """S3: a pending-restart row with no batch_id never matches the
        correlated EXISTS the candidate queries use to skip batches that are
        still mid-provision-retry, so — unlike every other incomplete-batch
        case, which clears on its own — it would otherwise pin the bounded
        restart-enqueue page forever. Should be unreachable (every writer of
        pending_restart also stamps batch_id); fail it once, loudly, rather
        than re-log and re-skip it on every tick.
        """
        now = datetime.now(UTC)
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_pending_restart_missing_batch_id()
        for deployment in candidates:
            async with self._session_factory() as db, db.begin():
                applied = await GpuSessionDeploymentRepository(
                    db
                ).fail_pending_restart_missing_batch_id(deployment.id, at=now)
            if not applied:
                continue
            deployment.status = DeploymentStatus.failed
            deployment.pending_restart = False
            deployment.removed_at = now
            logger.error(
                "gpu_session.deployment.pending_restart_missing_batch_id",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(
                self._event_bus,
                deployment,
                error_message="pending_restart deployment has no batch_id",
            )

    async def _process_one_restart_session(
        self, session_id: UUID, ready_members: Sequence[GpuSessionDeployment]
    ) -> None:
        """Restart each ready marker cohort on one session exactly once.

        Every attach has its own provision batch, but ComfyUI restarts are a session
        concern. ``ready_members`` already excludes any batch still provisioning
        (N2) and any invariant-violating no-batch_id row (S3) — both predicates
        live in list_pending_restart_awaiting_operation_for_session now, not
        here. Different readiness markers cannot share an aisha restart
        command, because its contract carries only one ``node_class`` to verify.
        """
        if not ready_members:
            return

        # S5: suspend routing for every currently-active deployment on this
        # session, in the same transaction as the in-flight count, before that
        # count is read — a restart takes the whole node down, not only the
        # deployment(s) being restarted, so a generation must not be able to
        # resolve this session's routable deployments between this drain check
        # and the restart actually happening. The suspension is deliberately
        # never rolled back here on a non-zero count (only cleared once the
        # whole restart cycle resolves, in step 4) — that monotonicity is what
        # makes D35's drain timeout an actual bound rather than a hope.
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionDeploymentRepository(db)
            await repo.acquire_removal_lock(session_id)
            routing_suspended = await repo.suspend_routing_for_session(session_id)
            in_flight = await JobRepository(db).count_in_flight_for_session(session_id)

        for deployment in routing_suspended:
            await publish_deployment_event(self._event_bus, deployment)

        cohorts: dict[str | None, list[GpuSessionDeployment]] = {}
        for member in ready_members:
            cohorts.setdefault(member.readiness_marker_node_class, []).append(member)

        for marker, members in cohorts.items():
            await self._enqueue_restart_cohort(
                session_id=session_id,
                members=members,
                in_flight=in_flight,
                readiness_marker_node_class=marker,
            )

    async def _enqueue_restart_cohort(
        self,
        *,
        session_id: UUID,
        members: Sequence[GpuSessionDeployment],
        in_flight: int,
        readiness_marker_node_class: str | None,
    ) -> None:
        """Drain and enqueue one restart for deployments sharing one marker."""
        now = datetime.now(UTC)
        # D35 is an upper bound for the oldest ready deployment, not the most
        # recent attach in a busy session.  Cohorts are marker-specific, so take
        # the oldest timestamp within this cohort.
        pending_restart_since = min(
            (
                member.pending_restart_since
                for member in members
                if member.pending_restart_since is not None
            ),
            default=None,
        )
        timed_out = (
            pending_restart_since is not None
            and (now - pending_restart_since).total_seconds()
            >= self._settings.gpu_deployment_restart_drain_timeout_seconds
        )

        if in_flight > 0:
            if not timed_out:
                return
            logger.warning(
                "gpu_session.deployment.restart_drain_timeout",
                session_id=str(session_id),
                batch_ids=sorted({member.batch_id for member in members if member.batch_id}),
                in_flight_count=in_flight,
                readiness_marker_node_class=readiness_marker_node_class,
            )

        try:
            # The restart command and every member's pointer form one invariant:
            # a pending deployment must never see a committed restart command
            # without pointing at it. Keeping both writes in this transaction
            # eliminates the duplicate-enqueue crash window outright.
            async with self._session_factory() as db, db.begin():
                command = await self._command_service.enqueue(
                    session_id=session_id,
                    product_id=members[0].product_id,
                    command=RestartCommand(node_class=readiness_marker_node_class),
                    db=db,
                )
                updated = await GpuSessionDeploymentRepository(db).set_restart_pointer(
                    [member.id for member in members], operation_id=command.operation_id
                )
                _require_restart_pointer_update(updated, len(members))
        except CommandEnqueueSessionError:
            # Session went terminal — the D15 chokepoint cascade already flipped these
            # deployments away from 'deploying'/pending_restart, so they'll drop out of
            # the next tick's candidate query on their own. Nothing to compensate here.
            logger.info(
                "gpu_session.deployment.restart_enqueue_skipped_session_unavailable",
                session_id=str(session_id),
                batch_ids=sorted({member.batch_id for member in members if member.batch_id}),
            )
            return
        except _RestartCohortChangedError:
            # The exception rolls the enclosing transaction back, including the
            # freshly-created command and operation. A later tick re-reads the
            # cohort instead of leaving an orphaned command behind.
            logger.info(
                "gpu_session.deployment.restart_enqueue_skipped_cohort_changed",
                session_id=str(session_id),
                batch_ids=sorted({member.batch_id for member in members if member.batch_id}),
            )
            return
        logger.info(
            "gpu_session.deployment.restart_enqueued",
            session_id=str(session_id),
            batch_ids=sorted({member.batch_id for member in members if member.batch_id}),
            operation_id=str(command.operation_id),
            deployment_count=updated,
            readiness_marker_node_class=readiness_marker_node_class,
        )
        for member in members:
            member.restart_operation_id = command.operation_id
            await publish_deployment_event(self._event_bus, member)

    # ------------------------------------------------------------------
    # Step 4 — restart finished
    # ------------------------------------------------------------------

    async def _process_restart_results(self) -> None:
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_pending_restart_awaiting_outcome()
        sessions: dict[UUID, list[GpuSessionDeployment]] = {}
        for deployment in candidates:
            if deployment.restart_operation_id is None:  # pragma: no cover — query guarantees this
                continue
            sessions.setdefault(deployment.session_id, []).append(deployment)
        for session_id, members in sessions.items():
            try:
                await self._process_one_restart_cycle(session_id, members)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.restart_outcome_error",
                    session_id=str(session_id),
                )

    async def _process_one_restart_cycle(
        self, session_id: UUID, members: Sequence[GpuSessionDeployment]
    ) -> None:
        """S6: a restart cycle activates only once EVERY restart enqueued
        within it has succeeded, and fails as a whole the moment ANY of them
        fails or is expired by the P3 sweep — never on mere terminality
        (which says nothing about whether ComfyUI actually came back up), and
        never only the one cohort whose own command happened to resolve.

        R3 splits one session's restart into a cohort per readiness marker,
        and the command queue serializes them one in-flight at a time, so
        cohorts in one cycle can go terminal on different ticks. A failure or
        expiry anywhere in the cycle means Apex no longer knows which node
        classes ComfyUI actually has loaded, so every deployment in the cycle
        is failed immediately — D34 already rules out a retry, so there is
        nothing to gain by waiting on a still-pending sibling's own outcome.
        Success is the opposite: it can only be declared once every member's
        command is terminal, since any one of them could still turn out to be
        the failure that dooms the whole cycle.

        ``members`` is exactly "every deployment on this session still
        pending_restart with a restart_operation_id assigned" — the whole
        cycle, by construction: a cohort whose restart gets enqueued after
        this tick started shows up in this same query on the tick that
        processes it, pulling it into the group too.
        """
        operation_ids = sorted(
            {member.restart_operation_id for member in members if member.restart_operation_id},
            key=str,
        )
        if not operation_ids:  # pragma: no cover — every member has one by construction
            return

        now = datetime.now(UTC)
        commands: dict[UUID, GpuSessionCommand] = {}
        operations: dict[UUID, GpuSessionOperation] = {}
        # S4: fetch every command and the operation it governs in one session —
        # same data as two separate lookups per operation, fewer connections.
        async with self._session_factory() as db:
            command_repo = GpuSessionCommandRepository(db)
            operation_repo = GpuSessionOperationRepository(db)
            for operation_id in operation_ids:
                command = await command_repo.get_by_operation(operation_id)
                if command is not None:
                    commands[operation_id] = command
                operation = await operation_repo.get(operation_id)
                if operation is not None:
                    operations[operation_id] = operation

        failed_commands = [
            command
            for command in commands.values()
            if command.status in TERMINAL_COMMAND_STATUSES
            and command.status != CommandStatus.succeeded
        ]
        if not failed_commands:
            still_pending = len(commands) < len(operation_ids) or any(
                command.status not in TERMINAL_COMMAND_STATUSES for command in commands.values()
            )
            if still_pending:
                logger.info(
                    "gpu_session.deployment.restart_cycle_pending",
                    session_id=str(session_id),
                    deployment_count=len(members),
                )
                return

        succeeded = not failed_commands
        ids = [member.id for member in members]
        routing_released: Sequence[GpuSessionDeployment] = ()
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionDeploymentRepository(db)
            updated = await repo.resolve_restart_outcome(ids, succeeded=succeeded, at=now)
            if succeeded:
                # Per-row hygiene above clears the cycle members' flags.  This
                # session-wide release handles non-members such as the primary.
                routing_released = await repo.clear_routing_suspended_for_session(session_id)
            else:
                cancellation_reason = "restart cycle failed before command could be claimed"
                cancelled = await GpuSessionCommandRepository(db).cancel_queued_by_session_and_kind(
                    session_id,
                    OperationKind.comfyui_restart,
                    at=now,
                    reason=cancellation_reason,
                )
                operation_repo = GpuSessionOperationRepository(db)
                for command in cancelled:
                    await operation_repo.close_failed(
                        command.operation_id,
                        at=now,
                        error=cancellation_reason,
                    )

                # A claimed sibling may already be restarting ComfyUI.  It must
                # keep the session-wide suspension until it reaches a terminal
                # state; cancelling that row would be a lie.
                has_remaining_restart = await GpuSessionCommandRepository(
                    db
                ).has_non_terminal_by_session_and_kind(session_id, OperationKind.comfyui_restart)
                if not has_remaining_restart:
                    # resolve_restart_outcome is per-row hygiene for the cycle
                    # members. This session-wide release is what reopens active
                    # non-members such as the primary.
                    routing_released = await repo.clear_routing_suspended_for_session(session_id)
        for deployment in routing_released:
            await publish_deployment_event(self._event_bus, deployment)

        # A terminal session cascade can win after the candidate query.  Its
        # guarded UPDATE intentionally returns zero; do not publish a stale
        # active/failed deployment event in that case. The routing-release
        # events above remain current and are deliberately still published.
        if updated == 0:
            return

        error_message: str | None = None
        if not succeeded:
            failing_description = ", ".join(
                f"{command.operation_id} ({command.error or command.status})"
                for command in failed_commands
            )
            error_message = f"restart cycle failed: {failing_description}"
            logger.warning(
                "gpu_session.deployment.restart_cycle_failed",
                session_id=str(session_id),
                deployment_count=updated,
                failing_operation_ids=[str(command.operation_id) for command in failed_commands],
            )
        else:
            logger.info(
                "gpu_session.deployment.activated",
                session_id=str(session_id),
                deployment_count=updated,
            )

        for member in members:
            operation = (
                operations.get(member.restart_operation_id) if member.restart_operation_id else None
            )
            member.pending_restart = False
            if succeeded:
                member.status = DeploymentStatus.active
                member.activated_at = now
                await publish_deployment_event(self._event_bus, member, operation=operation)
            else:
                member.status = DeploymentStatus.failed
                member.removed_at = now
                await publish_deployment_event(
                    self._event_bus, member, operation=operation, error_message=error_message
                )

    # ------------------------------------------------------------------
    # Step 5 — removal finished
    # ------------------------------------------------------------------

    async def _process_removal_results(self) -> None:
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(db).list_removing()
        for deployment in candidates:
            try:
                await self._process_one_removal_result(deployment)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.removal_result_error",
                    deployment_id=str(deployment.id),
                )

    async def _process_one_removal_result(self, deployment: GpuSessionDeployment) -> None:
        # S4: fetch the command and the operation it governs in the same
        # session — one fewer connection checkout than fetching them
        # separately, since the operation lookup only ever needs command.id.
        async with self._session_factory() as db:
            command = await GpuSessionCommandRepository(db).get_latest_by_deployment_and_kind(
                deployment.id, OperationKind.bundle_removal
            )
            if command is None or command.status not in TERMINAL_COMMAND_STATUSES:
                return
            operation = await GpuSessionOperationRepository(db).get(command.operation_id)

        succeeded = self._removal_succeeded(command)
        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            applied = await GpuSessionDeploymentRepository(db).resolve_removal_outcome(
                deployment.id, succeeded=succeeded, at=now
            )
        if not applied:
            return
        if succeeded:
            deployment.status = DeploymentStatus.removed
            deployment.removed_at = now
            logger.info(
                "gpu_session.deployment.removed",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(self._event_bus, deployment, operation=operation)
        else:
            deployment.status = DeploymentStatus.active
            logger.warning(
                "gpu_session.deployment.removal_failed",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
                command_status=str(command.status),
                error=command.error,
            )
            await publish_deployment_event(
                self._event_bus, deployment, operation=operation, error_message=command.error
            )

    # ------------------------------------------------------------------
    # Step 6 — release a suspension that no longer has a restart owner
    # ------------------------------------------------------------------

    async def _reconcile_orphaned_routing_suspensions(self) -> None:
        """Repair a session-wide suspension stranded outside the normal cycle path."""
        async with self._session_factory() as db:
            session_ids = await GpuSessionDeploymentRepository(
                db
            ).list_orphaned_routing_suspension_session_ids()
        for session_id in session_ids:
            async with self._session_factory() as db, db.begin():
                released = await GpuSessionDeploymentRepository(
                    db
                ).clear_orphaned_routing_suspension(session_id)
            if released:
                logger.warning(
                    "gpu_session.deployment.routing_suspension_released",
                    session_id=str(session_id),
                    deployment_count=len(released),
                )
                for deployment in released:
                    await publish_deployment_event(self._event_bus, deployment)

    @staticmethod
    def _removal_succeeded(command: GpuSessionCommand) -> bool:
        if command.status == CommandStatus.succeeded:
            return True
        # CO5: tolerate a re-claimed removal racing an already-completed one.
        return _is_already_removed_error(command.error)
