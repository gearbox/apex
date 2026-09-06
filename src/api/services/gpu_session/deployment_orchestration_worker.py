"""Background worker: drives a deployment through provision -> restart -> active (P4).

OperationEventService is a pure writer (P1); the state machine for what happens
*after* a command reaches a terminal state belongs to a worker (D32). This worker is
the P4 counterpart to GpuSessionCommandSweepWorker: each tick it looks for deployments
whose current command finished and advances them, exactly mirroring the shape (and the
LeaderLease) of every other PeriodicWorker in this codebase.

Six independent steps per tick, in this order so a batch that finishes and drains in
the same tick doesn't wait a full interval for its restart:
  0. Recover a deployment's provision pointer lost between commit and enqueue.
  1. Reap an attach that committed but never enqueued a provision command.
  1b. Reap a removal stranded without its removal command.
  2. Provision finished  -> pending_restart=true (success) or 'failed' (failure).
  3. Session drain       -> enqueue one comfyui_restart per readiness marker.
  4. Restart finished    -> 'active' (success) or 'failed', no retry (D34).
  5. Removal finished    -> 'removed' (success) or back to 'active' (failure).
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
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
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

logger = structlog.get_logger(__name__)

# CO5: aisha's BundleRemover.remove() (gearbox/aisha @ a7a65d864231c31709124dfe3c6c74a99ea5a0a9,
# src/ai_content_service/remover.py) raises RemovalError(f"bundle {bundle_name!r} is not
# recorded in residency; resident bundles: {names}") when the bundle was already removed from
# this node's residency manifest. A D23 re-claim after an agent crash can legitimately run the
# removal command twice; the second run's failure must not fail an operation whose weights are
# already gone.
_ALREADY_REMOVED_ERROR_MARKER = "is not recorded in residency"
_ORPHANED_DEPLOYMENT_GRACE = timedelta(minutes=1)


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
            await publish_deployment_event(
                self._event_bus,
                deployment,
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
        async with self._session_factory() as db:
            command = await GpuSessionCommandRepository(db).get_latest_by_deployment_and_kind(
                deployment.id, OperationKind.bundle_provision
            )
        if command is None or command.status not in TERMINAL_COMMAND_STATUSES:
            return

        succeeded = command.status == CommandStatus.succeeded
        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            applied = await GpuSessionDeploymentRepository(db).resolve_provision_outcome(
                deployment.id, succeeded=succeeded, at=now
            )
        if not applied:
            return
        # The command was just fetched above — the operation it governs is one
        # cheap lookup away, so the SSE event can carry the node's own terminal
        # phase/progress rather than falling back to id-only (N4).
        async with self._session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(command.operation_id)
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
                await self._process_one_restart_session(session_id, members)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.restart_enqueue_error", session_id=str(session_id)
                )

    async def _process_one_restart_session(
        self, session_id: UUID, candidates: Sequence[GpuSessionDeployment]
    ) -> None:
        """Restart each ready marker cohort on one session exactly once.

        Every attach has its own provision batch, but ComfyUI restarts are a session
        concern. ``candidates`` already excludes any batch still provisioning (N2's
        batch-readiness predicate lives in
        list_pending_restart_awaiting_operation_for_session now, not here), so this
        only needs to filter the one invariant-violation case that predicate can't
        express. Different readiness markers cannot share an aisha restart command,
        because its contract carries only one ``node_class`` to verify.
        """
        ready_members: list[GpuSessionDeployment] = []
        for deployment in candidates:
            if deployment.batch_id is None:
                # Every attach stamps batch_id at enqueue time (CO4) — a
                # pending-restart deployment with no batch_id means that write
                # was skipped somewhere.
                logger.error(
                    "gpu_session.deployment.pending_restart_missing_batch_id",
                    deployment_id=str(deployment.id),
                )
                continue
            ready_members.append(deployment)

        if not ready_members:
            return

        async with self._session_factory() as db:
            in_flight = await JobRepository(db).count_in_flight_for_session(session_id)

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
            command = await self._command_service.enqueue(
                session_id=session_id,
                product_id=members[0].product_id,
                command=RestartCommand(node_class=readiness_marker_node_class),
            )
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

        async with self._session_factory() as db, db.begin():
            updated = await GpuSessionDeploymentRepository(db).set_restart_pointer(
                [member.id for member in members], operation_id=command.operation_id
            )
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
        groups: dict[UUID, list[GpuSessionDeployment]] = {}
        for deployment in candidates:
            if deployment.restart_operation_id is None:  # pragma: no cover — query guarantees this
                continue
            groups.setdefault(deployment.restart_operation_id, []).append(deployment)
        for operation_id, members in groups.items():
            try:
                await self._process_one_restart_outcome(operation_id, members)
            except Exception:
                logger.exception(
                    "gpu_session.deployment.restart_outcome_error",
                    operation_id=str(operation_id),
                )

    async def _process_one_restart_outcome(
        self, operation_id: UUID, members: Sequence[GpuSessionDeployment]
    ) -> None:
        async with self._session_factory() as db:
            command = await GpuSessionCommandRepository(db).get_by_operation(operation_id)
        if command is None or command.status not in TERMINAL_COMMAND_STATUSES:
            return

        succeeded = command.status == CommandStatus.succeeded
        now = datetime.now(UTC)

        if succeeded:
            # N1: R3 splits one session's restart into a cohort per readiness
            # marker, and the command queue serializes them one in-flight at a
            # time — so this cohort's restart can go terminal while a sibling
            # cohort's restart for the same session is still queued or claimed.
            # Activating this cohort now would make it briefly routable right
            # before that sibling restart takes ComfyUI down again. Defer: the
            # pending sibling either succeeds, fails, or is expired by the P3
            # sweep, all of which are terminal and clear this gate on a later tick.
            session_id = members[0].session_id
            async with self._session_factory() as db:
                other_restart = await GpuSessionCommandRepository(
                    db
                ).get_non_terminal_by_session_and_kind(session_id, OperationKind.comfyui_restart)
            if other_restart is not None:
                logger.info(
                    "gpu_session.deployment.restart_outcome_deferred_pending_sibling",
                    operation_id=str(operation_id),
                    session_id=str(session_id),
                )
                return

        ids = [m.id for m in members]
        async with self._session_factory() as db, db.begin():
            updated = await GpuSessionDeploymentRepository(db).resolve_restart_outcome(
                ids, succeeded=succeeded, at=now
            )
        # A terminal session cascade can win after the candidate query.  Its
        # guarded UPDATE intentionally returns zero; do not publish a stale
        # active/failed deployment event in that case.
        if updated == 0:
            return
        # The command was just fetched above — the operation it governs is one
        # cheap lookup away, so the SSE event can carry the node's own terminal
        # phase/progress rather than falling back to id-only (N4).
        async with self._session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
        if succeeded:
            logger.info(
                "gpu_session.deployment.activated",
                operation_id=str(operation_id),
                deployment_count=updated,
            )
            for member in members:
                member.status = DeploymentStatus.active
                member.pending_restart = False
                member.activated_at = now
                await publish_deployment_event(self._event_bus, member, operation=operation)
        else:
            logger.warning(
                "gpu_session.deployment.restart_failed",
                operation_id=str(operation_id),
                deployment_count=updated,
                command_status=str(command.status),
                error=command.error,
            )
            for member in members:
                member.status = DeploymentStatus.failed
                member.pending_restart = False
                member.removed_at = now
                await publish_deployment_event(
                    self._event_bus, member, operation=operation, error_message=command.error
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
        async with self._session_factory() as db:
            command = await GpuSessionCommandRepository(db).get_latest_by_deployment_and_kind(
                deployment.id, OperationKind.bundle_removal
            )
        if command is None or command.status not in TERMINAL_COMMAND_STATUSES:
            return

        succeeded = self._removal_succeeded(command)
        now = datetime.now(UTC)
        async with self._session_factory() as db, db.begin():
            applied = await GpuSessionDeploymentRepository(db).resolve_removal_outcome(
                deployment.id, succeeded=succeeded, at=now
            )
        if not applied:
            return
        # The command was just fetched above — the operation it governs is one
        # cheap lookup away, so the SSE event can carry the node's own terminal
        # phase/progress rather than falling back to id-only (N4).
        async with self._session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(command.operation_id)
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

    @staticmethod
    def _removal_succeeded(command: GpuSessionCommand) -> bool:
        if command.status == CommandStatus.succeeded:
            return True
        # CO5: tolerate a re-claimed removal racing an already-completed one.
        return _is_already_removed_error(command.error)
