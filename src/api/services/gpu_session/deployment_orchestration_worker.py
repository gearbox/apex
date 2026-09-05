"""Background worker: drives a deployment through provision -> restart -> active (P4).

OperationEventService is a pure writer (P1); the state machine for what happens
*after* a command reaches a terminal state belongs to a worker (D32). This worker is
the P4 counterpart to GpuSessionCommandSweepWorker: each tick it looks for deployments
whose current command finished and advances them, exactly mirroring the shape (and the
LeaderLease) of every other PeriodicWorker in this codebase.

Four independent steps per tick, in this order so a batch that finishes and drains in
the same tick doesn't wait a full interval for its restart:
  1. Provision finished  -> pending_restart=true (success) or 'failed' (failure).
  2. Batch drained        -> enqueue one comfyui_restart, stamp restart_operation_id.
  3. Restart finished     -> 'active' (success) or 'failed', no retry (D34).
  4. Removal finished     -> 'removed' (success) or back to 'active' (failure).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.api.services.gpu_session._events import publish_deployment_event
from src.api.services.gpu_session.command_payload import RestartCommand
from src.api.services.gpu_session.command_service import CommandEnqueueSessionError
from src.core.enums import TERMINAL_COMMAND_STATUSES, CommandStatus, DeploymentStatus, OperationKind
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
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
        await self._process_provision_results()
        await self._process_restart_enqueue()
        await self._process_restart_results()
        await self._process_removal_results()

    # ------------------------------------------------------------------
    # Step 1 — provision finished
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
        if succeeded:
            deployment.pending_restart = True
            logger.info(
                "gpu_session.deployment.pending_restart",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(self._event_bus, deployment)
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
            await publish_deployment_event(self._event_bus, deployment, error_message=command.error)

    # ------------------------------------------------------------------
    # Step 2 — batch drained -> enqueue restart
    # ------------------------------------------------------------------

    async def _process_restart_enqueue(self) -> None:
        async with self._session_factory() as db:
            candidates = await GpuSessionDeploymentRepository(
                db
            ).list_pending_restart_awaiting_operation()
        groups: dict[str | None, list[GpuSessionDeployment]] = {}
        for deployment in candidates:
            groups.setdefault(deployment.batch_id, []).append(deployment)
        for batch_id, members in groups.items():
            try:
                await self._process_one_restart_batch(batch_id, members)
            except Exception:
                logger.exception("gpu_session.deployment.restart_enqueue_error", batch_id=batch_id)

    async def _process_one_restart_batch(
        self, batch_id: str | None, members: list[GpuSessionDeployment]
    ) -> None:
        if batch_id is None:
            # Every attach stamps batch_id at enqueue time (CO4) — a pending-restart
            # deployment with no batch_id means that write was skipped somewhere.
            logger.error(
                "gpu_session.deployment.pending_restart_missing_batch_id",
                deployment_ids=[str(m.id) for m in members],
            )
            return

        async with self._session_factory() as db:
            batch_commands = await GpuSessionCommandRepository(db).list_by_batch(batch_id)
        if not batch_commands or any(
            c.status not in TERMINAL_COMMAND_STATUSES for c in batch_commands
        ):
            return  # not every member has finished yet

        session_id = members[0].session_id
        product_id = members[0].product_id

        async with self._session_factory() as db:
            in_flight = await JobRepository(db).count_in_flight_for_session(session_id)

        now = datetime.now(UTC)
        batch_complete_at = max(
            (c.terminal_at for c in batch_commands if c.terminal_at is not None), default=None
        )
        timed_out = (
            batch_complete_at is not None
            and (now - batch_complete_at).total_seconds()
            >= self._settings.gpu_deployment_restart_drain_timeout_seconds
        )

        if in_flight > 0:
            if not timed_out:
                return
            logger.warning(
                "gpu_session.deployment.restart_drain_timeout",
                session_id=str(session_id),
                batch_id=batch_id,
                in_flight_count=in_flight,
            )

        newest = max(members, key=lambda m: m.created_at)
        try:
            command = await self._command_service.enqueue(
                session_id=session_id,
                product_id=product_id,
                command=RestartCommand(node_class=newest.readiness_marker_node_class),
            )
        except CommandEnqueueSessionError:
            # Session went terminal — the D15 chokepoint cascade already flipped these
            # deployments away from 'deploying'/pending_restart, so they'll drop out of
            # the next tick's candidate query on their own. Nothing to compensate here.
            logger.info(
                "gpu_session.deployment.restart_enqueue_skipped_session_unavailable",
                session_id=str(session_id),
                batch_id=batch_id,
            )
            return

        async with self._session_factory() as db, db.begin():
            updated = await GpuSessionDeploymentRepository(db).set_restart_pointer(
                [m.id for m in members], operation_id=command.operation_id
            )
        logger.info(
            "gpu_session.deployment.restart_enqueued",
            session_id=str(session_id),
            batch_id=batch_id,
            operation_id=str(command.operation_id),
            deployment_count=updated,
        )
        for member in members:
            member.restart_operation_id = command.operation_id
            await publish_deployment_event(self._event_bus, member)

    # ------------------------------------------------------------------
    # Step 3 — restart finished
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
        ids = [m.id for m in members]
        async with self._session_factory() as db, db.begin():
            updated = await GpuSessionDeploymentRepository(db).resolve_restart_outcome(
                ids, succeeded=succeeded, at=now
            )
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
                await publish_deployment_event(self._event_bus, member)
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
                await publish_deployment_event(self._event_bus, member, error_message=command.error)

    # ------------------------------------------------------------------
    # Step 4 — removal finished
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
        if succeeded:
            deployment.status = DeploymentStatus.removed
            deployment.removed_at = now
            logger.info(
                "gpu_session.deployment.removed",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
            )
            await publish_deployment_event(self._event_bus, deployment)
        else:
            deployment.status = DeploymentStatus.active
            logger.warning(
                "gpu_session.deployment.removal_failed",
                session_id=str(deployment.session_id),
                deployment_id=str(deployment.id),
                command_status=str(command.status),
                error=command.error,
            )
            await publish_deployment_event(self._event_bus, deployment, error_message=command.error)

    @staticmethod
    def _removal_succeeded(command: GpuSessionCommand) -> bool:
        if command.status == CommandStatus.succeeded:
            return True
        # CO5: tolerate a re-claimed removal racing an already-completed one.
        return _is_already_removed_error(command.error)
