"""P4: attach/remove additional model deployments on a running GPU session.

Sessions and model deployments stay distinct domain concerns (D12 Option B) — this
service is the counterpart to GpuSessionService for the /deployments sub-resource. It
never touches GpuSession.status; DeploymentOrchestrationWorker owns every deployment
status transition that happens after a command is enqueued (see that module).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import structlog
from sqlalchemy.exc import IntegrityError

from src.api.services.gpu_session.command_payload import ProvisionCommand, RemovalCommand
from src.api.services.gpu_session.command_service import CommandEnqueueSessionError
from src.core.enums import DeploymentStatus, GpuSessionStatus
from src.core.uid import new_id
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.job import JobRepository

from ._events import publish_deployment_event
from .exceptions import (
    DeploymentAlreadyLiveError,
    DeploymentHasInFlightJobsError,
    DeploymentNotLiveError,
    GpuSessionError,
    InvalidSessionStateError,
    LastDeploymentRequiresForceError,
    RetainBundlesUnresolvableError,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.bundle_index import BundleIndexService
    from src.api.services.event_bus import EventBus
    from src.api.services.gpu_session.command_service import GpuSessionCommandService
    from src.core.enums import ModelType
    from src.db.models.gpu_session_deployment import GpuSessionDeployment

logger = structlog.get_logger(__name__)


class GpuSessionDeploymentService:
    """Attach a model to a running session, or remove one that's already there."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        bundle_index: BundleIndexService,
        command_service: GpuSessionCommandService,
        event_bus: EventBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._bundles = bundle_index
        self._command_service = command_service
        self._event_bus = event_bus

    async def attach(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
    ) -> tuple[GpuSessionDeployment, UUID]:
        """Attach a new model to a running session (P1).

        Returns (deployment, provision_operation_id). Raises GpuSessionError (session
        not found/owned), InvalidSessionStateError (session not active), BundleNotFoundError
        (model not indexed), or DeploymentAlreadyLiveError (D36).
        """
        async with self._session_factory() as db:
            session_row = await GpuSessionRepository(db).get_by_id_for_user(
                session_id, user_id, product_id
            )
        if session_row is None:
            raise GpuSessionError(f"Session {session_id} not found or access denied")
        if session_row.status != GpuSessionStatus.active:
            raise InvalidSessionStateError(
                f"Cannot attach a deployment to session in state '{session_row.status}'",
                current_status=session_row.status,
                operation="attach",
            )

        # Raises BundleNotFoundError if the model isn't indexed — propagates uncaught,
        # the route maps it to a 404.
        bundle = self._bundles.resolve_bundle(model_type.value)

        async with self._session_factory() as db:
            existing = await GpuSessionDeploymentRepository(db).get_live_for_model(
                user_id, product_id, model_type.value
            )
        if existing is not None:
            raise DeploymentAlreadyLiveError(
                f"User {user_id} already has a live deployment for model_type="
                f"{model_type.value} in product={product_id}"
            )

        deployment_id = new_id()
        readiness_marker_node_class = (
            bundle.readiness_marker.node_class if bundle.readiness_marker is not None else None
        )
        async with self._session_factory() as db, db.begin():
            try:
                deployment = await GpuSessionDeploymentRepository(db).create(
                    id=deployment_id,
                    session_id=session_id,
                    user_id=user_id,
                    product_id=product_id,
                    model_type=model_type.value,
                    bundle_name=bundle.bundle_name,
                    bundle_version=bundle.bundle_version,
                    readiness_marker_node_class=readiness_marker_node_class,
                    status=DeploymentStatus.deploying,
                    pending_restart=False,
                    is_primary=False,
                )
            except IntegrityError:
                # D36 race: a concurrent attach for the same model slipped past the
                # pre-check above. No command is ever enqueued on this path.
                raise DeploymentAlreadyLiveError(
                    f"Concurrent attach conflict for user={user_id}, model_type={model_type.value}"
                ) from None

        bundle_spec = (
            f"{bundle.bundle_name}:{bundle.bundle_version}"
            if bundle.bundle_version
            else bundle.bundle_name
        )
        try:
            # D8: every provision is a batch, even a batch of one (invariant #1).
            commands = await self._command_service.enqueue_batch(
                session_id=session_id,
                product_id=product_id,
                commands=[ProvisionCommand(bundle=bundle_spec, mode="additive")],
                deployment_id=deployment_id,
                declared_model_bytes=bundle.declared_model_bytes,
            )
        except Exception as exc:
            # The row is committed before enqueue_batch opens its own transaction. Any
            # enqueue failure, not only a session-state race, must therefore release
            # the live-deployment slot. If this compensating write also fails, the
            # worker's orphan reaper will repair the row after its short grace period.
            try:
                async with self._session_factory() as db, db.begin():
                    await GpuSessionDeploymentRepository(db).resolve_provision_outcome(
                        deployment_id, succeeded=False, at=datetime.now(UTC)
                    )
            except Exception:
                logger.exception(
                    "gpu_session.deployment.attach_enqueue_compensation_failed",
                    session_id=str(session_id),
                    deployment_id=str(deployment_id),
                )

            if isinstance(exc, CommandEnqueueSessionError):
                logger.warning(
                    "gpu_session.deployment.attach_race_session_unavailable",
                    session_id=str(session_id),
                    deployment_id=str(deployment_id),
                )
                raise InvalidSessionStateError(
                    f"Session {session_id} became unavailable while attaching: {exc}",
                    current_status="unknown",
                    operation="attach",
                ) from exc
            raise

        command = commands[0]
        # enqueue_batch always stamps a fresh batch_id (D8: every provision is a
        # batch, even a batch of one) — cast rather than assert: this is a static-
        # typing note about enqueue_batch's own contract, not a runtime check we
        # want stripped under -O.
        batch_id = cast("str", command.batch_id)
        async with self._session_factory() as db, db.begin():
            await GpuSessionDeploymentRepository(db).set_provision_pointer(
                deployment_id, operation_id=command.operation_id, batch_id=batch_id
            )
        deployment.provision_operation_id = command.operation_id
        deployment.batch_id = batch_id

        logger.info(
            "gpu_session.deployment.attached",
            session_id=str(session_id),
            deployment_id=str(deployment_id),
            model_type=model_type.value,
            operation_id=str(command.operation_id),
        )
        await publish_deployment_event(self._event_bus, deployment)
        return deployment, command.operation_id

    async def remove(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
        force: bool,
    ) -> GpuSessionDeployment:
        """Remove a model from a running session (P2). Does not restart ComfyUI (D37)."""
        async with self._session_factory() as db:
            session_row = await GpuSessionRepository(db).get_by_id_for_user(
                session_id, user_id, product_id
            )
        if session_row is None:
            raise GpuSessionError(f"Session {session_id} not found or access denied")

        # P4 removal is one serialized state transition.  In particular, do not
        # split active -> removing from the in-flight count or command enqueue:
        # after the transition no new request can resolve this deployment for
        # routing, and an exception rolls the entire transition back to active.
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionDeploymentRepository(db)
            await repo.acquire_removal_lock(session_id)

            deployment = await repo.get_live_for_session_and_model(session_id, model_type.value)
            if deployment is None:
                raise DeploymentNotLiveError(
                    f"No live deployment for model_type={model_type.value} on session {session_id}"
                )

            marked = await repo.mark_removing(deployment.id, require_another_active=not force)
            if not marked:
                if not force and deployment.status == DeploymentStatus.active:
                    # With the per-session removal lock held, an active target
                    # which missed the correlated UPDATE is necessarily the last
                    # active deployment.  This read is error classification only;
                    # the guard itself lives in mark_removing's UPDATE.
                    raise LastDeploymentRequiresForceError(
                        f"Removing the last deployment on session {session_id} requires "
                        "?force=true; the session will keep running and billing with "
                        "nothing deployed on it"
                    )
                raise DeploymentNotLiveError(
                    f"Deployment {deployment.id} is no longer active (concurrent state change)"
                )

            # Transition before counting: once the transaction commits this row
            # is no longer routable, so the count closes the arrival window for
            # new same-model work.  A non-zero count raises and rolls the status
            # transition back to active with the surrounding transaction.
            job_repo = JobRepository(db)
            in_flight = await job_repo.count_in_flight_for_session_and_model(
                session_id, model_type.value
            )
            if in_flight > 0:
                raise DeploymentHasInFlightJobsError(
                    deployment_id=deployment.id, in_flight_count=in_flight
                )

            # Build retention from the post-transition live set.  The target is
            # still live as 'removing' until its command succeeds, so exclude it
            # explicitly; every other live bundle must be retained by aisha.
            live = await repo.list_live_for_session(session_id)
            retain_bundles: list[str] = []
            for other in live:
                if other.id == deployment.id:
                    continue
                # Fail loud (D11) rather than ship a short retain list — bundle_name is
                # NOT NULL on every deployment row, so an empty value here means a data
                # invariant was violated elsewhere, not a normal user-facing condition.
                if not other.bundle_name:
                    raise RetainBundlesUnresolvableError(
                        f"Deployment {other.id} on session {session_id} has no resolvable "
                        "bundle name; refusing to remove without a complete retain list"
                    )
                # aisha's residency store is keyed by bare bundle name, never name:version
                # (BundleRemover.remove looks up resident[bundle_name]) — never append a
                # version suffix here.
                retain_bundles.append(other.bundle_name)

            # Supplying the active transaction makes the queue write atomic with
            # the state transition.  Any enqueue exception rolls both back, which
            # is the active-state compensation required for a retry to remain safe.
            try:
                command = await self._command_service.enqueue(
                    session_id=session_id,
                    product_id=product_id,
                    command=RemovalCommand(
                        bundle=deployment.bundle_name, retain_bundles=tuple(retain_bundles)
                    ),
                    deployment_id=deployment.id,
                    db=db,
                )
            except CommandEnqueueSessionError as exc:
                raise InvalidSessionStateError(
                    f"Session {session_id} became unavailable while removing: {exc}",
                    current_status="unknown",
                    operation="remove",
                ) from exc

        deployment.status = DeploymentStatus.removing
        logger.info(
            "gpu_session.deployment.remove_enqueued",
            session_id=str(session_id),
            deployment_id=str(deployment.id),
            model_type=model_type.value,
            operation_id=str(command.operation_id),
            retain_bundles=retain_bundles,
        )
        # S1: name the removal's own operation explicitly — the fallback in
        # publish_deployment_event would otherwise report a stale
        # restart_operation_id from a restart that finished long ago, since
        # resolve_restart_outcome never clears that pointer on activation.
        await publish_deployment_event(
            self._event_bus, deployment, operation_id=command.operation_id
        )
        return deployment
