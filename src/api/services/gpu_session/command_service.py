"""Enqueue and claim GPU session commands (P3).

Enqueueing is an internal service method only in this phase (D24) — no route calls
it yet, only tests. The claim path is what the internal controller calls; it is the
only externally reachable surface this module exposes.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from src.api.services.gpu_session.command_payload import (
    BatchPosition,
    CommandBuildError,
    CommandInput,
    ProvisionCommand,
    build_envelope,
    build_payload,
)
from src.core.enums import TERMINAL_GPU_SESSION_STATUSES, OperationKind
from src.core.uid import new_id
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.core.config import Settings
    from src.db.models.gpu_session_command import GpuSessionCommand

logger = structlog.get_logger(__name__)


def _validate_token(presented: str, stored_hash: str | None) -> bool:
    """Constant-time comparison of presented bearer token against stored SHA-256 hash.

    Deliberately duplicated from operation_event_service.py rather than imported —
    D27 scopes P3's only change to that file to the terminal-close addition.
    """
    if not stored_hash:
        return False
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    return hmac.compare_digest(presented_hash, stored_hash)


def _validate(command: CommandInput, *, batch: BatchPosition | None) -> dict[str, Any]:
    """Reject, before any DB write, anything parse_command would reject (D24)."""
    payload = build_payload(command, batch=batch)
    build_envelope(
        command_id="placeholder",
        operation_id="placeholder",
        kind=command.kind,
        batch=batch,
        payload=payload,
    )
    return payload


def _provision_target(command: CommandInput) -> tuple[str | None, str | None, str | None]:
    """(target_bundle, target_bundle_version, target_mode) for a provision command.

    None/None/None for removal and restart — those columns' established meaning
    (apply_event's version-preservation coalesce) is specifically about the
    provisioning target, not overloaded for other kinds.
    """
    if not isinstance(command, ProvisionCommand):
        return None, None, None
    if ":" in command.bundle:
        name, version = command.bundle.split(":", 1)
        return name, version, command.mode
    return command.bundle, None, command.mode


class GpuSessionCommandService:
    """Enqueue commands atomically with their operation; claim them for a node agent."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    async def enqueue(
        self,
        *,
        session_id: UUID,
        product_id: str,
        command: CommandInput,
        deployment_id: UUID | None = None,
    ) -> GpuSessionCommand:
        """Enqueue one command outside of a batch."""
        payload = _validate(command, batch=None)
        async with self._session_factory() as db, db.begin():
            return await self._create_one(
                db,
                session_id=session_id,
                product_id=product_id,
                deployment_id=deployment_id,
                command=command,
                payload=payload,
                batch_id=None,
                batch_index=None,
                batch_total=None,
            )

    async def enqueue_batch(
        self,
        *,
        session_id: UUID,
        product_id: str,
        commands: Sequence[CommandInput],
        deployment_id: UUID | None = None,
        declared_model_bytes: int | None = None,
    ) -> list[GpuSessionCommand]:
        """Enqueue a batch atomically (D25). F8's declared_model_bytes lands only on
        index 0 — pass it here, never set it on an individual member."""
        if not commands:
            raise CommandBuildError("enqueue_batch requires at least one command")
        if declared_model_bytes is not None and not isinstance(commands[0], ProvisionCommand):
            raise CommandBuildError(
                "declared_model_bytes requires a ProvisionCommand at batch index 0"
            )

        batch_id = str(new_id())
        total = len(commands)
        validated: list[tuple[CommandInput, dict[str, Any]]] = []
        for index, member in enumerate(commands):
            if isinstance(member, ProvisionCommand) and member.declared_model_bytes is not None:
                raise CommandBuildError(
                    "set declared_model_bytes via enqueue_batch's own parameter, not per-member"
                )
            resolved: CommandInput = member
            if index == 0 and declared_model_bytes is not None:
                # Guaranteed a ProvisionCommand by the guard above this loop.
                provision_member = cast("ProvisionCommand", member)
                resolved = ProvisionCommand(
                    bundle=provision_member.bundle,
                    mode=provision_member.mode,
                    verify=provision_member.verify,
                    declared_model_bytes=declared_model_bytes,
                )
            batch = BatchPosition(batch_id=batch_id, index=index, total=total)
            validated.append((resolved, _validate(resolved, batch=batch)))

        async with self._session_factory() as db, db.begin():
            created: list[GpuSessionCommand] = []
            for index, (command, payload) in enumerate(validated):
                created.append(
                    await self._create_one(
                        db,
                        session_id=session_id,
                        product_id=product_id,
                        deployment_id=deployment_id,
                        command=command,
                        payload=payload,
                        batch_id=batch_id,
                        batch_index=index,
                        batch_total=total,
                    )
                )
            return created

    async def claim(
        self, *, session_id: UUID, bearer_token: str, agent_id: str
    ) -> tuple[int, dict[str, Any] | None]:
        """Validate node auth + session state, then attempt one D22/D23 claim.

        Returns (status, envelope_or_None) mapped directly onto the HTTP response.
        Never raises for a normal-path outcome (D29 — the endpoint must never 5xx).
        """
        async with self._session_factory() as db, db.begin():
            session_repo = GpuSessionRepository(db)
            session = await session_repo.get_by_id(session_id)
            if session is None:
                return 401, None
            if not _validate_token(bearer_token, session.callback_token_hash):
                return 401, None
            if session.status in TERMINAL_GPU_SESSION_STATUSES:
                # The node is going away; an ERROR log + 60s backoff is the wrong
                # response to normal shutdown (D29) — not 404, not 409.
                return 204, None

            now = datetime.now(UTC)
            claimed = await GpuSessionCommandRepository(db).claim(
                session_id, agent_id, now=now, deadline_for=self._deadline_for
            )
            if claimed is None:
                return 204, None

            batch: BatchPosition | None = None
            if claimed.batch_id is not None:
                # batch_id/batch_index/batch_total are always written together by
                # _create_one — never independently None once batch_id is set.
                batch = BatchPosition(
                    batch_id=claimed.batch_id,
                    index=cast("int", claimed.batch_index),
                    total=cast("int", claimed.batch_total),
                )
            envelope = build_envelope(
                command_id=str(claimed.id),
                operation_id=str(claimed.operation_id),
                kind=OperationKind(claimed.kind),
                batch=batch,
                payload=claimed.payload,
            )
            claimed_id, operation_id, kind = claimed.id, claimed.operation_id, claimed.kind

        logger.info(
            "gpu_session.command.claimed",
            session_id=str(session_id),
            command_id=str(claimed_id),
            operation_id=str(operation_id),
            agent_id=agent_id,
            kind=kind,
        )
        return 200, envelope

    async def _create_one(
        self,
        db: AsyncSession,
        *,
        session_id: UUID,
        product_id: str,
        deployment_id: UUID | None,
        command: CommandInput,
        payload: dict[str, Any],
        batch_id: str | None,
        batch_index: int | None,
        batch_total: int | None,
    ) -> GpuSessionCommand:
        operation_id = new_id()
        command_id = new_id()
        target_bundle, target_bundle_version, target_mode = _provision_target(command)
        await GpuSessionOperationRepository(db).create(
            id=operation_id,
            session_id=session_id,
            product_id=product_id,
            kind=command.kind,
            target_bundle=target_bundle,
            target_bundle_version=target_bundle_version,
            target_mode=target_mode,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
            command_id=command_id,
        )
        return await GpuSessionCommandRepository(db).create(
            id=command_id,
            session_id=session_id,
            product_id=product_id,
            operation_id=operation_id,
            deployment_id=deployment_id,
            kind=command.kind,
            payload=payload,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
        )

    def _deadline_for(self, kind: str) -> int:
        mapping = {
            OperationKind.bundle_provision.value: self._settings.gpu_command_provision_timeout_seconds,
            OperationKind.bundle_removal.value: self._settings.gpu_command_removal_timeout_seconds,
            OperationKind.comfyui_restart.value: self._settings.gpu_command_restart_timeout_seconds,
        }
        return mapping[kind]
