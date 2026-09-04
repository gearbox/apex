"""Single source of truth for the Apex -> aisha-agent command wire format (D28).

Mirrors gearbox/aisha's src/ai_content_service/agent_contract.py::parse_command at
SHA a7a65d864231c31709124dfe3c6c74a99ea5a0a9 (read-only contract, not vendored here).
Every rule that module enforces via ``raise context.error(...)`` has a corresponding
check below — Apex must never provoke one of those 400s by constructing a bad
envelope. Used by GpuSessionCommandService for both directions: ``build_payload`` at
enqueue time (validate + persist) and ``build_envelope`` at claim time (re-validate +
serialize straight from the stored row, so there is zero drift between what was
written and what is served).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from src.core.enums import OperationKind

# The exact literal set gearbox/aisha's DeployMode accepts. Not modeled as a shared
# enum with aisha (separate repos) — these three strings are the wire contract.
_VALID_MODES: frozenset[str] = frozenset({"full", "models_only", "additive"})

# What Apex may ever emit. session_bootstrap is deliberately absent — it is not a
# command Apex enqueues (P1 owns that operation kind entirely).
_ALLOWED_KINDS: frozenset[OperationKind] = frozenset(
    {OperationKind.bundle_provision, OperationKind.bundle_removal, OperationKind.comfyui_restart}
)


class CommandBuildError(ValueError):
    """A typed command input would violate the aisha agent_contract on the wire."""


@dataclass(frozen=True, slots=True)
class BatchPosition:
    """One member's position within a batch — mirrors aisha's BatchRef shape."""

    batch_id: str
    index: int
    total: int


@dataclass(frozen=True, slots=True)
class ProvisionCommand:
    """Typed input for one bundle_provision command."""

    bundle: str
    """'{name}' or '{name}:{version}' — passed through opaquely; Apex doesn't parse it."""
    mode: str
    """One of 'full', 'models_only', 'additive'."""
    verify: bool = True
    declared_model_bytes: int | None = None
    """F8: summed models[].files[].size_bytes for this bundle. None omits the wire
    key entirely; it is only emitted on batch index 0. Every provision is wrapped in
    a batch, including a batch of one, so the arc-level batch-headroom guard runs for
    every provision."""

    kind: ClassVar[OperationKind] = OperationKind.bundle_provision


@dataclass(frozen=True, slots=True)
class RemovalCommand:
    """Typed input for one bundle_removal command."""

    bundle: str
    retain_bundles: tuple[str, ...] = ()

    kind: ClassVar[OperationKind] = OperationKind.bundle_removal


@dataclass(frozen=True, slots=True)
class RestartCommand:
    """Typed input for one comfyui_restart command."""

    node_class: str | None = None

    kind: ClassVar[OperationKind] = OperationKind.comfyui_restart


CommandInput = ProvisionCommand | RemovalCommand | RestartCommand


def build_payload(command: CommandInput, *, batch: BatchPosition | None) -> dict[str, Any]:
    """Build and validate the wire-format 'payload' object for one command.

    Raises CommandBuildError for anything parse_command would reject.
    """
    if isinstance(command, ProvisionCommand):
        return _build_provision_payload(command, batch)
    if isinstance(command, RemovalCommand):
        return _build_removal_payload(command)
    if isinstance(command, RestartCommand):
        return _build_restart_payload(command)
    raise CommandBuildError(
        f"unsupported command input type {type(command).__name__}"
    )  # pragma: no cover


def build_envelope(
    *,
    command_id: str,
    operation_id: str,
    kind: OperationKind,
    batch: BatchPosition | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the full claim-response envelope, re-validating kind/batch/payload.

    Called both right after build_payload (enqueue time) and again at claim time
    from stored row fields — the re-validation is defense in depth against DB drift,
    not a second opinion on already-trusted input.
    """
    if kind not in _ALLOWED_KINDS:
        raise CommandBuildError(f"{kind.value!r} is not a command Apex may enqueue")
    if not command_id:
        raise CommandBuildError("command_id must be a non-empty string")
    if not operation_id:
        raise CommandBuildError("operation_id must be a non-empty string")
    if "force" in payload:
        raise CommandBuildError("payload must not contain a 'force' key")

    envelope: dict[str, Any] = {
        "command_id": command_id,
        "operation_id": operation_id,
        "kind": kind.value,
        "batch": None,
        "payload": payload,
    }
    if batch is not None:
        if batch.total <= 0 or batch.index < 0 or batch.index >= batch.total:
            raise CommandBuildError("batch index must satisfy 0 <= index < total")
        envelope["batch"] = {"batch_id": batch.batch_id, "index": batch.index, "total": batch.total}
    return envelope


def _build_provision_payload(
    command: ProvisionCommand, batch: BatchPosition | None
) -> dict[str, Any]:
    if not command.bundle:
        raise CommandBuildError("bundle must be non-empty")
    if command.mode not in _VALID_MODES:
        raise CommandBuildError(f"unknown deployment mode {command.mode!r}")
    payload: dict[str, Any] = {
        "bundle": command.bundle,
        "mode": command.mode,
        "verify": command.verify,
    }
    if command.declared_model_bytes is not None:
        if command.declared_model_bytes < 0:
            raise CommandBuildError("declared_model_bytes must be a non-negative integer")
        if batch is None or batch.index != 0:
            raise CommandBuildError("declared_model_bytes is only permitted on batch index 0")
        payload["batch_declared_bytes"] = command.declared_model_bytes
    return payload


def _build_removal_payload(command: RemovalCommand) -> dict[str, Any]:
    if not command.bundle:
        raise CommandBuildError("bundle must be non-empty")
    if not all(command.retain_bundles):
        raise CommandBuildError("retain_bundles must be a list of non-empty strings")
    return {"bundle": command.bundle, "retain_bundles": list(command.retain_bundles)}


def _build_restart_payload(command: RestartCommand) -> dict[str, Any]:
    if command.node_class is None or command.node_class:
        return {"node_class": command.node_class}
    raise CommandBuildError("node_class must be a non-empty string or null")
