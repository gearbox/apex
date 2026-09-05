"""Contract tests for command_payload.py against every parse_command rule (D28).

Mirrors gearbox/aisha's agent_contract.py::parse_command at SHA
a7a65d864231c31709124dfe3c6c74a99ea5a0a9 — each test here targets one rule that
module enforces via ``raise context.error(...)``. This is the closest thing to
importing parse_command directly, and what catches drift when aisha tightens it.
"""

from __future__ import annotations

import pytest

from src.api.services.gpu_session.command_payload import (
    BatchPosition,
    CommandBuildError,
    ProvisionCommand,
    RemovalCommand,
    RestartCommand,
    build_envelope,
    build_payload,
)
from src.core.enums import OperationKind


class TestSessionBootstrapRejected:
    """Invariant #1: session_bootstrap is not a command Apex may enqueue."""

    def test_build_envelope_rejects_session_bootstrap(self) -> None:
        with pytest.raises(CommandBuildError, match="session_bootstrap"):
            build_envelope(
                command_id="c1",
                operation_id="o1",
                kind=OperationKind.session_bootstrap,
                batch=None,
                payload={},
            )

    def test_provision_command_kind_is_never_session_bootstrap(self) -> None:
        assert ProvisionCommand(bundle="b", mode="full").kind == OperationKind.bundle_provision
        assert RemovalCommand(bundle="b").kind == OperationKind.bundle_removal
        assert RestartCommand().kind == OperationKind.comfyui_restart


class TestNoForceKey:
    """Invariant #2: no builder output ever contains a 'force' key, for any kind."""

    def test_provision_payload_has_no_force_key(self) -> None:
        payload = build_payload(ProvisionCommand(bundle="wan_2.2_i2v", mode="full"), batch=None)
        assert "force" not in payload

    def test_removal_payload_has_no_force_key(self) -> None:
        payload = build_payload(RemovalCommand(bundle="wan_2.2_i2v"), batch=None)
        assert "force" not in payload

    def test_restart_payload_has_no_force_key(self) -> None:
        payload = build_payload(RestartCommand(node_class="WanImageToVideo"), batch=None)
        assert "force" not in payload

    def test_build_envelope_rejects_a_force_key_if_present(self) -> None:
        with pytest.raises(CommandBuildError, match="force"):
            build_envelope(
                command_id="c1",
                operation_id="o1",
                kind=OperationKind.bundle_provision,
                batch=None,
                payload={"bundle": "b", "mode": "full", "verify": True, "force": True},
            )


class TestBatchDeclaredBytes:
    """Invariant #3: batch_declared_bytes only on index 0; non-batch is rejected."""

    def test_declared_bytes_present_on_index_0(self) -> None:
        batch = BatchPosition(batch_id="b1", index=0, total=3)
        payload = build_payload(
            ProvisionCommand(bundle="b", mode="full", declared_model_bytes=1024), batch=batch
        )
        assert payload["batch_declared_bytes"] == 1024

    def test_declared_bytes_absent_on_non_zero_index(self) -> None:
        payload = build_payload(
            ProvisionCommand(bundle="b", mode="full"), batch=BatchPosition("b1", 1, 3)
        )
        assert "batch_declared_bytes" not in payload

    def test_declared_bytes_on_non_zero_index_is_rejected(self) -> None:
        batch = BatchPosition(batch_id="b1", index=1, total=3)
        with pytest.raises(CommandBuildError, match="batch index 0"):
            build_payload(
                ProvisionCommand(bundle="b", mode="full", declared_model_bytes=1024), batch=batch
            )

    def test_declared_bytes_on_non_batch_provision_is_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="batch index 0"):
            build_payload(
                ProvisionCommand(bundle="b", mode="full", declared_model_bytes=1024), batch=None
            )

    def test_negative_declared_bytes_is_rejected(self) -> None:
        batch = BatchPosition(batch_id="b1", index=0, total=1)
        with pytest.raises(CommandBuildError, match="non-negative"):
            build_payload(
                ProvisionCommand(bundle="b", mode="full", declared_model_bytes=-1), batch=batch
            )

    def test_no_fabricated_size_none_omits_the_key_entirely(self) -> None:
        """Invariant #18: None omits the key — never 0, never null."""
        batch = BatchPosition(batch_id="b1", index=0, total=1)
        payload = build_payload(
            ProvisionCommand(bundle="b", mode="full", declared_model_bytes=None), batch=batch
        )
        assert "batch_declared_bytes" not in payload


class TestBatchBounds:
    """Invariant #4: every emitted batch satisfies 0 <= index < total and total > 0."""

    @pytest.mark.parametrize(
        "index,total",
        [(-1, 3), (3, 3), (0, 0), (5, 3)],
    )
    def test_out_of_bounds_batch_is_rejected(self, index: int, total: int) -> None:
        with pytest.raises(CommandBuildError, match="batch index"):
            build_envelope(
                command_id="c1",
                operation_id="o1",
                kind=OperationKind.bundle_provision,
                batch=BatchPosition(batch_id="b1", index=index, total=total),
                payload={"bundle": "b", "mode": "full", "verify": True},
            )

    def test_valid_batch_bounds_are_accepted(self) -> None:
        envelope = build_envelope(
            command_id="c1",
            operation_id="o1",
            kind=OperationKind.bundle_provision,
            batch=BatchPosition(batch_id="b1", index=0, total=3),
            payload={"bundle": "b", "mode": "full", "verify": True},
        )
        assert envelope["batch"] == {"batch_id": "b1", "index": 0, "total": 3}

    def test_no_batch_serializes_as_null(self) -> None:
        envelope = build_envelope(
            command_id="c1",
            operation_id="o1",
            kind=OperationKind.comfyui_restart,
            batch=None,
            payload={"node_class": None},
        )
        assert envelope["batch"] is None


class TestProvisionRules:
    def test_empty_bundle_is_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="bundle"):
            build_payload(ProvisionCommand(bundle="", mode="full"), batch=None)

    @pytest.mark.parametrize("mode", ["full", "models_only", "additive"])
    def test_valid_modes_accepted(self, mode: str) -> None:
        payload = build_payload(ProvisionCommand(bundle="b", mode=mode), batch=None)
        assert payload["mode"] == mode

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="mode"):
            build_payload(ProvisionCommand(bundle="b", mode="dry_run"), batch=None)

    def test_verify_defaults_true_and_is_a_bool(self) -> None:
        payload = build_payload(ProvisionCommand(bundle="b", mode="full"), batch=None)
        assert payload["verify"] is True

    def test_bundle_with_version_passes_through_opaquely(self) -> None:
        payload = build_payload(
            ProvisionCommand(bundle="wan_2.2_i2v:260105-01", mode="full"), batch=None
        )
        assert payload["bundle"] == "wan_2.2_i2v:260105-01"


class TestRemovalRules:
    def test_empty_bundle_is_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="bundle"):
            build_payload(RemovalCommand(bundle=""), batch=None)

    def test_retain_bundles_must_be_non_empty_strings(self) -> None:
        with pytest.raises(CommandBuildError, match="retain_bundles"):
            build_payload(RemovalCommand(bundle="b", retain_bundles=("ok", "")), batch=None)

    def test_valid_retain_bundles_pass_through(self) -> None:
        payload = build_payload(RemovalCommand(bundle="b", retain_bundles=("a", "b")), batch=None)
        assert payload["retain_bundles"] == ["a", "b"]

    def test_default_retain_bundles_is_empty_list(self) -> None:
        payload = build_payload(RemovalCommand(bundle="b"), batch=None)
        assert payload["retain_bundles"] == []


class TestRestartRules:
    def test_node_class_none_is_valid(self) -> None:
        payload = build_payload(RestartCommand(node_class=None), batch=None)
        assert payload["node_class"] is None

    def test_node_class_non_empty_string_is_valid(self) -> None:
        payload = build_payload(RestartCommand(node_class="WanImageToVideo"), batch=None)
        assert payload["node_class"] == "WanImageToVideo"

    def test_empty_node_class_is_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="node_class"):
            build_payload(RestartCommand(node_class=""), batch=None)


class TestEnvelopeShape:
    def test_full_envelope_matches_wire_contract_shape(self) -> None:
        envelope = build_envelope(
            command_id="0199-cmd",
            operation_id="0199-op",
            kind=OperationKind.bundle_provision,
            batch=BatchPosition(batch_id="0199-batch", index=0, total=3),
            payload={
                "bundle": "wan_2.2_i2v:260105-01",
                "mode": "additive",
                "verify": True,
                "batch_declared_bytes": 9663676416,
            },
        )
        assert envelope == {
            "command_id": "0199-cmd",
            "operation_id": "0199-op",
            "kind": "bundle_provision",
            "batch": {"batch_id": "0199-batch", "index": 0, "total": 3},
            "payload": {
                "bundle": "wan_2.2_i2v:260105-01",
                "mode": "additive",
                "verify": True,
                "batch_declared_bytes": 9663676416,
            },
        }

    def test_empty_command_id_or_operation_id_rejected(self) -> None:
        with pytest.raises(CommandBuildError, match="command_id"):
            build_envelope(
                command_id="",
                operation_id="o1",
                kind=OperationKind.comfyui_restart,
                batch=None,
                payload={},
            )
        with pytest.raises(CommandBuildError, match="operation_id"):
            build_envelope(
                command_id="c1",
                operation_id="",
                kind=OperationKind.comfyui_restart,
                batch=None,
                payload={},
            )
