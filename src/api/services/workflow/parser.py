"""Fail-loud parsing for the bundle workflow contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from src.api.services.workflow.contract import (
    MEDIA_SLOT_KINDS,
    REQUIRED_ROLES,
    ROLE_PARAMETERS,
    SUPPORTED_CONTRACT_VERSION,
    VIDEO_ONLY_PARAMETERS,
    MediaSlot,
    WorkflowMap,
    WorkflowMediaInput,
    WorkflowModelInput,
    WorkflowNode,
    WorkflowRole,
)
from src.core.bundle_config import BundleDefinitionError
from src.core.enums import MediaKind

if TYPE_CHECKING:
    from pathlib import Path


class WorkflowContractError(BundleDefinitionError):
    """A bundle's workflow declaration is invalid."""


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else None


def _node_id(value: object, *, field: str, errors: list[str]) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        errors.append(f"{field} must be an integer or non-empty string")
        return None
    normalized = str(value)
    if not normalized:
        errors.append(f"{field} must not be empty")
        return None
    return normalized


def _non_blank(value: object, *, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty string")
        return None
    return value


def _enum[E: str](enum_cls: type[E], value: object, *, field: str, errors: list[str]) -> E | None:
    try:
        return enum_cls(str(value))
    except ValueError:
        errors.append(f"{field} has unsupported value {value!r}")
        return None


def parse_workflow_map(data: Mapping[str, object], source: Path) -> WorkflowMap:
    """Parse a v2 ``workflow:`` mapping and report all semantic failures once."""
    errors: list[str] = []
    version = data.get("contract_version")
    if isinstance(version, bool) or not isinstance(version, int):
        errors.append("workflow.contract_version is required and must be an integer")
        contract_version = -1
    else:
        contract_version = version
        if contract_version != SUPPORTED_CONTRACT_VERSION:
            errors.append(
                "workflow.contract_version must be "
                f"{SUPPORTED_CONTRACT_VERSION}, got {contract_version!r}"
            )

    raw_media = data.get("media")
    if raw_media is None:
        errors.append("workflow.media is required")
        media = None
    else:
        media = _enum(MediaKind, raw_media, field="workflow.media", errors=errors)

    raw_nodes = _as_mapping(data.get("nodes"))
    if raw_nodes is None:
        errors.append("workflow.nodes is required and must be a mapping")
        raw_nodes = {}

    nodes: dict[WorkflowRole, WorkflowNode] = {}
    id_owners: dict[str, list[str]] = {}
    for raw_role, raw_node in raw_nodes.items():
        role = _enum(WorkflowRole, raw_role, field=f"workflow.nodes.{raw_role}", errors=errors)
        if role is None:
            continue
        node = _as_mapping(raw_node)
        if node is None:
            errors.append(f"workflow.nodes.{role.value} must be a mapping")
            continue
        node_id = _node_id(node.get("id"), field=f"workflow.nodes.{role.value}.id", errors=errors)
        class_name = _non_blank(
            node.get("class"), field=f"workflow.nodes.{role.value}.class", errors=errors
        )
        raw_inputs = _as_mapping(node.get("inputs", {}))
        if raw_inputs is None:
            errors.append(f"workflow.nodes.{role.value}.inputs must be a mapping")
            raw_inputs = {}
        inputs: dict[str, str] = {}
        for parameter, input_name in raw_inputs.items():
            if parameter not in ROLE_PARAMETERS[role]:
                errors.append(
                    f"workflow.nodes.{role.value}.inputs has unsupported parameter {parameter!r}"
                )
                continue
            parsed_input = _non_blank(
                input_name,
                field=f"workflow.nodes.{role.value}.inputs.{parameter}",
                errors=errors,
            )
            if parsed_input is not None:
                inputs[parameter] = parsed_input
        if media is MediaKind.IMAGE:
            for parameter in sorted(set(inputs) & VIDEO_ONLY_PARAMETERS):
                errors.append(
                    f"workflow.nodes.{role.value}.inputs parameter {parameter!r} is video-only "
                    "for media image"
                )
        if (
            role in {WorkflowRole.POSITIVE_PROMPT, WorkflowRole.NEGATIVE_PROMPT}
            and "text" not in inputs
        ):
            errors.append(f"workflow.nodes.{role.value}.inputs must include 'text'")
        if node_id is not None:
            id_owners.setdefault(node_id, []).append(f"nodes.{role.value}")
        if node_id is not None and class_name is not None:
            nodes[role] = WorkflowNode(
                id=node_id,
                class_name=class_name,
                inputs=MappingProxyType(inputs),
            )

    for role in sorted(REQUIRED_ROLES - nodes.keys(), key=lambda item: item.value):
        errors.append(f"workflow.nodes is missing required role {role.value!r}")

    raw_media_inputs = data.get("media_inputs", [])
    if not isinstance(raw_media_inputs, list):
        errors.append("workflow.media_inputs must be a list")
        raw_media_inputs = []
    media_inputs: list[WorkflowMediaInput] = []
    seen_slots: set[tuple[MediaKind, MediaSlot]] = set()
    first_frame = False
    for index, raw_item in enumerate(raw_media_inputs):
        item = _as_mapping(raw_item)
        prefix = f"workflow.media_inputs[{index}]"
        if item is None:
            errors.append(f"{prefix} must be a mapping")
            continue
        node_id = _node_id(item.get("id"), field=f"{prefix}.id", errors=errors)
        class_name = _non_blank(item.get("class"), field=f"{prefix}.class", errors=errors)
        input_name = _non_blank(item.get("input", "image"), field=f"{prefix}.input", errors=errors)
        kind = _enum(MediaKind, item.get("kind"), field=f"{prefix}.kind", errors=errors)
        slot = _enum(MediaSlot, item.get("slot"), field=f"{prefix}.slot", errors=errors)
        target_role = _enum(
            WorkflowRole,
            item.get("target_role", WorkflowRole.POSITIVE_PROMPT.value),
            field=f"{prefix}.target_role",
            errors=errors,
        )
        target_input = _non_blank(
            item.get("target_input"), field=f"{prefix}.target_input", errors=errors
        )
        if node_id is not None:
            id_owners.setdefault(node_id, []).append(f"media_inputs[{index}]")
        if kind is not None and slot is not None:
            if kind is not MEDIA_SLOT_KINDS[slot]:
                errors.append(
                    f"{prefix}.slot {slot.value!r} requires kind {MEDIA_SLOT_KINDS[slot].value!r}"
                )
            if slot is not MediaSlot.REFERENCE:
                key = (kind, slot)
                if key in seen_slots:
                    errors.append(f"workflow.media_inputs has duplicate {slot.value!r} slot")
                seen_slots.add(key)
            if slot is MediaSlot.FIRST_FRAME:
                first_frame = True
            if media is MediaKind.IMAGE and slot in {
                MediaSlot.FIRST_FRAME,
                MediaSlot.LAST_FRAME,
                MediaSlot.SOURCE,
            }:
                errors.append(f"{prefix}.slot {slot.value!r} requires media video")
        if target_role is not None:
            target_node = nodes.get(target_role)
            if target_node is None:
                errors.append(
                    f"{prefix}.target_role {target_role.value!r} is not declared in workflow.nodes"
                )
            elif target_input is not None and target_input in target_node.inputs.values():
                errors.append(
                    f"{prefix}.target_input {target_input!r} collides with a mapped "
                    f"parameter on {target_role.value}"
                )
        if all(
            value is not None
            for value in (node_id, class_name, input_name, kind, slot, target_role, target_input)
        ):
            media_inputs.append(
                WorkflowMediaInput(
                    id=cast("str", node_id),
                    class_name=cast("str", class_name),
                    input=cast("str", input_name),
                    kind=cast("MediaKind", kind),
                    slot=cast("MediaSlot", slot),
                    target_role=cast("WorkflowRole", target_role),
                    target_input=cast("str", target_input),
                )
            )
    if any(item.slot is MediaSlot.LAST_FRAME for item in media_inputs) and not first_frame:
        errors.append("workflow.media_inputs slot 'last_frame' requires a first_frame slot")

    raw_model_inputs = data.get("model_inputs", [])
    if not isinstance(raw_model_inputs, list):
        errors.append("workflow.model_inputs must be a list")
        raw_model_inputs = []
    model_inputs: list[WorkflowModelInput] = []
    for index, raw_item in enumerate(raw_model_inputs):
        item = _as_mapping(raw_item)
        prefix = f"workflow.model_inputs[{index}]"
        if item is None:
            errors.append(f"{prefix} must be a mapping")
            continue
        node_id = _node_id(item.get("id"), field=f"{prefix}.id", errors=errors)
        class_name = _non_blank(item.get("class"), field=f"{prefix}.class", errors=errors)
        input_name = _non_blank(item.get("input"), field=f"{prefix}.input", errors=errors)
        model_type = item.get("model_type")
        filename = item.get("filename")
        label = item.get("label")
        if model_type is not None:
            model_type = _non_blank(model_type, field=f"{prefix}.model_type", errors=errors)
        if filename is not None:
            filename = _non_blank(filename, field=f"{prefix}.filename", errors=errors)
        if label is not None:
            label = _non_blank(label, field=f"{prefix}.label", errors=errors)
        if model_type is None and filename is None:
            errors.append(f"{prefix} requires model_type or filename")
        if node_id is not None:
            id_owners.setdefault(node_id, []).append(f"model_inputs[{index}]")
        if (
            node_id is not None
            and class_name is not None
            and input_name is not None
            and (model_type or filename)
        ):
            model_inputs.append(
                WorkflowModelInput(
                    id=node_id,
                    class_name=class_name,
                    input=input_name,
                    model_type=model_type,
                    filename=filename,
                    label=label,
                )
            )

    for node_id, owners in sorted(id_owners.items()):
        if len(owners) > 1:
            errors.append(f"workflow node id {node_id!r} is reused by {', '.join(owners)}")
    if errors:
        raise WorkflowContractError(f"{source}: {'; '.join(errors)}")
    if media is None:
        raise WorkflowContractError(f"{source}: workflow.media is required")
    return WorkflowMap(
        contract_version=contract_version,
        media=media,
        nodes=MappingProxyType(nodes),
        media_inputs=tuple(media_inputs),
        model_inputs=tuple(model_inputs),
    )
