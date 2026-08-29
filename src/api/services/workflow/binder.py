"""Bind a parsed workflow map to the API graph shipped by its bundle."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from src.api.services.workflow.contract import BoundWorkflow, WorkflowMap, WorkflowRole
from src.api.services.workflow.parser import WorkflowContractError


def _is_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def bind_workflow(map_: WorkflowMap, api_graph: Mapping[str, object]) -> BoundWorkflow:
    """Prove every map address exists in an API graph before it is usable."""
    errors: list[str] = []
    graph: dict[str, Mapping[str, Any]] = {}
    for raw_id, raw_node in api_graph.items():
        node_id = str(raw_id)
        if not isinstance(raw_node, Mapping):
            errors.append(f"workflow API node {node_id!r} must be a mapping")
            continue
        graph[node_id] = cast("Mapping[str, Any]", raw_node)

    def node_inputs(node_id: str, owner: str) -> Mapping[str, Any] | None:
        node = graph.get(node_id)
        if node is None:
            errors.append(f"{owner} declares node {node_id!r}, absent from workflow API graph")
            return None
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            errors.append(f"workflow API node {node_id!r}.inputs must be a mapping")
            return None
        return cast("Mapping[str, Any]", inputs)

    for role, declared in map_.nodes.items():
        node = graph.get(declared.id)
        if node is None:
            errors.append(
                f"workflow.nodes.{role.value} id {declared.id!r} is absent from workflow API graph"
            )
            continue
        if node.get("class_type") != declared.class_name:
            errors.append(
                f"workflow.nodes.{role.value} id {declared.id!r} class {declared.class_name!r} "
                f"does not match API class_type {node.get('class_type')!r}"
            )
        inputs = node_inputs(declared.id, f"workflow.nodes.{role.value}")
        if inputs is not None:
            for parameter, input_name in declared.inputs.items():
                if input_name not in inputs:
                    errors.append(
                        f"workflow.nodes.{role.value}.inputs.{parameter} names absent API input "
                        f"{input_name!r}"
                    )

    if WorkflowRole.SAVE not in map_.nodes:
        errors.append("workflow.nodes must include the save role for Apex output collection")

    for index, media_declared in enumerate(map_.media_inputs):
        owner = f"workflow.media_inputs[{index}]"
        node = graph.get(media_declared.id)
        if node is None:
            errors.append(f"{owner} id {media_declared.id!r} is absent from workflow API graph")
        else:
            if node.get("class_type") != media_declared.class_name:
                errors.append(
                    f"{owner} id {media_declared.id!r} class {media_declared.class_name!r} does not match "
                    f"API class_type {node.get('class_type')!r}"
                )
            inputs = node_inputs(media_declared.id, owner)
            if inputs is not None and media_declared.input not in inputs:
                errors.append(f"{owner}.input names absent API input {media_declared.input!r}")
        target = map_.nodes.get(media_declared.target_role)
        if target is None:
            errors.append(f"{owner}.target_role {media_declared.target_role.value!r} is absent")
            continue
        target_inputs = node_inputs(target.id, f"workflow.nodes.{media_declared.target_role.value}")
        if target_inputs is None:
            continue
        target_value = target_inputs.get(media_declared.target_input)
        if media_declared.target_input not in target_inputs:
            errors.append(
                f"{owner}.target_input names absent API input {media_declared.target_input!r}"
            )
        elif not _is_link(target_value):
            errors.append(
                f"{owner}.target_input {media_declared.target_input!r} must hold a graph link"
            )

    for index, model_declared in enumerate(map_.model_inputs):
        owner = f"workflow.model_inputs[{index}]"
        node = graph.get(model_declared.id)
        if node is None:
            errors.append(f"{owner} id {model_declared.id!r} is absent from workflow API graph")
            continue
        if node.get("class_type") != model_declared.class_name:
            errors.append(
                f"{owner} id {model_declared.id!r} class {model_declared.class_name!r} does not match "
                f"API class_type {node.get('class_type')!r}"
            )
        inputs = node_inputs(model_declared.id, owner)
        if inputs is not None and model_declared.input not in inputs:
            errors.append(f"{owner}.input names absent API input {model_declared.input!r}")

    if errors:
        raise WorkflowContractError("workflow bind failed: " + "; ".join(errors))
    return BoundWorkflow(map=map_, api_graph=MappingProxyType(graph))
