"""Apply a request to a graph that has already passed workflow binding."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from src.api.services.workflow.contract import PARAMETER_ACCESSORS, PARAMETER_HAS_REQUEST_SOURCE

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from src.api.services.workflow.contract import BoundWorkflow, MediaSlot


class ModelInputResolutionError(ValueError):
    """A declared model loader cannot be supplied with a bundle filename."""


class WorkflowApplyError(ValueError):
    """A request cannot be applied to the bundle's declared workflow."""


def _select_model_filename(
    *,
    model_type: str | None,
    declared_filename: str | None,
    filenames: Callable[[str], list[str] | None],
) -> str:
    if model_type is None:
        if declared_filename is None:
            raise ModelInputResolutionError("Model input has neither model_type nor filename")
        return declared_filename
    available = filenames(model_type)
    if not available:
        raise ModelInputResolutionError(
            f"Bundle declares model input type {model_type!r}, but it has no filename"
        )
    if declared_filename is not None:
        if declared_filename not in available:
            raise ModelInputResolutionError(
                f"Bundle declares model input filename {declared_filename!r}, which is not "
                f"present in model type {model_type!r}"
            )
        return declared_filename
    if len(available) != 1:
        raise ModelInputResolutionError(
            f"Bundle model type {model_type!r} has {len(available)} filenames; "
            "workflow.model_inputs must select one with filename"
        )
    return available[0]


def apply(
    bound: BoundWorkflow,
    request: object,
    *,
    media_filenames: Mapping[MediaSlot, list[str]],
    filename_prefix: str,
    model_filenames: Callable[[str], list[str] | None],
) -> dict[str, Any]:
    """Return a configured deep copy of the exact API graph in ``bound``."""
    graph = copy.deepcopy(dict(bound.api_graph))
    for role, node in bound.map.nodes.items():
        inputs = graph[node.id]["inputs"]
        for parameter, input_name in node.inputs.items():
            key = f"{role.value}.{parameter}"
            value = PARAMETER_ACCESSORS[(role, parameter)](request, filename_prefix)
            if value is None:
                if key in PARAMETER_HAS_REQUEST_SOURCE:
                    raise WorkflowApplyError(
                        f"{key} is declared by the bundle and has a request source, "
                        f"but the request supplied no value"
                    )
                continue
            inputs[input_name] = value

    for media_input in bound.map.media_inputs:
        graph[bound.map.nodes[media_input.target_role].id]["inputs"].pop(
            media_input.target_input, None
        )
    for slot, filenames in media_filenames.items():
        declared = [item for item in bound.map.media_inputs if item.slot is slot]
        if not declared:
            raise WorkflowApplyError(
                f"slot {slot.value!r} received {len(filenames)} filename(s) "
                f"but the bundle declares no {slot.value!r} media input"
            )
        if len(filenames) > len(declared):
            raise WorkflowApplyError(
                f"slot {slot.value!r} received {len(filenames)} filenames "
                f"but the bundle declares {len(declared)}"
            )
        for index, filename in enumerate(filenames):
            media_input = declared[index]
            graph[media_input.id]["inputs"][media_input.input] = filename
            graph[bound.map.nodes[media_input.target_role].id]["inputs"][
                media_input.target_input
            ] = [media_input.id, 0]

    for model_input in bound.map.model_inputs:
        graph[model_input.id]["inputs"][model_input.input] = _select_model_filename(
            model_type=model_input.model_type,
            declared_filename=model_input.filename,
            filenames=model_filenames,
        )
    return graph
