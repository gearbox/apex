"""Mechanical capability derivation from a bound workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.workflow.contract import (
    PARAMETER_HAS_REQUEST_SOURCE,
    BoundWorkflow,
    BundleCapabilities,
    MediaSlot,
    WorkflowRole,
)
from src.api.services.workflow.parser import WorkflowContractError
from src.core.enums import GenerationType, MediaKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from src.core.generation_config import BundleGenerationConfig


def _image_generation_types(
    slots: set[tuple[MediaKind, MediaSlot]],
) -> set[GenerationType]:
    generation_types = {GenerationType.T2I}
    if (MediaKind.IMAGE, MediaSlot.REFERENCE) in slots:
        generation_types.add(GenerationType.I2I)
    return generation_types


def _video_generation_types(
    slots: set[tuple[MediaKind, MediaSlot]],
) -> set[GenerationType]:
    generation_types = {GenerationType.T2V}
    if (MediaKind.IMAGE, MediaSlot.FIRST_FRAME) in slots:
        generation_types.add(GenerationType.I2V)
        if (MediaKind.IMAGE, MediaSlot.LAST_FRAME) in slots:
            generation_types.add(GenerationType.FLF2V)
    if (MediaKind.VIDEO, MediaSlot.SOURCE) in slots:
        generation_types.add(GenerationType.V2V)
    return generation_types


_GENERATION_TYPES_BY_MEDIA: Mapping[
    MediaKind, Callable[[set[tuple[MediaKind, MediaSlot]]], set[GenerationType]]
] = {
    MediaKind.IMAGE: _image_generation_types,
    MediaKind.VIDEO: _video_generation_types,
}


def derive_capabilities(
    bound: BoundWorkflow, generation: BundleGenerationConfig
) -> BundleCapabilities:
    """Return only the capabilities a bound bundle can actually honour."""
    nodes = bound.map.nodes
    declared_writable = {
        f"{role.value}.{parameter}" for role, node in nodes.items() for parameter in node.inputs
    }
    writable = frozenset(declared_writable & PARAMETER_HAS_REQUEST_SOURCE)
    negative = nodes.get(WorkflowRole.NEGATIVE_PROMPT)
    supports_negative_prompt = negative is not None and "text" in negative.inputs
    slots = {(item.kind, item.slot) for item in bound.map.media_inputs}
    try:
        generation_types = _GENERATION_TYPES_BY_MEDIA[bound.media](slots)
    except KeyError as exc:
        raise WorkflowContractError(
            f"no capability derivation declared for media {bound.media.value!r}"
        ) from exc
    max_batch_size = generation.constraints.max_batch_size if "latent.batch_size" in writable else 1
    return BundleCapabilities(
        media=bound.media,
        generation_types=frozenset(generation_types),
        supports_negative_prompt=supports_negative_prompt,
        writable=writable,
        max_batch_size=max_batch_size,
        max_reference_images=sum(
            item.kind is MediaKind.IMAGE and item.slot is MediaSlot.REFERENCE
            for item in bound.map.media_inputs
        ),
    )
