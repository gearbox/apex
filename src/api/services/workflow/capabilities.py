"""Mechanical capability derivation from a bound workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.workflow.contract import (
    BoundWorkflow,
    BundleCapabilities,
    MediaSlot,
    WorkflowRole,
)
from src.core.enums import GenerationType, MediaKind

if TYPE_CHECKING:
    from src.core.generation_config import BundleGenerationConfig


def derive_capabilities(
    bound: BoundWorkflow, generation: BundleGenerationConfig
) -> BundleCapabilities:
    """Return only the capabilities a bound bundle can actually honour."""
    nodes = bound.map.nodes
    writable = frozenset(
        f"{role.value}.{parameter}" for role, node in nodes.items() for parameter in node.inputs
    )
    negative = nodes.get(WorkflowRole.NEGATIVE_PROMPT)
    supports_negative_prompt = negative is not None and "text" in negative.inputs
    slots = {(item.kind, item.slot) for item in bound.map.media_inputs}
    generation_types: set[GenerationType] = set()
    if bound.media is MediaKind.IMAGE:
        generation_types.add(GenerationType.T2I)
        if (MediaKind.IMAGE, MediaSlot.REFERENCE) in slots:
            generation_types.add(GenerationType.I2I)
    elif bound.media is MediaKind.VIDEO:
        generation_types.add(GenerationType.T2V)
        if (MediaKind.IMAGE, MediaSlot.FIRST_FRAME) in slots:
            generation_types.add(GenerationType.I2V)
            if (MediaKind.IMAGE, MediaSlot.LAST_FRAME) in slots:
                generation_types.add(GenerationType.FLF2V)
        if (MediaKind.VIDEO, MediaSlot.SOURCE) in slots:
            generation_types.add(GenerationType.V2V)
    max_batch_size = generation.constraints.max_batch_size if "latent.batch_size" in writable else 1
    return BundleCapabilities(
        media=bound.media,
        generation_types=frozenset(generation_types),
        supports_negative_prompt=supports_negative_prompt,
        writable=writable,
        max_batch_size=max_batch_size,
        max_reference_images=sum(
            item.slot is MediaSlot.REFERENCE for item in bound.map.media_inputs
        ),
    )
