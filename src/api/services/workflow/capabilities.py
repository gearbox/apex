"""Mechanical capability derivation from a bound workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.workflow.contract import (
    PARAMETER_HAS_REQUEST_SOURCE,
    BoundWorkflow,
    BundleCapabilities,
    WorkflowMediaInput,
    WorkflowRole,
)
from src.api.services.workflow.parser import WorkflowContractError
from src.core.enums import GenerationType, MediaKind, MediaSlot
from src.core.generation_mode import GenerationModeMeta, SourceMediaConstraints

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from src.core.generation_config import BundleGenerationConfig


def _image_generation_modes(
    media_inputs: Sequence[WorkflowMediaInput],
) -> dict[GenerationType, GenerationModeMeta]:
    """Derive image-mode contracts from declared reference slots."""
    modes = {GenerationType.T2I: GenerationModeMeta()}
    if references := sum(
        item.kind is MediaKind.IMAGE and item.slot is MediaSlot.REFERENCE for item in media_inputs
    ):
        modes[GenerationType.I2I] = GenerationModeMeta(
            SourceMediaConstraints(
                min=1,
                max=references,
                media_types=frozenset({MediaKind.IMAGE}),
            )
        )
    return modes


def _video_generation_modes(
    media_inputs: Sequence[WorkflowMediaInput],
) -> dict[GenerationType, GenerationModeMeta]:
    """Derive video-mode contracts from declared frame/source slots."""
    slots = {(item.kind, item.slot) for item in media_inputs}
    modes = {GenerationType.T2V: GenerationModeMeta()}
    if (MediaKind.IMAGE, MediaSlot.FIRST_FRAME) in slots:
        modes[GenerationType.I2V] = GenerationModeMeta(
            SourceMediaConstraints(
                min=1,
                max=1,
                media_types=frozenset({MediaKind.IMAGE}),
                roles=(MediaSlot.FIRST_FRAME,),
            )
        )
        if (MediaKind.IMAGE, MediaSlot.LAST_FRAME) in slots:
            modes[GenerationType.FLF2V] = GenerationModeMeta(
                SourceMediaConstraints(
                    min=2,
                    max=2,
                    media_types=frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME, MediaSlot.LAST_FRAME),
                )
            )
    if (MediaKind.VIDEO, MediaSlot.SOURCE) in slots:
        modes[GenerationType.V2V] = GenerationModeMeta(
            SourceMediaConstraints(
                min=1,
                max=1,
                media_types=frozenset({MediaKind.VIDEO}),
                roles=(MediaSlot.SOURCE,),
            )
        )
    return modes


_GENERATION_TYPES_BY_MEDIA: Mapping[
    MediaKind, Callable[[Sequence[WorkflowMediaInput]], dict[GenerationType, GenerationModeMeta]]
] = {
    MediaKind.IMAGE: _image_generation_modes,
    MediaKind.VIDEO: _video_generation_modes,
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
    try:
        generation_modes = _GENERATION_TYPES_BY_MEDIA[bound.media](bound.map.media_inputs)
    except KeyError as exc:
        raise WorkflowContractError(
            f"no capability derivation declared for media {bound.media.value!r}"
        ) from exc
    max_batch_size = generation.constraints.max_batch_size if "latent.batch_size" in writable else 1
    return BundleCapabilities(
        media=bound.media,
        generation_modes=generation_modes,
        supports_negative_prompt=supports_negative_prompt,
        writable=writable,
        max_batch_size=max_batch_size,
    )
