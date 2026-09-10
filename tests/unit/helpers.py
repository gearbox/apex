"""Shared test helpers that are not fixtures.

Kept out of conftest.py so conftest stays fixtures-only. Helpers here are
plain functions imported directly by the tests that need them.
"""

from __future__ import annotations

from src.api.services.workflow.capabilities import derive_capabilities
from src.api.services.workflow.contract import (
    BoundWorkflow,
    BundleCapabilities,
    WorkflowMap,
    WorkflowMediaInput,
    WorkflowRole,
)
from src.core.enums import MediaKind, MediaSlot, Resolution, Sampler, Scheduler
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)


def aisha_video_capabilities() -> BundleCapabilities:
    """Derive the representative Aisha video contract from a bound workflow."""
    media_inputs = tuple(
        WorkflowMediaInput(
            id=slot.value,
            class_name="LoadImage",
            input="image",
            kind=MediaKind.IMAGE,
            slot=slot,
            target_role=WorkflowRole.POSITIVE_PROMPT,
            target_input=slot.value,
        )
        for slot in (MediaSlot.FIRST_FRAME, MediaSlot.LAST_FRAME)
    )
    bound = BoundWorkflow(
        map=WorkflowMap(
            contract_version=2,
            media=MediaKind.VIDEO,
            nodes={},
            media_inputs=media_inputs,
            model_inputs=(),
        ),
        api_graph={},
    )
    return derive_capabilities(
        bound,
        BundleGenerationConfig(
            defaults=GenerationDefaults(
                resolution=Resolution.STANDARD,
                steps=12,
                cfg=1.1,
                sampler=Sampler.EULER,
                scheduler=Scheduler.BETA,
                denoise=1.0,
            ),
            constraints=GenerationConstraints(
                max_megapixels=1.0,
                latent_multiple=16,
                max_edge=1536,
                min_steps=1,
                max_steps=20,
                min_cfg=0.0,
                max_cfg=30.0,
                allowed_samplers=frozenset(),
                allowed_schedulers=frozenset(),
                max_batch_size=1,
            ),
        ),
    )
