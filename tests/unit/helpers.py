"""Shared test helpers that are not fixtures.

Kept out of conftest.py so conftest stays fixtures-only. Helpers here are
plain functions imported directly by the tests that need them.
"""

from __future__ import annotations

from typing import Any

from src.api.services.workflow.capabilities import derive_capabilities
from src.api.services.workflow.contract import (
    BoundWorkflow,
    BundleCapabilities,
    WorkflowMap,
    WorkflowMediaInput,
    WorkflowNode,
    WorkflowRole,
)
from src.core.config import Settings
from src.core.enums import MediaKind, MediaSlot, Resolution, Sampler, Scheduler
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)


def hermetic_settings(**overrides: Any) -> Settings:
    """Construct Settings ignoring any local ``.env`` file (S4 remediation).

    ``Settings.model_config`` sets ``env_file=".env"``, so a bare ``Settings(...)``
    call in a test silently inherits whatever a developer's shell has in a local
    ``.env`` file (commonly created via ``cp .env.example .env``) — e.g.
    ``.env.example`` sets a non-empty ``NOWPAYMENTS_IPN_CALLBACK_URL``, which
    flips ``test_unset_accepted``'s meaning on a machine with that file present.
    Passing ``_env_file=None`` makes construction depend only on explicit
    overrides and real process env vars (via ``monkeypatch.setenv``), never on
    developer disk state. Use this in every config test instead of a bare
    ``Settings(...)``.
    """
    # pydantic's pyright plugin synthesizes __init__ purely from model fields,
    # so it doesn't see BaseSettings' own hand-written _env_file kwarg.
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


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
    return derive_capabilities(bound, bundle_generation_config())


def bundle_generation_config() -> BundleGenerationConfig:
    """A permissive, representative generation config for capability derivation."""
    return BundleGenerationConfig(
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
    )


# Node ids of the synthetic qwen.rapid.aio graph below.
QWEN_ENCODER_NODE = "3"
QWEN_LOAD_IMAGE_NODES = ("7", "8")


def qwen_rapid_aio_bound_workflow(reference_slots: int = 2) -> BoundWorkflow:
    """A bound workflow shaped like ``qwen.rapid.aio``: one edit encoder fed by N loaders.

    ``TextEncodeQwenImageEditPlus`` (node 3) takes ``image1``/``image2`` from two
    ``LoadImage`` nodes (7, 8). The template graph already links every declared
    slot; the applier unlinks them all and relinks only as many as it is given.
    """
    loader_ids = QWEN_LOAD_IMAGE_NODES[:reference_slots]
    media_inputs = tuple(
        WorkflowMediaInput(
            id=node_id,
            class_name="LoadImage",
            input="image",
            kind=MediaKind.IMAGE,
            slot=MediaSlot.REFERENCE,
            target_role=WorkflowRole.POSITIVE_PROMPT,
            target_input=f"image{index}",
        )
        for index, node_id in enumerate(loader_ids, start=1)
    )
    api_graph: dict[str, dict[str, Any]] = {
        QWEN_ENCODER_NODE: {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                **{f"image{i}": [node_id, 0] for i, node_id in enumerate(loader_ids, start=1)},
            },
        },
        **{
            node_id: {"class_type": "LoadImage", "inputs": {"image": f"template_{node_id}.png"}}
            for node_id in loader_ids
        },
    }
    return BoundWorkflow(
        map=WorkflowMap(
            contract_version=2,
            media=MediaKind.IMAGE,
            nodes={
                WorkflowRole.POSITIVE_PROMPT: WorkflowNode(
                    id=QWEN_ENCODER_NODE,
                    class_name="TextEncodeQwenImageEditPlus",
                    inputs={},
                )
            },
            media_inputs=media_inputs,
            model_inputs=(),
        ),
        api_graph=api_graph,
    )


def qwen_rapid_aio_capabilities(reference_slots: int = 2) -> BundleCapabilities:
    """Capabilities derived from :func:`qwen_rapid_aio_bound_workflow` (i2i max = slots)."""
    return derive_capabilities(
        qwen_rapid_aio_bound_workflow(reference_slots), bundle_generation_config()
    )
